# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Dict, List, Optional, Tuple

import esm
import numpy as np
import pandas as pd
import torch
from esm import pretrained
from esm.data import FastaBatchedDataset
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

from bionemo.data.diffdock.docking_dataset import read_mol
from bionemo.data.diffdock.embedding_preprocess import compute_ESM_embeddings, get_sequences_from_pdbfile
from bionemo.data.diffdock.process_mols import (
    extract_receptor_structure,
    generate_conformer,
    get_lig_graph_with_matching,
    get_rec_graph,
    parse_pdb_from_path,
    read_molecule,
)
from bionemo.model.molecule.diffdock.utils.utils import get_symmetry_rmsd


def set_nones(l):
    return [s if str(s) != "nan" else None for s in l]


def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile([None, protein_files[i]], to_fasta=False))
        else:
            new_sequences.append(protein_sequences[i])
    return new_sequences


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory on chunk_size", chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


class InferenceDataset(Dataset):
    def __init__(
        self,
        out_dir,
        complex_names,
        protein_files,
        ligand_descriptions,
        protein_sequences,
        lm_embeddings,
        receptor_radius=30,
        c_alpha_max_neighbors=None,
        precomputed_lm_embeddings=None,
        remove_hs=False,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        seed=None,
    ):
        super(InferenceDataset, self).__init__()
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors

        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences

        self.seed = seed

        # generate LM embeddings
        if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = get_sequences(protein_files, protein_sequences)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(":")
                sequences.extend(s)
                labels.extend([complex_names[i] + "_chain_" + str(j) for j in range(len(s))])

            dataset = FastaBatchedDataset(labels, sequences)
            lm_embeddings = compute_ESM_embeddings(model, alphabet, dataset)

            self.lm_embeddings = []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(":")
                self.lm_embeddings.append(
                    [lm_embeddings[complex_names[i] + "_chain_" + str(j)] for j in range(len(s))]
                )

        elif not lm_embeddings:
            self.lm_embeddings = [None] * len(self.complex_names)

        else:
            self.lm_embeddings = precomputed_lm_embeddings

        # generate structures with ESMFold
        if None in protein_files:
            print("generating missing structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()

            for i in range(len(protein_files)):
                if protein_files[i] is None:
                    self.protein_files[i] = f"{out_dir}/{complex_names[i]}/{complex_names[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(model, self.protein_files[i], protein_sequences[i])

    def len(self):
        return len(self.complex_names)

    def get(self, idx):
        name, protein_file, ligand_description, lm_embedding = (
            self.complex_names[idx],
            self.protein_files[idx],
            self.ligand_descriptions[idx],
            self.lm_embeddings[idx],
        )

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph["name"] = name

        # parse the ligand, either from file or smile
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path

            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol, seed=self.seed)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception("RDKit could not read the molecule ", ligand_description)
                mol.RemoveAllConformers()
                mol = MolFromSmiles(
                    MolToSmiles(mol)
                )  # To avoid code freeze in the following generate_conformer() with using certain sdf files, convert to smiles and reload the molecule object
                mol = AddHs(mol)
                generate_conformer(mol, seed=self.seed)
        except Exception as e:
            print(
                "Failed to read molecule ", ligand_description, " We are skipping it. The reason is the exception: ", e
            )
            complex_graph["success"] = False
            return complex_graph

        try:
            # parse the receptor from the pdb file
            rec_model = parse_pdb_from_path(protein_file)
            get_lig_graph_with_matching(
                mol,
                complex_graph,
                popsize=None,
                maxiter=None,
                matching=False,
                keep_original=False,
                num_conformers=1,
                remove_hs=self.remove_hs,
                seed=self.seed,
            )
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(
                rec_model, mol, lm_embedding_chains=lm_embedding
            )
            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                print(
                    f"LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}."
                )
                complex_graph["success"] = False
                return complex_graph

            get_rec_graph(
                rec,
                rec_coords,
                c_alpha_coords,
                n_coords,
                c_coords,
                complex_graph,
                rec_radius=self.receptor_radius,
                c_alpha_max_neighbors=self.c_alpha_max_neighbors,
                all_atoms=self.all_atoms,
                atom_radius=self.atom_radius,
                atom_max_neighbors=self.atom_max_neighbors,
                remove_hs=self.remove_hs,
                lm_embeddings=lm_embeddings,
            )

        except Exception as e:
            print(f"Skipping {name} because of the error:")
            print(e)
            complex_graph["success"] = False
            return complex_graph

        protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
        complex_graph["receptor"].pos -= protein_center
        if self.all_atoms:
            complex_graph["atom"].pos -= protein_center

        ligand_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        complex_graph["ligand"].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = mol
        complex_graph["success"] = True
        return complex_graph


def build_inference_datasets(cfg: DictConfig) -> Tuple[List, Dataset, Dataset, DataLoader]:
    """Preprocess the conf arguments and build the datasets and dataloader for inference.

    Args:
        cfg (DictConfig): Inference conf

    Returns:
        Tuple[List, Dataset, Dataset, DataLoader]: List of complex names for doing inferences,
                                                   Dataset with the complex graphs for score model,
                                                   Dataset with the complex graphs for the confidence model,
                                                   Dataloader for the inference.
    """
    if cfg.protein_ligand_csv is not None:
        df = pd.read_csv(cfg.protein_ligand_csv)
        complex_name_list = set_nones(df["complex_name"].tolist())
        protein_path_list = set_nones(df["protein_path"].tolist())
        protein_sequence_list = set_nones(df["protein_sequence"].tolist())
        ligand_description_list = set_nones(df["ligand_description"].tolist())
    else:
        complex_name_list = [cfg.complex_name]
        protein_path_list = [cfg.protein_path]
        protein_sequence_list = [cfg.protein_sequence]
        ligand_description_list = [cfg.ligand_description]

    complex_name_list = [f"{name}" if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]

    score_model_cfg = cfg.score_infer

    # preprocessing of complexes into geometric graphs
    test_dataset = InferenceDataset(
        out_dir=cfg.out_dir,
        complex_names=complex_name_list,
        protein_files=protein_path_list,
        ligand_descriptions=ligand_description_list,
        protein_sequences=protein_sequence_list,
        lm_embeddings=True,
        receptor_radius=score_model_cfg.data.receptor_radius,
        remove_hs=score_model_cfg.data.remove_hs,
        c_alpha_max_neighbors=score_model_cfg.data.c_alpha_max_neighbors,
        all_atoms=score_model_cfg.data.all_atoms,
        atom_radius=score_model_cfg.data.atom_radius,
        atom_max_neighbors=score_model_cfg.data.atom_max_neighbors,
        seed=cfg.seed,
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if cfg.confidence_infer is not None:
        confidence_model_cfg = cfg.confidence_infer
        if not (
            confidence_model_cfg.use_original_model_cache or confidence_model_cfg.transfer_weights
        ):  # if the confidence model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            logging.info(
                "HAPPENING | confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now."
            )
            confidence_test_dataset = InferenceDataset(
                out_dir=cfg.out_dir,
                complex_names=complex_name_list,
                protein_files=protein_path_list,
                ligand_descriptions=ligand_description_list,
                protein_sequences=protein_sequence_list,
                lm_embeddings=True,
                receptor_radius=confidence_model_cfg.data.receptor_radius,
                remove_hs=confidence_model_cfg.data.remove_hs,
                c_alpha_max_neighbors=confidence_model_cfg.data.c_alpha_max_neighbors,
                all_atoms=confidence_model_cfg.data.all_atoms,
                atom_radius=confidence_model_cfg.data.atom_radius,
                atom_max_neighbors=confidence_model_cfg.data.atom_max_neighbors,
                precomputed_lm_embeddings=test_dataset.lm_embeddings,
                seed=cfg.seed,
            )
    else:
        confidence_test_dataset = None

    return complex_name_list, test_dataset, confidence_test_dataset, test_loader


def compute_rmsd(
    cfg: DictConfig, name: str, results_path_containments: List, ref_ligand: Optional[str] = None
) -> Tuple[List, np.ndarray, np.ndarray]:
    """Read the reference and predicted ligand molecules and compute RMSDs.

    Args:
        cfg (DictConfig): Inference conf
        name (str): pdbid, used as the folder name for the reference pdb and also the prediction results
        results_path_containments (List): List of the directory with names of predicted pdb.
        ref_ligand (optional, str): path to the ligand file

    Raises:
        Exception: If can't find the pdb with the given name,
                   or can read the molecule sdf file,
                   or rdkit can not process the molecule.

    Returns:
        Tuple[List, np.ndarray, np.ndarray]: List with computed RMSDs or the given pdb name,
                                             predicted ligand positions, [num. of predictions, num. of atoms, 3]
                                             reference ligand positions, [num. of atoms, 3]
    """
    if ref_ligand is not None:
        mol = read_molecule(ref_ligand, remove_hs=True, sanitize=True)
    else:
        mol = read_mol(cfg.protein_data_dir, name, remove_hs=True)
    mol = Chem.RemoveAllHs(mol)
    orig_ligand_pos = np.array(mol.GetConformer().GetPositions())

    if cfg.all_dirs_in_results:
        directory_with_name = [directory for directory in results_path_containments if name in directory][0]
        ligand_pos = []
        debug_paths = []
        for i in range(cfg.num_predictions):
            file_paths = sorted(os.listdir(os.path.join(cfg.results_path, directory_with_name)))
            if cfg.file_to_exclude is not None:
                file_paths = [path for path in file_paths if cfg.file_to_exclude not in path]
            try:
                file_path = [path for path in file_paths if f"rank{i+1}_" in path][0]
                mol_pred = read_molecule(
                    os.path.join(cfg.results_path, directory_with_name, file_path), remove_hs=True, sanitize=True
                )
                mol_pred = Chem.RemoveAllHs(mol_pred)
            except Exception as e:
                logging.warning(
                    f"In {name} rank={i+1}, error encountered: {e}, " + f"Please redo inference with {name}"
                )
                return None, None, None

            ligand_pos.append(mol_pred.GetConformer().GetPositions())
            debug_paths.append(file_path)
        ligand_pos = np.asarray(ligand_pos)
    else:
        if not os.path.exists(
            os.path.join(cfg.results_path, name, f'{"" if cfg.no_id_in_filename else name}{cfg.file_suffix}')
        ):
            raise Exception(
                "path did not exists:",
                os.path.join(cfg.results_path, name, f'{"" if cfg.no_id_in_filename else name}{cfg.file_suffix}'),
            )
        mol_pred = read_molecule(
            os.path.join(cfg.results_path, name, f'{"" if cfg.no_id_in_filename else name}{cfg.file_suffix}'),
            remove_hs=True,
            sanitize=True,
        )
        if mol_pred is None:
            logging.warning(f"Skipping {name} because RDKIT could not read it.")
            return None, None, None
        mol_pred = Chem.RemoveAllHs(mol_pred)
        ligand_pos = np.asarray(
            [np.array(mol_pred.GetConformer(i).GetPositions()) for i in range(cfg.num_predictions)]
        )
    try:
        rmsd = get_symmetry_rmsd(mol, orig_ligand_pos, list(ligand_pos), mol_pred)
    except Exception as e:
        logging.warning(f"Using non corrected RMSD for {name} because of the error: {e}")
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

    return rmsd, ligand_pos, orig_ligand_pos


def compute_performance_metrics(
    cfg: DictConfig,
    rmsds_list: List,
    min_cross_distances_list: List,
    min_self_distances_list: List,
    centroid_distances_list: List,
    successful_names_list: List,
    without_rec_overlap_list: List,
    save_results: bool = True,
) -> Dict:
    """Compute the performance metrics given the prediction results. Used inside the evaluate_files.py

    Args:
        cfg (DictConfig): Inference conf
        rmsds_list (List): List of RMSDs from predicted ligand conformers
        min_cross_distances_list (List): list of minimal cross distance between ligand and receptor.
        min_self_distances_list (List): list of minimal self distance in ligand.
        centroid_distances_list (List): List of distance between the centers of predicted and reference ligands
        successful_names_list (List): List of complex name with inference successful
        without_rec_overlap_list (List): List of 1 or 0 to indicate if the names
            are in the list of names_no_rec_overlap specified in cfg
        save_results (bool, optional): If save output numpy array to result path. Defaults to True.

    Returns:
        Dict: A dictionary of performance metrics for top 1, top 5, top 10 predictions.
    """
    performance_metrics = {}
    for overlap in ["", "no_overlap_"]:
        if "no_overlap_" == overlap:
            without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
            rmsds = np.array(rmsds_list)[without_rec_overlap]
            centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
            min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
            min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
            successful_names = np.array(successful_names_list)[without_rec_overlap]
        else:
            rmsds = np.array(rmsds_list)
            centroid_distances = np.array(centroid_distances_list)
            min_cross_distances = np.array(min_cross_distances_list)
            min_self_distances = np.array(min_self_distances_list)
            successful_names = np.array(successful_names_list)

        if save_results:
            np.save(os.path.join(cfg.results_path, f"{overlap}rmsds.npy"), rmsds)
            np.save(os.path.join(cfg.results_path, f"{overlap}names.npy"), successful_names)
            np.save(os.path.join(cfg.results_path, f"{overlap}min_cross_distances.npy"), np.array(min_cross_distances))
            np.save(os.path.join(cfg.results_path, f"{overlap}min_self_distances.npy"), np.array(min_self_distances))
            np.save(os.path.join(cfg.results_path, f"{overlap}centroid_distances.npy"), np.array(centroid_distances))

        performance_metrics.update(
            {
                f"{overlap}steric_clash_fraction": (
                    100 * (min_cross_distances < 0.4).sum() / len(min_cross_distances) / cfg.num_predictions
                ).__round__(2),
                f"{overlap}self_intersect_fraction": (
                    100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / cfg.num_predictions
                ).__round__(2),
                f"{overlap}mean_rmsd": rmsds[:, 0].mean(),
                f"{overlap}rmsds_below_2": (100 * (rmsds[:, 0] < 2).sum() / len(rmsds[:, 0])),
                f"{overlap}rmsds_below_5": (100 * (rmsds[:, 0] < 5).sum() / len(rmsds[:, 0])),
                f"{overlap}rmsds_percentile_25": np.percentile(rmsds[:, 0], 25).round(2),
                f"{overlap}rmsds_percentile_50": np.percentile(rmsds[:, 0], 50).round(2),
                f"{overlap}rmsds_percentile_75": np.percentile(rmsds[:, 0], 75).round(2),
                f"{overlap}mean_centroid": centroid_distances[:, 0].mean().__round__(2),
                f"{overlap}centroid_below_2": (
                    100 * (centroid_distances[:, 0] < 2).sum() / len(centroid_distances[:, 0])
                ).__round__(2),
                f"{overlap}centroid_below_5": (
                    100 * (centroid_distances[:, 0] < 5).sum() / len(centroid_distances[:, 0])
                ).__round__(2),
                f"{overlap}centroid_percentile_25": np.percentile(centroid_distances[:, 0], 25).round(2),
                f"{overlap}centroid_percentile_50": np.percentile(centroid_distances[:, 0], 50).round(2),
                f"{overlap}centroid_percentile_75": np.percentile(centroid_distances[:, 0], 75).round(2),
            }
        )

        top5_rmsds = np.min(rmsds[:, :5], axis=1)
        top5_centroid_distances = centroid_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
        ][:, 0]
        top5_min_cross_distances = min_cross_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
        ][:, 0]
        top5_min_self_distances = min_self_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
        ][:, 0]
        performance_metrics.update(
            {
                f"{overlap}top5_steric_clash_fraction": (
                    100 * (top5_min_cross_distances < 0.4).sum() / len(top5_min_cross_distances)
                ).__round__(2),
                f"{overlap}top5_self_intersect_fraction": (
                    100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)
                ).__round__(2),
                f"{overlap}top5_rmsds_below_2": (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
                f"{overlap}top5_rmsds_below_5": (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
                f"{overlap}top5_rmsds_percentile_25": np.percentile(top5_rmsds, 25).round(2),
                f"{overlap}top5_rmsds_percentile_50": np.percentile(top5_rmsds, 50).round(2),
                f"{overlap}top5_rmsds_percentile_75": np.percentile(top5_rmsds, 75).round(2),
                f"{overlap}top5_centroid_below_2": (
                    100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)
                ).__round__(2),
                f"{overlap}top5_centroid_below_5": (
                    100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)
                ).__round__(2),
                f"{overlap}top5_centroid_percentile_25": np.percentile(top5_centroid_distances, 25).round(2),
                f"{overlap}top5_centroid_percentile_50": np.percentile(top5_centroid_distances, 50).round(2),
                f"{overlap}top5_centroid_percentile_75": np.percentile(top5_centroid_distances, 75).round(2),
            }
        )

        top10_rmsds = np.min(rmsds[:, :10], axis=1)
        top10_centroid_distances = centroid_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
        ][:, 0]
        top10_min_cross_distances = min_cross_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
        ][:, 0]
        top10_min_self_distances = min_self_distances[
            np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
        ][:, 0]
        performance_metrics.update(
            {
                f"{overlap}top10_self_intersect_fraction": (
                    100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)
                ).__round__(2),
                f"{overlap}top10_steric_clash_fraction": (
                    100 * (top10_min_cross_distances < 0.4).sum() / len(top10_min_cross_distances)
                ).__round__(2),
                f"{overlap}top10_rmsds_below_2": (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
                f"{overlap}top10_rmsds_below_5": (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
                f"{overlap}top10_rmsds_percentile_25": np.percentile(top10_rmsds, 25).round(2),
                f"{overlap}top10_rmsds_percentile_50": np.percentile(top10_rmsds, 50).round(2),
                f"{overlap}top10_rmsds_percentile_75": np.percentile(top10_rmsds, 75).round(2),
                f"{overlap}top10_centroid_below_2": (
                    100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)
                ).__round__(2),
                f"{overlap}top10_centroid_below_5": (
                    100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)
                ).__round__(2),
                f"{overlap}top10_centroid_percentile_25": np.percentile(top10_centroid_distances, 25).round(2),
                f"{overlap}top10_centroid_percentile_50": np.percentile(top10_centroid_distances, 50).round(2),
                f"{overlap}top10_centroid_percentile_75": np.percentile(top10_centroid_distances, 75).round(2),
            }
        )

    with open(os.path.join(cfg.results_path, "performance_metrics.txt"), "w") as f:
        for k in performance_metrics:
            f.write(f"{k} {performance_metrics[k]}\n")
    return performance_metrics
