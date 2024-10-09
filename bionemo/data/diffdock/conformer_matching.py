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


import copy

import networkx as nx
import numpy as np
from nemo.utils import logging
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.AllChem import AlignMol as RMSD
from scipy.optimize import differential_evolution


"""
    Conformer matching routines from Torsional Diffusion
"""


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol


def optimize_rotatable_bonds(
    mol,
    true_mol,
    rotable_bonds,
    probe_id=-1,
    ref_id=-1,
    seed=0,
    popsize=15,
    maxiter=500,
    mutation=(0.5, 1),
    recombination=0.8,
):
    opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    result = differential_evolution(
        opt.score_conformation,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        disp=False,
        seed=seed,
    )
    opt_mol = apply_changes(opt.mol, result["x"], opt.rotable_bonds, conf_id=probe_id)

    return opt_mol


class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return RMSD(self.mol, self.true_mol, self.probe_id, self.ref_id)


def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2:
            continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append((n0[0], e[0], e[1], n1[0]))
    return torsions_list, nx.is_connected(G)


# GeoMol
def get_torsions(mol_list):
    logging.info("USING GEOMOL GET TORSIONS FUNCTION")
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if b1.GetIdx() == bond.GetIdx():
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if idx4 == idx1:
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter)
                        )
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter)
                        )
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def A_transpose_matrix(alpha):
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]], dtype=np.double)


def S_vec(alpha):
    return np.array([[np.cos(alpha)], [np.sin(alpha)]], dtype=np.double)


def GetDihedralFromPointCloud(Z, atom_idx):
    p = Z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum("...i,...i", v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)


def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2, 1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.arctan2(v[1], v[0])


def get_von_mises_rms(mol, mol_rdkit, rotable_bonds, conf_id):
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(
            mol_rdkit, mol_rdkit.GetConformer(conf_id), r, mol.GetConformer().GetPositions()
        )
    mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
    return RMSD(mol_rdkit, mol, conf_id)


def mmff_func(mol):
    mol_mmff = copy.deepcopy(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol_mmff, mmffVariant="MMFF94s")
    for i in range(mol.GetNumConformers()):
        coords = mol_mmff.GetConformers()[i].GetPositions()
        for j in range(coords.shape[0]):
            mol.GetConformer(i).SetAtomPosition(j, Geometry.Point3D(*coords[j]))
