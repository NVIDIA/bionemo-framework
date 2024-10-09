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


import logging
from typing import List

from rdkit import Chem

from bionemo.model.molecule.infer import MolInference
from bionemo.model.molecule.megamolbart import MegaMolBARTInference


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SMIS_FOR_TEST = [
    "c1cc2ccccc2cc1",
    "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC",
]


MAX_GEN_LEN: int = 64


def run_hidden_to_smis(inferer: MolInference, smis: List[str]):
    hidden_state, pad_masks = inferer.seq_to_hiddens(smis)
    infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks, override_generate_num_tokens=MAX_GEN_LEN)
    log.info(f"Input SMILES and Infered: {smis}, {infered_smis}")

    assert len(infered_smis) == len(smis)

    for smi, infered_smi in zip(smis, infered_smis):
        log.info(f"Input and Infered:{smi},  {infered_smi}")
        input_mol = Chem.MolFromSmiles(smi)
        infer_mol = Chem.MolFromSmiles(infered_smi)
        assert input_mol is not None

        canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
        # FIXME remove this distinction once we have a good checkpoint for MolMIM
        if isinstance(inferer, MegaMolBARTInference):
            canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
            log.info(f"Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}")
            assert canonical_smi == canonical_infered_smi
        else:
            if infer_mol is not None:
                canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
                if canonical_smi != canonical_infered_smi:
                    log.warning(f"Difference in SMILE {canonical_smi} vs {canonical_infered_smi}")
            else:
                log.warning(f"Did not get a valid smile: {infered_smi}")


def run_sample_not_beam(inferer: MolInference, smis: List[str], sampling_method: str):
    samples = inferer.sample(
        num_samples=3,
        sampling_method=sampling_method,
        scaled_radius=0,
        topk=4,
        temperature=2,
        topp=0.0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    nl = "\n"
    for smi_i, samples_i in zip(smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")
    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f"Sample: {smi},  {isvalid}")

    log.info("Valid Molecules" + "\n".join(valid_molecules))
    log.info(
        f"Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))}  valids {len(valid_molecules)}"
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


def run_beam_search(inferer: MolInference, smis: List[str], beam_search_method: str):
    num_samples = 3
    beam_size = 5
    samples = inferer.sample(
        num_samples=num_samples,
        beam_size=beam_size,  # internally beam_size will be set to num_samples
        sampling_method=beam_search_method,
        beam_alpha=0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(samples) == len(smis)
    assert len(samples[0]) == num_samples

    nl = "\n"
    for smi_i, samples_i in zip(smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")

    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f"Sample: {smi},  {isvalid}")

    log.info("Valid Molecules" + "\n".join(valid_molecules))

    log.info(
        f"Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} "
        f"valids {len(valid_molecules)}"
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    if beam_search_method != "beam-search-single-sample":
        # "beam-search-single-sample" is not very good, only one gaussian is sampled and top beam_size are sampled from that.
        # otherwise test that we get at least one valid molecule.
        assert len(valid_molecules) != 0


def run_beam_search_product(inferer: MolInference, smis: List[str]):
    num_samples = 3
    beam_size = 2
    samples = inferer.sample(
        num_samples=num_samples,
        beam_size=beam_size,
        sampling_method="beam-search-perturbate-sample",
        beam_alpha=0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    # Samples shoudl be batch (original) x num_samples x beam_size for this sampler
    assert len(samples) == len(smis)
    assert len(samples[0]) == num_samples
    assert len(samples[0][0]) == beam_size

    nl = "\n"
    for smi_i, ssamples_i in zip(smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join([nl.join(samples_i) for samples_i in ssamples_i])}\n")

    samples_flat = [item for subsublist in samples for sublist in subsublist for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f"Sample: {smi},  {isvalid}")

    log.info("Valid Molecules" + "\n".join(valid_molecules))

    log.info(
        f"Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} "
        f"valids {len(valid_molecules)}"
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


def run_interpolate(inferer: MolInference, smis: List[str], sampling_method: str):
    interpolations = inferer.interpolate_samples(
        sample1=smis[0],
        sample2=smis[1],
        num_interpolations=3,
        sampling_method=sampling_method,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(interpolations) == 3
