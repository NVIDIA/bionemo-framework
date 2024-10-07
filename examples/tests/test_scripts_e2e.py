# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
import subprocess
from glob import glob

import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.utils.tests import teardown_apex_megatron_cuda


BIONEMO_HOME = os.environ["BIONEMO_HOME"]
TEST_DATA_DIR = os.path.join(BIONEMO_HOME, "examples/tests/test_data")


@pytest.fixture
def train_args():
    return {
        "trainer.devices": torch.cuda.device_count(),
        "trainer.num_nodes": 1,
        "trainer.max_steps": 20,
        "trainer.val_check_interval": 10,
        "trainer.limit_val_batches": 2,
        "model.data.val.use_upsampling": True,
        "trainer.limit_test_batches": 1,
        "model.data.test.use_upsampling": True,
        "exp_manager.create_wandb_logger": False,
        "exp_manager.create_tensorboard_logger": False,
        "model.micro_batch_size": 2,
    }


@pytest.fixture
def data_args():
    return {
        "model.data.dataset.train": "x000",
        "model.data.dataset.val": "x000",
        "model.data.dataset.test": "x000",
    }


# TODO: can we assume that we always run these tests from main bionemo dir?
DIRS_TO_TEST = [
    "examples/",
    "examples/molecule/megamolbart/",
    "examples/protein/downstream/",
    "examples/protein/esm1nv/",
    "examples/protein/esm2nv/",
    "examples/protein/prott5nv/",
    "examples/protein/openfold/",
    "examples/dna/dnabert/",
    "examples/molecule/diffdock/",
    "examples/molecule/molmim/",
    "examples/singlecell/geneformer/",
]

TRAIN_SCRIPTS = []
for subdir in DIRS_TO_TEST:
    TRAIN_SCRIPTS += list(glob(os.path.join(subdir, "*train*.py")))
    TRAIN_SCRIPTS += [f for f in glob(os.path.join(subdir, "downstream*.py")) if not f.endswith("test.py")]
    #  TODO add required test files for finetune to work without first manually calling preprocess.
    # TRAIN_SCRIPTS += [f for f in glob(os.path.join(subdir, 'finetune*.py')) if not f.endswith('test.py')]


INFERENCE_CONFIGS = []
for subdir in DIRS_TO_TEST:
    INFERENCE_CONFIGS += list(glob(os.path.join(subdir, "conf", "infer*yaml")))


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    DATA = " ++model.data"
    MAIN = f"{DATA}.dataset_path={TEST_DATA_DIR}/%s"
    DOWNSTREAM = f" ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s"

    root, domain, model, *conf, script = script_or_cfg_path.split("/")
    assert root == "examples" and model in (
        "megamolbart",
        "esm1nv",
        "esm2nv",
        "prott5nv",
        "downstream",
        "openfold",
        "dnabert",
        "diffdock",
        "molmim",
        "geneformer",
    ), "update this function, patterns might be wrong"

    task = {
        "molecule": "physchem/SAMPL",
        "protein": "flip",
        "dna": "downstream",
        "singlecell": "downstream",
    }

    train_data_folder = {
        "molecule": "zinc15",
        "protein": "uniref50",
    }

    if model == "geneformer":
        sample_data_path = f"{TEST_DATA_DIR}/singlecell/cellxgene_2023-12-15_small/processed_data"
        return (
            # This is what we run inference on when running infer.py. This is not checked or used during pretraining.
            f" {DATA}.dataset_path={sample_data_path}/test"
            # The following three paths are used for pretrain.py, but also are required to support model loading currently when running inference.
            f" {DATA}.train_dataset_path={sample_data_path}/train"
            f" {DATA}.val_dataset_path={sample_data_path}/val"
            f" {DATA}.test_dataset_path={sample_data_path}/test"
        )
    if conf == ["conf"]:
        if model in ("megamolbart", "openfold", "molmim"):
            return ""
        else:
            return MAIN % f"{domain}/{task[domain]}/test/x000"

    if "retro" in script:
        return MAIN % f"{domain}/uspto50k"
    elif model == "openfold":
        return MAIN % f"{domain}/{model}"
    elif model == "diffdock":
        return (
            f" ++data.split_train={TEST_DATA_DIR}/molecule/diffdock/splits/split_train"
            f" ++data.split_val={TEST_DATA_DIR}/molecule/diffdock/splits/split_train"
            f" ++data.split_test={TEST_DATA_DIR}/molecule/diffdock/splits/split_train"
            f" ++data.cache_path={TEST_DATA_DIR}/molecule/diffdock/data_cache"
        )
    elif "downstream" in script:
        if model == "dnabert":
            fasta_directory = os.path.join(TEST_DATA_DIR, "dna/downstream")
            fasta_pattern = fasta_directory + "/test-chr1.fa"
            splicesite_overrides = (
                f"++model.data.fasta_directory={fasta_directory} "
                "++model.data.fasta_pattern=" + fasta_pattern + " "
                f"++model.data.train_file={fasta_directory}/train.csv "
                f"++model.data.val_file={fasta_directory}/val.csv "
                f"++model.data.predict_file={fasta_directory}/test.csv "
            )
            return splicesite_overrides
        else:
            return MAIN % f"{domain}/{task[domain]}"
    elif model == "dnabert":
        DNABERT_TEST_DATA_DIR = os.path.join(TEST_DATA_DIR, domain)
        sample_filename = "chr1-test.fa"
        dnabert_overrides = (
            f"++model.data.dataset_path={DNABERT_TEST_DATA_DIR} "
            f"++model.data.dataset.train={sample_filename} "
            f"++model.data.dataset.val={sample_filename} "
            f"++model.data.dataset.test={sample_filename} "
        )
        return dnabert_overrides
    elif model == "esm2nv" and "infer" not in script:
        # TODO(dorotat) Simplify this case when data-related utils for ESM2 are refactored
        UNIREF_FOLDER = "protein/uniref50_90/"
        MAIN = f"{DATA}.train.dataset_path={TEST_DATA_DIR}/%s"
        esm2_overwrites = (
            MAIN % f"{UNIREF_FOLDER}/uf50"
            + " do_preprocessing=False"
            + f"{DATA}.train.cluster_mapping_tsv={TEST_DATA_DIR}/{UNIREF_FOLDER}/mapping.tsv"
            f"{DATA}.train.index_mapping_dir={TEST_DATA_DIR}/{UNIREF_FOLDER}"
            f"{DATA}.train.uf90.uniref90_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf90/"
            f"{DATA}.val.dataset_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf50/"
            f"{DATA}.test.dataset_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf50/" + DOWNSTREAM % f"{domain}/{task[domain]}"
        )
        return esm2_overwrites

    else:
        return (MAIN + DOWNSTREAM) % (f"{domain}/{train_data_folder[domain]}/processed", f"{domain}/{task[domain]}")


def get_train_args_overrides(script_or_cfg_path, train_args):
    root, domain, model, *conf, script = script_or_cfg_path.split("/")
    if model == "openfold":
        # FIXME: provide even smaller data sample or do not generate MSA features
        pytest.skip(reason="CI infrastructure is too limiting")
        train_args["model.micro_batch_size"] = 1
        train_args["model.train_ds.num_workers"] = 1
        train_args["model.train_sequence_crop_size"] = 32
        # do not use kalign as it requires third-party-download and it not essential for testing
        train_args["model.data.realign_when_required"] = False
    elif model == "diffdock":
        # Use size aware batching, and set the size control to default
        train_args["trainer.devices"] = 1
        train_args["model.micro_batch_size"] = 2
        train_args["model.estimate_memory_usage.maximal"] = "null"
        train_args["model.max_total_size"] = "null"
        train_args["model.train_ds.num_workers"] = 1
        train_args["model.validation_ds.num_workers"] = 1

    return train_args


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("script_path", TRAIN_SCRIPTS)
def test_train_scripts(script_path, train_args, data_args, tmp_path):
    data_str = get_data_overrides(script_path)
    train_args = get_train_args_overrides(script_path, train_args)
    # Lookup a free socket to fix errors with DDP on a single node, e.g. this pytest.
    # TODO(@cye): Why did this solution regress in PyTorch Lightning?
    open_port = find_free_network_port()
    cmd = (
        f"export MASTER_PORT={open_port} && python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} "
        + " ".join(f"++{k}={v}" for k, v in train_args.items())
    )
    # TODO(dorotat) Try to simplify when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += " " + " ".join(f"++{k}={v}" for k, v in data_args.items())
    print(cmd)
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode("utf-8")
    teardown_apex_megatron_cuda()
    assert process_handle.returncode == 0, f"Command failed:\n{cmd}\n Error log:\n{error_out}"

    if "esm" in script_path and train_args["trainer.devices"] > 1:
        # Additional check for training an ESM model with pipeline parallel and val-in-loop
        cmd += f" ++model.dwnstr_task_validation.enabled=True ++model.pipeline_model_parallel_size=2 ++exp_manager.exp_dir={tmp_path}-pipeline"
        print(f"Pipeline Parallel command:\n {cmd}")
        process_handle = subprocess.run(cmd, shell=True, capture_output=True)
        error_out = process_handle.stderr.decode("utf-8")
        assert process_handle.returncode == 0, f"Command failed:\n{cmd}\n Error log:\n{error_out}"


def get_infer_args_overrides(config_path, tmp_path):
    if "openfold" in config_path:
        return {
            # cropped 7YVT_B  # cropped 7ZHL
            # predicting on longer sequences will result in CUDA OOM.
            # TODO: if preparing MSA is to be tested, the model has to be further scaled down
            "sequences": r"\['GASTATVGRWMGPAEYQQMLDTGTVVQSSTGTTHVAYPAD','MTDSIKTLSAHRSFGGVQHFHEHASREIGLPMRFAAYLPP'\]"
        }
    if "diffdock" in config_path:
        return {
            # save the inference results to tmp_path.
            "out_dir": f"{tmp_path}",
        }
    return {}


@pytest.mark.needs_fork
@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize("config_path", INFERENCE_CONFIGS)
def test_infer_script(config_path, tmp_path):
    if "dnabert" in config_path:
        # Inference scripts make assumptions that are not met for DNABERT.
        return
    config_dir, config_name = os.path.split(config_path)
    script_path = os.path.join(os.path.dirname(config_dir), "infer.py")
    infer_args = get_infer_args_overrides(config_path, tmp_path)
    if not os.path.exists(script_path):
        script_path = "bionemo/model/infer.py"
    # Lookup a free socket to fix errors with DDP on a single node, e.g. this pytest.
    # TODO(@cye): Why does this depend on how `pytest` is executed, i.e. with a single vs. multiple tests?
    # Some tests succeed when run in batch / fail otherwise, other tests succeed when run alone / fail otherwise.
    open_port = find_free_network_port()
    cmd: str = (
        f"export MASTER_PORT={open_port} && python {script_path} --config-dir {config_dir} --config-name {config_name} ++exp_manager.exp_dir={tmp_path} "
        + " ".join(f"++{k}={v}" for k, v in infer_args.items())
    )

    # FIXME: WARs for unavailable checkpoints
    if "retro" in config_path:
        model_checkpoint_path = os.path.join(BIONEMO_HOME, "models/molecule/megamolbart/megamolbart.nemo")
        cmd += f" model.downstream_task.restore_from_path={model_checkpoint_path}"

    cmd += get_data_overrides(config_path)
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode("utf-8")
    teardown_apex_megatron_cuda()
    assert process_handle.returncode == 0, f"Command failed:\n{cmd}\n Error log:\n{error_out}"
