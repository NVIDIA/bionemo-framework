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
from typing import List, TypedDict

import pytest
import torch


BIONEMO_HOME = os.environ["BIONEMO_HOME"]
TEST_DATA_DIR = os.path.join(BIONEMO_HOME, "examples/tests/test_data")


class StopAndGoTestParams(TypedDict):
    script_path: str
    metadata_keys: List[str]


# possible values for metadata_list defined in getter_function_map in bionemo/callbacks/testing_callbacks.py
# new values can be defined there by adding new getter functions
TEST_PARAMS: List[StopAndGoTestParams] = [
    {"script_path": "examples/protein/openfold/train.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/dna/dnabert/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/singlecell/geneformer/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/molecule/molmim/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/protein/esm2nv/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
]

TRAINING_SCRIPTS_PATH = [params["script_path"] for params in TEST_PARAMS]
METADATA_LIST = [params["metadata_keys"] for params in TEST_PARAMS]


@pytest.fixture
def train_args():
    return {
        'trainer.devices': torch.cuda.device_count(),
        'trainer.num_nodes': 1,
        'trainer.max_steps': 8,
        'trainer.val_check_interval': 2,  # check validation set every 2 training batches
        'trainer.limit_val_batches': 1,  # run validation for 2 validation batches
        'model.data.val.use_upsampling': False,
        'trainer.limit_test_batches': 1,
        'model.data.test.use_upsampling': True,
        'exp_manager.create_wandb_logger': False,
        'exp_manager.create_tensorboard_logger': False,
        'model.micro_batch_size': 2,
    }


@pytest.fixture
def data_args():
    return {
        'model.data.dataset.train': 'x000',
        'model.data.dataset.val': 'x000',
        'model.data.dataset.test': 'x000',
    }


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    DATA = " ++model.data"
    MAIN = f'{DATA}.dataset_path={TEST_DATA_DIR}/%s'
    DOWNSTREAM = f' ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s'

    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    assert root == 'examples' and model in (
        'megamolbart',
        'esm1nv',
        'esm2nv',
        'prott5nv',
        'downstream',
        'openfold',
        'dnabert',
        'diffdock',
        'molmim',
        'geneformer',
    ), 'update this function, patterns might be wrong'

    task = {
        'molecule': 'physchem/SAMPL',
        'protein': 'downstream',
        'dna': 'downstream',
        'singlecell': 'downstream',
    }
    if conf == ['conf']:
        if model in ('megamolbart', 'openfold', 'molmim'):
            return ''
        elif model == 'geneformer':
            return MAIN % 'singlecell'
        else:
            return MAIN % f'{domain}/{task[domain]}/test/x000'

    if 'retro' in script:
        return MAIN % 'reaction'
    elif model == 'geneformer':
        return (
            # This is what we run inference on when running infer.py. This is not checked or used during pretraining.
            f' {DATA}.dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/test'
            # The following three paths are used for pretrain.py, but also are required to support model loading currently when running inference.
            f' {DATA}.train_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/train'
            f' {DATA}.val_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/val'
            f' {DATA}.test_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/test'
        )
    elif model == 'openfold':
        return MAIN % 'openfold_data/processed_sample'
    elif model == 'diffdock':
        return (
            f' ++data.split_train={TEST_DATA_DIR}/molecule/diffdock/splits/split_train'
            f' ++data.split_val={TEST_DATA_DIR}/molecule/diffdock/splits/split_val'
            f' ++data.split_test={TEST_DATA_DIR}/molecule/diffdock/splits/split_test'
            f' ++data.cache_path={TEST_DATA_DIR}/molecule/diffdock/data_cache'
        )
    elif 'downstream' in script:
        if model == 'dnabert':
            fasta_directory = os.path.join(TEST_DATA_DIR, 'dna/downstream')
            fasta_pattern = fasta_directory + '/test-chr1.fa'
            splicesite_overrides = (
                f"++model.data.fasta_directory={fasta_directory} "
                "++model.data.fasta_pattern=" + fasta_pattern + " "
                f"++model.data.train_file={fasta_directory}/train.csv "
                f"++model.data.val_file={fasta_directory}/val.csv "
                f"++model.data.predict_file={fasta_directory}/test.csv "
            )
            return splicesite_overrides
        else:
            return MAIN % f'{domain}/{task[domain]}'
    elif model == 'dnabert':
        DNABERT_TEST_DATA_DIR = os.path.join(BIONEMO_HOME, 'examples/dna/dnabert/data/small-example')
        dnabert_overrides = (
            f"++model.data.dataset_path={DNABERT_TEST_DATA_DIR} "
            "++model.data.dataset.train=chr1-trim-train.fna "
            "++model.data.dataset.val=chr1-trim-val.fna "
            "++model.data.dataset.test=chr1-trim-test.fna "
        )
        return dnabert_overrides
    elif model == 'esm2nv' and "infer" not in script:
        UNIREF_FOLDER = "uniref202104_esm2_qc_test200_val200"
        esm2_overwrites = MAIN % UNIREF_FOLDER + DOWNSTREAM % f'{domain}/{task[domain]}'
        return esm2_overwrites

    else:
        return (MAIN + DOWNSTREAM) % (domain, f'{domain}/{task[domain]}')


def get_train_args_overrides(script_or_cfg_path, train_args):
    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    if model == "openfold":
        train_args['model.micro_batch_size'] = 1
        train_args['model.train_ds.num_workers'] = 1
        train_args['model.train_sequence_crop_size'] = 16
        # do not use kalign as it requires third-party-download and it not essential for testing
        train_args['model.data.realign_when_required'] = False
    elif model == "diffdock":
        # Use size aware batch sampler, and set the size control to default
        train_args['model.micro_batch_size'] = 2
        train_args['model.estimate_memory_usage.maximal'] = 'null'
        train_args['model.max_total_size'] = 'null'
        train_args['model.tensor_product.type'] = 'fast_tp'

    return train_args


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize('script_path, metadata_keys', list(zip(TRAINING_SCRIPTS_PATH, METADATA_LIST)))
def test_stop_and_go(script_path: str, metadata_keys: List[str], train_args, data_args, tmp_path):
    data_str = get_data_overrides(script_path)
    train_args = get_train_args_overrides(script_path, train_args)
    # add kill-after-checkpoint and metadata-save callbacks for first run
    train_args['create_kill_after_signal_callback'] = True
    train_args['kill_after_signal_callback_kwargs.metadata_path'] = tmp_path
    train_args['create_metadata_save_callback'] = True
    train_args['metadata_save_callback_kwargs.metadata_path'] = tmp_path

    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    cmd = cmd + f' "++metadata_save_callback_kwargs.metadata_keys={metadata_keys}"'
    # TODO(dorotat) Trye to simplify when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += ' ' + ' '.join(f'++{k}={v}' for k, v in data_args.items())
    print(cmd)
    # run initial training run to save a checkpoint and some metadata and kill the job afterwards
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode('utf-8')
    assert process_handle.returncode == 0, f"Initial training command failed:\n{cmd}\n Error log:\n{error_out}"

    # assert that metadata was saved correctly
    assert os.path.isfile(
        tmp_path / 'checkpoints/metadata.pkl'
    ), f"No file found at {tmp_path / 'checkpoints/metadata.pkl'}"

    # add check checkpoint integrity callback for second run
    train_args['create_checkpoint_integrity_callback'] = True
    train_args['checkpoint_integrity_callback_kwargs.metadata_path'] = tmp_path
    # remove kill after checkpoint and metadata save callbacks for second run
    train_args['create_kill_after_signal_callback'] = False
    train_args['kill_after_signal_callback_kwargs'] = None
    train_args['create_metadata_save_callback'] = False
    train_args['metadata_save_callback_kwargs.metadata_path'] = None

    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    cmd = cmd + f' "++checkpoint_integrity_callback_kwargs.metadata_keys={metadata_keys}"'
    # TODO(dorotat) Trye to simplify  when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += ' ' + ' '.join(f'++{k}={v}' for k, v in data_args.items())
    print(cmd)
    # run resume training run to load checkpoint and check against saved metadata
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode('utf-8')
    assert process_handle.returncode == 0, f"Resuming training command failed:\n{cmd}\n Error log:\n{error_out}"
