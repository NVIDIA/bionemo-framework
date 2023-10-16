import os
import subprocess
from glob import glob

import pytest
import torch


@pytest.fixture
def train_args():
    return {
        'trainer.devices': torch.cuda.device_count(),
        'trainer.num_nodes': 1,
        'trainer.max_steps': 20,
        'trainer.val_check_interval': 10,
        'trainer.limit_val_batches': 2,
        'exp_manager.create_wandb_logger': False,
        'exp_manager.create_tensorboard_logger': False,
        'model.data.dataset.train': 'x000',
        'model.data.dataset.val': 'x000',
        'model.data.dataset.test': 'x000',
        'model.micro_batch_size': 2,
    }


# TODO: can we assume that we always run these tests from main bionemo dir?
DIRS_TO_TEST = [
    'examples/',
    'examples/molecule/megamolbart/',
    'examples/protein/downstream/',
    'examples/protein/esm1nv/',
    'examples/protein/prott5nv/',
]

TRAIN_SCRIPTS = []
for subdir in DIRS_TO_TEST:
    TRAIN_SCRIPTS += list(glob(os.path.join(subdir, 'pretrain*.py')))
    TRAIN_SCRIPTS += [f for f in glob(os.path.join(subdir, 'downstream*.py')) if not f.endswith('test.py')]


INFERENCE_CONFIGS = []
for subdir in DIRS_TO_TEST:
    INFERENCE_CONFIGS += list(glob(os.path.join(subdir, 'conf', 'infer*yaml')))


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    TEST_DATA_DIR = '/workspace/bionemo/examples/tests/test_data'
    MAIN = f' ++model.data.dataset_path={TEST_DATA_DIR}/%s'
    DOWNSTREAM = f' ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s'

    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    assert root == 'examples' and model in (
        'megamolbart',
        'esm1nv',
        'prott5nv',
        'downstream',
    ), 'update this function, patterns might be wrong'

    task = {
        'molecule': 'phys_chem/SAMPL',
        'protein': 'downstream',
    }

    if conf == ['conf']:
        if model == 'megamolbart':
            return ''
        else:
            return MAIN % f'{domain}/{task[domain]}/test/x000'

    if 'retro' in script:
        return MAIN % 'reaction'
    elif 'downstream' in script:
        return MAIN % f'{domain}/{task[domain]}'
    else:
        return (MAIN + DOWNSTREAM) % (domain, f'{domain}/{task[domain]}')


@pytest.mark.needs_gpu
@pytest.mark.parametrize('script_path', TRAIN_SCRIPTS)
def test_train_scripts(script_path, train_args, tmp_path):
    data_str = get_data_overrides(script_path)
    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    assert process_handle.returncode == 0


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('config_path', INFERENCE_CONFIGS)
def test_infer_script(config_path, tmp_path):
    config_dir, config_name = os.path.split(config_path)
    cmd = f'python examples/infer.py --config-dir {config_dir} --config-name {config_name} ++exp_manager.exp_dir={tmp_path}'
    # FIXME: WAR for retro checkpoint not being released
    if 'retro' in config_path:
        cmd += ' model.downstream_task.restore_from_path=/model/molecule/megamolbart/megamolbart.nemo'
    cmd += get_data_overrides(config_path)
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    assert process_handle.returncode == 0
