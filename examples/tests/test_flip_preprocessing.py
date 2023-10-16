import glob
import os

import pytest
from omegaconf import OmegaConf

from bionemo.data import FLIPPreprocess
from bionemo.utils.tests import get_directory_hash


# FLIP secondary structure benchmark dataset is small and will be fully downloaded in this test
os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
ROOT_DIR = 'FLIP'
CONFIG = {'url': None, 'num_csv_files': 1}
HEADER = 'id,sequence,3state,resolved'
NUM_ENTRIES = 11156
TRAIN_VAL_TEST_HASHES = {
    'train': 'af653e0e56e63442b4f3bafe68de7d70',
    'val': 'fa5d9cbdd729180b8401d18b31ecc0a9',
    'test': 'a92c394b0017cd573d25fbd86cc6abf7',
}

##############


@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory, root_directory=ROOT_DIR):
    """Create tmp directory"""
    tmp_path_factory.mktemp(root_directory)
    return tmp_path_factory.getbasetemp()


@pytest.mark.parametrize(
    'config, header, num_entries, hash_dict', [(CONFIG, HEADER, NUM_ENTRIES, TRAIN_VAL_TEST_HASHES)]
)
def test_prepare_dataset(tmp_directory, config, header, num_entries, hash_dict):
    cfg = OmegaConf.create(config)
    preproc = FLIPPreprocess(root_directory=tmp_directory)
    processed_directory = os.path.join(tmp_directory, 'processed')
    print(processed_directory)
    preproc.prepare_dataset(num_csv_files=cfg.num_csv_files, output_dir=processed_directory)

    total_lines = 0
    for split in ['train', 'val', 'test']:
        split_directory = os.path.join(processed_directory, split)
        assert get_directory_hash(split_directory) == hash_dict[split]

        csv_file_list = glob.glob(os.path.join(split_directory, '*.csv'))
        assert len(csv_file_list) == cfg.num_csv_files

        for file in csv_file_list:
            with open(file, 'r') as fh:
                lines = fh.readlines()
                total_lines += len(lines) - 1
                assert lines[0].strip() == header

    assert total_lines == num_entries
