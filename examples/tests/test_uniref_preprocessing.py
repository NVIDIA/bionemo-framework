import glob
import os
import pathlib
import shutil

import pytest
from omegaconf import OmegaConf

from bionemo.data import UniRef50Preprocess
from bionemo.utils.tests import get_directory_hash


# THe UniRef file takes ~20 minutes to download so the default test uses a smaller (local) file.
os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
ROOT_DIR = 'uniref'
SAMPLE_DATA = os.path.join(
    os.environ['PROJECT_MOUNT'], 'examples/tests/test_data/preprocessing/test/uniref2022_small.fasta'
)
SAMPLE_NGC_FILE = os.path.join(
    os.environ['PROJECT_MOUNT'], 'examples/tests/test_data/preprocessing/uniref2022_UR50.fasta'
)
NGC_REGISTRY_TARGET = "uniref50_2022_05"
NGC_REGISTRY_VERSION = "v23.06"
MD5_CHECKSUM = '415cd74fda2c95c46b2496feb5d55d17'
CONFIG = {'url': None, 'num_csv_files': 5, 'val_size': 10, 'test_size': 5, 'random_seed': 0}
HEADER = 'record_id,record_name,sequence_length,sequence'
NUM_ENTRIES = 100
TRAIN_VAL_TEST_HASHES = {
    'train': 'eaa0161c6870605f51cbafff21bcf7e1',
    'val': 'fc8864b075ebd7553ab345cf8d82cf6a',
    'test': '243a2b10c32ef82bbd8c457fcd1728d8',
}

##############


@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory, root_directory=ROOT_DIR):
    """Create tmp directory"""
    tmp_path_factory.mktemp(root_directory)
    return tmp_path_factory.getbasetemp()


@pytest.fixture()
def download_directory(tmp_directory):
    """Create the temporary directory for testing and the download directory"""
    # TODO mock a download when preprocessing code is refactored
    download_directory = pathlib.Path(os.path.join(tmp_directory, 'raw'))
    download_directory.mkdir(parents=True, exist_ok=True)
    return download_directory


@pytest.fixture()
def mock_url(download_directory, sample_data=SAMPLE_DATA):
    """Preprocessing expects a url with 'fasta.gz' extension, must mimic that for local file"""
    dest_path = os.path.join(download_directory, os.path.basename(sample_data))
    shutil.copyfile(sample_data, dest_path)
    mock_url = f'http://{dest_path}.gz'
    return mock_url


def test_process_files_uniprot(tmp_directory, download_directory, mock_url):
    preproc = UniRef50Preprocess(root_directory=tmp_directory, checksum=MD5_CHECKSUM)
    preproc.process_files_uniprot(url=mock_url, download_dir=download_directory)
    fasta_file = os.path.splitext(os.path.basename(mock_url))[0]
    uniref_file = os.path.join(download_directory, fasta_file)
    assert os.path.isfile(uniref_file)


# TODO reconsider this test after switch to NGC resources since it can't be mocked
def test_process_files_ngc(tmp_directory):
    fasta_file = os.path.basename(SAMPLE_NGC_FILE)
    output_dir = "/".join(SAMPLE_NGC_FILE.split("/")[:-1])
    preproc = UniRef50Preprocess(root_directory=tmp_directory, checksum=MD5_CHECKSUM)

    preproc.process_files_ngc(
        ngc_registry_target=NGC_REGISTRY_TARGET,
        ngc_registry_version=NGC_REGISTRY_VERSION,
        download_dir=tmp_directory,
        output_dir=output_dir,
        checksum=MD5_CHECKSUM,
    )
    uniref_file = os.path.join(output_dir, fasta_file)
    assert os.path.isfile(uniref_file)


@pytest.mark.parametrize(
    'config, header, num_entries, hash_dict', [(CONFIG, HEADER, NUM_ENTRIES, TRAIN_VAL_TEST_HASHES)]
)
def test_prepare_dataset(tmp_directory, mock_url, config, header, num_entries, hash_dict):
    cfg = OmegaConf.create(config)
    preproc = UniRef50Preprocess(root_directory=tmp_directory)
    preproc.prepare_dataset(
        url=mock_url,
        num_csv_files=cfg.num_csv_files,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_seed=cfg.random_seed,
        source='uniprot',
    )

    processed_directory = os.path.join(tmp_directory, 'processed')

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
