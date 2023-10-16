import os

import pandas as pd
import pytest

from bionemo.data.preprocess.molecule.uspto50k_preprocess import USPTO50KPreprocess


ROOT_DIR = 'uspto50k'


@pytest.mark.parametrize('root_directory', [ROOT_DIR])
def test_uspto50k_preprocess(tmp_path_factory, root_directory: str):
    ngc_registry_target = 'uspto_50k_dataset'
    ngc_registry_version = 'v23.06'
    tmp_directory = tmp_path_factory.mktemp(root_directory)
    filename = "uspto_50.pickle"
    data_preprocessor = USPTO50KPreprocess(data_dir=tmp_directory)
    data_preprocessor.prepare_dataset(
        ngc_registry_target=ngc_registry_target, ngc_registry_version=ngc_registry_version, force=True
    )

    raw_data_filepath = os.path.join(data_preprocessor.download_dir, filename)
    assert raw_data_filepath == data_preprocessor.datapath_raw
    assert os.path.join(tmp_directory, "raw", filename) == data_preprocessor.datapath_raw
    assert os.listdir(data_preprocessor.download_dir)[0] == filename
    assert os.path.join(tmp_directory, "processed") == data_preprocessor.processed_dir
    # reading raw data
    df_raw = pd.read_pickle(raw_data_filepath)

    assert all(folder in data_preprocessor.splits for folder in os.listdir(data_preprocessor.processed_dir))

    assert len(df_raw) == 50037
    assert all(col in ['reactants_mol', 'products_mol', 'reaction_type', 'set'] for col in df_raw.columns)

    assert all(s in ['train', 'valid', 'test'] for s in df_raw.set.unique())
    assert all(s in ['train', 'val', 'test'] for s in data_preprocessor.splits)

    split_sizes = {'train': 40029, 'val': 5004, 'test': 5004}
    for split in data_preprocessor.splits:
        filepath = os.path.join(data_preprocessor.get_split_dir(split), data_preprocessor.data_file)
        # reading processed split data
        df = pd.read_csv(filepath)
        assert len(df) == split_sizes[split]
        assert all(col in ['reaction_type', 'set', 'reactants', 'products'] for col in df.columns)

    assert df.products[54] == 'O=[N+]([O-])c1ccccc1CCN1CCN(c2ccccc2)CC1'
    assert df.reactants[565] == 'Nc1ccc(O)c(C(=O)O)c1.O=C(O)C1CCCCC1'
