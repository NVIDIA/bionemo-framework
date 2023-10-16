import logging
import os
from typing import List, Optional, Union

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVMemMapDataset
from nemo.core import Dataset
from omegaconf import OmegaConf

from bionemo.data.mapped_dataset import NeMoUpsampling
from bionemo.data.memmap_csv_fields_dataset import CSVFieldsMemmapDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.utils import check_paths_exist, expand_dataset_paths


_CSV_FIELDS_MMAP_TYPE = 'csv_fields_mmap'
_CSV_MMAP_TYPE = 'csv_mmap'
_FASTA_FIELDS_MMAP_TYPE = 'fasta_fields_mmap'

_DATA_IMPL_TYPE_CLS = {
    _CSV_FIELDS_MMAP_TYPE: CSVFieldsMemmapDataset,
    _CSV_MMAP_TYPE: CSVMemMapDataset,
    _FASTA_FIELDS_MMAP_TYPE: FASTAFieldsMemmapDataset,
}


def build_typed_dataset(
    dataset_paths: Union[str, List[str]],
    data_impl: str,
    use_upsampling: bool,
    cfg: OmegaConf,
    num_samples: Optional[int] = None,
) -> Dataset:
    """
    Builds dataset based on preferred implementation given provided paths to the files with data and
    optionally down/upsamples it to num_samples.
    Args:
        dataset_paths: local path or list of paths to the files with data
        data_impl: dataset implementation type specified as key in _DATA_IMPL_TYPE_CLS
        cfg: config to be passed to a dataset constructor
        num_samples: down/upsample size of the dataset, if applicable. If None, then the num_samples equals len(dataset)
    Returns:
        Dataset
    """

    assert (
        data_impl in _DATA_IMPL_TYPE_CLS.keys()
    ), f'Argument data_impl must be set to: {", ".join(_DATA_IMPL_TYPE_CLS.keys())}'
    dataset_cls = _DATA_IMPL_TYPE_CLS[data_impl]

    assert 'data_impl_kwargs' in cfg, (
        f"Config 'cfg' should contain 'data_impl_kwargs.{data_impl}' key being "
        f"a dictionary with arguments to the constructor of {dataset_cls.__name__}"
    )

    data_impl_kwargs = cfg.data_impl_kwargs.get(data_impl, {})
    if data_impl_kwargs == {}:
        logging.info(f'Default values of the arguments are used to initialize dataset {dataset_cls.__name__}')

    if data_impl == _FASTA_FIELDS_MMAP_TYPE:
        ext = ".fasta"
    else:
        ext = ".csv"

    if isinstance(dataset_paths, list):
        dataset_paths: List[str] = [
            path for dataset_path in dataset_paths for path in expand_dataset_paths(dataset_path, ext=ext)
        ]

    elif isinstance(dataset_paths, str):
        dataset_paths: List[str] = expand_dataset_paths(dataset_paths, ext=ext)
    else:
        raise ValueError('Argument dataset_paths should be a str or list of str corresponding to paths to data')

    errors = check_paths_exist(dataset_paths)
    assert len(errors) == 0, "Following files do not exist %s" % ' '.join(errors)
    logging.info(f'Loading data from {", ".join(dataset_paths)}')

    index_mapping_dir = cfg.get("index_mapping_dir", os.path.dirname(dataset_paths[0]))
    dataset: Dataset = dataset_cls(
        dataset_paths=dataset_paths, index_mapping_dir=index_mapping_dir, **data_impl_kwargs
    )

    if use_upsampling:
        assert num_samples is not None, (
            'To enable upsampling, "num_samples" need to be specified as '
            'the number of samples in the upsampled dataset'
        )
        data_prefix = cfg.get('data_prefix', None)
        if data_prefix is None:
            data_prefix = os.path.commonprefix(dataset_paths)
        dataset = NeMoUpsampling(
            dataset,
            num_samples=num_samples,
            cfg=cfg,
            data_prefix=data_prefix,
            index_mapping_dir=index_mapping_dir,
            max_seq_length=cfg.max_seq_length,
            seed=cfg.seed,
        )
    return dataset
