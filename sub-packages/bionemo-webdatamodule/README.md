# bionemo-webdatamodule

To install, execute the following:
```bash
pip install -e .
```

To run unit tests, execute:
```bash
pytest -v .
```

## WebDataModule

```python
class WebDataModule(L.LightningDataModule)
```

A LightningDataModule for using webdataset tar files to setup dataset and
dataloader. This data module takes as input a dictionary: Split -> tar file
directory and vaiours webdataset config settings. In its setup() function, it
creates the webdataset object chaining up the input `pipeline_wds` workflow. In
its train/val/test_dataloader(), it creates the WebLoader object chaining up the
`pipeline_prebatch_wld` workflow

Examples
--------

1. create the data module with input directory to webdataset tar files.
Depending on which of the downstream Lightning.Trainer methods are called,
e.g., `Trainer.fit()`, `Trainer.validate()`, `Trainer.test()` or
`Trainer.predict()`, only a subset of the train, val and test splits need to
be specified in the various input options to the data module:

- `Trainer.fit()` requires the `train` and `val` splits
- `Trainer.validate()` requires the `val` split
- `Trainer.test()` requires the `test` splits
- `Trainer.predict()` requires the `test` splits

Here is an example of constructing the data module for `Trainer.fit()`:
```
>>> from bionemo.webdatamodule.datamodule import Split, WebDataModule
>>>
>>> tar_file_prefix = "shards"
>>>
>>> dirs_of_tar_files = {
>>>     Split.train: "/path/to/train/split/tars",
>>>     Split.val: "/path/to/val/split/tars",
>>> }
>>>
>>> n_samples {
>>>     Split.train: 1000,
>>>     Split.val: 100,
>>> }
>>>
>>> # this is the string to retrieve the corresponding data object from the
>>> # webdataset file (see
>>> # https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-format
>>> # for details)
>>> suffix_keys_wds = "tensor.pyd"
>>>
>>> # see the API doc for the definition of global_batch_size
>>> global_batch_size = 16
>>>
>>> seed = 27193781
>>>
>>> # Specify the routines to process the samples in the WebDataset object.
>>> # The routine is a generator of an Iterable of generators that are chained
>>> # together by nested function calling. The following is equivalent of
>>> # defining a overall generator of `shuffle(untuple(...))` which
>>> # untuples the samples and shuffles them. See webdataset's Documentation
>>> # for details.
>>> # NOTE: the `untuple` is almost always necessary due to the webdataset's
>>> # file parsing rule.
>>>
>>> untuple = lambda source : (sample for (sample,) in source)
>>>
>>> from webdatast import shuffle
>>> pipeline_wds = {
>>>     Split.train : [untuple, shuffle(n_samples[Split.train],
>>>                                     rng=random.Random(seed_rng_shfl))],
>>>     Split.val: untuple
>>> }
>>>
>>> # Similarly the user can optionally define the processing routine on the
>>> # WebLoader (the dataloader of webdataset).
>>> # NOTE: these routines by default take unbatched sample as input so the
>>> # user can customize their batching routines here
>>>
>>> batch = batched(local_batch_size, collation_fn=lambda
                    list_samples : torch.vstack(list_samples))
>>> pipeline_prebatch_wld = {
        Split.train: [shuffle(n_samples[Split.train],
                              rng=random.Random(seed_rng_shfl)), batch],
        Split.val : batch,
        Split.test : batch
    }
>>>
>>> # the user can optionally specify the kwargs for WebDataset and
>>> # WebLoader
>>>
>>> kwargs_wds = {
>>>     split : {'shardshuffle' : split == Split.train,
>>>              'nodesplitter' : wds.split_by_node,
>>>              'seed' : seed_rng_shfl}
>>>     for split in Split
>>>     }
>>>
>>> kwargs_wld = {
>>>     split : {"num_workers": 2} for split in Split
>>>     }
>>>
>>> # construct the data module
>>> data_module = WebDataModule(dirs_of_tar_files, n_samples, suffix_keys_wds,
                                global_batch_size,
                                prefix_tars_wds=tar_file_prefix,
                                pipeline_wds=pipeline_wds,
                                pipeline_prebatch_wld=pipeline_prebatch_wld,
                                kwargs_wds=kwargs_wds,
                                kwargs_wld=kwargs_wld)
```

<a id="datamodule.WebDataModule.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        dirs_tars_wds: Dict[Split, str],
        n_samples: Dict[Split, int],
        suffix_keys_wds: Union[str, Iterable[str]],
        global_batch_size: int,
        prefix_tars_wds: str = "wdshards",
        pipeline_wds: Optional[Dict[Split, Union[Iterable[Iterable[Any]],
                                                 Iterable[Any]]]] = None,
        pipeline_prebatch_wld: Optional[Dict[Split,
                                             Union[Iterable[Iterable[Any]],
                                                   Iterable[Any]]]] = None,
        kwargs_wds: Optional[Dict[Split, Dict[str, Any]]] = None,
        kwargs_wld: Optional[Dict[Split, Dict[str, Any]]] = None)
```

constructor

**Arguments**:

- `dirs_tars_wds` _Dict[Split, str]_ - input dictionary: Split -> tar file
  directory that contains the webdataset tar files for each split
- `n_samples` _Dict[Split, int]_ - input dictionary: Split -> number of
  data samples for each split
- `suffix_keys_wds` _Union[str, Iterable[str]]_ - a set of keys each
  corresponding to a data object in the webdataset tar file
  dictionary. The data objects of these keys will be extracted and
  tupled for each sample in the tar files
- `global_batch_size` _int_ - size of batch summing across nodes in Data
  Distributed Parallel, i.e., local_batch_size * n_nodes. NOTE:
  this data module doesn't rely on the input `global_batch_size`
  for batching the samples. The batching is supposed to be done as
  a part of the input `pipeline_prebatch_wld`. `global_batch_size`
  is only used to compute a (pseudo-) epoch length for the data
  loader so that the loader yield approximately n_samples //
  global_batch_size batches
  Kwargs:
- `prefix_tars_wds` _str_ - name prefix of the input webdataset tar
  files. The input tar files are globbed by
  "{dirs_tars_wds[split]}/{prefix_tars_wds}-*.tar"
  pipeline_wds (Optional[Dict[Split, Union[Iterable[Iterable[Any]],
- `Iterable[Any]]]])` - a dictionary of webdatast composable, i.e.,
  functor that maps a iterator to another iterator that
  transforms the data sample yield from the dataset object, for
  different splits, or an iterable to such a sequence of such
  iterators. For example, this can be used to transform the
  sample in the worker before sending it to the main process of
  the dataloader
  pipeline_prebatch_wld (Optional[Dict[Split,
  Union[Iterable[Iterable[Any]], Iterable[Any]]]]): a dictionary
  of webloader composable, i.e., functor that maps a iterator to
  another iterator that transforms the data sample yield from the
  WebLoader object, for different splits, or an iterable to a
  seuqnence of such iterators. For example, this can be used for
  batching the samples. NOTE: this is applied before batching is
  yield from the WebLoader
- `kwargs_wds` _Optional[Dict[Split, Dict[str,  Any]]]_ - kwargs for the
  WebDataset.__init__()
- `kwargs_wld` _Optional[Dict[Split, Dict[str,  Any]]]_ - kwargs for the
  WebLoader.__init__(), e.g., num_workers, of each split

<a id="datamodule.WebDataModule.prepare_data"></a>

#### prepare\_data

```python
def prepare_data() -> None
```

This is called only by the main process by the Lightning workflow. Do
not rely on this data module object's state update here as there is no
way to communicate the state update to other subprocesses.

Returns: None

<a id="datamodule.WebDataModule.setup"></a>

#### setup

```python
def setup(stage: str) -> None
```

This is called on all Lightning-managed nodes in a multi-node
training session


**Arguments**:

- `stage` _str_ - "fit", "test" or "predict"
- `Returns` - None

## PickledDataWDS

```python
class PickledDataWDS(WebDataModule)
```

A LightningDataModule to process pickled data into webdataset tar files
and setup dataset and dataloader. This inherits the webdataset setup from
its parent module `WebDataModule`. This data module takes a directory of
pickled data files, data filename prefixes for train/val/test splits, data
filename suffixes and prepare webdataset tar files by globbing the specific
pickle data files `{dir_pickles}/{name_subset[split]}.{suffix_pickles}` and
outputing to webdataset tar file with the dict structure:
```
    {"__key__" : name.replace(".", "-"),
     suffix_pickles : pickled.dumps(data) }
```
NOTE: this assumes only one pickled file is processed for each sample. In
its setup() function, it creates the webdataset object chaining up the input
`pipeline_wds` workflow. In its train/val/test_dataloader(), it creates the
WebLoader object chaining up the `pipeline_prebatch_wld` workflow.

Examples
--------

1. create the data module with a directory of pickle files and the file name
prefix thereof for different splits to used by `Lightning.Trainer.fit()`

```
>>> from bionemo.webdatamodule.datamodule import Split, PickledDataWDS

>>> dir_pickles = "/path/to/my/pickles/dir"

>>> # the following will use `sample1.mydata.pt` and `sample2.mydata.pt` as the
>>> # training dataset and `sample4.mydata.pt` and `sample5.mydata.pt` as the
>>> # validation dataset

>>> suffix_pickles = "mydata.pt"

>>> names_subset = {
>>>     Split.train: [sample1, sample2],
>>>     Split.val: [sample4, sample5],
>>> }

>>> # the following setting will attempt to create at least 5 tar files in
>>> # `/path/to/output/tars/dir/myshards-00000{0-5}.tar`

>>> n_tars_wds = 5
>>> prefix_tars_wds = "myshards"
>>> output_dir_tar_files = "/path/to/output/tars/dir"

>>> # see the `WebDataModule` API doc for the definition of global_batch_size
>>> global_batch_size = 16

>>> # user can optionally customize the data processing routines and kwargs used
>>> # in the WebDataset and WebLoader (see the examples in `WebDataModule`)

>>> pipeline_wds = { Split.train: ... }

>>> pipeline_prebatch_wld = { Split.train: ... }

>>> kwargs_wds = { Split.train: ..., Split.val: ... }

>>> kwargs_wld = { Split.train: ..., Split.val: ... }

>>> # create the data module
>>> data_module = PickledDataWDS(
>>>     dir_pickles,
>>>     suffix_pickles,
>>>     names_subset,
>>>     output_dir_tar_files,
>>>     global_batch_size, # `WebDataModule` args
>>>     n_tars_wds=n_tars_wds,
>>>     prefix_tars_wds=prefix_tars_wds, # `WebDataModule` kwargs
>>>     pipeline_wds=pipeline_wds, # `WebDataModule` kwargs
>>>     pipeline_prebatch_wld=pipelines_wdl_batch, # `WebDataModule` kwargs
>>>     kwargs_wds=kwargs_wds, # `WebDataModule` kwargs
>>>     kwargs_wld=kwargs_wld, # `WebDataModule` kwargs
>>> )

```

<a id="datamodule.PickledDataWDS.__init__"></a>

#### \_\_init\_\_

```python
def __init__(dir_pickles: str,
             suffix_pickles: str,
             names_subset: Dict[Split, List[str]],
             prefix_dir_tars_wds: str,
             *args,
             n_tars_wds: Optional[int] = None,
             **kwargs)
```

constructor

**Arguments**:

- `dir_pickles` _str_ - input directory of pickled data files
- `suffix_pickles` _str_ - filename suffix of the input data in
  dir_pickles. This is also used as the key mapped to the
  tarballed pickled object in the webdataset
- `names_subset` _Dict[Split, List[str]]_ - list of filename prefix of
  the data samples to be loaded in the dataset and dataloader for
  each of the split
- `prefix_dir_tars_wds` _str_ - directory name prefix to store the output
  webdataset tar files. The actual directories storing the train, val
  and test sets will be suffixed with "train", "val" and "test"
  respectively.
- `*args` - arguments passed to the parent WebDataModule

  Kwargs:
- `n_tars_wds` _int_ - attempt to create at least this number of
  webdataset shards
- `**kwargs` - arguments passed to the parent WebDataModule

<a id="datamodule.PickledDataWDS.prepare_data"></a>

#### prepare\_data

```python
def prepare_data() -> None
```

This is called only by the main process by the Lightning workflow. Do
not rely on this data module object's state update here as there is no
way to communicate the state update to other subprocesses. The nesting
`pickles_to_tars` function goes through the data name prefixes in the
different splits, read the corresponding pickled file and output a
webdataset tar archive with the dict structure: {"__key__" :
name.replace(".", "-"), suffix_pickles : pickled.dumps(data) }.

Returns: None
