# bionemo-size-aware-batching

To install, execute the following:
```bash
pip install -e .
```

To run unit tests, execute:
```bash
pytest -v .
```

# **Summary of Usage**

This package provides a simple way to create mini-batches in a memory consumption-aware (or size-aware) manner, making it useful for tasks like training models on datasets with varying memory requirements. The usage typically consists of the following steps:

1. Use the `collect_cuda_peak_alloc` function to collect CUDA peak memory
   allocation statistics for a user-defined workflow. It's expected that the
   user-defined workflow will return a list of features extracted from the data
   so that the memory model in the following step can use these features to
   predict the memory allocation.
2. User defines and trains a memory model using the features and memory allocation
   data from previous step. This memory model can then be used to predict memory
   consumption.
3. Use `SizeAwareBatchSampler` or `size_aware_batching` with the memory model
   prediction (from the previous step) to build batch of data so that the
   resulting mini-batches do not exceed a specified maximum total memory size.

Refer to the later sections for the API documentation and examples on how to achieve each of the steps above.

### utils Module
---------------

*   [**collect_cuda_peak_alloc**](#utils.collect_cuda_peak_alloc): A function that
    collects CUDA peak memory allocation statistics and features to be used for
    memory usage prediction for a given workflow.

### sampler Module
-----------------

*   [**size_aware_batching**](#sampler.size_aware_batching): A generator that batches elements from an iterable while ensuring that the total size of each batch does not exceed a specified maximum.
*   [**SizeAwareBatchSampler**](#sampler.SizeAwareBatchSampler): A class that batches elements of varying sizes while ensuring that the total size of each batch does not exceed a specified maximum.


# API reference and examples

<a id="utils"></a>

## utils

<a id="utils.collect_cuda_peak_alloc"></a>

#### collect\_cuda\_peak\_alloc

```python
def collect_cuda_peak_alloc(
    dataset: Iterable[Data],
    work: Callable[[Data], Feature],
    device: torch.device,
    cleanup: Optional[Callable[[], None]] = None
) -> Tuple[List[Feature], List[int]]
```

Collects CUDA peak memory allocation statistics for a given workflow.

This function iterates through the provided dataset, applies the given feature function to each data point,
and records the peak CUDA memory allocation during this process. The features extracted from the data points
are collected along with their corresponding memory usage statistics.

Note that the first few iterations of the workflow might result in smaller memory allocations due to uninitialized
data (e.g., internal PyTorch buffers). Therefore, users may want to skip these initial data points when analyzing the results.

**Arguments**:

- `dataset` - An iterable containing the input data.
- `work` - A function that takes a data point and returns its corresponding feature. This is where
  the main computation happens and memory allocations are tracked.
- `device` - The target Torch CUDA device.
- `cleanup` - A function that is called after each iteration to perform any necessary cleanup.


**Returns**:

  A tuple containing the collected features and their corresponding memory usage statistics.


**Raises**:

- `ValueError` - If the provided device is not a CUDA device.

  -------

**Examples**:


```python
>>> import torch
>>> from bionemo.size_aware_batching.utils import collect_cuda_peak_alloc


>>> # prepare dataset, model and other components of a workflow
>>> # for which the user want to collect CUDA peak memory allocation statistics
>>> dataset, model, optimizer = ...
>>> # Set the target Torch CUDA device.
>>> device = torch.device("cuda:0")
>>> model = model.to(device)

>>> # Define a function that takes an element of the dataset as input and
>>> # do a training step
>>> def work(data):
...     # example body of a training loop
...     optimizer.zero_grad()
...     output = model(data.to(device))
...     loss = compute_loss(output)
...     loss.backward()
...     optimizer.step()
...     # extract the feature for later to be modeled or analyzed
...     return featurize(data)

>>> # can optionally use a cleanup function to release the references
>>> # hold during the work(). This cleanup function will be called
>>> # at the end of each step before garbage collection and memory allocations measurement
>>> def cleanup():
...     model.zero_grad(set_to_none=True)

>>> # Collect features (i.e., model outputs) and memory usage statistics for the workflow.
>>> features, alloc_peaks = collect_cuda_peak_alloc(
...     dataset=batches,
...     work=work,
...     device=device,
...     cleanup=cleanup,
... )


>>> # use features and alloc_peaks as needed, e.g., fit a model
>>> # that can use these statistics to predict memory usage
>>> memory_model = ...
>>> memory_model.fit(features, alloc_peaks)
```

<a id="sampler"></a>

## sampler

<a id="sampler.size_aware_batching"></a>

#### size\_aware\_batching

```python
def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: Real,
    collate_fn: Optional[Callable[[Iterable[Data]], BatchCollated]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None
) -> Iterator[Union[List[Data], BatchCollated]]
```

A generator that batches elements from an iterable while ensuring that the
total size of each batch does not exceed a specified maximum. Here the size
can be a measurement of memory consumption of the elements in the batch.
This can be useful for both indexible data or non-indexible but iterable data.

**Arguments**:

- `dataset` - The input iterable.
- `sizeof` - A function or mapping that returns the "size" of each element in `dataset`.
  E.g., this can used to determine how much memory an element consumes. Its return
  type must be comparable with `max_total_size` and it must be addable (operator `+`).
- `max_total_size` - The maximum total "size" of each batch. The semantics of "size"
  is defined by the `sizeof` argument. The type of this value must be comparable
  with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
- `collate_fn` - An optional function to collate batches. Defaults to None, in which case
  each batch is a list of elements from the input dataset
- `info_logger` - A function to log info. Defaults to None.
- `warn_logger` - A function to log warnings. Defaults to None.


**Yields**:

  A generator that yields batches from `dataset`.

  -----------
  Assumptions
  1. Linear complexity. This function consumes the given Iterable of data (`dataset`) once,
  by going over the data item one by one to build a batch and yield it as soon as the
  addition of the next data item to the batch would exceed `max_total_size` or if the
  batch is the last one (end of iteration)
  2. Additive size measurement. For the general usage case of building mini-batches with
  a threshold of the batch's memory consumption, it assumes that the size of the batch is
  the sum of all elements in the batch (additive property).
  3. Comparable type of `max_total_size` and `sizeof`'s return. `sizeof`'s return values
  must be compared with `max_total_size` to threshold the size of batches


  ------
  Caveat
- `1` - The generated batch sizes may have large variance
  - how to workaround: filter the output of this generator using a batch size threshold
- `2` - The number of batches may vary a lot across different epochs.
  - how to workaround: increase the number of steps that compose an epoch,
  e.g., in the Lightning training/validation loop, which effectively increases the input
  dataset size per epoch


  -------
  Example

```python
>>> import torch
>>> from torch.utils.data import default_collate
>>> from bionemo.size_aware_batching.sampler import size_aware_batching

>>> # Define a sample dataset with torch.tensor
>>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
...            torch.tensor([7, 8]), torch.tensor([9, 10])]

>>> # Define a sizeof function that returns the size of each tensor
>>> def sizeof(x):
...     return x.numel()

>>> # Create a generator with max_total_size=4 and default_collate_fn
>>> gen = size_aware_batching(dataset, sizeof, 4, collate_fn=default_collate)
>>> batches = list(gen)
>>> print(batches)
    [tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]), tensor([[9, 10]])]
```

<a id="sampler.SizeAwareBatchSampler"></a>

## SizeAwareBatchSampler Objects

```python
class SizeAwareBatchSampler(Sampler[List[int]])
```

A sampler that batches elements of varying sizes while ensuring
that the total size of each batch does not exceed a specified maximum.

This is useful when dealing with datasets where each element has a
different size, such as graphs or sequences of varying lengths.
The sampler uses a provided `sizeof` function to determine the size
of each element in the dataset and ensures that the total size of
each batch does not exceed the specified `max_total_size`.

---------

**Examples**:


```python
>>> import torch
>>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


>>> # Define a sample dataset with torch.tensor
>>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
...            torch.tensor([7, 8]), torch.tensor([9, 10])]


>>> # Define a function that returns the size of each element in the dataset.
>>> def sizeof(index):
...     return dataset[index].numel()


>>> # Create a SizeAwareBatchSampler with a maximum total batch size of 10.
>>> batch_sampler = SizeAwareBatchSampler(
...     sampler=torch.utils.data.SequentialSampler(dataset),
...     sizeof=sizeof,
...     max_total_size=4
... )


>>> # Iterate over batches of indices that do not exceed the maximum total size.
>>> print(list(batch_sampler))
    [[0, 1], [2, 3], [4]]
```

<a id="sampler.SizeAwareBatchSampler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(sampler: Union[Sampler[List[int]], Iterable[int]],
             sizeof: Callable[[int], Real],
             max_total_size: Real,
             info_logger: Optional[Callable[[str], None]] = None,
             warn_logger: Optional[Callable[[str], None]] = None) -> None
```

Initializes the SizeAwareBatchSampler.

**Arguments**:

- `sampler` - The underlying sampler.
- `sizeof` - A function that returns the size at each index. E.g., this can used to
  determine how much memory an element consumes. Its return type must be
  comparable with `max_total_size` and it must be addable (operator `+`).
- `max_total_size` - The maximum total size of a mini-batch. The semantics of "size"
  is defined by the `sizeof` argument. The type of this value must be comparable
  with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
- `info_logger` - A function to log info. Defaults to None.
- `warn_logger` - A function to log warnings. Defaults None.


**Raises**:

- `TypeError` - If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
- `ValueError` - If max_total_size is not a positive number.

<a id="sampler.SizeAwareBatchSampler.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[List[int]]
```

Iterate over batches of indices.

This function yields batches of indices that do not exceed the maximum total size.

**Yields**:

  A batch of indices that do not exceed the maximum total size.
