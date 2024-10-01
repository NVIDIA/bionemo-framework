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

In addition, this package provides one solution to create homogeneous mini-batches, which can be useful to reduce the padding in the network from the input tensors with varying sizes. This `BucketBatchSampler` can be used in conjunction with `torch.utils.data.BatchSampler`, `SizeAwareBatchSampler` or other user-defined batch samplers.

Refer to the later sections for the API documentation and examples on how to achieve each of the steps above.

### utils Module
---------------

*   [**collect_cuda_peak_alloc**](#collect_cuda_peak_alloc): A function that
    collects CUDA peak memory allocation statistics and features to be used for
    memory usage prediction for a given workflow.

*   [**create_buckets**](#create_buckets): A function to create buckets for a
    list of integers with pre-defined maximal range of interval and minimal
    bucket sizes.

### sampler Module
-----------------

*   [**size_aware_batching**](#size_aware_batching): A generator that batches elements from an iterable while ensuring that the total size of each batch does not exceed a specified maximum.
*   [**SizeAwareBatchSampler**](#sizeawarebatchsampler): A class that batches elements of varying sizes while ensuring that the total size of each batch does not exceed a specified maximum.
*   [**BucketBatchSampler**](#BucketBatchSampler): A class that groups elements of varying sizes based on predefined bucket ranges, and batches elements from each bucket to ensure that each batch has elements with homogeneous sizes.

# API reference and examples

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

<a id="utils.create_buckets"></a>

#### create\_buckets

```python
def create_buckets(sizes: Iterable[int], max_range: int,
                   min_bucket_count: int) -> Tuple[np.ndarray, np.ndarray]
```

Create buckets for a list of integers with pre-defined maximal range of interval and minimal bucket sizes.

**Arguments**:

- `sizes` _Iterable[int]_ - An iterable of integers representing sizes.
- `max_range` _int_ - The maximum range of a bucket.
- `min_bucket_count` _int_ - The minimum count of a bucket.
  Bucket size may be smaller than min_bucket_count if its range reaches max_range.


**Raises**:

- `ValueError` - If the provided sizes is empty, or not integers.
- `ValueError` - If max_range is not non-negative integer or min_bucket_count is not positive integer.


**Returns**:

  Tuple[np.ndarray, np.ndarray]: A tuple containing bucket ranges in ascending order and the number of elements in each bucket.
  e.g. np.array([[0, 5], [7,10]]), np.array([3,2]): specifies 2 buckets: 0<= sizes <= 5, 7 <= sizes <= 10, with 3 and 2 elements.

  ---------

**Examples**:


```python
>>> import numpy as np
>>> from bionemo.size_aware_batching.utils import create_buckets

>>> sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 22, 22, 22, 22])
>>> bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=20, min_bucket_count=20)
>>> print(bucket_ranges)
[[ 1  3]
[22 22]]
>>> print(bucket_sizes)
[12  4]
```

## sampler

<a id="sampler.size_aware_batching"></a>

#### size\_aware\_batching

```python
def size_aware_batching(
        dataset: Iterable[Data],
        sizeof: Callable[[Data], Real],
        max_total_size: Real,
        collate_fn: Optional[Callable[[Iterable[Data]], Any]] = None,
        info_logger: Optional[Callable[[str], None]] = None,
        warn_logger: Optional[Callable[[str], None]] = None) -> Iterator[Any]
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
- `collate_fn` - An optional function to collate batches. Defaults to None.
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
def __init__(
    sampler: Union[Sampler[List[int]], Iterable[int]],
    sizeof: Callable[[int], Real],
    max_total_size: Real,
    info_logger: Optional[Callable[[str], None]] = lambda msg: print(msg),
    warn_logger: Optional[Callable[[str], None]] = lambda msg: warn(msg)
) -> None
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
- `info_logger` - A function to log info. Defaults to a lambda function that print.
- `warn_logger` - A function to log warnings. Defaults to a lambda function that warns.


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


<a id="sampler.BucketBatchSampler"></a>

## BucketBatchSampler Objects

```python
class BucketBatchSampler(Sampler[List[int]])
```

A batch sampler to create batches with sizes of elements from each pre-defined bucket ranges.
A base batch sampler will be used for each bucket.

Modified from https://github.com/rssrwn/semla-flow/blob/main/semlaflow/data/util.py

**Arguments**:

- `sizes` _np.ndarray_ - A 1D numpy array of real numbers representing the size of each element in the dataset.
- `bucket_ranges` _np.ndarray_ - A 2D numpy array of real numbers with shape (num_buckets, 2) with each row representing the closed boundary of each bucket interval.
- `base_batch_sampler_class` _Sampler_ - Base batch sampler class type, which will be used for each bucket.
- `base_batch_sampler_shared_kwargs` _Dict[str, Any], optional_ - Shared keyword argument dictionary used to initialize all base batch samplers for all buckets.
  Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_individual_kwargs`. Default to  {}.
- `base_batch_sampler_individual_kwargs` _Dict[str, Iterable], optional_ - Keyword argument dictionary used to initialize each bucket batch sampler with the corresponding key value pairs.
  Length of each value in this dict must be equal to len(`bucket_ranges`) (the number of buckets).
  e.g. {'batch_size': [8,10,12]} will be used to create 3 batch samplers with batch_size = 8, 10, 12 for 3 buckets.
  Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_shared_kwargs`.
  Default to  {}.
- `shuffle` _bool_ - A boolean indicating whether to shuffle the dataset and buckets. Defaults to True.


**Raises**:

- `ValueError` - If `sizes` is not a 1D numpy array of real numbers.
- `ValueError` - If `bucket_ranges` is not a 2D numpy array with shape (num_buckets, 2), or each row is not a valid interval, or the intervals overlap.
- `ValueError` - If `base_batch_sampler_individual_kwargs` or `base_batch_sampler_individual_kwargs` is not a keyword argument dictionary.
- `ValueError` - If the length of values in the dict of `base_batch_sampler_individual_kwargs` must be equal to len(bucket_ranges).
- `RuntimeError` - If there is no elements with sizes inside the `bucket_ranges`.

  ---------

**Examples**:


```python
>>> import torch
>>> from bionemo.size_aware_batching.sampler import BucketBatchSampler

>>> # Define the sizes for a dataset
>>> import numpy as np
>>> sizes = np.arange(25)
>>> # Define bucket ranges
>>> bucket_ranges = np.array([[0,5],[6,14],[15,24]])

>>> # Create a bucket batch sampler with torch.utils.data.BatchSampler as base batch sampler
>>> # As there are 3 buckets, there will be 3 base batch samplers with batch sizes 2, 3, and 5.
>>> batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=torch.utils.data.BatchSampler,
        base_batch_sampler_shared_kwargs={'drop_last': False},
        base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
        shuffle=False,
    )

>>> # Iterate over batches of indices that lies in the same bucket and with different batch sizes.
>>> print(list(batch_sampler))
[[0, 1], [2, 3], [4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]

>>> # randomize the dataset and buckets
>>> np.random.seed(0)
>>> batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=torch.utils.data.BatchSampler,
        base_batch_sampler_shared_kwargs={'drop_last': False},
        base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
        shuffle=True,
    )
>>> print(list(batch_sampler))
[[9, 7, 13], [20, 17, 18, 19, 16], [12, 14, 6], [15, 24, 23, 22, 21], [5, 2], [10, 8, 11], [1, 3], [0, 4]]
>>> print(list(batch_sampler))
[[6, 14, 13], [5, 2], [12, 11, 10], [8, 7, 9], [17, 21, 20, 15, 16], [18, 22, 24, 19, 23], [1, 0], [3, 4]]
```
  >>> # Combine with SizeAwareBatchSampler to control the cost of each batch
  >>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler
  >>> item_costs = np.copy(sizes).tolist()
  >>> def cost_of_element(index):
  return item_costs[index]
  >>> np.random.seed(0)
  >>> batch_sampler = BucketBatchSampler(
  sizes=sizes,
  bucket_ranges=bucket_ranges,
  base_batch_sampler_class=SizeAwareBatchSampler,
- `base_batch_sampler_shared_kwargs={"sizeof"` - cost_of_element, "max_total_size": 40},
  base_batch_sampler_individual_kwargs={},
  shuffle=True,
  )
  >>> print(list(iter(batch_sampler)))
  [[9, 7, 13], [20, 17], [12, 14, 6], [18, 19], [5, 2, 1, 3, 0, 4], [16, 15], [24], [23], [10, 8, 11], [22], [21]]

<a id="sampler.BucketBatchSampler.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Generator[List[int], None, None]
```

Iterate over batches of indices.

This function yields batches of indices of elements with sizes from each bucket range.

**Yields**:

- `List[int]` - A batch of indices of elements with sizes from each bucket range.
