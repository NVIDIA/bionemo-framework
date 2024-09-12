# bionemo-size-aware-batching

To install, execute the following:
```bash
pip install -e .
```

To run unit tests, execute:
```bash
pytest -v .
```

<a id="utils"></a>

# utils

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

- `dataset` _Iterable[Data]_ - An iterable containing the input data.
- `work` _Callable[[Data], Feature]_ - A function that takes a data point and returns its corresponding feature. This is where
  the main computation happens and memory allocations are tracked.
- `device` _torch.device_ - The target Torch CUDA device.
- `cleanup` _Optional[Callable[[], None]]_ - A function that is called after each iteration to perform any necessary cleanup.


**Returns**:

  Tuple[List[Feature], List[int]]: A tuple containing the collected features and their corresponding memory usage statistics.


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
>>> # holded during the work(). This cleanup function will be called
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

# sampler

<a id="sampler.size_aware_batching"></a>

#### size\_aware\_batching

```python
def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: int,
    collate_fn: Optional[Callable[[Iterable[Data]], Any]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None
) -> Generator[Any, None, None]
```

A generator that batches elements from an iterable while ensuring that the
total size of each batch does not exceed a specified maximum. This can be
useful for both indexible data or non-indexible but iterable data.

**Arguments**:

- `dataset` _Iterable[Data]_ - The input iterable.
- `max_total_size` _int_ - The maximum total size of each batch.
  sizeof (Callable[[Data], Real]):
  A function or mapping that returns the size of each element in `dataset`.
  collate_fn (Optional[Callable[[Iterable[Data]], Any]], optional):
  An optional function to collate batches. Defaults to None.
- `info_logger` _Optional[Callable[[str], None]], optional_ - A function to log info.
  Defaults to None.
- `warn_logger` _Optional[Callable[[str], None]], optional_ - A function to log warnings.
  Defaults to None.


**Yields**:

  Generator[Any, None, None]: A generator that yields batches from `dataset`.

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

- `sampler` _Union[Sampler[List[int]], Iterable[int]]_ - The underlying sampler.
- `sizeof` _Callable[[int], Real]_ - A function that returns the size at each index.
- `max_total_size` _Real_ - The maximum total size of a mini-batch.
- `info_logger` _Optional[Callable[[str], None]], optional_ - A function to log info.
  Defaults to a lambda function that print.
- `warn_logger` _Optional[Callable[[str], None]], optional_ - A function to log warnings.
  Defaults to a lambda function that warns.


**Raises**:

- `TypeError` - If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
- `ValueError` - If max_total_size is not a positive number.

<a id="sampler.SizeAwareBatchSampler.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Generator[List[int], None, None]
```

Iterate over batches of indices.

This function yields batches of indices that do not exceed the maximum total size.

**Yields**:

- `List[int]` - A batch of indices that do not exceed the maximum total size.
