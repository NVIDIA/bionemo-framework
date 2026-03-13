# Writing Megatron-LM Compatible Datamodules

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) relies on determinism in the training dataset classes to ensure
that input tensors are initialized correctly across model-parallel ranks (see [NeMo2 Parallelism](./nemo2.md)). As a
consequence, ensure that the new dataset classes preserve the required determinism. Common operations such as data
augmentation and masking can cause `dataset[i]` to return random results for a given index, breaking this megatron
contract.

## Multi-Epoch Training

One training regime where this limitation is most apparent is multi-epoch training, where standard training recipes
would apply different random masks or different data augmentation strategies each time the data is encountered. BioNeMo
provides some utilities that make multi-epoch training easier, while obeying the determinism requirements of
megatron.

The [MultiEpochDatasetResampler][bionemo.core.data.multi_epoch_dataset.MultiEpochDatasetResampler] class simplifies the
process of multi-epoch training, where the data should both be re-shuffled each epoch with different random effects
applied each time the data is seen. To be compatible with this resampler, the provided dataset class's `__getitem__`
method should accept a [EpochIndex][bionemo.core.data.multi_epoch_dataset.EpochIndex] tuple that contains both an epoch
and index value. Random effects can then be performed by setting the torch random seed based on the epoch value:

```python
class MyDataset:
    def __getitem__(self, idx: EpochIndex):
        rng = torch.Generator()
        rng.manual_seed(idx.epoch)
        ...
```

!!! bug "Avoid `torch.manual_seed`"

```
Megatron-LM handles torch seeding internally. Calling `torch.cuda.manual_seed` inside the user-provided dataset
can cause issues with model parallelism. See [megatron/core/tensor_parallel/random.py#L198-L199](
https://github.com/NVIDIA/Megatron-LM/blob/dddecd19/megatron/core/tensor_parallel/random.py#L198-L199) for more
details.
```

For deterministic datasets that still want to train for multiple epochs with epoch-level shuffling, the
[IdentityMultiEpochDatasetWrapper][bionemo.core.data.multi_epoch_dataset.IdentityMultiEpochDatasetWrapper] class can
simplify this process by wrapping a dataset that accepts integer indices and passes along the
[EpochIndex][bionemo.core.data.multi_epoch_dataset.EpochIndex] index values from the resampled dataset.

```python
class MyDeterministicDataset:
    def __getitem__(self, index: int): ...


dataset = IdentityMultiEpochDatasetWrapper(MyDeterministicDataset())
for sample in MultiEpochDatasetResampler(dataset, num_epochs=3, shuffle=True):
    ...
```

## Training Resumption

The actively maintained framework no longer ships the older Megatron-specific datamodule helpers that used to manage
sample-exact training resumption. If you are maintaining a legacy Megatron data pipeline, preserve the same contract
explicitly in your datamodule:

- persist enough dataloader state to resume from the correct sample index
- distinguish train, validation, and test loader behavior explicitly
- avoid reusing training-resume state for validation or test loaders

For new training code, prefer the self-contained implementations in `bionemo-recipes`, where checkpointing and
dataloader state management live alongside the training entrypoints.

## Testing Datasets for Megatron Compatibility

For legacy Megatron-compatible datasets, the key invariant is still determinism: repeated calls with the same effective
index should yield the same sample. In practice, tests should confirm that:

- repeated indexing is deterministic
- epoch-aware randomization is driven only by the epoch component of the index
- `torch.manual_seed` is not called inside dataset access paths

Current large-scale training examples live in `bionemo-recipes`, so recipe-local tests are the best reference for how
to validate these assumptions today.
