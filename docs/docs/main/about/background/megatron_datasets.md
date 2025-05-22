# Writing Megatron-LM Compatible Datamodules

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) relies on determinism in the training dataset classes to ensure
that input tensors are initialized correctly across model-parallel ranks (see [NeMo2 Parallelism](./nemo2.md)). As a
consequence, new dataset classes must be careful to preserve the required determinism. Common operations such as data
augmentation, masking, etc. can cause `dataset[i]` to return random results for a given index, breaking this megatron
contract.


## Multi-Epoch Training

One training regime where this limitation is most apparent is is multi-epoch training, where standard training recipes
would apply different random masks or different data augmentation strategies each time the data is encountered. BioNeMo
provides a number of utilities that make multi-epoch training easier while still obeying the determinism requirements of
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

    Megatron-LM handles torch seeding internally. Calling `torch.cuda.manual_seed` inside the user-provided dataset
    can cause issues with model parallelism. See [megatron/core/tensor_parallel/random.py#L198-L199](
    https://github.com/NVIDIA/Megatron-LM/blob/dddecd19/megatron/core/tensor_parallel/random.py#L198-L199) for more
    details.

For deterministic datasets that still want to train for multiple epochs with epoch-level shuffling, the
[IdentityMultiEpochDatasetWrapper][bionemo.core.data.multi_epoch_dataset.IdentityMultiEpochDatasetWrapper] class can
simplify this process by wrapping a dataset that accepts integer indices and passing along the
[EpochIndex][bionemo.core.data.multi_epoch_dataset.EpochIndex] index values from the resampled dataset.

```python
class MyDeterministicDataset:
    def __getitem__(self, index: int):
        ...

dataset = IdentityMultiEpochDatasetWrapper(MyDeterministicDataset())
for sample in MultiEpochDatasetResampler(dataset, num_epochs=3, shuffle=True):
    ...
```

## Training Resumption
To ensure identical behavior with and without job interruption, BioNeMo provides [MegatronDataModule][bionemo.llm.data.datamodule.MegatronDataModule] to save and load state dict for training resumption, and provides [WrappedDataLoader][nemo.lightning.data.WrappedDataLoader] to add a `mode` attribute to [DataLoader][torch.utils.data.DataLoader].

```python
class MyDataModule(MegatronDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ...

    def train_dataloader(self):
        self.update_init_global_step()  # required to set the correct `global_step` for resumption
        return WrappedDataLoader(
            ...,  # any other arguments for DataLoader
            mode="train",
        )

    def val_dataloader(self):
        self.update_init_global_step()  # required to set the correct `global_step` for resumption
        return WrappedDataLoader(
            ...,  # any other arguments for DataLoader
            mode="validation",
        )

    def test_dataloader(self):
        self.update_init_global_step()  # required to set the correct `global_step` for resumption
        return WrappedDataLoader(
            ...,  # any other arguments for DataLoader
            mode="test",
        )
```

!!! note "MegatronDataModule"

    Users will see non-overlapping training curve if their datamodule is not inheritting from `MegatronDataModule`, unless similar logics are handled by the users. In `MegatronDataModule`, `self.update_init_global_step()` must be called right before the dataloaders are returned to ensure that training resumes with the correct sample index instead of restarting from 0 everytime. We recommend users to inherit from `MegatronDataModule` similar to the pattern above.

!!! note "WrappedDataLoader"

    The `WrappedDataLoader` class is a wrapper around the PyTorch DataLoader class that adds the `mode` attribute to the dataloader. The dataloader will resume from the last sample index only when mode is 'train'. `val_dataloader` and `test_dataloader` are unaffected.

    WARNING: 'train' is the default value of `mode` in `WrappedDataLoader`. If not set, users might find their validation/test dataloader changes behavior by resuming from a non-zero sample index.

## Testing Datasets For Megatron Compatibility

BioNeMo also provides utility functions for test suites to validate that datasets conform to the megatron data model.
The [assert_dataset_compatible_with_megatron][bionemo.testing.data_utils.assert_dataset_compatible_with_megatron]
function calls the dataset with identical indices and ensures the outputs are identical, while also checking to see if
`torch.manual_seed` was used.

!!! example "Example datasets in BioNeMo"

    The [ESMMaskedResidueDataset][bionemo.esm2.data.dataset.ESMMaskedResidueDataset] demonstrates one approach for
    leveraging [EpochIndex][bionemo.core.data.multi_epoch_dataset.EpochIndex] indices to perform epoch-level
    randomization within the confines of megatron's data model.
