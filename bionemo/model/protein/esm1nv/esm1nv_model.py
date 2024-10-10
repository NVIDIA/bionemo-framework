# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import functools
import inspect
import os
from typing import Dict, List, Optional, Union

import torch
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVFieldsMemmapDataset
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.core import Dataset
from nemo.core.neural_types import NeuralType
from nemo.utils import AppState, logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from bionemo.data.dataloader import ProteinBertCollate
from bionemo.data.dataloader.protein_collate import ESM2BertCollate
from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.mapped_dataset import Uniref90ClusterMappingDataset
from bionemo.model.protein.esm1nv.base import ESMnvMegatronBertModel


try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ["ESM1nvModel", "ESM2nvModel", "ESM2nvLoRAModel"]


def esm1nv_build_train_valid_test_datasets(
    cfg: DictConfig, train_valid_test_num_samples: Dict[str, Optional[int]]
) -> List[Dataset]:
    """
    Build train, validation and test for pretraining of MegaMolBartModel.
    Args:
        cfg: config of data components
        train_valid_test_num_samples: dict that specifies split-specific size of loaded dataset
    Returns:
        list of dataset for splits
    """
    cfg = copy.deepcopy(cfg)

    # setting
    use_upsampling: bool = cfg.get("use_upsampling", True)
    data_impl: str = cfg.get("data_impl", None)
    # assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
    dataset_path: str = cfg.get("dataset_path", None)
    assert dataset_path is not None, 'Config "cfg" should contain field "cfg.dataset_path"'

    assert all(
        split in ["train", "val", "test"] for split in train_valid_test_num_samples.keys()
    ), "Incorrect key in train_valid_test_num_samples!"

    datasets = []
    # Build individual datasets.
    for split in train_valid_test_num_samples.keys():
        num_samples = train_valid_test_num_samples[split]
        print(f"{split}:{num_samples}")
        if num_samples is None or num_samples > 0:
            ds_name: Optional[Union[str, List[Union[int, str]]]] = cfg.dataset.get(split, None)
            assert ds_name is not None, (
                f'Config "cfg" should contain field "cfg.dataset.{split}" with name or list of '
                f"names corresponding to the data files used to construct the dataset"
            )
            filepath: str = os.path.join(dataset_path, split, ds_name)
            dataset = build_typed_dataset(
                dataset_paths=filepath,
                data_impl=data_impl,
                use_upsampling=use_upsampling if num_samples is not None else False,
                cfg=cfg,
                num_samples=num_samples,
            )
        else:
            dataset = None
        datasets.append(dataset)  # These can be my train/ val/ test datasets.

    return datasets


class ESM1nvModel(ESMnvMegatronBertModel):
    """
    ESM1nv pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        """
        model_name = self._cfg.tokenizer.get("model_name")
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=model_name,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            legacy=False,
        )
        # patch tokenizer for use with HF esm tokenizer
        if self._cfg.tokenizer.library == "huggingface" and str(model_name).startswith("facebook/esm2"):
            self.tokenizer.tokenizer.vocab = self.tokenizer.tokenizer.get_vocab()

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers=None):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == "single", AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )

        # NOTE (SKH) this was taken directly from megatron, this is the 'single' dataloader type.
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=self.cfg.get("drop_last", True),
        )

        if num_workers is None:
            num_workers = self.cfg.data.num_workers
        # Torch dataloader.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,  # Needs to be set to zero.
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False  # must be False with CSV dataset TODO check with binary
        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False
        if self.cfg.pipeline_model_parallel_size > 1 and self.cfg.data.dynamic_padding:
            raise ValueError("Pipeline model parallelism does not support dynamic_padding.")
        perturb_percent = self._cfg.data.get("perturb_percent", 0.1)
        mask_percent = self._cfg.data.get("mask_percent", 0.8)
        identity_percent = self._cfg.data.get("identity_percent", max(1 - (mask_percent + perturb_percent), 0))
        dataloader.collate_fn = ProteinBertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=self._cfg.data.modify_percent,
            perturb_token_percent=perturb_percent,
            mask_token_percent=mask_percent,
            identity_token_percent=identity_percent,
            dynamic_padding=self.cfg.data.dynamic_padding,
        ).collate_fn

        return dataloader

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        if hasattr(self, "_validation_ds"):
            if self._validation_ds is not None:
                consumed_samples = 0
                self._validation_dl = self.build_pretraining_data_loader(
                    self._validation_ds, consumed_samples, num_workers=0
                )

    def setup_test_data(self, cfg):
        if hasattr(self, "_test_ds"):
            if self._test_ds is not None:
                consumed_samples = 0
                self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg):
        logging.info("Building Bert datasets.")
        global_batch_size = trainer.world_size * model_cfg.micro_batch_size / model_cfg.tensor_model_parallel_size
        # Compute training micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        # limit_val_batches specifies the number of batches.
        max_train_steps = trainer.max_steps * trainer.accumulate_grad_batches

        eval_iters = (max_train_steps // trainer.val_check_interval + 1) * trainer.limit_val_batches
        test_iters = trainer.limit_test_batches

        train_valid_test_num_samples = {
            "train": int(max_train_steps * global_batch_size),
            "val": int(eval_iters * global_batch_size),
            "test": int(test_iters * global_batch_size),
        }

        # Note(@jomitchell) ESM should not be calling megamolbart's dataset builder.
        _train_ds, _validation_ds, _test_ds = esm1nv_build_train_valid_test_datasets(
            cfg=model_cfg.data, train_valid_test_num_samples=train_valid_test_num_samples
        )

        logging.info(f"Length of train dataset: {len(_train_ds)}")
        logging.info(f"Length of val dataset: {len(_validation_ds)}")
        logging.info(f"Length of test dataset: {len(_test_ds) if _test_ds is not None else None}")
        logging.info("Finished building Bert datasets.")
        return _train_ds, _validation_ds, _test_ds

    def build_train_valid_test_datasets(self):
        train, val, test = self._build_train_valid_test_datasets(self.trainer, self._cfg)
        self._train_ds = train
        self._validation_ds = val
        self._test_ds = test
        return self._train_ds, self._validation_ds, self._test_ds

    def on_validation_epoch_end(self) -> None:
        """The function computes and logs three scores:
            - The average cross entropy loss over the validation data
            - The exponential of the averaged loss
            - The average perplexity score over the validation data.

        The estimate of the perplexity is defined as the exponential of the average of the masked CE losses: `exp(-loss_mean)`

        Args:
            outputs: A list of dictionaries, where each dictionary represents the output of a validation step.
            The computed values are logged using the Lightning logger.
        """
        outputs = self.validation_step_outputs
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        average_perplexity = averaged_loss.exp()
        self.log("val_loss", averaged_loss, prog_bar=True, batch_size=1)
        self.log("val_perplexity", average_perplexity, batch_size=1)
        self.log(
            "val_loss_ECE", pow(2, averaged_loss), batch_size=1
        )  # calculate exponential cross entropy loss for logs
        self.log(
            "consumed_samples",
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            batch_size=1,
        )
        self.validation_step_outputs.clear()

    def encode(
        self, tokens_enc: torch.Tensor, enc_mask: torch.Tensor, reconfigure_microbatch: bool = True
    ) -> torch.Tensor:
        """
        Encodes input tokens and return the model's hidden embeddings, supporting different parallel processing modes.

        Parameters:
        - tokens_enc: The encoder input tokens ids, expected to be in the shape of [batch_size, sequence_length].
        - enc_mask: The attention mask associated with the input tokens.
        - reconfigure_microbatch (bool): Flag to reconfigure the micro-batch size for pipeline parallel mode. When running inference
            as an interactive session, you should set it to False.

        Returns:
        - torch.Tensor: The output embeddings of shape [batch_size, sequence_length, hidden_size].
        """
        # TODO: upstream the `encode` function to NeMo as the current BERT class does not have one

        # TODO [sichu] Potential duplication to bionemo/model/utils.py
        # Check whether the DDP is initialized. This is needed when running inference outside of the training loop.
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            if reconfigure_microbatch:
                _reconfigure_microbatch_calculator(
                    rank=0,  # This doesn't matter since it is only used for logging
                    rampup_batch_size=None,
                    global_batch_size=1,
                    micro_batch_size=1,  # Make sure that there is no "grad acc" while encoding.
                    data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
                )

        app_state = AppState()

        global_batch_per_gpu = tokens_enc.size(0)
        encoder_seq_length = tokens_enc.size(1)

        # Reconfigure microbatch calculator here to set num microbatches as expected by the encoding step.
        if reconfigure_microbatch:
            global_batch_size = self.cfg.global_batch_size // self.cfg.micro_batch_size * global_batch_per_gpu
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_size,
                micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while encoding.
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        tensor_shape = [
            encoder_seq_length,
            global_batch_per_gpu,
            self.cfg.hidden_size,
        ]  # expected output shape from the encoder

        # build input arguments description as expected by the encoder's forward method
        batch_for_pipeline = [tokens_enc, enc_mask]
        arg_names = ["bert_model_input", "attention_mask"]

        forward_step_func = self._get_forward_output_only_func(
            arg_names=arg_names,
            output_name="hiddens",  # Set the name of the encoder's outputs
        )

        fwd_bwd_func = (
            get_forward_backward_func()
        )  # Use Megatron's util function to get the correct output tensor in pipeline parallel mode

        output_tensor = fwd_bwd_func(
            forward_step_func=forward_step_func,
            data_iterator=iter(
                [
                    batch_for_pipeline,
                ]
            ),
            model=[self.model],  # Use the encoder module of ESM to get the hidden representations
            forward_only=True,
            num_microbatches=1,
            seq_length=encoder_seq_length,
            micro_batch_size=get_micro_batch_size(),
        )

        if output_tensor:
            output_tensor = output_tensor[0]["hiddens"]
        else:
            # Only the last pipeline stage has the actual output_tensor, for all other model-parallel ranks the output_tensor is None.
            output_tensor = torch.zeros(tensor_shape, dtype=self.autocast_dtype).cuda()

        if self.cfg.get("pipeline_model_parallel_size", 1) > 1:
            # Broadcast the output_tensor from the last pipeline stage to all other model-parallel ranks.
            torch.distributed.broadcast(
                output_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

        # Reset microbatch calculator to what it was before encoding. This adjustment is necessary during pre-training with validation-in-loop,
        # as the main ESM model and the downstream task model have differing batch sizes.
        if reconfigure_microbatch:
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_size,
                micro_batch_size=global_batch_per_gpu,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        # Return the output tensor of the encoder and transpose from [seq_len, batch, hidden] to [batch, seq_len, hidden]
        return output_tensor.transpose(1, 0)

    def _get_forward_output_only_func(self, arg_names: List[str], output_name: str, **kwargs):
        """
        Creates a function that prepare the batches in the order expected by model's forward signature and returns
        its output.

        Parameters:
        ----------
            arg_names: A list of argument names that maps the batch inputs to the positons of arguments expected by the forward method.
            output_name (str): The name of the output to be extracted.
            kwargs - shared arguments (non tensors)

        Returns:
        --------
            function: A function that takes a dataloader iterator and a model as inputs, processes the next batch from the
                    dataloader through the model, and returns the specified output along with a function that return tensor identity.
                    this format is expected by the Megatron's util function `get_forward_backward_func`
        """

        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]

            # map the batch and shared args into forward's positional arguments
            args = self._build_forward_args_from_kwargs(args_name=arg_names, args=batch, **kwargs)
            output = model(*args).contiguous()

            def id_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations ("output" is the default output)
                    output_tensor = output_tensor["output"]

                return output_tensor, {output_name: output_tensor}

            return output, id_func

        return fwd_output_only_func

    def _build_forward_args_from_kwargs(self, args_name: List[str], args: List, **kwargs):
        """
        A helper method that converts arguments into positional arguments (by name)

        Parameters:
        -----------
            args: A list of positional arguments, typically tensors, that are passed to self.model.
            args_name: A list of names corresponding to each positional argument. These names are used to
                match against the allowed 'kwargs' and to order the final argument list.
            kwargs: A dict of {arg name: arg value} (used for non-tensor values).

        Returns:
        --------
            List: Ordered list of arguments, as expected by the self.model's forward method
        """
        # sanity checks
        if len(args) != len(args_name):
            raise ValueError(f"Mismatch between length in args_name ({len(args_name)}) and args ({len(args)})")
        if any(n in kwargs for n in args_name):
            raise ValueError(f"args_name = {args_name} cannot overlap kwargs = {list(kwargs.keys())}")

        # get mapping of kwarg names to arg index based on the self.model.forward's signature
        kwargs_to_arg_idx = self._kwargs_to_arg_idx()

        # collect all arguments
        all_args_name = args_name[:]
        all_args = args[:]
        for k, v in kwargs.items():
            all_args_name.append(k)
            all_args.append(v)

        args_idx = [kwargs_to_arg_idx[n] for n in all_args_name]

        # construct args ordered by name (with None as place-holder)
        forward_args = [None] * (max(args_idx) + 1)
        for i, v in zip(args_idx, all_args):
            forward_args[i] = v

        return forward_args

    @functools.lru_cache(maxsize=None)
    def _kwargs_to_arg_idx(self):
        """
        Returns a dict {kwarg name: arg index} to be used when mapping
        kwargs into a list of args.

        Computed on first call, and then cached.
        """
        # build mapping of kwargs to arg index at first run
        module = self.model.forward
        args_name = inspect.getfullargspec(module)[0][1:]
        kwargs_to_arg_idx = dict(zip(args_name, range(len(args_name))))

        return kwargs_to_arg_idx

    def on_test_epoch_end(self) -> None:
        """
        The function computes and logs three scores:
            - the average cross-entropy loss over the test data,
            - the exponential of the averaged loss, and
            - the average perplexity score over the test data.

        The estimate of the perplexity is defined as the exponential of the average of the masked CE losses: `exp(-loss_mean)`
        This function is called at the end of the testing step `model.test()`.

        Args:
            outputs: A list of dictionaries, where each dictionary represents the output of a validation step.
            The computed values are logged using the NeMo logger.
        """
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f"test_loss: {averaged_loss[0]}")
        logging.info(f"test_loss_ECE: {pow(2, averaged_loss[0])}")
        logging.info(f"test_perplexity: {averaged_loss[0].exp()}")
        self.test_step_outputs.clear()

    @property
    def input_names(self):
        return [
            "input_ids",
            "attention_mask",
        ]

    @property
    def output_names(self):
        return ["output"]

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"input_ids": {0: "batch", 1: "time"}, "attention_mask": {0: "batch", 1: "time"}}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output": {0: "batch", 1: "time", 2: "size"}}


class ESM2nvModel(ESM1nvModel):
    """
    Extends the ESM1nv model by customizing dataset construction to mimic ESM2's implementation.

    This model introduces a dataset structure that not only retains elements from ESM1nv but also
    incorporates new fields specifically tailored for ESM2nv. One significant change involves using
    NeMoUpsampling to upsample UniRef50 cluster IDs and pre-computing samples from UniRef90 via a mapped dataset.

    The configuration files are designed to incorporate specific argument structures. A sample configuration
    demonstrating the dataset configuration can be viewed in `examples/protein/esm2nv/conf/base_config.yaml`

    Args:
        cfg (DictConfig): The configuration object, typically constructed from a YAML file.
        trainer (Trainer): The training instance associated with this model.

    Attributes:
        dataset_path (str): Parent directory containing train, test, and validation data.
        dataset (dict): Specifies the data file range for training, testing, and validation.
        data_impl (str): Implementation choice, like "csv_mmap".
        data_impl_kwargs (dict): Arguments specific to the data implementation choice.
        uf50_datapath (str): Path to the raw UniRef50 fasta file.
        uf90_datapath (str): Path to the raw UniRef90 fasta file.
        cluster_mapping_tsv (str): TSV file mapping UniRef50 cluster IDs to UniRef90 entries.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    def forward(self, *args, **kwargs):
        output_tensor = super().forward(*args, **kwargs)
        return output_tensor

    def build_train_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
    ) -> Union[Uniref90ClusterMappingDataset, CSVFieldsMemmapDataset]:
        """Constructs a train dataset either
            1. following the default training strategy in ESM2 publication, if model_cfg.data.train.custom_pretraining_fasta_path is None, or
            2. from the manually specified fasta (model_cfg.data.train.custom_pretraining_fasta_path)

        Args:
            num_samples: The number of samples in the dataset
            model_cfg: If no custom fasta is specified, a config file that contains certain keys that tell us:
                1. Where the UF50 dataset lies
                2. Where the UF90 dataset lies
                3. How we want to map between the two etc.
        Returns:
            train_dataset (Unionp[Uniref90ClusterMappingDataset, CSVFieldsMemmapDataset]): Training dataset.
        """
        # Create the custom training dataset
        if model_cfg.data.train.custom_pretraining_fasta_path is None:
            train_ds = self._build_uniref50_90_train_dataset(model_cfg=model_cfg, num_samples=num_samples)
        else:
            # overrride config with uf90 settings
            model_cfg = copy.deepcopy(model_cfg)
            model_cfg.data.train.data_impl = model_cfg.data.train.uf90.data_impl
            model_cfg.data.train.data_impl_kwargs = model_cfg.data.train.uf90.data_impl_kwargs

            # TODO raise exception if cannot read the dataset properly
            train_ds = build_typed_dataset(
                dataset_paths=os.path.join(model_cfg.data.train.dataset_path, "train", model_cfg.data.train.range),
                data_impl=model_cfg.data.train.uf90.data_impl,
                cfg=model_cfg.data.train,
                use_upsampling=model_cfg.data.train.use_upsampling,
                num_samples=num_samples,
            )

        return train_ds

    @staticmethod
    def _build_uniref50_90_train_dataset(
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
    ) -> Uniref90ClusterMappingDataset:
        """Build a pretraining train dataset as described in ESM2 publication, in which each uniref90
        sequence is mapped to a "cluster-center" in uniref50. Sampling is done at uniref50 level while
        the actual training is on uniref90 sequences.

        Ref: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf

        Args:
            num_samples (int) : The number of samples in the dataset.
            model_cfg (DictConfig): ESM2 pretraining config.
        Returns:
            train_dataset (Uniref90ClusterMappingDataset): A UF90 cluster mapping train dataset.
        """
        train_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.train.dataset_path, "train", model_cfg.data.train.range),
            data_impl=model_cfg.data.train.data_impl,
            cfg=model_cfg.data.train,
            use_upsampling=model_cfg.data.train.use_upsampling,
            num_samples=num_samples,
        )
        uniref90_dataset = build_typed_dataset(
            dataset_paths=os.path.join(
                model_cfg.data.train.uf90.uniref90_path, "uf90_csvs", model_cfg.data.train.range
            ),
            data_impl=model_cfg.data.train.uf90.data_impl,
            cfg=model_cfg.data.train.uf90,
            use_upsampling=False,  # TODO(@jomitchell): Jasleen can these be upsampled?
            num_samples=None,
        )

        index_mapping_dir = train_ds.index_mapping_dir
        # TODO(@jomitchell): Dataset refactor: Path to mmap_dataset should be full path to `train`
        mmap_dataset_path: str = os.path.join(model_cfg.data.train.dataset_path, "train")
        # Setup the resampling
        train_dataset = Uniref90ClusterMappingDataset(
            uniref50_dataset=train_ds,
            uniref90_dataset=uniref90_dataset,
            data_prefix="train",  # used for index creation
            seed=model_cfg.seed,  # used for rng, although awkward because global statehood
            index_mapping_dir=index_mapping_dir,  # stores index
            cluster_map_starts_fn=f"{mmap_dataset_path}/starts.mmap",
            cluster_map_counts_fn=f"{mmap_dataset_path}/counts.mmap",
            name=train_ds.name,
        )
        return train_dataset

    def build_val_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
        limit_batches_scale_factor: Optional[float] = None,
    ):
        # TODO: If `num_samples is None` do we load the full dataset?
        # IF not default none -> raise warning.
        val_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.val.dataset_path, "val", model_cfg.data.val.range),
            data_impl=model_cfg.data.val.data_impl,
            cfg=model_cfg.data.val,
            use_upsampling=model_cfg.data.val.use_upsampling,
            num_samples=num_samples,
            limit_batches_scale_factor=limit_batches_scale_factor,
        )
        return val_ds

    def build_test_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
        limit_batches_scale_factor: Optional[float] = None,
    ):
        """Constructs the test dataset."""
        test_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.test.dataset_path, "test", model_cfg.data.test.range),
            data_impl=model_cfg.data.test.data_impl,
            cfg=model_cfg.data.test,
            use_upsampling=model_cfg.data.test.use_upsampling,
            num_samples=num_samples,
            limit_batches_scale_factor=limit_batches_scale_factor,
        )
        return test_ds

    def build_train_valid_test_datasets(self):
        if self.trainer.limit_test_batches == 0:
            raise ValueError("trainer.limit_test_batches is set to 0 which means you will have no test data.")

        if self.trainer.limit_val_batches == 0:
            raise ValueError(
                "trainer.limit_val_batches is set to 0 which means you will have no val data."
                "Please use fractional values 0<x<1 to get fractional data, or 1 for full data."
                "Or an int > 1 to specify the exact number of batches that you want."
            )

        # If val upsampling is false, and we expect to be up/ downsampling val.
        if not self._cfg.data.val.use_upsampling and self.trainer.limit_val_batches != 1.0:
            raise ValueError(
                f"config.model.data.val.use_upsampling is {self._cfg.data.val.use_upsampling} but self.trainer.limit_val_batches is: "
                f"{self.trainer.limit_val_batches} which is not 1.0"
            )

        # If test upsampling is false, and we expect to be downsampling test.
        if not self._cfg.data.test.use_upsampling and self.trainer.limit_test_batches != 1.0:
            raise ValueError(
                f"config.model.data.test.use_upsampling is {self._cfg.data.test.use_upsampling} but self.trainer.limit_test_batches is: {self.trainer.limit_test_batches}"
                "which is not equal to 1.0"
            )

        global_batch_size = self.trainer.world_size * self._cfg.micro_batch_size / self._cfg.tensor_model_parallel_size
        # Compute training micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        num_train_samples = int(max_train_steps * global_batch_size)

        # Num_val_runs is the number of times we will run validation.
        num_val_runs = max_train_steps // self.trainer.val_check_interval + 1

        limit_val_batches_scale_factor = None

        # If we want to use a set number of batches during validation, and that number is > 1.
        if isinstance(self.trainer.limit_val_batches, int):
            eval_iters = num_val_runs * self.trainer.limit_val_batches
            num_val_samples = int(eval_iters * global_batch_size)
        # If we use a float for a fractional piece of the dataset. Then we need to scale the number of iterations accordingly
        elif isinstance(self.trainer.limit_val_batches, float):
            # Note: we need to do the equation:
            # total_val_iterations = (len(dataset) * limit_val_batches) / global_batch_size but we don't have len(dataset) until further downstream.
            # So we are going to save the scale factor here and multiply by len(dataset) * limit_val_batches_scale_factor later on.
            total_val_iterations = 1.0 / global_batch_size
            eval_iters = num_val_runs * total_val_iterations
            limit_val_batches_scale_factor = self.trainer.limit_val_batches
            num_val_samples = eval_iters * global_batch_size
        else:
            raise ValueError(
                f"self.trainer.limit_val_batches is of type {type(self.trainer.limit_val_batches)}"
                "which is not supported."
            )

        limit_test_batches_scale_factor = None
        if isinstance(self.trainer.limit_test_batches, int):
            test_iters = self.trainer.limit_test_batches
            num_test_samples = int(test_iters * global_batch_size)
        elif isinstance(self.trainer.limit_test_batches, float):
            limit_test_batches_scale_factor = self.trainer.limit_test_batches
            num_test_samples = 1
        else:
            raise ValueError(
                f"self.trainer.limit_test_batches is of type {type(self.trainer.limit_test_batches)}"
                "which is not supported."
            )

        self._train_ds = self.build_train_dataset(model_cfg=self._cfg, num_samples=num_train_samples)
        logging.info(f"Length of train dataset: {len(self._train_ds)}")

        self._validation_ds = self.build_val_dataset(
            model_cfg=self._cfg,
            num_samples=num_val_samples,
            limit_batches_scale_factor=limit_val_batches_scale_factor,
        )
        logging.info(f"Length of val dataset: {len(self._validation_ds)}")

        self._test_ds = self.build_test_dataset(
            model_cfg=self._cfg,
            num_samples=num_test_samples,
            limit_batches_scale_factor=limit_test_batches_scale_factor,
        )
        logging.info(f"Length of test dataset: {len(self._test_ds)}")

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers=None):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == "single", AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )
        dataloader = super().build_pretraining_data_loader(
            dataset=dataset, consumed_samples=consumed_samples, num_workers=num_workers
        )

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False

        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False
        if self.cfg.pipeline_model_parallel_size > 1 and self.cfg.data.dynamic_padding:
            raise ValueError("Pipeline model parallelism does not support dynamic_padding.")
        modify_percentage = self._cfg.data.get("modify_percent", 0.15)
        perturb_percentage = self._cfg.data.get("perturb_percent", 0.1)
        mask_percentage = self._cfg.data.get("mask_percent", 0.8)
        identity_percentage = self._cfg.data.get(
            "identity_percent", max(1 - (mask_percentage + perturb_percentage), 0)
        )

        dataloader.collate_fn = ESM2BertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=modify_percentage,
            perturb_token_percent=perturb_percentage,
            mask_token_percent=mask_percentage,
            identity_token_percent=identity_percentage,
            dynamic_padding=self.cfg.data.dynamic_padding,
        ).collate_fn
        return dataloader

    def setup_validation_data(self, cfg):
        if hasattr(self, "_validation_ds"):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, num_workers=0
            )

    def setup_test_data(self, cfg):
        if hasattr(self, "_test_ds"):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)


class ESM2nvLoRAModel(NLPAdapterModelMixin, ESM2nvModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.setup_complete = True
