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


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from rdkit import Chem, RDLogger

from bionemo.data import DatasetTypes
from bionemo.model.utils import get_from_encoder_or_model


# Disable logging of invalid SMILES moloecules
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = ["MolEncDecModelBase"]


class MolEncDecModelBase(MegatronLMEncoderDecoderModel, ABC):
    """
    Base class for encoder-decoder molecule models with abstract methods to be implemented:
    - _setup_collate
    - _load_train_valid_test_datasets
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self._check_scheduler(cfg)
        super().__init__(cfg, trainer=trainer)
        pad_size_divisible_by_8 = True if get_from_encoder_or_model(self._cfg, "masked_softmax_fusion") else False
        self._collate_fn = self._setup_collate(pad_size_divisible_by_8)
        self.sequence_search_method = getattr(self._cfg.decoder, "sequence_sampling_method", "greedy-search")
        # NOTE: if using 'beam-search' make sure you also set some kwargs in your config
        #   the defaults for beam_search are to have beam_size=1 which is very similar to greedy-search.
        #  model:
        #    decoder:
        #      sequence_sampling_method: "beam-search"
        #      sequence_sampling_kwargs:
        #        beam_size: 3 # larger numbers will be slower, but will return closer to the global maximum P(seq|hidden)
        #        keep_only_best_tokens: True # Set this so that only the best result is returned.
        self.sequence_search_kwargs = getattr(self._cfg.decoder, "sequence_sampling_kwargs", {})

    @abstractmethod
    def _setup_collate(self, pad_size_divisible_by_8: bool):
        """
        Sets up collate fn that is required by dataloader used to finetune MolMIM
        Args:
            pad_size_divisible_by_8: should torch.Tensors be padded to sizes divisible by 8?
        Returns:
            callable collate fn
        """
        raise NotImplementedError

    @abstractmethod
    def _load_train_valid_test_datasets(self, train_n_samples: Optional[int] = None):
        """
        Helper method that sets instance variables corresponding to train, val and test datasets
        Args:
            train_n_samples: number of samples to limit training set to, if defined. Validation/Test sets should be controlled with trainer.limit_val_batches and trainer.limit_test_batches
        """
        raise NotImplementedError

    def _check_scheduler(self, cfg):
        """Warn if maximum learning rate with Noam is less than minimum learning rate"""
        # TODO add to Noam Scheduler in NeMo
        if cfg.optim.sched.name == "NoamAnnealing":
            if cfg.optim.sched.warmup_steps:
                warmup_steps = cfg.optim.sched.warmup_steps
            else:
                warmup_steps = int(cfg.optim.sched.warmup_ratio * cfg.optim.sched.max_steps)
            max_lr = cfg.optim.lr * cfg.optim.sched.d_model ** (-0.5) * warmup_steps ** (-0.5)
            min_lr = cfg.optim.sched.min_lr
            if max_lr <= min_lr:
                logging.warning(
                    f"Warning: maximum learning rate for Noam Scheduler ({max_lr}) is less than minimum ({min_lr})."
                )
        return

    def _get_train_num_samples(self) -> int:
        """
        Helper method that calculates number of samples needed for train, val and test steps
        based on training length (max_steps, limited batches or number), batch size and number of devises used
        Returns:
            dicts with number of samples needed for train, val and test steps
        """
        tensor_model_parallel_size = self._cfg.get("tensor_model_parallel_size", 1)
        global_batch_size = (
            self.trainer.world_size * self._cfg.micro_batch_size / tensor_model_parallel_size
        ) * self.trainer.limit_train_batches
        return int(self.trainer.max_steps * global_batch_size)

    def build_train_valid_test_datasets(self):
        logging.info(f"Building datasets for {type(self).__name__}")

        if self._cfg.data.get("dataset_type", None) is not None:
            dataset_types = DatasetTypes.__members__
            if self._cfg.data.get("dataset_type") not in dataset_types:
                raise ValueError(
                    f"dataset_type must be in {dataset_types}. Found {self._cfg.data.get('dataset_type')}"
                )

        train_n_samples = self._get_train_num_samples()
        self._load_train_valid_test_datasets(train_n_samples=train_n_samples)

        if self._train_ds is not None:
            logging.info(f"Length of train dataset: {len(self._train_ds)}")
        else:
            logging.info("No train dataset is used")
        if self._validation_ds is not None:
            logging.info(f"Length of val dataset: {len(self._validation_ds)}")
        else:
            logging.info("No validation dataset is used")
        if self._test_ds is not None:
            logging.info(f"Length of test dataset: {len(self._test_ds)}")
        else:
            logging.info("No test dataset is used")
        logging.info("Finished building MegaMolBART datasets.")
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers=None):
        """Build dataloader given an input dataset. You can try setting model.data.dataloader_type to one of [single, cyclic] for different sampling procedures."""

        dataloader = super().build_pretraining_data_loader(dataset, consumed_samples, num_workers)
        if dataloader is not None:
            # set to our custom colate function
            dataloader.collate_fn = self._collate_fn
        return dataloader

    def process_global_batch(self, global_batch):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = (
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"],
        )
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = [
            t.to(self.device) for t in (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask)
        ]

        return (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask)

    def get_step_logits(self, batch, batch_idx):
        """Teacher forcing"""
        arg_names = ["enc_input_ids", "enc_attn_mask", "dec_input_ids", "dec_attn_mask"]
        (
            encoder_input_ids,
            decoder_input_ids,
            loss_mask,
            lm_labels,
            encoder_attn_mask,
            decoder_attn_mask,
        ) = self.process_global_batch(batch)

        batch_for_pipeline = [
            encoder_input_ids,
            encoder_attn_mask,
            decoder_input_ids,
            decoder_attn_mask,
        ]
        forward_step_func = self._get_forward_output_only_func(arg_names=arg_names, output_name="token_logits")
        fwd_bwd_function = get_forward_backward_func()

        encoder_seq_length = encoder_input_ids.size(1)

        output_tensor = fwd_bwd_function(
            forward_step_func=forward_step_func,
            data_iterator=iter(
                [
                    batch_for_pipeline,
                ]
            ),
            model=[self.enc_dec_model],
            forward_only=True,
            num_microbatches=get_num_microbatches(),
            seq_length=encoder_seq_length,
            decoder_seq_length=encoder_seq_length,
            micro_batch_size=get_micro_batch_size(),
        )

        if output_tensor:
            # collect across micro batches
            logits_tensors_list = [o["token_logits"] for o in output_tensor]
            logits_tensor = torch.concat(logits_tensors_list)
        else:
            # we're not on the last pipeline stage so no output
            logits_tensor = []

        return logits_tensor

    def _test_validation_step(self, step_outputs, dataloader_iter, batch_idx, dataloader_idx=0):
        """
        Shared code for validation and test step. See parent NeMo class for how this gets called by def validation_step() along with
            any other nemo specific optimizations that may happen outside.
        """
        # check if the dataloader is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            # NeMo expects us to return None once we are beyond the last batch.
            return
        batch = next(dataloader_iter)
        dataloader_iter = iter([batch])
        # store the outputs of the forward pass in last element of step_outputs
        loss_dict = super()._test_validation_step(step_outputs, dataloader_iter, batch_idx, dataloader_idx)

        # update loss_dict, these are the logits given teacher forcing, eg given the target sample.
        token_logits = self.get_step_logits(batch, batch_idx)
        (
            tokens_enc,
            tokens_dec,
            loss_mask,
            labels,
            enc_mask,
            dec_mask,
        ) = self.process_global_batch(batch)

        target_smiles = batch["target_smiles"]
        token_logits[:, :, self.tokenizer.vocab_size :] = -float("Inf")  # never pick padded tokens

        log_n_batches = 10
        log_mol = True if batch_idx < log_n_batches else False  # TODO enable logging in yaml config
        metrics = self.calculate_metrics(
            token_logits=token_logits,
            loss_mask=loss_mask,
            labels=labels,
            tokens_enc=tokens_enc,
            enc_mask=enc_mask,
            target_smiles=target_smiles,
            batch_idx=batch_idx,
            log_char=False,
            log_mol=log_mol,
        )

        # add metrics to loss_dict
        loss_dict.update(metrics)

        # store back in last element of step_outputs
        step_outputs[-1] = loss_dict

        return loss_dict

    def _test_validation_epoch_end(self, step_outputs, prefix):
        """
        Shared logging for validation and test
        """
        logging.info(f"Finishing {prefix} epoch")
        averaged_loss = super()._test_validation_epoch_end(step_outputs, prefix)

        # Log results
        log_list = []
        for metric_name, metric_val in averaged_loss.items():
            metric_name = metric_name.replace("_", " ").title()
            log_list.append(f"{metric_name}: {metric_val:.2f}")
        logging.info(f"{prefix.title()} Results: " + ", ".join(log_list))

        return averaged_loss

    def get_hiddens_mask(self, enc_attn_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Convenience function for getting the perceiver aware hiddenes mask given a token based hiddens mask"""
        return self.enc_dec_model.enc_dec_model.get_hiddens_mask(enc_attn_mask, *args, **kwargs)

    def sample_molecules(
        self,
        tokens_enc,
        enc_mask,
        hidden_states=None,
        sampling_method: str = "greedy-search",
        sampling_kwargs: dict = {},
    ):
        """Autoregressively sample SMILES molecules from encoder hidden state, if provided, or encode tokens_enc into a hidden state and sample that.

        Args:
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            sampling_method (str): a sampling method to use in the decoding iterations
            sampling_kwargs (dict): dict with arguments to be passed to the sampling function. If
                using `sampling_method="beam-search"` make sure to pass
                `sampling_kwargs={"beam_size": 5, "keep_only_best_tokens": True}`
                so that the single best result from beam search is returned given some desired memory.
                Please refer to the method get_sampling_token_fn in NeMO to see which arguments are relevant
                for a chosen sampling_method.
        Returns:
            sampled_smiles (list[str]): a list of sampled SMILES strings
        """

        self.freeze()

        predicted_tokens_ids, log_probs = self.decode(
            tokens_enc,
            enc_mask,
            self._cfg.max_position_embeddings,
            enc_output=hidden_states,
            sampling_method=sampling_method,
            sampling_kwargs=sampling_kwargs,
        )

        supported_dims = [2, 3]
        tensor_dim = len(predicted_tokens_ids.size())
        predicted_tokens_ids = predicted_tokens_ids.cpu().detach().numpy().tolist()
        if tensor_dim == 2:
            sampled_smiles, _ = self._process_predicted_tokens_ids(predicted_tokens_ids)
        elif tensor_dim == 3:
            sampled_smiles = []
            for predicted_tokens_ids_i in predicted_tokens_ids:
                sampled_smiles_i, predicted_tokens_ids_i = self._process_predicted_tokens_ids(predicted_tokens_ids_i)
                sampled_smiles.append(sampled_smiles_i)
        else:
            raise ValueError(
                f"The shape of the tensor with token_ids is not supported. "
                f"Supported numbers of dims: {supported_dims}"
            )
        self.unfreeze()
        return sampled_smiles

    def _process_predicted_tokens_ids(
        self, predicted_tokens_ids: List[List[int]]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Prunes tokens ids by eos / padding and convert them to SMILES
        Args:
            predicted_tokens_ids: list with sequences of tokens
        Returns:
            list with corresponding SMILES str
        """
        #
        # Prune tokens by eos / padding and convert to SMILES
        predicted_tokens_ids = deepcopy(predicted_tokens_ids)
        for item, predicted_tokens_ in enumerate(predicted_tokens_ids):
            if self.tokenizer.eos_id in predicted_tokens_:
                idx = predicted_tokens_.index(self.tokenizer.eos_id)
                predicted_tokens_ids[item] = predicted_tokens_[:idx]
            else:
                # NB: this is slightly different from previous version in that pad tokens can be in the middle of sequence
                predicted_tokens_ids[item] = [id for id in predicted_tokens_ if id != self.tokenizer.pad_id]
        predicted_tokens_text = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        sampled_smiles = self.tokenizer.tokens_to_text(predicted_tokens_text)
        return sampled_smiles, predicted_tokens_ids

    def calculate_character_accuracy(self, token_logits, loss_mask, labels, batch_idx=None, log=False):
        """Character (token) level accuracy when model forward has been guided by the true tokens at the i-1 positions.

        Args:
            token_logits (torch.Tensor, float): softmax values for all tokens
            loss_mask (torch.Tensor, float): binary mask for ignored data (1=active, 0=mask), must be float
            labels (torch.Tensor, long): token IDs for correct output

        Returns:
            float: character accuracy value
        """

        # Get most probable token
        _, predicted_tokens = torch.max(token_logits, dim=2)
        correct_tokens = torch.eq(labels, predicted_tokens) * loss_mask  # NB: mask includes EOS in calculation

        # Calculate percent of correct tokens
        num_correct = correct_tokens.detach().sum()
        total = loss_mask.detach().sum()
        character_accuracy = num_correct / total

        if log:
            logging.info(f"Character accuracy for batch {batch_idx}:")
            for idx in range(predicted_tokens.shape[0]):
                mask = loss_mask[idx].to(int)
                correct_ = labels[idx][mask] == predicted_tokens[idx][mask]
                logging.info(f"     Sample {idx} has {correct_} / {sum(mask)}")

        return character_accuracy

    def calculate_molecular_accuracy(self, tokens_enc, enc_mask, target_smiles, batch_idx=None, log=False):
        """Calculate molecular accuracy (with canonicalization)

        Args:
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            target_smiles (str): ground truth for canonicalized SMILES

        Returns:
            float, float: molecular accuracy and percent invalid
        """

        sampled_smiles = self.sample_molecules(
            tokens_enc,
            enc_mask,
            sampling_method=self.sequence_search_method,
            sampling_kwargs=self.sequence_search_kwargs,
        )
        sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]

        invalid = [mol is None for mol in sampled_mols]

        invalid_mol_str = "Unknown"

        def try_canonicalize(mol) -> str:
            if mol is None:
                return invalid_mol_str
            try:
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                return invalid_mol_str

        canonical_smiles = [try_canonicalize(mol) for mol in sampled_mols]
        correct_smiles = [target_smiles[idx] == smi for idx, smi in enumerate(canonical_smiles)]

        num_correct = sum(correct_smiles)
        total = len(correct_smiles)
        num_invalid = sum(invalid)
        percent_invalid = torch.tensor([num_invalid / total]).to(tokens_enc.device)
        molecular_accuracy = torch.tensor([num_correct / total]).to(tokens_enc.device)

        if log:
            logging.info(f"Molecular accuracy for batch {batch_idx}:")
            for idx, (invalid_, correct_) in enumerate(zip(invalid, correct_smiles)):
                if invalid_:
                    result = "invalid"
                elif correct_:
                    result = "correct"
                else:
                    result = "incorrect"
                logging.info(
                    f"     Sample {idx} is {result}, target: {target_smiles[idx]}, sample: {sampled_smiles[idx]}"
                )

        return molecular_accuracy, percent_invalid

    def calculate_metrics(
        self,
        token_logits,
        loss_mask,
        labels,
        tokens_enc,
        enc_mask,
        target_smiles,
        batch_idx=None,
        log_char=False,
        log_mol=False,
    ):
        """Calculate metrics for character accuracy, molecular accuracy, and invalid molecules

        Args:
            token_logits (torch.Tensor, float): softmax values for all tokens
            loss_mask (torch.Tensor, float): binary mask for ignored data (1=active, 0=mask), must be float
            labels (torch.Tensor, long): token IDs for correct output
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            target_smiles (str): ground truth for canonicalized SMILES

        Returns:
            dict: dictionary of metric values
        """
        character_accuracy = self.calculate_character_accuracy(
            token_logits, loss_mask, labels, batch_idx, log=log_char
        )
        molecular_accuracy, percent_invalid = self.calculate_molecular_accuracy(
            tokens_enc, enc_mask, target_smiles, batch_idx, log=log_mol
        )
        metrics = {
            "character_accuracy": character_accuracy,
            "molecular_accuracy": molecular_accuracy,
            "percent_invalid": percent_invalid,
        }
        return metrics

    def list_available_models(self):
        pass
