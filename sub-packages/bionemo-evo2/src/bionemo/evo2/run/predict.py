# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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


import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.utils import get_batch_on_this_cp_rank
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from torch import Tensor

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset

# Add import for Mamba models
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from bionemo.llm.utils.callbacks import PredictionWriter


CheckpointFormats = Literal["torch_dist", "zarr"]


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--fasta", type=Path, required=True, help="Fasta path from which to generate logit predictions.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="NeMo2 checkpoint directory for inference.")
    ap.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences. Defaults to False.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help="When using TP, skip sequence parallelism. Otherwise sequence parallelism is used whenever tensor "
        "parallelism is used. sequence parallelism should save a small amount of GPU memory so it's on"
        " by default.",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for prediction. Defaults to 1.")
    ap.add_argument(
        "--model-type",
        type=str,
        choices=["hyena", "mamba"],
        default="hyena",
        help="Model architecture family to use. Choose between 'hyena' and 'mamba'.",
    )
    ap.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=sorted(list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys())),
        help="Model size to use. Defaults to '7b'.",
    )
    # output args:
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir that will contain the generated text produced by the Evo2 model. If not provided, the output will be logged.",
    )
    ap.add_argument(
        "--full-fp8",
        action="store_true",
        help="Use full FP8 precision (faster but less accurate) rather than vortex style which "
        "only applies FP8 to the projection layer of the hyena mixer, when using FP8.",
    )
    ap.add_argument("--fp8", action="store_true", help="Use FP8 precision. Defaults to BF16.")
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )
    ap.add_argument(
        "--output-log-prob-seqs", action="store_true", help="Output log probability of sequences. Defaults to False."
    )
    ap.add_argument(
        "--log-prob-collapse-option",
        choices=["sum", "mean"],
        default="mean",
        help="How to collapse the log probabilities across the sequence dimension.",
    )
    ap.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies hyena layer ordering and type).",
    )
    ap.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )
    ap.add_argument(
        "--seq-len-interpolation-factor",
        type=int,
        help="If set, override the sequence length interpolation factor specified in the requested config. If you "
        "know a model was trained with a specific interpolation factor for ROPE, provide it here, it can make a big "
        "difference in accuracy.",
    )
    return ap.parse_args()


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the last dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class HyenaPredictor(LightningPassthroughPredictionMixin, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean"] = "mean",
        **kwargs,
    ):
        """Initialize the predictor with our needs around computing log probabilities."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option

    def predict_step(self, batch, batch_idx: int | None = None) -> Tensor:
        """Alias for forward_step, also log the pad mask since sequences may not all have the same length."""
        if len(batch) == 0:
            return
        forward_out = self.forward_step(batch)
        if not isinstance(forward_out, Tensor):
            return forward_out
        # Reminder: the model's predictions for input i land at output i+1. To get everything to align, we prepend the
        # EOS token to the input sequences and take the outputs for all but the first token.
        forward_out_tp_gathered = _gather_along_last_dim(
            forward_out, group=parallel_state.get_tensor_model_parallel_group()
        )
        # else:
        #     forward_out_tp_gathered = _collect_into_dim(forward_out, dim=-1)
        forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
        assert self.tokenizer.vocab_size == forward_out_gathered.shape[-1]
        if self.output_log_prob_seqs:
            softmax_logprobs = torch.log_softmax(forward_out_gathered, dim=-1)
            softmax_logprobs = softmax_logprobs[:, :-1]
            input_ids = batch["tokens"][:, 1:]
            assert softmax_logprobs.shape[1] == input_ids.shape[1]

            logprobs = torch.gather(
                softmax_logprobs,  # Gather likelihoods...
                2,  # along the vocab dimension...
                input_ids.unsqueeze(-1),  # using the token ids to index.
            ).squeeze(-1)
            log_prob_seqs = torch.sum(logprobs * batch["loss_mask"][:, 1:].float(), dim=-1)
            if self.log_prob_collapse_option == "mean":
                log_prob_seqs = log_prob_seqs / (batch["loss_mask"][:, 1:].float().sum(dim=-1) + 1e-8)
            return {"log_probs_seqs": log_prob_seqs.cpu(), "seq_idx": batch["seq_idx"].cpu()}
        else:
            # If the user wants to match back to logits, then they will need to do the offsetting logic themselves.
            return {
                "token_logits": forward_out_gathered.cpu(),
                "pad_mask": batch["loss_mask"].cpu(),
                "seq_idx": batch["seq_idx"].cpu(),
            }


def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Performs a forward step for the Hyena model.

    Args:
        model: The Hyena model
        batch: Dictionary containing input batch data with keys:
            - tokens: Input token IDs
            - position_ids: Position IDs
            - labels: Labels for loss computation
            - loss_mask: Mask for loss computation

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        # "labels": batch["labels"],
        # "loss_mask": batch["loss_mask"],
    }

    forward_args["attention_mask"] = None
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)
    return model(**forward_args)


def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step for the Hyena model prediction. Modified from the original gpt data step to include the seq_idx."""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask", "seq_idx"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 1):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataloader."""
        pass

    def predict_dataloader(self):
        """Create a dataloader for prediction."""
        # need to use this to communicate that we are in predict mode and safe to not drop last batch
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
        )


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    model_size: str = "7b",
    model_type: str = "hyena",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    work_dir: Path | None = None,
    batch_size: int = 1,
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean"] = "mean",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    num_layers: int | None = None,
    seq_len_interpolation_factor: int | None = None,
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists, files will be written here.
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )
    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=model_parallel_size,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=tensor_parallel_size > 1 and sequence_parallel,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=batch_size,
                global_batch_size=batch_size,
                seq_len=8192,
                output_log=False,  # this is needed for predict step to work
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        callbacks=[
            PredictionWriter(
                output_dir=output_dir,
                write_interval="epoch",
                batch_dim_key_defaults={"token_logits": 0},
                seq_dim_key_defaults={"token_logits": 1},
            )
        ],
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )
    # The following two config options are really only used for testing, but may also be useful for getting output from
    #   specific layers of the model.
    config_modifiers_init = {}
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern
    if num_layers is not None:
        config_modifiers_init["num_layers"] = num_layers
    # Select model config based on model type
    if model_type == "hyena":
        if "-1m" in model_size and "nv" not in model_size and seq_len_interpolation_factor is None:
            # TODO remove this override once we add this as a default upstream in NeMo.
            #  if you see this, just check the pointed to model option for the 1m model in nemo and see if it already
            #  has this option set.
            config_modifiers_init["seq_len_interpolation_factor"] = 128

        if model_size not in HYENA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Hyena: {model_size}")
        config = HYENA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,  # , attention_backend=AttnBackend.fused,
            distribute_saved_activations=False if sequence_parallel and tensor_parallel_size > 1 else True,
            # Only use vortex style FP8 in the model config if using FP8 and not full FP8. This will only apply FP8 to
            #   the projection layer of the hyena mixer.
            vortex_style_fp8=fp8 and not full_fp8,
            **config_modifiers_init,
        )
    else:  # mamba
        if model_size not in MAMBA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Mamba: {model_size}")
        config = MAMBA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,  # Can reuse the same forward steps
            data_step_fn=hyena_predict_data_step,
            distribute_saved_activations=False if sequence_parallel and tensor_parallel_size > 1 else True,
            **config_modifiers_init,
        )

    trainer.strategy._setup_optimizers = False

    nemo_logger = NeMoLogger(log_dir=work_dir)
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        restore_config=nl.RestoreConfig(
            path=str(ckpt_dir),  # NeMo expects a string path.
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )
    tokenizer = get_nmt_tokenizer("byte-level")

    # Create appropriate model based on type
    if model_type == "hyena":
        model = HyenaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
        )
    else:  # mamba
        model = MambaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
        )

    resume.setup(trainer, model)  # this pulls weights from the starting checkpoint.

    dataset = SimpleFastaDataset(fasta_path, tokenizer, prepend_bos=prepend_bos)
    datamodule = PredictDataModule(dataset, batch_size=batch_size)
    trainer.predict(model, datamodule=datamodule)
    dataset.write_idx_map(
        output_dir
    )  # Finally write out the index map so we can match the predictions to the original sequences.


def main():
    """Entrypoint for Evo2 prediction (single inference step, no new tokens)."""
    args = parse_args()
    predict(
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        output_dir=args.output_dir,
        model_size=args.model_size,
        model_type=args.model_type,
        ckpt_format=args.ckpt_format,
        fp8=args.fp8,
        full_fp8=args.full_fp8,
        batch_size=args.batch_size,
        output_log_prob_seqs=args.output_log_prob_seqs,
        log_prob_collapse_option=args.log_prob_collapse_option,
        prepend_bos=args.prepend_bos,
        no_sequence_parallel=args.no_sequence_parallel,
        hybrid_override_pattern=args.hybrid_override_pattern,
        seq_len_interpolation_factor=args.seq_len_interpolation_factor,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()


# Create MambaPredictor class, similar to HyenaPredictor
class MambaPredictor(MambaModel, LightningPassthroughPredictionMixin):
    """Mamba model for prediction with additional metrics."""

    def __init__(
        self,
        config,
        tokenizer=None,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean"] = "mean",
    ):
        """Initialize the MambaPredictor, which wraps the mamba model for prediction handling model parallelism.

        Args:
            config: Model Config
            tokenizer: tokenizer for the model. Defaults to None.
            output_log_prob_seqs: If you want to output the log probabilities of the sequences. Defaults to False.
            log_prob_collapse_option: collapse the log probabilities of the sequences with this prior to return.
                Defaults to "mean".
        """
        super().__init__(config, tokenizer)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option
        # Storage for the predictions
        self.predictions = []
        self.log_probabilities = []
        self.tokens = []

    def forward(
        self,
        input_ids,
        position_ids,
        inference_params=None,
        attention_mask=None,
    ):
        """Forward pass for prediction. Overrides the base forward method to store predictions."""
        # Call parent forward (which doesn't calculate loss)
        logits = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_params=inference_params,
        )

        # Save information for each sequence
        if self.output_log_prob_seqs:
            for i, sequence in enumerate(input_ids):
                logits_seq = logits[i]  # shape: [seq_len, vocab_size]
                sequence_tokens = sequence.tolist()
                # Get the probabilities for each token
                log_probs = torch.log_softmax(logits_seq, dim=-1)
                # Get the log probabilities of the actual tokens
                token_log_probs = []
                for j, token in enumerate(sequence_tokens):
                    if j < len(log_probs):  # Check if we have a prediction for this position
                        token_log_probs.append(log_probs[j, token].item())

                # Depending on the option, sum or average the log probabilities
                if self.log_prob_collapse_option == "sum":
                    sequence_log_prob = sum(token_log_probs)
                else:  # mean
                    sequence_log_prob = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0

                self.tokens.append(sequence_tokens)
                self.log_probabilities.append(sequence_log_prob)

        # Return the logits
        return logits

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Prediction step, saving the results."""
        # Get the tokens and attention mask
        if batch == {}:
            batch = {"tokens": [], "position_ids": [], "attention_mask": []}
        inference_data = {"tokens": batch["tokens"], "position_ids": batch["position_ids"]}

        if self.trainer.strategy._cp_size > 1:
            inference_data = get_batch_on_this_context_parallel_rank(
                inference_data,
                num_micro_batches=1,
                micro_batch_idx=0,
                sequence_parallel=False,
                micro_batch_size_per_context_rank=None,
                context_parallel_size=self.trainer.strategy._cp_size,
                sequence_length=self.config.seq_length,
                multiple_of=1,
            )
            inference_data["attention_mask"] = None

        with nullcontext() if torch.is_inference_mode_enabled() else torch.no_grad():
            output = self(
                input_ids=inference_data["tokens"],
                position_ids=inference_data["position_ids"],
                inference_params=None,
                attention_mask=inference_data.get("attention_mask", None),
            )

            if self.trainer.strategy._tp_size > 1:
                # TP > 1 case - all-gather the output tensor along last dimension from all TP ranks for each token
                # [b, s, v/p] -> [b, s, v]
                output = _gather_along_last_dim(output)

            # Cache predictions
            self.predictions.extend(torch.argmax(output, dim=-1).cpu().numpy().tolist())

        # Return as dict for PredictionWriter callback
        return {"predictions": self.predictions, "log_probabilities": self.log_probabilities, "tokens": self.tokens}
