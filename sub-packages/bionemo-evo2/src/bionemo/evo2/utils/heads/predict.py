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
import functools
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch import callbacks as nl_callbacks

# from bionemo.evo2.run.peft import Evo2LoRA
from nemo.utils import logging

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.evo2.utils.heads.parallel_head import (
    ParallelHeadTransform,
    parallel_head_data_step_fn,
    parallel_head_forward_step_fn,
)
from bionemo.llm.data import collate
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.utils.callbacks import PredictionWriter


CheckpointFormats = Literal["torch_dist", "zarr"]

# Enable detailed logging for debugging purposes
LOGGING = False


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
        "--model-size",
        type=str,
        default="7b",
        choices=sorted(HYENA_MODEL_OPTIONS.keys()),
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

    # TODO: FIX PREDICTION WITH LORA
    ap.add_argument("--lora-checkpoint-path", type=Path, default=None, help="LoRA checkpoint path")
    ap.add_argument("--lora-finetune", action="store_true", help="Use LoRA fine-tuning")

    # Parallel Head
    ap.add_argument(
        "--parallel-heads",
        action="store_true",
        help="Train with parallel-heads. NOTE: Add adaptor to prediction scirpt.",
    )
    ap.add_argument(
        "--parallel-dna-head", action="store_true", help="Add dna token prediction head to parallel-heads."
    )
    ap.add_argument(
        "--parallel-rna-seq-head",
        action="store_true",
        help="Add rna seq expression prediction head to parallel-heads.",
    )
    ap.add_argument(
        "--parallel-pep-map-head",
        action="store_true",
        help="Add peptide map expression prediction head to parallel-heads.",
    )

    return ap.parse_args()


SHUFFLE_MESSAGE = (
    "Per token log probabilities are not supported when using context parallelism. The results will be "
    "zigzag shuffled along the sequence dimension. Raise a feature request if you need this and do "
    "not want to manually do the unshuffling yourself. You need to undo the shuffling that happened in "
    "`megatron.core.utils.get_batch_on_this_cp_rank`."
)


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
            collate_fn=functools.partial(
                collate.padding_collate_fn,
                padding_values={"tokens": 0, "position_ids": 0, "loss_mask": False},
                min_length=None,
                max_length=None,
            ),
        )


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the last dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    # TODO: handle zigzag packing here. Currently this just gathers along ranks, but if you want to see the sequence in
    #   the original order you need to undo the zigzag packing that happens in
    #   `megatron.core.utils.get_batch_on_this_cp_rank`.
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=parallel_state.get_context_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class BasePredictor(LightningPassthroughPredictionMixin):
    """Base predictor for GPT-style models."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
        **kwargs,
    ):
        """Initialize the base predictor with arguments needed for writing predictions."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option
        self.shuffle_warning_raised = False

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced predict_step that handles both single-head and parallel-head inference."""
        if LOGGING:
            logging.info(f"🔍 Predict step - Model type: {type(self)}")
            logging.info("🔍 Model has parallel attributes:")
            logging.info(f"   - parallel_dna: {getattr(self, 'parallel_dna', 'Not set')}")
            logging.info(f"   - parallel_rna: {getattr(self, 'parallel_rna', 'Not set')}")
            logging.info(f"   - parallel_pep: {getattr(self, 'parallel_pep', 'Not set')}")
            logging.info(f"   - _original_forward: {hasattr(self, '_original_forward')}")
        with torch.no_grad():
            forward_out = self.forward_step(batch)  # type: ignore

        # 🔄 Process each head's output separately
        gathered_outputs = {}
        for head_name, logits in forward_out.items():
            # Skip None values (which indicate unused heads)
            if logits is None:
                logging.info(f"Skipping {head_name}: None value") if LOGGING else None
                continue

            # Gather the logits
            gathered_logits = self._gather_parallel_output(logits)

            # Reshape rnaseq head output if needed
            if head_name == "rna_seq_logits" and len(gathered_logits.shape) == 2:
                gathered_logits = gathered_logits.transpose(0, 1)

            # Log shape info if LOGGING is enabled
            logging.info(
                f"Head: {head_name}, Original shape: {logits.shape}, Gathered shape: {gathered_logits.shape}"
            ) if LOGGING else None

            # Store the gathered logits for this head
            gathered_outputs[head_name] = gathered_logits.cpu()

        # 📤 Return all head outputs plus metadata
        result = {
            **gathered_outputs,
            "pad_mask": batch["loss_mask"].cpu() if "loss_mask" in batch else None,
            "seq_idx": batch["seq_idx"].cpu() if "seq_idx" in batch else None,
        }

        return result

    def _gather_parallel_output(self, tensor_output):
        """Helper to gather tensor output across both tensor parallel and context parallel dimensions."""
        # Gather across tensor parallel dimension
        tp_gathered = _gather_along_last_dim(tensor_output, group=parallel_state.get_tensor_model_parallel_group())
        # Gather across context parallel dimension
        cp_gathered = _gather_along_cp_dim(tp_gathered)
        return cp_gathered


class HyenaPredictor(BasePredictor, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def configure_model(self, *args, **kwargs) -> None:
        """Configure the model."""
        super().configure_model(*args, **kwargs)
        self.trainer.strategy._init_model_parallel = True  # type: ignore
        # Apply model transform if it exists
        if self.model_transform is not None:
            self.model_transform.__call__(self)


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    args: argparse.Namespace,
    num_nodes: int = 1,
    devices: int | None = None,
    model_size: str = "7b",
    model_type: str = "hyena",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    work_dir: Path | None = None,
    micro_batch_size: int = 1,
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
    write_interval: Literal["epoch", "batch"] = "epoch",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    num_layers: int | None = None,
    seq_len_interpolation_factor: int | None = None,
    files_per_subdir: int | None = None,
    lora_checkpoint_path: Path | None = None,
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    if files_per_subdir is None and write_interval == "batch":
        logging.warning(
            "--files-per-subdir is not set with --write-interval batch, will write all predictions to a "
            "single directory. This may cause problems if you are predicting on a very large dataset."
        )
    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists, files will be written here.
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    if devices is None:
        devices = model_parallel_size
    world_size = num_nodes * devices
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size must be divisible by model_parallel_size, got {world_size} and"
            f" {model_parallel_size}. Please set --num-nodes and --devices such that num_nodes * devices is divisible "
            "by model_parallel_size, which is TP * CP * PP."
        )
    global_batch_size = micro_batch_size * world_size // model_parallel_size

    callbacks = [
        PredictionWriter(
            output_dir=output_dir,
            write_interval=write_interval,
            batch_dim_key_defaults={"token_logits": 0},
            seq_dim_key_defaults={"token_logits": 1},
            files_per_subdir=files_per_subdir,
            save_all_model_parallel_ranks=False,  # only write one copy of predictions.
        )
    ]

    # The following two config options are really only used for testing, but may also be useful for getting output from
    #   specific layers of the model.
    config_modifiers_init = {}
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern
    if num_layers is not None:
        config_modifiers_init["num_layers"] = num_layers

    tokenizer = get_nmt_tokenizer("byte-level")

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
            forward_step_fn=partial(parallel_head_forward_step_fn, predict=True),
            data_step_fn=partial(parallel_head_data_step_fn, predict=True),  # , attention_backend=AttnBackend.fused,
            distribute_saved_activations=False if sequence_parallel and tensor_parallel_size > 1 else True,
            # Only use vortex style FP8 in the model config if using FP8 and not full FP8. This will only apply FP8 to
            #   the projection layer of the hyena mixer.
            vortex_style_fp8=fp8 and not full_fp8,
            **config_modifiers_init,
        )

        if args.parallel_heads:
            model_transform = ParallelHeadTransform(
                parallel_dna=args.parallel_dna_head,
                parallel_rna=args.parallel_rna_seq_head,
                parallel_pep=args.parallel_pep_map_head,
                predict=True,
            )
            callbacks.append(nl_callbacks.ModelTransform())  # type: ignore
        else:
            model_transform = None

        model = HyenaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
            model_transform=model_transform,
        )

        # if args.parallel_heads:
        #     model.model_transform.__call__(model)  # type: ignore

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=num_nodes,
        devices=devices,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=sequence_parallel,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",  # type: ignore
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                seq_len=8192,
                output_log=False,  # this is needed for predict step to work
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,  # type: ignore
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,  # type: ignore
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )
    trainer.strategy._setup_optimizers = False  # type: ignore

    nemo_logger = NeMoLogger(log_dir=work_dir)  # type: ignore
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        resume_from_path=str(ckpt_dir),
        restore_config=None,
    )

    resume.setup(trainer, model)  # this pulls weights from the starting checkpoint.

    dataset = SimpleFastaDataset(fasta_path, tokenizer, prepend_bos=prepend_bos)
    datamodule = PredictDataModule(dataset, batch_size=micro_batch_size)
    trainer.predict(model, datamodule=datamodule)  # TODO return_predictions=False
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
        ckpt_format=args.ckpt_format,
        fp8=args.fp8,
        full_fp8=args.full_fp8,
        output_log_prob_seqs=args.output_log_prob_seqs,
        log_prob_collapse_option=args.log_prob_collapse_option,
        prepend_bos=args.prepend_bos,
        no_sequence_parallel=args.no_sequence_parallel,
        hybrid_override_pattern=args.hybrid_override_pattern,
        num_layers=args.num_layers,
        args=args,
    )


if __name__ == "__main__":
    main()
