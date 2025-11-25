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
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch import callbacks as nl_callbacks

# from bionemo.evo2.run.peft import Evo2LoRA
from torch import Tensor, nn

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.evo2.utils.heads.parallel_head import (
    ParallelHeadTransform,
    parallel_head_data_step_fn,
    parallel_head_forward_step_fn,
)
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
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


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 1):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
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


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the context parallel dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Use the correct parallel group for context parallelism
    cp_group = parallel_state.get_context_parallel_group()

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=cp_group)
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class HyenaPredictor(LightningPassthroughPredictionMixin, llm.HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean"] = "mean",
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        **kwargs,
    ):
        """Initialize the predictor with our needs around computing log probabilities."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option
        self.model_transform = model_transform

        if self.model_transform is not None:
            self.model_transform.__call__(self)

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """Enhanced predict_step that handles both single-head and parallel-head inference."""
        print(f"🔍 Predict step - Model type: {type(self)}")
        print("🔍 Model has parallel attributes:")
        print(f"   - parallel_dna: {getattr(self, 'parallel_dna', 'Not set')}")
        print(f"   - parallel_rna: {getattr(self, 'parallel_rna', 'Not set')}")
        print(f"   - parallel_pep: {getattr(self, 'parallel_pep', 'Not set')}")
        print(f"   - _original_forward: {hasattr(self, '_original_forward')}")

        forward_out = self.forward_step(batch)
        print(f"Batch: \n{batch}")
        print(f"Batch type: {type(batch)}")
        try:
            print(f"Batch keys: {batch}")
        except Exception:
            pass
        print(f"Forward out type: {type(forward_out)}")
        try:
            print(f"Forward shape: {forward_out.shape}")
        except Exception:
            pass
        print(f"Forward out: {forward_out}")

        # 🔍 CHECK: Are we using parallel heads?
        using_parallel_heads = (
            hasattr(self, "parallel_dna") or hasattr(self, "parallel_rna") or hasattr(self, "parallel_pep")
        )

        print(f"🔍 Using parallel heads: {using_parallel_heads}")

        if using_parallel_heads:
            # 🎯 PARALLEL HEADS: Handle multiple outputs
            return self._handle_parallel_head_outputs(forward_out, batch)  # type: ignore
        else:
            # 🎯 SINGLE HEAD: Original DNA-only logic
            if not isinstance(forward_out, Tensor):
                print(f"⚠️ Warning: Expected tensor for single head, got {type(forward_out)}")
                return forward_out
            return self._handle_single_head_outputs(forward_out, batch)  # type: ignore

    def _handle_parallel_head_outputs(self, forward_out, batch):
        """Handle forward outputs when using parallel heads."""
        print(f"Handling parallel head outputs, type: {type(forward_out)}")

        # Handle dictionary output (expected case)
        if isinstance(forward_out, dict):
            gathered_outputs = {}

            # 🔄 Process each head's output separately
            for head_name, logits in forward_out.items():
                # Skip None values
                if logits is None:
                    print(f"Skipping {head_name}: None value")
                    continue

                print(f"Processing {head_name} with shape: {logits.shape}")

                # Gather the logits
                gathered_logits = self._gather_parallel_output(logits)
                gathered_outputs[head_name] = gathered_logits.cpu()

        elif isinstance(forward_out, (tuple, list)):
            # Alternative: if forward_out is a tuple/list of tensors
            gathered_outputs = {}
            head_names = ["dna_logits", "rna_seq_logits", "pep_map_logits"]

            for i, logits in enumerate(forward_out):
                if logits is None:
                    continue

                if i < len(head_names):
                    head_name = head_names[i]
                    print(f"Processing {head_name} with shape: {logits.shape}")

                    gathered_logits = self._gather_parallel_output(logits)
                    gathered_outputs[head_name] = gathered_logits.cpu()

        elif isinstance(forward_out, torch.Tensor):
            # Single tensor case - this might happen if only one head is active
            print("⚠️ Got single tensor for parallel heads - treating as DNA logits")
            gathered_logits = self._gather_parallel_output(forward_out)
            gathered_outputs = {"dna_logits": gathered_logits.cpu()}

        else:
            # Unexpected case
            print(f"⚠️ Unexpected forward_out type for parallel heads: {type(forward_out)}")
            print(f"Forward out value: {forward_out}")

            # Try to handle as single tensor if it has tensor-like attributes
            if hasattr(forward_out, "shape") and hasattr(forward_out, "dtype"):
                gathered_logits = self._gather_parallel_output(forward_out)
                gathered_outputs = {"unknown_logits": gathered_logits.cpu()}
            else:
                # Return as-is with metadata
                gathered_outputs = {"raw_output": forward_out}

        # 📤 Return all head outputs plus metadata
        result = {
            **gathered_outputs,
            "pad_mask": batch["loss_mask"].cpu() if "loss_mask" in batch else None,
            "seq_idx": batch["seq_idx"].cpu() if "seq_idx" in batch else None,
        }

        print(f"Final result keys: {list(result.keys())}")
        return result

    def _handle_single_head_outputs(self, forward_out, batch):
        """Handle forward outputs for single-head (DNA-only) inference."""
        forward_out_gathered = self._gather_parallel_output(forward_out)

        # Verify DNA vocab size
        assert self.tokenizer.vocab_size == forward_out_gathered.shape[-1]  # type: ignore

        if self.output_log_prob_seqs:
            # 📊 Compute log probabilities
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
            # 📤 Return raw logits
            return {
                "token_logits": forward_out_gathered.cpu(),
                "pad_mask": batch["loss_mask"].cpu(),
                "seq_idx": batch["seq_idx"].cpu(),
            }

    def _gather_parallel_output(self, tensor_output):
        """Helper to gather tensor output across both tensor parallel and context parallel dimensions."""
        # Gather across tensor parallel dimension
        tp_gathered = _gather_along_last_dim(tensor_output, group=parallel_state.get_tensor_model_parallel_group())
        # Gather across context parallel dimension
        cp_gathered = _gather_along_cp_dim(tp_gathered)
        return cp_gathered

    def _gather_single_output(self, forward_out):
        """Helper to gather a single tensor output across parallel dimensions."""
        return self._gather_parallel_output(forward_out)


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    args: argparse.Namespace,
    model_size: str = "7b",
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
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    callback_list = [
        PredictionWriter(
            output_dir=output_dir,
            write_interval="epoch",
            batch_dim_key_defaults={"token_logits": 0, "dna_logits": 0, "logits": 0},
            seq_dim_key_defaults={"token_logits": 1, "dna_logits": 1, "logits": 1},
        )
    ]

    # Asserts for proper configuration of parallel heads
    if args.parallel_heads:
        heads = [args.parallel_dna_head, args.parallel_rna_seq_head, args.parallel_pep_map_head]
        callback_list.append(nl_callbacks.ModelTransform())  # type: ignore
        assert any(heads), "No heads added to parallel heads. Add two or more heads."

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
        strategy=nl.MegatronStrategy(  # TODO: MIGHT HAVE TO USE CUSTOM STRATEGY IF USING MODEL TRANSFORM
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
            ckpt_load_strictness="log_all",  # type: ignore
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
        callbacks=callback_list,  # type: ignore
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
    # The following two config options are really only used for testing, but may also be useful for getting output from
    #   specific layers of the model.
    config_modifiers_init = {}
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern
    if num_layers is not None:
        config_modifiers_init["num_layers"] = num_layers
    config = HYENA_MODEL_OPTIONS[model_size](
        forward_step_fn=partial(parallel_head_forward_step_fn, predict=True),
        data_step_fn=partial(parallel_head_data_step_fn, predict=True),  # Use parallel head data step when needed
        distribute_saved_activations=False if sequence_parallel and tensor_parallel_size > 1 else True,
        # Only use vortex style FP8 in the model config if using FP8 and not full FP8. This will only apply FP8 to
        #   the projection layer of the hyena mixer.
        vortex_style_fp8=fp8 and not full_fp8,
        **config_modifiers_init,
    )
    trainer.strategy._setup_optimizers = False  # type: ignore

    nemo_logger = NeMoLogger(log_dir=work_dir)  # type: ignore
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
    model = HyenaPredictor(
        config,
        tokenizer=tokenizer,
        output_log_prob_seqs=output_log_prob_seqs,
        log_prob_collapse_option=log_prob_collapse_option,
        model_transform=ParallelHeadTransform(
            dna_loss_weight=1.0,
            rna_loss_weight=0.5,
            pep_loss_weight=0.5,
            parallel_dna=args.parallel_dna_head,
            parallel_rna=args.parallel_rna_seq_head,
            parallel_pep=args.parallel_pep_map_head,
        )
        if args.parallel_heads
        else None,
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
        ckpt_format=args.ckpt_format,
        fp8=args.fp8,
        full_fp8=args.full_fp8,
        batch_size=args.batch_size,
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
