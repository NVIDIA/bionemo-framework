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
from pathlib import Path
from typing import Literal

import nemo.lightning as nl
import torch
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.utils import logging as logger
from torch import Tensor
from torch.nn import functional as F

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset

# Add import for Mamba models
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.evo2.run.predict import (
    BasePredictor,
    HyenaPredictor,
    MambaPredictor,
    _to_cpu,
    hyena_predict_data_step,
    hyena_predict_forward_step,
)
from bionemo.llm.utils.callbacks import PredictionWriter


# TODO consider just adding this to predict...
class LogitsPredictor(BasePredictor):
    def __init__(
        self, *args, output_log_prob_seqs: bool = True, include_tokens_with_logprob_seqs: bool = True, **kwargs
    ):
        super().__init__(
            *args,
            output_log_prob_seqs=output_log_prob_seqs,
            include_tokens_with_logprob_seqs=include_tokens_with_logprob_seqs,
            **kwargs,
        )

    def predict_step(
        self, batch, batch_idx: int | None = None, to_cpu: bool = True
    ) -> Tensor | dict[str, Tensor] | None:
        _result_any = super().predict_step(batch, batch_idx, to_cpu=False)
        if not isinstance(_result_any, dict):
            return _result_any
        result: dict[str, Tensor] = _result_any
        shifted_token_logits = result["token_logits"][:, :-1]
        shifted_pad_mask = result["pad_mask"][:, 1:]
        shifted_tokens = result["tokens"][:, 1:]
        lm_loss_full = (
            F.cross_entropy(shifted_token_logits[shifted_pad_mask], shifted_tokens[shifted_pad_mask], reduction="none")
            * shifted_pad_mask
        )
        n_tokens = shifted_pad_mask.sum(dim=-1)
        nll_sum = lm_loss_full.sum(dim=-1)
        # These have been TP and CP gathered, so they will be the same on all CP+TP ranks. DP may differ unless
        #  predict gathers them for us.
        # TODO: idea get a sum per seq_idx, then later we can gather on the various samples in the fasta and sum
        #   by group.
        return _to_cpu({"nll_sum": nll_sum, "n_tokens": n_tokens, "seq_idx": result["seq_idx"]})


class MambaLogitsPredictor(LogitsPredictor, MambaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HyenaLogitsPredictor(LogitsPredictor, HyenaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


CheckpointFormats = Literal["torch_dist", "zarr"]


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for prediction, defaults to 1.")
    ap.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use for prediction, defaults to tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size.",
    )
    ap.add_argument("--fasta", type=Path, required=True, help="Fasta path from which to generate logit predictions.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="NeMo2 checkpoint directory for inference.")
    ap.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences. Defaults to False.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        choices=[1],
        default=1,
        help="Order of pipeline parallelism. Defaults to 1 and currently only 1 is supported.",
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
    ap.add_argument("--micro-batch-size", type=int, default=1, help="Batch size for prediction. Defaults to 1.")
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
        default="7b_arc_longcontext",
        choices=sorted(list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys())),
        help="Model size to use. Defaults to '7b_arc_longcontext'.",
    )
    # output args:
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir that will contain the generated text produced by the Evo2 model. If not provided, the output will be logged.",
    )
    ap.add_argument(
        "--files-per-subdir",
        type=int,
        help="Number of files to write to each subdirectory. If provided, subdirectories with N files each will be created. Ignored unless --write-interval is 'batch'.",
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


def eval(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    num_nodes: int = 1,
    devices: int | None = None,
    model_size: str = "7b",
    model_type: str = "hyena",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    work_dir: Path | None = None,
    micro_batch_size: int = 1,
    log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
    write_interval: Literal["epoch", "batch"] = "epoch",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    num_layers: int | None = None,
    seq_len_interpolation_factor: int | None = None,
    files_per_subdir: int | None = None,
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    if files_per_subdir is None and write_interval == "batch":
        logger.warning(
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
            ckpt_load_strictness="log_all",
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
        callbacks=[
            PredictionWriter(
                output_dir=output_dir,
                write_interval=write_interval,
                batch_dim_key_defaults={"token_logits": 0},
                seq_dim_key_defaults={"token_logits": 1},
                files_per_subdir=files_per_subdir,
                save_all_model_parallel_ranks=False,  # only write one copy of predictions.
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
        resume_from_path=str(ckpt_dir),
        restore_config=None,
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
    datamodule = PredictDataModule(dataset, batch_size=micro_batch_size)
    trainer.predict(model, datamodule=datamodule)  # TODO return_predictions=False
    dataset.write_idx_map(
        output_dir
    )  # Finally write out the index map so we can match the predictions to the original sequences.


def main():
    """Entrypoint for Evo2 prediction (single inference step, no new tokens)."""
    args = parse_args()
    eval(
        num_nodes=args.num_nodes,
        devices=args.devices,
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
        micro_batch_size=args.micro_batch_size,
        output_log_prob_seqs=args.output_log_prob_seqs,
        log_prob_collapse_option=args.log_prob_collapse_option,
        prepend_bos=args.prepend_bos,
        no_sequence_parallel=args.no_sequence_parallel,
        hybrid_override_pattern=args.hybrid_override_pattern,
        seq_len_interpolation_factor=args.seq_len_interpolation_factor,
        num_layers=args.num_layers,
        files_per_subdir=args.files_per_subdir,
        write_interval=args.write_interval,
    )


if __name__ == "__main__":
    main()
