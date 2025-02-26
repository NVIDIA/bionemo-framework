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
import json
import tempfile
from pathlib import Path
from typing import Literal, Optional

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from torch import Tensor

from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.utils.callbacks import PredictionWriter
from bionemo.noodles.nvfaidx import NvFaidx


CheckpointFormats = Literal["torch_dist", "zarr"]


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--fasta", type=Path, required=True, help="Fasta path from which to generate logit predictions.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="NeMo2 checkpoint directory for inference.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
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
    ap.add_argument("--fp8", action="store_true", help="Use FP8 precision. Defaults to BF16.")
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )

    return ap.parse_args()


class HyenaPredictor(LightningPassthroughPredictionMixin, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """Alias for forward_step, also log the pad mask since sequences may not all have the same length."""
        if len(batch) == 0:
            return
        forward_out = self.forward_step(batch)
        if isinstance(forward_out, Tensor):
            return {"token_logits": forward_out, "pad_mask": batch["loss_mask"], "seq_idx": batch["seq_idx"]}
        return forward_out


class SimpleFastaDataset(torch.utils.data.Dataset):
    """A simple dataset for Evo2 prediction."""

    def __init__(self, fasta_path: Path, tokenizer):
        """Initialize the dataset."""
        super().__init__()
        self.fasta = NvFaidx(fasta_path)
        self.seqids = list(self.fasta.keys())
        self.tokenizer = tokenizer

    def write_idx_map(self, output_dir: Path):
        """Write the index map to the output directory."""
        with open(output_dir / "seq_idx_map.json", "w") as f:
            json.dump({seqid: idx for idx, seqid in enumerate(self.seqids)}, f)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.seqids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset."""
        sequence = self.fasta[self.seqids[idx]].sequence().upper()
        tokens: list[int] = self.tokenizer.text_to_ids(sequence)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.arange(len(tokens), dtype=torch.long),
            "seq_idx": torch.tensor(idx, dtype=torch.long),
            "loss_mask": torch.ones_like(torch.tensor(tokens, dtype=torch.long), dtype=torch.long),
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
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output


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
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    model_size: str = "7b",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    work_dir: Path | None = None,
    batch_size: int = 1,
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
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
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=1,
                global_batch_size=1,
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
            fp8="hybrid" if fp8 else None,
            fp8_amax_history_len=16 if fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 else "most_recent",
        ),
    )
    config = HYENA_MODEL_OPTIONS[model_size](
        forward_step_fn=hyena_predict_forward_step, data_step_fn=hyena_predict_data_step
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
        ),
    )
    tokenizer = get_nmt_tokenizer("byte-level")
    model = HyenaPredictor(config, tokenizer=tokenizer)
    resume.setup(trainer, model)  # this pulls weights from the starting checkpoint.

    dataset = SimpleFastaDataset(fasta_path, tokenizer)
    datamodule = PredictDataModule(dataset, batch_size=batch_size)
    trainer.predict(model, datamodule.predict_dataloader())
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
