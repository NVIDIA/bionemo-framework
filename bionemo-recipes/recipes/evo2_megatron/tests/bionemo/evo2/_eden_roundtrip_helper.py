# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helper script for Eden Llama roundtrip test.

Run with torchrun to handle distributed init requirements:
    torchrun --nproc-per-node 1 tests/bionemo/evo2/_eden_roundtrip_helper.py \
        --ckpt-dir <mbridge_checkpoint> --hf-output-dir <hf_output> --mode export

    torchrun --nproc-per-node 1 tests/bionemo/evo2/_eden_roundtrip_helper.py \
        --hf-input-dir <hf_dir> --ckpt-output-dir <mbridge_output> --mode import
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.model_provider import ProcessGroupCollection
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_run_config_filename, read_run_config
from megatron.bridge.utils.instantiate_utils import instantiate
from transformers import LlamaConfig, LlamaForCausalLM

from bionemo.evo2.run.predict import initialize_inference_distributed, resolve_checkpoint_path


def export_mbridge_to_hf(ckpt_dir: Path, hf_output_dir: Path) -> None:
    """Export an mbridge Eden checkpoint to HuggingFace format using AutoBridge."""
    resolved = resolve_checkpoint_path(ckpt_dir)
    run_config = read_run_config(get_checkpoint_run_config_filename(str(resolved)))
    model_provider = instantiate(run_config["model"])

    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.context_parallel_size = 1
    model_provider.sequence_parallel = False

    mp_config = get_mixed_precision_config("bf16_mixed")
    mp_config.finalize()
    mp_config.setup(model_provider)
    model_provider.vocab_size = 512
    model_provider.should_pad_vocab = True

    rng_config = RNGConfig(seed=1234)
    dist_config = DistributedInitConfig()
    initialize_inference_distributed(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=1,
        rng_config=rng_config,
        dist_config=dist_config,
    )

    model_provider.finalize()
    # _pg_collection is a dataclass field on GPTModelProvider (megatron.bridge)
    model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    raw_model = model_provider.provide().eval().cuda()
    _load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved), model=[raw_model], dist_ckpt_strictness="ignore_all"
    )

    hf_config = LlamaConfig(
        hidden_size=model_provider.hidden_size,
        intermediate_size=model_provider.ffn_hidden_size,
        num_hidden_layers=model_provider.num_layers,
        num_attention_heads=model_provider.num_attention_heads,
        num_key_value_heads=model_provider.num_query_groups,
        vocab_size=model_provider.vocab_size,
        max_position_embeddings=model_provider.seq_length,
        rms_norm_eps=model_provider.layernorm_epsilon,
        rope_theta=model_provider.rotary_base,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=model_provider.share_embeddings_and_output_weights,
    )
    hf_config.architectures = ["LlamaForCausalLM"]

    hf_output_dir.mkdir(parents=True, exist_ok=True)
    hf_config.save_pretrained(str(hf_output_dir))

    with tempfile.TemporaryDirectory() as tmp_hf_dir:
        hf_model = LlamaForCausalLM(hf_config)
        hf_model.save_pretrained(tmp_hf_dir)
        del hf_model
        bridge = AutoBridge.from_hf_pretrained(tmp_hf_dir, torch_dtype=torch.bfloat16)
        bridge.save_hf_weights([raw_model], str(hf_output_dir))

    print(f"Exported mbridge -> HF at {hf_output_dir}")


def import_hf_to_mbridge(hf_input_dir: Path, ckpt_output_dir: Path) -> None:
    """Import an HF checkpoint into an mbridge model and save the mbridge state dict.

    The saved state dict uses bare mbridge keys (no ``module.`` prefix) so it
    can be compared directly against the original mbridge DCP state dict.
    """
    rng_config = RNGConfig(seed=1234)
    dist_config = DistributedInitConfig()
    initialize_inference_distributed(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=1,
        rng_config=rng_config,
        dist_config=dist_config,
    )

    bridge = AutoBridge.from_hf_pretrained(str(hf_input_dir), torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider()

    mp_config = get_mixed_precision_config("bf16_mixed")
    mp_config.finalize()
    mp_config.setup(provider)
    provider.finalize()

    models = provider.provide_distributed_model(
        ddp_config=None, wrap_with_ddp=False, data_parallel_random_init=False, bf16=True
    )

    raw_sd = models[0].state_dict()
    mbridge_sd = {k.removeprefix("module."): v for k, v in raw_sd.items()}

    ckpt_output_dir = Path(ckpt_output_dir)
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mbridge_sd, str(ckpt_output_dir / "state_dict.pt"))
    print(f"Saved roundtripped mbridge state dict to {ckpt_output_dir / 'state_dict.pt'}")


def main():
    """Entry point for the roundtrip helper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["export", "import"], required=True)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--hf-output-dir", type=Path, default=None)
    parser.add_argument("--hf-input-dir", type=Path, default=None)
    parser.add_argument("--ckpt-output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "export":
        assert args.ckpt_dir and args.hf_output_dir
        export_mbridge_to_hf(args.ckpt_dir, args.hf_output_dir)
    elif args.mode == "import":
        assert args.hf_input_dir and args.ckpt_output_dir
        import_hf_to_mbridge(args.hf_input_dir, args.ckpt_output_dir)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
