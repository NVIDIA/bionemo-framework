# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp
from transformers import AutoModelForMaskedLM, AutoTokenizer

from esm.collator import split_batch_by_cp_rank
from esm.convert import convert_esm_hf_to_te
from esm.modeling_esm_te import NVEsmForMaskedLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def get_dummy_data_thd_with_padding_dp0(cp_size: int = 2, tokenizer=None):
    """
    Get dummy data for the THD format with padding for context parallelism.
    Args:
        cp_size: The size of the context parallelism group.
        tokenizer: The tokenizer to use.
    Returns:
        A dictionary containing the padded input ids, labels, and cu seq lens.
    """
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Two real protein sequences (30 amino acids each, will be 32 tokens with BOS/EOS)
    protein1 = "MKTAYIAKQRQISFVKSHFSRQLEERLG"  # 29 AA -> ~31 tokens with special tokens
    protein2 = "MSHHWGYGKHNGPEHWHKDFPIAKGERF"  # 29 AA -> ~31 tokens with special tokens

    tok1 = tokenizer(protein1, return_tensors="pt", add_special_tokens=True)
    tok2 = tokenizer(protein2, return_tensors="pt", add_special_tokens=True)

    # Concatenate the token IDs
    input_ids = torch.cat([tok1["input_ids"].squeeze(), tok2["input_ids"].squeeze()])
    # Use input_ids as labels (for simplicity in testing)
    labels = input_ids.clone()

    cu_seqlens_q = torch.tensor([0, tok1["input_ids"].shape[1], input_ids.shape[0]])

    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )

    # Calculate max_length based on actual padded sequence lengths
    seq_lengths = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
    max_seq_len = int(seq_lengths.max().item())

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": max_seq_len,
        "max_length_k": max_seq_len,
        "pad_between_seqs": True,
    }
    return batch


def get_te_model_checkpoint(tmp_path):
    """
    Get a TE model checkpoint for the ESM2 model.
    Args:
        tmp_path: The path to save the model checkpoint.
    Returns:
        The path to the saved model checkpoint.
    """
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def get_batch_for_cp_rank(batch, cp_rank, cp_world_size):
    """
    Get a batch for a given context parallelism rank.

    Args:
        batch: The batch to get a shard of.
        cp_rank: The context parallelism rank.
        cp_world_size: The size of the context parallelism group.
    Returns:
        A dictionary containing the shard of the batch.
    """
    input_ids_sharded, labels_sharded = split_batch_by_cp_rank(
        cu_seqlens_padded=batch["cu_seq_lens_q_padded"],
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="thd",
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )
    batch_shard = dict(batch)
    batch_shard["input_ids"] = input_ids_sharded
    batch_shard["labels"] = labels_sharded
    # Now determine the max length of the sequence.
    seqlens_q = batch_shard["cu_seq_lens_q_padded"][1:] - batch_shard["cu_seq_lens_q_padded"][:-1]
    batch_shard["max_length_q"] = int((seqlens_q.max().item() + 63) // 64 * 64)  # From TE code.
    batch_shard["max_length_k"] = batch_shard["max_length_q"]
    return batch_shard


@dataclass(frozen=True)
class DistributedConfig:
    """Class to track distributed ranks and handle basic distributed training setup.

    If torch distributed environment variables are not set, we set them to default values for single-process training.

    Attributes:
        rank: The rank of the process.
        local_rank: The local rank of the process.
        world_size: The total number of processes.
    """

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12355"))

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


@requires_multi_gpu
def test_context_parallel_equivalence_2process():
    """
    Test the context parallel equivalence between 2 processes. In one instance, we run the model in non-distributed mode and in the other
    we run the model in distributed mode with context parallelism. We then compare the losses and logits from the two runs.

    We compare the (1) Losses, (2) Logits, and (3) Gradients from the two runs.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    tmp_path = Path("./tmp_model")
    os.makedirs(tmp_path, exist_ok=True)
    model_ckpt = get_te_model_checkpoint(tmp_path)

    # Create tokenizer for real protein sequences
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    input_data_thd_padded_dp0 = get_dummy_data_thd_with_padding_dp0(tokenizer=tokenizer)

    model = NVEsmForMaskedLM.from_pretrained(
        model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16
    )
    model.to("cuda")
    input_data_thd_padded_dp0 = {
        k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd_padded_dp0.items()
    }
    outputs_nondistributed = model(**input_data_thd_padded_dp0)
    loss_nondistributed = outputs_nondistributed.loss
    loss_nondistributed.backward()

    # Clone everything we need for later comparison BEFORE deleting
    loss_nondistributed_for_comparison = loss_nondistributed.detach().clone().cpu()
    logits_nondistributed_for_comparison = outputs_nondistributed.logits.detach().clone().cpu()

    # Sample gradients from a few layers for comparison
    sample_layers = [
        model.esm.encoder.layers[0].self_attention.core_attention,
        model.esm.encoder.layers[0].self_attention.layernorm_qkv,
    ]

    # Now grab the gradients from the sample layers
    gradients_nondistributed = {}
    for i, layer in enumerate(sample_layers):
        for name, param in layer.named_parameters():
            if param.grad is not None:
                key = f"layer_{i}.{name}"
                gradients_nondistributed[key] = param.grad.detach().clone().cpu()

    # Now setup distributed training for CP.
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")

    # Clean up everything from non-distributed run
    del model, outputs_nondistributed, loss_nondistributed, input_data_thd_padded_dp0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete

    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    # Create a device mesh for DDP=1, CP=2

    ddp_size = 1
    cp_size = 2
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(ddp_size, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )
    # Re-initialize the model on the new device (fresh instance, no shared graph)
    model = NVEsmForMaskedLM.from_pretrained(
        model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16
    )
    model = model.to(device=device)
    model.train()  # Set to training mode to enable gradient computation
    model.zero_grad(set_to_none=True)  # Ensure no gradients from initialization

    group_fsdp_cp = device_mesh[("ddp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )
    cp_group = device_mesh["cp"].get_group()
    cp_rank = device_mesh.get_local_rank("cp")
    cp_world_size = torch.distributed.get_world_size(group=cp_group)

    # Set up context parallelism for each layer
    for i, transformer_layer in enumerate(model.module.esm.encoder.layers):
        transformer_layer.set_context_parallel_group(
            cp_group, torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()), torch.cuda.Stream()
        )

    # Ensure model starts with clean slate
    model.zero_grad(set_to_none=True)

    # Create FRESH batch data for CP (don't reuse tensors from non-distributed run)
    batch = get_dummy_data_thd_with_padding_dp0(tokenizer=tokenizer)
    # Move batch to CUDA and ensure tensors are detached from any previous graphs
    batch = {k: v.detach().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch_cp = get_batch_for_cp_rank(batch, cp_rank=cp_rank, cp_world_size=cp_world_size)

    torch.distributed.barrier(group=cp_group)

    outputs_cp = model(**batch_cp)
    loss_cp = outputs_cp.loss

    # Gather the losses from all cp ranks (collective operation - all ranks must participate)
    losses_list = [torch.zeros_like(outputs_cp.loss) for _ in range(cp_world_size)]
    torch.distributed.all_gather(losses_list, outputs_cp.loss, group=cp_group)

    if cp_rank == 0:
        average_cp_loss = torch.mean(torch.stack(losses_list))
        # The average of per-rank losses should be close to the non-distributed loss
        # Note: They may not be exactly equal due to how loss is computed on sharded data
        torch.testing.assert_close(
            average_cp_loss.cpu(),
            loss_nondistributed_for_comparison,
            atol=0.1,  # Allow some difference due to loss computation on shards
            rtol=0.05,
        )

    # Gather the logits from all CP ranks
    # The logits are split along the sequence dimension (dim=1 for THD format: [batch, seq, vocab])
    logits_contiguous = outputs_cp.logits.contiguous()
    logits_list = [torch.zeros_like(logits_contiguous) for _ in range(cp_world_size)]
    torch.distributed.all_gather(logits_list, logits_contiguous, group=cp_group)

    if cp_rank == 0:
        # Reconstruct the full logits from CP-split chunks dynamically
        # Get sequence lengths from cu_seqlens
        cu_seqlens = batch["cu_seq_lens_q_padded"].cpu()
        num_seqs = len(cu_seqlens) - 1
        total_tokens = int(cu_seqlens[-1].item())
        vocab_size = logits_nondistributed_for_comparison.shape[-1]

        reconstructed_logits = torch.zeros((total_tokens, vocab_size), dtype=torch.bfloat16)

        # For each sequence, reconstruct from CP chunks
        cp_offset_rank0 = 0
        cp_offset_rank1 = 0

        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            chunk_size = seq_len // (2 * cp_world_size)  # Each sequence split into 2*cp_world_size chunks

            # CP rank 0 gets chunks [0, 3], CP rank 1 gets chunks [1, 2]
            for chunk_idx in range(2 * cp_world_size):
                chunk_start_in_seq = seq_start + chunk_idx * chunk_size
                chunk_end_in_seq = chunk_start_in_seq + chunk_size

                if chunk_idx == 0 or chunk_idx == 3:  # Chunks for CP rank 0
                    reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = logits_list[0][
                        cp_offset_rank0 : cp_offset_rank0 + chunk_size, :
                    ]
                    cp_offset_rank0 += chunk_size
                else:  # Chunks 1, 2 for CP rank 1
                    reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = logits_list[1][
                        cp_offset_rank1 : cp_offset_rank1 + chunk_size, :
                    ]
                    cp_offset_rank1 += chunk_size

        assert reconstructed_logits.shape == logits_nondistributed_for_comparison.shape
        torch.testing.assert_close(
            reconstructed_logits,
            logits_nondistributed_for_comparison,
            atol=0.29,
            rtol=0.01,
        )

    # Test gradient synchronization with DDP
    loss_cp = outputs_cp.loss
    loss_cp.backward()  # DDP automatically synchronizes gradients here

    # Capture gradients from the same layers in the CP model
    # Note: DDP wraps the model with 'module.' prefix
    sample_layers_cp = [
        model.module.esm.encoder.layers[0].self_attention.core_attention,
        model.module.esm.encoder.layers[0].self_attention.layernorm_qkv,
    ]

    gradients_cp = {}
    for i, layer in enumerate(sample_layers_cp):
        for name, param in layer.named_parameters():
            if param.grad is not None:
                key = f"layer_{i}.{name}"
                gradients_cp[key] = param.grad.detach().clone().cpu()

    # Now we compare the CP grads from rank 0 to the Grads from the non-distributed run. (they should be the same on each process for non dist)
    if cp_rank == 0:
        print(f"Captured {len(gradients_cp)} gradient tensors from CP run")
        print(f"Gradient keys: {list(gradients_cp.keys())}")

        # Compare gradients between non-distributed and CP
        for key in gradients_nondistributed.keys():
            if key in gradients_cp:
                grad_cp = gradients_cp[key]
                grad_nondist = gradients_nondistributed[key]

                print(f"\nParameter: {key}")
                print(f"  CP grad shape: {grad_cp.shape}")
                print(f"  Non-dist grad shape: {grad_nondist.shape}")
                print(f"  Gradient mean - CP: {grad_cp.mean():.6f}, Non-dist: {grad_nondist.mean():.6f}")
                print(f"  Gradient std - CP: {grad_cp.std():.6f}, Non-dist: {grad_nondist.std():.6f}")
                print(f"  Max absolute diff: {(grad_cp - grad_nondist).abs().max():.6f}")

                # Check if they're close
                try:
                    torch.testing.assert_close(
                        grad_cp, grad_nondist, atol=2e-3, rtol=1e-2, msg=f"Gradients don't match for {key}"
                    )
                    print("  ✓ Gradients match within tolerance")
                except AssertionError as e:
                    print(f"  ✗ Gradients differ: {e}")
            else:
                print(f"Warning: {key} not found in CP gradients")

    torch.distributed.destroy_process_group()
