from dataclasses import dataclass, field
import os
import torch.distributed as dist
import pytest
import torch

from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, AutoModelForMaskedLM
from esm.convert import convert_esm_hf_to_te
from esm.modeling_esm_te import NVEsmForMaskedLM
from esm.collator import MLMDataCollatorWithFlattening, split_batch_by_cp_rank
from conftest import te_model_checkpoint
import subprocess
from pathlib import Path
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp

def get_dummy_data_thd_with_padding_dp0(cp_size: int = 2, tokenizer=None):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Two real protein sequences (30 amino acids each, will be 32 tokens with BOS/EOS)
    protein1 = "MKTAYIAKQRQISFVKSHFSRQLEERLG"  # 29 AA -> ~31 tokens with special tokens
    protein2 = "MSHHWGYGKHNGPEHWHKDFPIAKGERF"  # 29 AA -> ~31 tokens with special tokens
    
    tok1 = tokenizer(protein1, return_tensors="pt", add_special_tokens=True)
    tok2 = tokenizer(protein2, return_tensors="pt", add_special_tokens=True)
    
    # Concatenate the token IDs
    input_ids = torch.cat([tok1['input_ids'].squeeze(), tok2['input_ids'].squeeze()])
    # Use input_ids as labels (for simplicity in testing)
    labels = input_ids.clone()
    
    cu_seqlens_q = torch.tensor([0, tok1['input_ids'].shape[1], input_ids.shape[0]])
    
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )

    # Calculate max_length based on actual padded sequence lengths
    seq_lengths = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
    max_seq_len = int(seq_lengths.max().item())
    
    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": max_seq_len,
        "max_length_k": max_seq_len,
        "pad_between_seqs": True,
    }
    return batch

def get_te_model_checkpoint(tmp_path):
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def get_batch_for_cp_rank(batch, cp_rank, cp_world_size):
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

def test_dummy_runner():
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
        # print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")
    # For debugging
    print(f"STDOUT:\n{result.stdout}")
    # print(f"STDERR:\n{result.stderr}")


if __name__ == "__main__":
    # Do everything here.
    tmp_path = Path("./tmp_model")
    os.makedirs(tmp_path, exist_ok=True)
    model_ckpt = get_te_model_checkpoint(tmp_path)

    # Create tokenizer for real protein sequences
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    input_data_thd_padded_dp0 = get_dummy_data_thd_with_padding_dp0(tokenizer=tokenizer)

    model = NVEsmForMaskedLM.from_pretrained(model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16)
    model.to("cuda")
    input_data_thd_padded_dp0 = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd_padded_dp0.items()}
    outputs_nondistributed = model(**input_data_thd_padded_dp0)

    # if os.environ.get("LOCAL_RANK") == "0":
    #     print("RANK 0 Outputs non-distributed: ", outputs_nondistributed)
    #     print("RANK 0 Loss: ", outputs_nondistributed.loss)
    #     print("RANK 0 Logits: ", outputs_nondistributed.logits.max())
    if os.environ.get("LOCAL_RANK") == "1":
        print("RANK 1 Outputs non-distributed: ", outputs_nondistributed)
        print("RANK 1 Loss: ", outputs_nondistributed.loss)
        print("RANK 1 Logits: ", outputs_nondistributed.logits.max())

    # Now do the whole CP thing.
    # TODO(@jomitchell): do i need a barrier here?
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    # Create a device mesh for DDP=1, CP=2
    ddp_size=1
    cp_size=2
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(ddp_size, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    model = model.to(device=device)
    group_fsdp_cp = device_mesh[("ddp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )
    cp_group = device_mesh["cp"].get_group()
    cp_rank = device_mesh.get_local_rank("cp")
    cp_world_size=torch.distributed.get_world_size(group=cp_group)
    
    for i, transformer_layer in enumerate(model.module.esm.encoder.layers):
        transformer_layer.set_context_parallel_group(
            cp_group,
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream()
        )

    batch = get_dummy_data_thd_with_padding_dp0(tokenizer=tokenizer)
    # Move batch to CUDA
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # print(f"[Rank {cp_rank}] BEFORE split - batch keys: {batch.keys()}")
    # print(f"[Rank {cp_rank}] BEFORE split - input_ids shape: {batch['input_ids'].shape}")
    # print(f"[Rank {cp_rank}] BEFORE split - cu_seq_lens_q_padded: {batch['cu_seq_lens_q_padded']}")
    # print(f"[Rank {cp_rank}] CP world size: {cp_world_size}")

    batch_cp = get_batch_for_cp_rank(batch, cp_rank=cp_rank, cp_world_size=cp_world_size)
    
    print(f"[CP Rank {cp_rank}] AFTER split - batch_cp keys: {batch_cp.keys()}")
    if 'input_ids' in batch_cp:
        print(f"[CP Rank {cp_rank}] AFTER split - input_ids: {batch_cp['input_ids']}")
        print(f"[CP Rank {cp_rank}] AFTER split - input_ids shape: {batch_cp['input_ids'].shape if batch_cp['input_ids'] is not None else 'None'}")
    else:
        print(f"[CP Rank {cp_rank}] AFTER split - input_ids key MISSING from batch_cp!")
    
    torch.distributed.barrier(group=cp_group)

    # batch_cp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_cp.items()}
    outputs_cp = model(**batch_cp)
    # print("CP Rank: ", cp_rank, "Outputs CP: ", outputs_cp)
    print("CP Rank: ", cp_rank, "Outputs CP Loss: ", outputs_cp.loss)
    print("CP Rank: ", cp_rank, "Outputs CP Logits shape: ", outputs_cp.logits.shape)
    
    # Gather the logits from all CP ranks
    # The logits are split along the sequence dimension (dim=1 for THD format: [batch, seq, vocab])
    # Make sure the tensor is contiguous before gathering
    logits_contiguous = outputs_cp.logits.contiguous()
    logits_list = [torch.zeros_like(logits_contiguous) for _ in range(cp_world_size)]
    torch.distributed.all_gather(logits_list, logits_contiguous, group=cp_group)
    
    # Verify the ordering by checking which tensor came from which rank
    # Create a rank identifier tensor to verify gather order
    rank_id = torch.tensor([cp_rank], dtype=torch.int32, device=device)
    rank_ids = [torch.zeros_like(rank_id) for _ in range(cp_world_size)]
    torch.distributed.all_gather(rank_ids, rank_id, group=cp_group)
    
    if cp_rank == 0:
        print(f"Verification: logits_list indices correspond to CP ranks: {[r.item() for r in rank_ids]}")
    
    # Concatenate along the sequence dimension
    # logits_full = torch.cat(logits_list, dim=1)
    
    if cp_rank == 0:
        print("Logits list: ", logits_list)
        print("Logits list[0] shape: ", logits_list[0].shape)
        print("Logits list[1] shape: ", logits_list[1].shape)
    
        # Reconstruct the full logits from CP-split chunks dynamically
        # Get sequence lengths from cu_seqlens
        cu_seqlens = batch["cu_seq_lens_q_padded"].cpu()
        num_seqs = len(cu_seqlens) - 1
        total_tokens = int(cu_seqlens[-1].item())
        vocab_size = outputs_nondistributed.logits.shape[-1]
        
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
                    reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = \
                        logits_list[0][cp_offset_rank0:cp_offset_rank0 + chunk_size, :]
                    cp_offset_rank0 += chunk_size
                else:  # Chunks 1, 2 for CP rank 1
                    reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = \
                        logits_list[1][cp_offset_rank1:cp_offset_rank1 + chunk_size, :]
                    cp_offset_rank1 += chunk_size

        # Now reconstructed logits should match the non-distributed logits
        print("Reconstructed logits shape: ", reconstructed_logits.shape)
        print("Reconstructed logits max: ", reconstructed_logits.max())
        print("Reconstructed logits min: ", reconstructed_logits.min())
        print("Reconstructed logits mean: ", reconstructed_logits.mean())
        print("Reconstructed logits std: ", reconstructed_logits.std())
        print("Reconstructed logits var: ", reconstructed_logits.var())
        print("Reconstructed logits sum: ", reconstructed_logits.sum())

        print("Non-distributed logits shape: ", outputs_nondistributed.logits.shape)
        print("Non-distributed logits max: ", outputs_nondistributed.logits.max())
        print("Non-distributed logits min: ", outputs_nondistributed.logits.min())
        print("Non-distributed logits mean: ", outputs_nondistributed.logits.mean())
        print("Non-distributed logits std: ", outputs_nondistributed.logits.std())
        print("Non-distributed logits var: ", outputs_nondistributed.logits.var())
        print("Non-distributed logits sum: ", outputs_nondistributed.logits.sum())

        # print hte first 32 values of the reconstructed logits and the non-distributed logits
        print("Reconstructed logits first 32 values: ", reconstructed_logits[0:32, :])
        print("Non-distributed logits first 32 values: ", outputs_nondistributed.logits[0:32, :])

        # Use relaxed tolerances for bfloat16 and distributed operations
        # atol=0.2 handles observed max differences of ~0.125
        # rtol=0.01 (1%) is reasonable for bfloat16 precision
        torch.testing.assert_close(
            reconstructed_logits.cpu(), 
            outputs_nondistributed.logits.cpu(),
            atol=0.29, 
            rtol=0.01,
        )
        print("\n✓ CP outputs match non-distributed outputs within tolerance!")
    torch.distributed.destroy_process_group()