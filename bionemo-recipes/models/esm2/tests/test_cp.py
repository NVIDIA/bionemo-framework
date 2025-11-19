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

def get_dummy_data_thd_with_padding_dp0(cp_size: int = 2):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data with longer sequences (64 tokens each)
    seq1 = torch.arange(2, 34, dtype=torch.long)  # 32 tokens
    seq2 = torch.arange(3, 35, dtype=torch.long)  # 32 tokens
    input_ids = torch.cat([seq1, seq2])  # 64 tokens total
    
    labels1 = torch.arange(10, 42, dtype=torch.long) % 33  # 32 labels
    labels2 = torch.arange(15, 47, dtype=torch.long) % 33  # 32 labels
    labels = torch.cat([labels1, labels2])  # 64 labels total
    
    cu_seqlens_q = torch.tensor([0, 32, 64])
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

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 64,  # Updated to match larger sequences
        "max_length_k": 64,  # Updated to match larger sequences
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
    batch_shard["max_length_q"] = int((seqlens_q.max().item() + 63) // 64 * 64) # TODO(@jomitchell): Not sure if I need this anymore.
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
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")
    # For debugging
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")


if __name__ == "__main__":
    # Do everything here.
    tmp_path = Path("./tmp_model")
    os.makedirs(tmp_path, exist_ok=True)
    model_ckpt = get_te_model_checkpoint(tmp_path)

    input_data_thd_padded_dp0 = get_dummy_data_thd_with_padding_dp0()

    model = NVEsmForMaskedLM.from_pretrained(model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16)
    model.to("cuda")
    input_data_thd_padded_dp0 = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd_padded_dp0.items()}
    outputs_nondistributed = model(**input_data_thd_padded_dp0)

    # if os.environ.get("LOCAL_RANK") == "0":
    #     print("RANK 0 Outputs non-distributed: ", outputs_nondistributed)
    #     print("RANK 0 Loss: ", outputs_nondistributed.loss)
    #     print("RANK 0 Logits: ", outputs_nondistributed.logits.max())
    # if os.environ.get("LOCAL_RANK") == "1":
    #     print("RANK 1 Outputs non-distributed: ", outputs_nondistributed)
    #     print("RANK 1 Loss: ", outputs_nondistributed.loss)
    #     print("RANK 1 Logits: ", outputs_nondistributed.logits.max())

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

    batch = get_dummy_data_thd_with_padding_dp0()
    # Move batch to CUDA
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # print(f"[Rank {cp_rank}] BEFORE split - batch keys: {batch.keys()}")
    # print(f"[Rank {cp_rank}] BEFORE split - input_ids shape: {batch['input_ids'].shape}")
    # print(f"[Rank {cp_rank}] BEFORE split - cu_seq_lens_q_padded: {batch['cu_seq_lens_q_padded']}")
    # print(f"[Rank {cp_rank}] CP world size: {cp_world_size}")

    batch_cp = get_batch_for_cp_rank(batch, cp_rank=cp_rank, cp_world_size=cp_world_size)
    
    print(f"[Rank {cp_rank}] AFTER split - batch_cp keys: {batch_cp.keys()}")
    if 'input_ids' in batch_cp:
        print(f"[Rank {cp_rank}] AFTER split - input_ids: {batch_cp['input_ids']}")
        print(f"[Rank {cp_rank}] AFTER split - input_ids shape: {batch_cp['input_ids'].shape if batch_cp['input_ids'] is not None else 'None'}")
    else:
        print(f"[Rank {cp_rank}] AFTER split - input_ids key MISSING from batch_cp!")
    
    torch.distributed.barrier(group=cp_group)

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    outputs_cp = model(**batch_cp)
    

    torch.distributed.destroy_process_group()