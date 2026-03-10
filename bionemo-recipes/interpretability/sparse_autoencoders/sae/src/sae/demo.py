import torch.distributed as dist
import os
import torch
from sae.process_group_manager import ProcessGroupManager

def setup_dist():
    dist.init_process_group("nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_dist()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}")
    print(f"Local rank: {local_rank}")
    is_on_local_rank = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    pg = ProcessGroupManager(dp_size=2, tp_size=2)
    print(pg)

if __name__ == "__main__":
    main()