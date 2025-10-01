"""Stop and go tests for the ESM2 Native TE recipe."""

import pytest
import torch
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForMaskedLM
from torch.optim import AdamW
from scheduler import get_linear_schedule_with_warmup
from checkpoint import save_checkpoint_ddp
from dataset import create_dataloader


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

@dataclass
class MockSingleProcessDistributedConfig:
    rank: int
    local_rank: int
    world_size: int
    def is_main_process(self):
        return self.rank == 0


# Custom run loop
def test_stop_and_go_single_gpu(tmp_path):
    # Setup the dataloader
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockSingleProcessDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )
    
    # Setup the model
    config = AutoConfig.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True, dtype=torch.bfloat16)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    # The huggingface model has a contact head that we don't use in masked language pre-training, so we delete it to
    # avoid errors with unused parameters.
    try:
        del model.esm.contact_head
    except AttributeError:
        pass

    # Create optimizer.
    adamw_kwargs = {
        "lr": 4e-4,
        "fused": True,
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "weight_decay": 0.01
    }
    lr_scheduler_kwargs = {
        "num_warmup_steps": 2_000,
        "num_training_steps": 500_000
    }

    optimizer = AdamW(model.parameters(), **adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)

    device = torch.device(f"cuda:{dist_config.local_rank}")
    model = model.to(device=device)

    # Training loop.
    model.train()
    for step in range(10):
        batch = next(reference_dataloader_info.iterator)
        # TODO: You have to set the labels equal to the input_ids so we just train on everything
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        # Backward pass.
        loss = outputs.loss
        loss.backward()

        # Step optimizer.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step == 5:
            dataloader_state = reference_dataloader_info.dataloader.state_dict()
            torch.save(dataloader_state, f"{tmp_path}_step_{step}_dataloader_state.pt")
            save_checkpoint_ddp(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=f"{tmp_path}_step_{step}",
                step=step,
                dist_config=dist_config,
            )

    # Now save the results after 10 steps
    save_checkpoint_ddp(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_path=f"{tmp_path}_step_{step}",
        step=step,
        dist_config=dist_config,
    )
    # TODO: Save the logits / loss / gradients etc
    # TODO: Now check after running this script multiple times that those are the same.
    
    print("tmp path is", tmp_path)
    return 1/0


    # Run ten steps of training. Then save the (1) Loss (2) Logits and (3) Gradients. and (4) Last batch of data.
    # At step 5, save (1) the model checkpoint, (2) dataloader state, (3) Optimizer state and (4) Global step etc.

    # Resume training from the checkpoint and continue for another 5 steps.
    # Then compare.


    # Dist config
