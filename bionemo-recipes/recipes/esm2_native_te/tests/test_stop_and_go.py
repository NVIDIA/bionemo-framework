"""Stop and go tests for the ESM2 Native TE recipe."""

import pytest
import torch
import os
from dataclasses import dataclass
import shutil
from transformers import AutoConfig, AutoModelForMaskedLM
from torch.optim import AdamW
from scheduler import get_linear_schedule_with_warmup
from checkpoint import save_checkpoint_ddp, load_checkpoint_ddp
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

    step5_path_reference = f"{tmp_path}_step_5"
    step10_path_reference = f"{tmp_path}_step_10"
    step5_path_reloaded = f"{tmp_path}_step_5_reloaded"
    if os.path.exists(step5_path_reference):
        shutil.rmtree(step5_path_reference)
    if os.path.exists(step10_path_reference):
        shutil.rmtree(step10_path_reference)
    if os.path.exists(step5_path_reloaded):
        shutil.rmtree(step5_path_reloaded)
    os.makedirs(step5_path_reference, exist_ok=True)
    os.makedirs(step10_path_reference, exist_ok=True)
    os.makedirs(step5_path_reloaded, exist_ok=True)
    # Training loop.
    model.train()
    for step, batch in enumerate(reference_dataloader_info.dataloader):
        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        # Backward pass.
        loss = outputs.loss
        logits = outputs.logits
        # TODO[LIGHT]: Get gradients
        loss.backward()

        # Step optimizer.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step == 5:
            dataloader_state = reference_dataloader_info.dataloader.state_dict()
            torch.save(dataloader_state, f"{step5_path_reference}_dataloader_state.pt")
            
            # torch.save(logits.cpu(), f"{step5_path_reference}_logits.pt")
            # torch.save(loss.cpu(), f"{step5_path_reference}_loss.pt")
            # Let's save a random layers weights to disk also
            save_checkpoint_ddp(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=step5_path_reference,
                step=step,
                dist_config=dist_config,
            )
        if step == 9:
            break

    # Now save the results after 10 steps
    save_checkpoint_ddp(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_path=step10_path_reference,
        step=step,
        dist_config=dist_config,
    )
    torch.save(logits.cpu(), f"{step10_path_reference}_logits.pt")
    torch.save(loss.cpu(), f"{step10_path_reference}_loss.pt")
    torch.save(batch, f"{step10_path_reference}_batch.pt")
    print("step5_path is", step5_path_reference)
    print("created the following files: os.listdir(step5_path_reference)", os.listdir(step5_path_reference))
    print("step10_path is", step10_path_reference)
    print("created the following files: os.listdir(step10_path_reference)", os.listdir(step10_path_reference))

    # Create fresh model, optimizer, scheduler for the resume test
    config = AutoConfig.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True, dtype=torch.bfloat16)
    resumed_model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    
    try:
        del resumed_model.esm.contact_head
    except AttributeError:
        pass
    
    resumed_model = resumed_model.to(device=device)
    resumed_optimizer = AdamW(resumed_model.parameters(), **adamw_kwargs)
    resumed_scheduler = get_linear_schedule_with_warmup(resumed_optimizer, **lr_scheduler_kwargs)
    
    # Load checkpoint from step 5 into the fresh model
    resumed_model, resumed_optimizer, resumed_scheduler, start_step = load_checkpoint_ddp(
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            ckpt_path=step5_path_reference,
            dist_config=dist_config,
        )
    dataloader_state = torch.load(f"{step5_path_reference}_dataloader_state.pt")
    # Now make a dataloader brand new and restore the state?
    new_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    new_dataloader = new_dataloader_info.dataloader
    new_dataloader.load_state_dict(dataloader_state)

    resumed_model.train()
    for step, batch in enumerate(new_dataloader):
        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = resumed_model(**batch)

        # Backward pass.
        loss = outputs.loss
        logits = outputs.logits
        # TODO[LIGHT]: Get gradients
        loss.backward()

        # Step optimizer.
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_optimizer.zero_grad()
        if step == 3:
            break


    # Now save the results after 5 steps from the new dataloader. Which should match 10 steps of the reference dataloader.
    torch.save(logits.cpu(), f"{step5_path_reloaded}_logits.pt")
    torch.save(loss.cpu(), f"{step5_path_reloaded}_loss.pt")
    torch.save(batch, f"{step5_path_reloaded}_batch.pt")
    torch.save(new_dataloader_info.dataloader.state_dict(), f"{step5_path_reloaded}_dataloader_state.pt")
    save_checkpoint_ddp(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ckpt_path=step5_path_reloaded,
        step=step,
        dist_config=dist_config,
    )

    # Let's compare the batches now.
    reference_batch_step_10 = torch.load(f"{step10_path_reference}_batch.pt")['input_ids']
    reloaded_batch_step_5 = torch.load(f"{step5_path_reloaded}_batch.pt")['input_ids']
    assert torch.equal(reference_batch_step_10, reloaded_batch_step_5), \
        "Final batches don't match - dataloader state restoration may have failed"


    # Let's compare logits now (using allclose for floating point tolerance)
    reference_logits_step_10 = torch.load(f"{step10_path_reference}_logits.pt")
    reloaded_logits_step_5 = torch.load(f"{step5_path_reloaded}_logits.pt")
    assert torch.allclose(reference_logits_step_10, reloaded_logits_step_5, rtol=1e-5, atol=1e-5), \
        "Logits don't match - model state may not have been restored correctly"
    