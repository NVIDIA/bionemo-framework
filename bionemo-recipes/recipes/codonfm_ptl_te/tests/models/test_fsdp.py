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

"""FSDP tests for EncodonPL model.

IMPORTANT: Multi-GPU distributed tests MUST run as subprocesses via torchrun.
You cannot run FSDP/DDP directly in pytest - it will hang due to process spawning issues.
"""

import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def run_distributed_training_script(script_path: Path, num_gpus: int = 2, timeout: int = 120):
    """Run a training script with torchrun for multi-GPU testing.

    This is the ONLY way to properly test FSDP/DDP. Running directly in pytest will hang.
    """
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node",
            str(num_gpus),
            "--standalone",
            str(script_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        cwd=str(script_path.parent),
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Training script failed with exit code {result.returncode}")

    return result


@requires_multi_gpu
def test_encodon_pl_fsdp_training_2gpus(tmp_path):
    """Test EncodonPL training with FSDP on 2 GPUs.

    This test runs as a subprocess with torchrun because FSDP requires
    proper distributed process initialization that cannot be done within pytest.
    """

    # Create a self-contained training script
    training_script = textwrap.dedent("""
        import torch
        from torch.utils.data import Dataset, DataLoader
        from src.models.encodon_te_pl import EncodonTEPL
        from src.utils.fsdp_config import get_fsdp_strategy
        from src.data.metadata import MetadataFields
        from lightning.pytorch import Trainer

        class SimpleCodonDataset(Dataset):
            def __init__(self, num_samples=8, seq_length=64, vocab_size=69):
                self.num_samples = num_samples
                self.seq_length = seq_length
                self.vocab_size = vocab_size

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {
                    MetadataFields.INPUT_IDS: torch.randint(0, self.vocab_size, (self.seq_length,)),
                    MetadataFields.LABELS: torch.randint(0, self.vocab_size, (self.seq_length,)),
                    MetadataFields.ATTENTION_MASK: torch.ones(self.seq_length),
                    MetadataFields.INPUT_MASK: torch.ones(self.seq_length, dtype=torch.bool),
                }

        def main():
            base_config = {
                "vocab_size": 69,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 3,
                "position_embedding_type": "rotary",
                "classifier_dropout": 0.1,
                "rotary_theta": 1e4,
                "ignore_index": -100,
                "loss_type": "regression",
                "lora": False,
                "lora_alpha": 32.0,
                "lora_r": 16,
                "lora_dropout": 0.1,
                "finetune_strategy": "full",
                "num_classes": 2,
                "use_downstream_head": False,
                "cross_attention_hidden_dim": 256,
                "cross_attention_num_heads": 8,
                "max_position_embeddings": 2048,
                "attn_input_format": "bshd",
                "optimizer": torch.optim.AdamW,
                "scheduler": None,
            }

            # Create dataset and model
            dataset = SimpleCodonDataset(num_samples=8, seq_length=64)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            model = EncodonTEPL(**base_config)
            model.configure_model()

            # Create trainer with FSDP strategy
            trainer = Trainer(
                max_steps=2,
                max_epochs=1,
                accelerator="gpu",
                devices=2,
                num_nodes=1,
                strategy=get_fsdp_strategy(),
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )

            # Run training
            trainer.fit(model, train_dataloaders=dataloader)
            print(f"[SUCCESS] Training completed! Global step: {trainer.global_step}")

        if __name__ == "__main__":
            main()
    """)

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tmp_path) as f:
        f.write(training_script)
        script_path = Path(f.name)

    try:
        # Run the training script with torchrun
        result = run_distributed_training_script(script_path, num_gpus=2)

        # Verify training completed successfully
        assert "[SUCCESS]" in result.stdout, "Training did not complete successfully"
    finally:
        # Cleanup temporary script
        script_path.unlink(missing_ok=True)
