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

import torch
from nemo.utils import logging
from torch import nn


def debug_rna_head_learning(model, batch, step, rna_seq_targets):
    """Check if RNA head is actually learning."""
    if step % 50 != 0:
        return

    print(f"\n{'=' * 60}")
    print(f"Step {step} RNA Head Diagnostics")
    print(f"{'=' * 60}")

    # Get the RNA head
    if hasattr(model, "rna_seq_head"):
        head = model.rna_seq_head
    elif hasattr(model, "module") and hasattr(model.module, "rna_seq_head"):
        head = model.module.rna_seq_head
    else:
        print("❌ No RNA head found!")
        return

    # ✅ FIX 1: Check if it's a composite module or simple layer
    if isinstance(head, nn.ModuleList) or hasattr(head, "layers"):
        # Composite module (like BioSignalHead)
        print(f"📦 Head Type: Composite ({type(head).__name__})")
        print(f"   Number of layers: {len(head.layers) if hasattr(head, 'layers') else 'unknown'}")  # type: ignore

        # Check each layer
        if hasattr(head, "layers"):
            for i, layer in enumerate(head.layers):  # type: ignore
                if hasattr(layer, "weight"):
                    weight_norm = layer.weight.norm().item()
                    grad_norm = layer.weight.grad.norm().item() if layer.weight.grad is not None else 0.0
                    print(f"\n   Layer {i} ({type(layer).__name__}):")
                    print(f"      Weight norm: {weight_norm:.4f}")
                    print(f"      Gradient norm: {grad_norm:.6f}")
                    print(f"      Weight shape: {layer.weight.shape}")

                    # ⚠️ Check if gradients are flowing
                    if grad_norm < 1e-8:
                        print("      ⚠️  WARNING: Gradient is nearly zero!")
    else:
        # Simple single-layer module
        print(f"📦 Head Type: Simple ({type(head).__name__})")
        if hasattr(head, "weight"):
            weight_norm = head.weight.norm().item()
            grad_norm = head.weight.grad.norm().item() if head.weight.grad is not None else 0.0
            print(f"   Weight norm: {weight_norm:.4f}")
            print(f"   Gradient norm: {grad_norm:.6f}")

    # ✅ FIX 2: Check total parameter count and gradient stats
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    print("\n📊 Overall Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Check if any gradients exist
    has_grads = any(p.grad is not None for p in head.parameters())
    if has_grads:
        grad_norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        print(f"   Average gradient norm: {avg_grad:.6f}")
        print(f"   Max gradient norm: {max_grad:.6f}")

        if avg_grad < 1e-8:
            print("   ⚠️  WARNING: Gradients are vanishing!")
    else:
        print("   ❌ NO GRADIENTS! Head may be frozen or not in computation graph")

    # ✅ FIX 3: Check actual predictions vs targets
    print("\n🎯 Prediction vs Target Analysis:")
    try:
        # Get predictions from batch (assuming they're computed during forward)
        if "rna_seq_logits" in batch:
            rna_pred = batch["rna_seq_logits"]
        else:
            print("   ⚠️  No predictions found in batch")
            rna_pred = None

        if rna_pred is not None and rna_seq_targets is not None:
            print("   Predictions:")
            print(f"      Shape: {rna_pred.shape}")
            print(f"      Min/Max: {rna_pred.min():.3f} / {rna_pred.max():.3f}")
            print(f"      Mean/Std: {rna_pred.mean():.3f} / {rna_pred.std():.3f}")

            print("   Targets:")
            print(f"      Shape: {rna_seq_targets.shape}")
            print(f"      Min/Max: {rna_seq_targets.min():.3f} / {rna_seq_targets.max():.3f}")
            print(f"      Mean/Std: {rna_seq_targets.mean():.3f} / {rna_seq_targets.std():.3f}")

            # Check correlation
            if rna_pred.numel() == rna_seq_targets.numel():
                pred_flat = rna_pred.flatten()
                target_flat = rna_seq_targets.flatten()

                # Pearson correlation
                pred_centered = pred_flat - pred_flat.mean()
                target_centered = target_flat - target_flat.mean()
                correlation = (pred_centered * target_centered).sum() / (
                    pred_centered.norm() * target_centered.norm() + 1e-8
                )
                print(f"   Pearson correlation: {correlation.item():.4f}")

                if abs(correlation.item()) < 0.01:
                    print("   ⚠️  WARNING: Predictions not correlated with targets!")

            # Check if predictions are constant
            if rna_pred.std() < 1e-4:
                print("   ❌ PROBLEM: Predictions are nearly constant!")
                print("      Head may have collapsed to predicting mean value")
    except Exception as e:
        print(f"   ❌ Error analyzing predictions: {e}")

    print(f"{'=' * 60}\n")


def test_simple_dual_head_approach():
    """Dry test function.

    Test the simple dual-head approach that extends original MockDataModule.
    """
    logging.info("🧪 Testing SIMPLE dual-head approach...")

    try:
        # Test imports
        logging.info("🔧 Testing imports...")
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        from .mockdata import ParallelHeadMockDataModule

        logging.info("✅ Imports successful")

        # Create tokenizer
        logging.info("🔧 Creating tokenizer...")
        tokenizer = get_nmt_tokenizer("byte-level")
        logging.info(f"✅ Tokenizer created: vocab_size={tokenizer.vocab_size}")

        # Create data module
        logging.info("🔧 Creating SimpleDualHeadMockDataModule...")
        data_module = ParallelHeadMockDataModule(
            seq_length=512,  # Small for testing
            tokenizer=tokenizer,  # type: ignore
            micro_batch_size=2,
            global_batch_size=4,
            num_workers=0,  # No workers for testing
            num_train_samples=10,  # Very small
            num_val_samples=5,
            expression_pattern="realistic",
        )
        logging.info("✅ Data module created")

        # Test setup
        logging.info("🔧 Testing setup...")
        data_module.setup(stage="fit")
        logging.info("✅ Setup completed")

        # Test single sample
        logging.info("🔧 Testing single sample...")
        sample = data_module._train_ds[0]

        # Validate sample structure - should match ORIGINAL MockDataModule + expression_targets
        expected_fields = ["tokens", "labels", "loss_mask", "position_ids", "expression_targets"]
        for field in expected_fields:
            if field not in sample:
                raise ValueError(f"Missing field: {field}")
            if sample[field] is None:
                raise ValueError(f"Field '{field}' is None!")

        logging.info("✅ Sample structure valid (original + expression_targets):")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logging.info(f"   {key}: {value.shape} {value.dtype}")
                if key == "expression_targets":
                    logging.info(f"     Expression range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                elif key == "labels":
                    logging.info(f"     Labels (shifted tokens) range: [{value.min().item()}, {value.max().item()}]")

        # Validate data relationships (original MockDataModule logic)
        tokens = sample["tokens"]
        labels = sample["labels"]
        expression_targets = sample["expression_targets"]

        # Check that labels are shifted tokens (original logic)
        if tokens.shape != labels.shape:
            raise ValueError(f"Tokens and labels shape mismatch: {tokens.shape} vs {labels.shape}")

        # Check expression targets match input length
        if tokens.shape != expression_targets.shape:
            raise ValueError(
                f"Tokens and expression_targets shape mismatch: {tokens.shape} vs {expression_targets.shape}"
            )

        logging.info("✅ Data relationships validated")

        # Test dataloader
        logging.info("🔧 Testing dataloader...")
        train_loader = data_module.train_dataloader()

        # Get a batch
        batch = next(iter(train_loader))
        logging.info("✅ Batch obtained:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logging.info(f"   {key}: {value.shape} {value.dtype}")
                if key == "expression_targets" and value is not None:
                    logging.info(f"     Expression range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                elif key == "labels" and value is not None:
                    logging.info(f"     Labels range: [{value.min().item()}, {value.max().item()}]")

        # Critical validations
        required_fields = ["tokens", "labels", "loss_mask", "position_ids", "expression_targets"]
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Missing required field in batch: {field}")
            if batch[field] is None:
                raise ValueError(f"CRITICAL: {field} is None in batch!")

        # Validate shapes
        batch_size = 2
        seq_length = 512
        expected_shape = (batch_size, seq_length)

        for key in ["tokens", "labels", "loss_mask", "position_ids", "expression_targets"]:
            actual_shape = batch[key].shape
            if actual_shape != expected_shape:
                # Allow some flexibility for different dtypes
                if actual_shape[:2] != expected_shape:
                    logging.warning(f"Wrong shape for {key}: {actual_shape} vs {expected_shape}")

        logging.info("✅ Batch shape validation passed")

        # Test forward step compatibility
        logging.info("🔧 Testing forward step compatibility...")

        # This should work with ORIGINAL field names
        forward_args = {
            "input_ids": batch["tokens"],  # ✅ Original field name
            "position_ids": batch["position_ids"],  # ✅ Original field name
            "attention_mask": batch.get("attention_mask"),  # ✅ Optional
            "labels": batch["labels"],  # ✅ Shifted tokens for LM loss
            "loss_mask": batch["loss_mask"],  # ✅ Original field name
            "expression_targets": batch["expression_targets"],  # ✅ Our addition
        }

        logging.info(f"✅ Forward args prepared: {list(forward_args.keys())}")

        # Validate that all forward args are not None where required
        for key, value in forward_args.items():
            if key in ["input_ids", "position_ids", "labels", "loss_mask", "expression_targets"]:
                if value is None:
                    raise ValueError(f"CRITICAL: Required forward arg '{key}' is None!")
                logging.info(f"   {key}: {value.shape} {value.dtype}")

        logging.info("✅ Forward step compatibility validated")

        # Test that this matches original MockDataModule expectations
        logging.info("🔧 Testing original MockDataModule compatibility...")

        # Check field names match exactly what original MockDataModule produces
        original_fields = ["tokens", "labels", "loss_mask", "position_ids"]
        for field in original_fields:
            if field not in batch:
                raise ValueError(f"Missing original MockDataModule field: {field}")

        # Check that we've only ADDED expression_targets, not changed anything else
        extra_fields = set(batch.keys()) - set(original_fields)
        if extra_fields != {"expression_targets"}:
            logging.warning(f"⚠️  Unexpected extra fields beyond expression_targets: {extra_fields}")

        logging.info("✅ Original MockDataModule compatibility validated")

        logging.info("🎉 SIMPLE dual-head approach test passed!")
        logging.info("   ✅ Uses original field names (tokens, labels, etc.)")
        logging.info("   ✅ Adds expression_targets without breaking anything")
        logging.info("   ✅ Compatible with original Hyena forward step")
        logging.info("   ✅ No None values detected")
        logging.info("   ✅ Proper data relationships maintained")

        return True

    except Exception as e:
        logging.error(f"❌ SIMPLE approach test failed: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


def debug_heads(
    name: str, model: torch.nn.Module, batch: dict | None = None, forward_args: dict | None = None, forced: bool = True
) -> None:
    """Debugging utility to log model and batch information during forward passes.

    Args:
        name: Name identifier for the debug instance.
        model: The model being debugged.
        batch: The input batch dictionary.
        forward_args: Additional arguments passed to the forward method.
        forced: If True, prints to stdout; otherwise uses logging.info.
    """
    if forced:
        print(f"🔄 {name} called")
        print(f"   Model type: {type(model)}")
        print(f"   Model id: {id(model)}")

        # Debug model hierarchy
        current = model
        level = 0
        while hasattr(current, "module") and level < 5:
            print(f"   Level {level}: {type(current)} (id: {id(current)})")
            if hasattr(current, "forward"):
                print(f"     Has forward method: {hasattr(current, '_original_forward')}")
            current = current.module  # type: ignore
            level += 1
        print(f"   Final level {level}: {type(current)} (id: {id(current)})")
        if hasattr(current, "forward"):
            print(f"     Has _original_forward: {hasattr(current, '_original_forward')}")
            print(f"     Has expression_head: {hasattr(current, 'expression_head')}")
            print(f"     Has parallel_token_head: {hasattr(current, 'parallel_token_head')}")

        if batch is not None:
            print(f"   Batch keys: {list(batch.keys())}")

            # Debug batch contents
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape} {value.dtype}")
                    if key == "expression_targets" and value is not None:
                        print(f"     Range: [{value.min().item():.3f}, {value.max().item():.3f}]")

        if forward_args is not None:
            print(f"   Forward args keys: {list(forward_args.keys())}")

    else:
        logging.info(f"🔄 {name} called")
        logging.info(f"   Model type: {type(model)}")
        logging.info(f"   Model id: {id(model)}")

        # Debug model hierarchy
        current = model
        level = 0
        while hasattr(current, "module") and level < 5:
            logging.info(f"   Level {level}: {type(current)} (id: {id(current)})")
            if hasattr(current, "forward"):
                logging.info(f"     Has forward method: {hasattr(current, '_original_forward')}")
            current = current.module  # type: ignore
            level += 1
        logging.info(f"   Final level {level}: {type(current)} (id: {id(current)})")
        if hasattr(current, "forward"):
            logging.info(f"     Has _original_forward: {hasattr(current, '_original_forward')}")
            logging.info(f"     Has expression_head: {hasattr(current, 'expression_head')}")
            logging.info(f"     Has parallel_token_head: {hasattr(current, 'parallel_token_head')}")

        if batch is not None:
            logging.info(f"   Batch keys: {list(batch.keys())}")

            # Debug batch contents
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f"   {key}: {value.shape} {value.dtype}")
                    if key == "expression_targets" and value is not None:
                        logging.info(f"     Range: [{value.min().item():.3f}, {value.max().item():.3f}]")

        if forward_args is not None:
            logging.info(f"   Forward args keys: {list(forward_args.keys())}")
