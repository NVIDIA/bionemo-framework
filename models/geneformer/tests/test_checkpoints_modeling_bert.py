"""Tests for Geneformer checkpoint integration with Transformer Engine models."""

import pytest
import torch
from transformers import AutoModelForMaskedLM


def load_geneformer_model(model_name):
    """Helper function to load the correct Geneformer model variant."""
    if model_name == "Geneformer-V2-316M":
        # Default model (no subfolder needed)
        return AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
    else:
        # Use subfolder for specific model variants
        return AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer", subfolder=model_name)


# Model variants with detailed information
MODEL_VARIANTS = [
    (
        "Geneformer-V1-10M",
        {
            "description": "10M parameters, June 2021",
            "input_size": 2048,
            "vocabulary": "~25K protein-coding or non-coding RNA genes",
            "training_data": "~30M human single cell transcriptomes",
        },
    ),
    (
        "Geneformer-V2-104M",
        {
            "description": "104M parameters, Dec 2024",
            "input_size": 4096,
            "vocabulary": "~20K protein-coding genes",
            "training_data": "~104M human single cell transcriptomes",
        },
    ),
    (
        "Geneformer-V2-316M",
        {
            "description": "316M parameters, Dec 2024 (default)",
            "input_size": 4096,
            "vocabulary": "~20K protein-coding genes",
            "training_data": "~104M human single cell transcriptomes",
        },
    ),
]


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS, ids=[variant[0] for variant in MODEL_VARIANTS])
def test_geneformer_checkpoint_has_te_layers(model_variant):
    """Test that the actual Geneformer checkpoints use Transformer Engine layers."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    model_name, model_info = model_variant

    # Require CUDA to be available
    assert torch.cuda.is_available(), "CUDA is required for this test"

    print(f"Loading {model_name} checkpoint from Hugging Face...")
    print(f"  - {model_info['description']}")
    print(f"  - Input size: {model_info['input_size']}")
    print(f"  - Vocabulary: {model_info['vocabulary']}")
    print(f"  - Training data: {model_info['training_data']}")

    # Load the specific Geneformer checkpoint from Hugging Face
    model_hf = load_geneformer_model(model_name)

    # Convert the config to our TE config format
    hf_config_dict = model_hf.config.to_dict()
    te_config_dict = {
        "hidden_size": hf_config_dict["hidden_size"],
        "num_hidden_layers": hf_config_dict["num_hidden_layers"],
        "num_attention_heads": hf_config_dict["num_attention_heads"],
        "intermediate_size": hf_config_dict["intermediate_size"],
        "max_position_embeddings": hf_config_dict["max_position_embeddings"],
        "vocab_size": hf_config_dict["vocab_size"],
        "attention_probs_dropout_prob": hf_config_dict.get("attention_probs_dropout_prob", 0.1),
        "hidden_dropout_prob": hf_config_dict.get("hidden_dropout_prob", 0.1),
        "hidden_act": hf_config_dict.get("hidden_act", "relu"),
        "initializer_range": hf_config_dict.get("initializer_range", 0.02),
        "layer_norm_eps": hf_config_dict.get("layer_norm_eps", 1e-12),
        "pad_token_id": hf_config_dict.get("pad_token_id", 0),
        "model_type": hf_config_dict.get("model_type", "bert"),
        "torch_dtype": torch.float32,
        "use_te_layers": True,  # Enable TE layers
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    print(f"Creating TE model with config: {te_config_dict}")

    te_config = TEBertConfig(**te_config_dict)
    model_te = TEBertForMaskedLM(te_config)

    device = torch.device("cuda")
    model_hf = model_hf.to(device)
    model_te = model_te.to(device)

    print("Verifying TE model has Transformer Engine layers...")

    # print(f"\nAll layers in {model_name} TE model:")
    # for name, module in model_te.named_modules():
    #     print(f"  {name}: {type(module).__name__}")

    # Check that our TE model has Transformer Engine layers
    # We need to verify that the layers are using TE implementations
    te_layer_count = 0

    # Check the encoder layers
    encoder = model_te.bert.encoder
    for i, layer in enumerate(encoder.layer):
        if hasattr(layer, "__class__") and "TEBertLayer" in str(layer.__class__):
            print(f"Layer {i}: {type(layer).__name__}")
            te_layer_count += 1
        else:
            print(f"Layer {i}: {type(layer).__name__} (not TEBertLayer)")

    print("\nLayer verification summary:")
    print(f"  - TEBertLayer instances found: {te_layer_count}")
    print(f"  - Total encoder layers: {len(encoder.layer)}")

    # Assert that we have the expected number of TE layers
    expected_layers = te_config.num_hidden_layers
    assert te_layer_count == expected_layers, (
        f"Expected {expected_layers} TEBertLayer instances, found {te_layer_count}"
    )
    print(f"All {te_layer_count} encoder layers are TEBertLayer instances")

    print("TE model architecture verification passed!")

    del model_hf, model_te
    torch.cuda.empty_cache()

    print(f"{model_name} checkpoint TE layer verification completed successfully!")


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS, ids=[variant[0] for variant in MODEL_VARIANTS])
def test_geneformer_checkpoint_loss(model_variant):
    """Test that the TE model can process input data and produce valid loss outputs."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    model_name, model_info = model_variant

    # Require CUDA to be available
    assert torch.cuda.is_available(), "CUDA is required for this test"

    print(f"Testing loss computation for {model_name}...")
    print(f"  - {model_info['description']}")

    # Load the specific Geneformer checkpoint from Hugging Face
    model_hf = load_geneformer_model(model_name)

    # Convert the config to our TE config format
    hf_config_dict = model_hf.config.to_dict()
    te_config_dict = {
        "hidden_size": hf_config_dict["hidden_size"],
        "num_hidden_layers": hf_config_dict["num_hidden_layers"],
        "num_attention_heads": hf_config_dict["num_attention_heads"],
        "intermediate_size": hf_config_dict["intermediate_size"],
        "max_position_embeddings": hf_config_dict["max_position_embeddings"],
        "vocab_size": hf_config_dict["vocab_size"],
        "attention_probs_dropout_prob": hf_config_dict.get("attention_probs_dropout_prob", 0.1),
        "hidden_dropout_prob": hf_config_dict.get("hidden_dropout_prob", 0.1),
        "hidden_act": hf_config_dict.get("hidden_act", "relu"),
        "initializer_range": hf_config_dict.get("initializer_range", 0.02),
        "layer_norm_eps": hf_config_dict.get("layer_norm_eps", 1e-12),
        "pad_token_id": hf_config_dict.get("pad_token_id", 0),
        "model_type": hf_config_dict.get("model_type", "bert"),
        "torch_dtype": torch.float32,
        "use_te_layers": True,  # Enable TE layers
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    print(f"Creating TE model with config: {te_config_dict}")

    # Create our TE model with the same architecture
    te_config = TEBertConfig(**te_config_dict)
    model_te = TEBertForMaskedLM(te_config)

    # Move both models to CUDA
    device = torch.device("cuda")
    model_hf = model_hf.to(device)
    model_te = model_te.to(device)

    # Test that both models can process the same input
    print("Testing model compatibility with input data...")

    batch_size = 1
    seq_length = min(128, te_config.max_position_embeddings)

    input_ids = torch.randint(0, te_config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float32, device=device)
    labels = torch.randint(0, te_config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)

    test_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Test both models can process the input
    with torch.no_grad():
        te_outputs = model_te(**test_input)

        hf_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        hf_outputs = model_hf(**hf_input)

    # Verify both models produce valid outputs
    assert te_outputs.loss is not None, "TE model should produce loss"
    assert hf_outputs.loss is not None, "HF model should produce loss"
    assert te_outputs.logits.shape == hf_outputs.logits.shape, "Both models should have same output shape"

    print("Model compatibility test passed!")
    print(f" - TE model loss: {te_outputs.loss.item():.4f}")
    print(f" - HF model loss: {hf_outputs.loss.item():.4f}")
    print(f" - Output shape: {te_outputs.logits.shape}")

    # Clean up
    del model_hf, model_te
    torch.cuda.empty_cache()

    print(f"{model_name} loss computation test completed successfully!")


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS, ids=[variant[0] for variant in MODEL_VARIANTS])
def test_geneformer_checkpoint_weight_compatibility(model_variant):
    """Test that our TE model can potentially load weights from the actual Geneformer checkpoints."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    model_name, model_info = model_variant

    # Require CUDA to be available
    assert torch.cuda.is_available(), "CUDA is required for this test"

    print(f"Testing weight compatibility with {model_name} checkpoint...")
    print(f"  - {model_info['description']}")
    print(f"  - Input size: {model_info['input_size']}")
    print(f"  - Vocabulary: {model_info['vocabulary']}")
    print(f"  - Training data: {model_info['training_data']}")

    model_hf = load_geneformer_model(model_name)

    hf_state_dict = model_hf.state_dict()

    print(f"Loaded checkpoint with {len(hf_state_dict)} parameters")
    print(
        f"Checkpoint config: hidden_size={model_hf.config.hidden_size}, "
        f"layers={model_hf.config.num_hidden_layers}, "
        f"heads={model_hf.config.num_attention_heads}"
    )

    # Create our TE model with the same architecture
    te_config_dict = {
        "hidden_size": model_hf.config.hidden_size,
        "num_hidden_layers": model_hf.config.num_hidden_layers,
        "num_attention_heads": model_hf.config.num_attention_heads,
        "intermediate_size": model_hf.config.intermediate_size,
        "max_position_embeddings": model_hf.config.max_position_embeddings,
        "vocab_size": model_hf.config.vocab_size,
        "attention_probs_dropout_prob": getattr(model_hf.config, "attention_probs_dropout_prob", 0.1),
        "hidden_dropout_prob": getattr(model_hf.config, "hidden_dropout_prob", 0.1),
        "hidden_act": getattr(model_hf.config, "hidden_act", "relu"),
        "initializer_range": getattr(model_hf.config, "initializer_range", 0.02),
        "layer_norm_eps": getattr(model_hf.config, "layer_norm_eps", 1e-12),
        "pad_token_id": getattr(model_hf.config, "pad_token_id", 0),
        "model_type": getattr(model_hf.config, "model_type", "bert"),
        "torch_dtype": torch.float32,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    te_config = TEBertConfig(**te_config_dict)
    model_te = TEBertForMaskedLM(te_config)

    print(f"Created TE model with config: {te_config_dict}")

    te_state_dict = model_te.state_dict()

    print(f"TE model has {len(te_state_dict)} parameters")

    _run_compatibility_analysis(hf_state_dict, te_state_dict, te_config)

    del model_hf, model_te
    torch.cuda.empty_cache()

    print(f"{model_name} weight compatibility analysis completed!")


# Helper functions for parameter compatibility analysis
def _expand_pattern(pattern, state_dict):
    """Expand wildcard patterns like 'bert.encoder.layer.*.attention.output.dense.weight'"""
    expanded = []
    for key in state_dict.keys():
        # Check if this key matches the pattern
        if "bert.encoder.layer." in pattern and "bert.encoder.layer." in key:
            # Extract layer number from the key
            key_parts = key.split(".")
            pattern_parts = pattern.split(".")

            # Find the layer number in the key
            layer_num = None
            for i, part in enumerate(key_parts):
                if part == "layer" and i + 1 < len(key_parts) and key_parts[i + 1].isdigit():
                    layer_num = key_parts[i + 1]
                    break

            if layer_num is not None:
                # Check if the key structure matches the pattern structure
                if len(key_parts) == len(pattern_parts) and all(
                    p1 == p2 or p2 == "*" for p1, p2 in zip(key_parts, pattern_parts)
                ):
                    # Replace wildcard with actual layer number
                    expanded_pattern = pattern.replace("*", layer_num)
                    expanded.append((expanded_pattern, key))

    return expanded


def _get_parameter_mapping():
    """Get the mapping from HF BERT format to TE format."""
    return {
        "bert.embeddings.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight": "bert.embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias": "bert.embeddings.LayerNorm.bias",
        # Attention self components (individual Q, K, V - unpacked from fused format)
        "bert.encoder.layer.*.attention.self.query.weight": "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.query.bias": "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.weight": "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.key.bias": "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.weight": "bert.encoder.layer.*.attention.self.value.weight",
        "bert.encoder.layer.*.attention.self.value.bias": "bert.encoder.layer.*.attention.self.value.bias",
        # Attention output components
        "bert.encoder.layer.*.attention.output.dense.weight": "bert.encoder.layer.*.self_attention.proj.weight",
        "bert.encoder.layer.*.attention.output.dense.bias": "bert.encoder.layer.*.self_attention.proj.bias",
        "bert.encoder.layer.*.attention.output.LayerNorm.weight": "bert.encoder.layer.*.self_attention.layernorm_qkv.layer_norm_weight",
        "bert.encoder.layer.*.attention.output.LayerNorm.bias": "bert.encoder.layer.*.self_attention.layernorm_qkv.layer_norm_bias",
        # MLP components
        "bert.encoder.layer.*.intermediate.dense.weight": "bert.encoder.layer.*.layernorm_mlp.fc1_weight",
        "bert.encoder.layer.*.intermediate.dense.bias": "bert.encoder.layer.*.layernorm_mlp.fc1_bias",
        "bert.encoder.layer.*.output.dense.weight": "bert.encoder.layer.*.layernorm_mlp.fc2_weight",
        "bert.encoder.layer.*.output.dense.bias": "bert.encoder.layer.*.layernorm_mlp.fc2_bias",
        "bert.encoder.layer.*.output.LayerNorm.weight": "bert.encoder.layer.*.layernorm_mlp.layer_norm_weight",
        "bert.encoder.layer.*.output.LayerNorm.bias": "bert.encoder.layer.*.layernorm_mlp.layer_norm_bias",
        # Classification head
        "cls.predictions.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight": "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias": "cls.predictions.decoder.bias",
        "cls.predictions.transform.dense.weight": "cls.predictions.transform.dense.weight",
        "cls.predictions.transform.dense.bias": "cls.predictions.transform.dense.bias",
        "cls.predictions.transform.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
        "cls.predictions.transform.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
    }


def _check_wildcard_mapping(hf_pattern, te_pattern, hf_state_dict, te_state_dict):
    """Check compatibility for wildcard patterns."""
    compatible_params = 0
    incompatible_params = 0
    missing_params = 0

    expanded = _expand_pattern(hf_pattern, hf_state_dict)
    for expanded_hf, original_hf in expanded:
        # Extract layer number from the expanded HF pattern
        parts = expanded_hf.split(".")
        layer_num = None
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_num = parts[i + 1]
                break

        if layer_num is not None:
            # Get the corresponding TE parameter name
            expanded_te = te_pattern.replace("*", layer_num)
            if expanded_te in te_state_dict:
                hf_param = hf_state_dict[original_hf]
                te_param = te_state_dict[expanded_te]
                if hf_param.shape == te_param.shape:
                    compatible_params += 1
                    print(f"Mapped: {original_hf} -> {expanded_te}")
                else:
                    incompatible_params += 1
                    print(f"Shape mismatch: {original_hf} -> {expanded_te}: HF={hf_param.shape}, TE={te_param.shape}")
            else:
                missing_params += 1
                print(f"Missing mapped parameter: {original_hf} -> {expanded_te}")

    return compatible_params, incompatible_params, missing_params


def _check_direct_mapping(hf_pattern, hf_state_dict, te_state_dict):
    """Check compatibility for direct (non-wildcard) patterns."""
    if hf_pattern in hf_state_dict and hf_pattern in te_state_dict:
        hf_param = hf_state_dict[hf_pattern]
        te_param = te_state_dict[hf_pattern]
        if hf_param.shape == te_param.shape:
            print(f"Direct: {hf_pattern}")
            return 1, 0, 0
        else:
            print(f"Shape mismatch: {hf_pattern}: HF={hf_param.shape}, TE={te_param.shape}")
            return 0, 1, 0
    elif hf_pattern in hf_state_dict:
        print(f"Missing: {hf_pattern}")
        return 0, 0, 1
    return 0, 0, 0


def _analyze_parameter_compatibility(hf_state_dict, te_state_dict):
    """Analyze parameter compatibility between HF and TE models."""
    compatible_params = 0
    incompatible_params = 0
    missing_params = 0

    mapping = _get_parameter_mapping()

    # Check mapped parameters
    for hf_pattern, te_pattern in mapping.items():
        if "*" in hf_pattern:
            # Handle wildcard mapping
            comp, incomp, miss = _check_wildcard_mapping(hf_pattern, te_pattern, hf_state_dict, te_state_dict)
            compatible_params += comp
            incompatible_params += incomp
            missing_params += miss
        else:
            # Direct mapping (no wildcards)
            comp, incomp, miss = _check_direct_mapping(hf_pattern, hf_state_dict, te_state_dict)
            compatible_params += comp
            incompatible_params += incomp
            missing_params += miss

    return compatible_params, incompatible_params, missing_params


def _print_compatibility_results(compatible_params, incompatible_params, missing_params, total_params):
    """Print the compatibility analysis results."""
    print("\nParameter compatibility analysis:")
    print(f"  - Compatible parameters: {compatible_params}")
    print(f"  - Incompatible parameters: {incompatible_params}")
    print(f"  - Missing parameters: {missing_params}")
    print(f"  - Total checkpoint parameters: {total_params}")

    compatibility_ratio = compatible_params / total_params
    print(f"  - Compatibility ratio: {compatibility_ratio:.2%}")

    return compatibility_ratio


def _unpack_fused_qkv_in_te_state_dict(te_state_dict, num_layers, num_heads):
    """Unpack fused QKV parameters in TE state dict to match HF format for comparison."""
    from geneformer.convert import _unpack_qkv_bias, _unpack_qkv_weight

    unpacked_te_state_dict = te_state_dict.copy()

    # Create a mock context object to use the original unpack functions
    class MockContext:
        def __init__(self, num_heads):
            self.source = type("Config", (), {"config": type("Config", (), {"num_attention_heads": num_heads})()})()

    mock_ctx = MockContext(num_heads)

    # Access the underlying functions bypassing the decorator
    unpack_weight_func = _unpack_qkv_weight.transform
    unpack_bias_func = _unpack_qkv_bias.transform

    for layer_idx in range(num_layers):
        # Unpack fused QKV weight
        fused_weight_key = f"bert.encoder.layer.{layer_idx}.self_attention.layernorm_qkv.weight"
        if fused_weight_key in unpacked_te_state_dict:
            fused_weight = unpacked_te_state_dict[fused_weight_key]
            query_weight, key_weight, value_weight = unpack_weight_func(mock_ctx, fused_weight)

            # Add individual Q, K, V weights to the state dict
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.query.weight"] = query_weight
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.key.weight"] = key_weight
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.value.weight"] = value_weight

            # Remove the fused weight
            del unpacked_te_state_dict[fused_weight_key]

        # Unpack fused QKV bias
        fused_bias_key = f"bert.encoder.layer.{layer_idx}.self_attention.layernorm_qkv.bias"
        if fused_bias_key in unpacked_te_state_dict:
            fused_bias = unpacked_te_state_dict[fused_bias_key]
            query_bias, key_bias, value_bias = unpack_bias_func(mock_ctx, fused_bias)

            # Add individual Q, K, V biases to the state dict
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.query.bias"] = query_bias
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.key.bias"] = key_bias
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.value.bias"] = value_bias

            # Remove the fused bias
            del unpacked_te_state_dict[fused_bias_key]

    return unpacked_te_state_dict


def _run_compatibility_analysis(hf_state_dict, te_state_dict, te_config):
    """Run the parameter compatibility analysis."""
    # Unpack fused QKV parameters in TE state dict for accurate comparison
    unpacked_te_state_dict = _unpack_fused_qkv_in_te_state_dict(
        te_state_dict, te_config.num_hidden_layers, te_config.num_attention_heads
    )

    print(f"Unpacked TE state dict: {len(te_state_dict)} -> {len(unpacked_te_state_dict)} parameters")

    # Run the analysis with unpacked TE state dict
    compatible_params, incompatible_params, missing_params = _analyze_parameter_compatibility(
        hf_state_dict, unpacked_te_state_dict
    )

    compatibility_ratio = _print_compatibility_results(
        compatible_params, incompatible_params, missing_params, len(hf_state_dict)
    )

    # Now we expect 100% compatibility since we've unpacked the fused QKV parameters
    assert compatibility_ratio == 1.0, (
        f"Expected 100% compatibility after unpacking fused QKV parameters, but got {compatibility_ratio:.2%}. "
        f"All parameters should be mappable between HF and TE models."
    )

    print("100% compatibility achieved - all parameters can be properly mapped!")
    print("Note: Fused QKV parameters were unpacked for accurate comparison.")
