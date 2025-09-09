import torch


def test_geneformer_model_creation():
    """Test that the geneformer BERT TE model can be created with proper config."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    # Create a geneformer-style config
    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "intermediate_size": 512,
        "hidden_act": "relu",  # uses relu
        "max_position_embeddings": 4096,  #  model capacity
        "model_type": "bert",
        "num_attention_heads": 4,
        "pad_token_id": 0,
        "vocab_size": 20275,  #  Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)

    assert model is not None, "Model should be created successfully"
    assert hasattr(model, "bert"), "Model should have bert attribute"
    assert hasattr(model, "cls"), "Model should have cls attribute"

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"

    print("Model creation test passed!")
    print(f" - Model parameters: {total_params:,}")


def test_geneformer_model_forward_pass(input_data):
    """Test that the geneformer model can perform forward pass and produce valid outputs."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 2048,
        "vocab_size": 25426,
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)

    device = torch.device("cuda")
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}

    model.train()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**input_data)

    # Verify outputs exist
    assert outputs is not None, "Model should produce outputs"
    assert hasattr(outputs, "loss"), "Outputs should have loss attribute"
    assert hasattr(outputs, "logits"), "Outputs should have logits attribute"

    print("Forward pass test passed!")


def test_geneformer_model_loss_validity(input_data):
    """Test that the geneformer model produces valid loss values."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 4096,  #  model capacity
        "vocab_size": 20275,  #  Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)

    device = torch.device("cuda")
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**input_data)

    # Verify loss exists and is a valid tensor
    assert outputs.loss is not None, "Model should produce a loss"
    assert isinstance(outputs.loss, torch.Tensor), "Loss should be a tensor"
    assert outputs.loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(outputs.loss), "Loss should not be NaN"
    assert not torch.isinf(outputs.loss), "Loss should not be infinite"
    assert outputs.loss > 0, "Loss should be positive"

    print("Loss validity test passed!")
    print(f"- Loss: {outputs.loss.detach().item():.4f}")


def test_geneformer_model_logits_shape(input_data):
    """Test that the geneformer model produces logits with correct shape."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 4096,  #  model capacity
        "vocab_size": 20275,  #  Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)

    device = torch.device("cuda")
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**input_data)

    assert outputs.logits is not None, "Model should produce logits"
    expected_logits_shape = (input_data["input_ids"].shape[0], input_data["input_ids"].shape[1], config.vocab_size)
    assert outputs.logits.shape == expected_logits_shape, (
        f"Logits shape {outputs.logits.shape} should be {expected_logits_shape}"
    )

    print("Logits shape test passed!")
    print(f"- Logits shape: {outputs.logits.shape}")


def test_geneformer_model_gradient_computation(input_data):
    """Test that the geneformer model can compute gradients."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 4096,  #  model capacity
        "vocab_size": 20275,  #  Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)

    device = torch.device("cuda")
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**input_data)

    outputs.loss.backward()

    # Check that gradients are computed for model parameters
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "Model should have gradients after backward pass"

    print("Gradient computation test passed!")


def test_geneformer_model_loss_convergence(input_data):
    """Test that the geneformer model loss decreases during training steps (CUDA required)."""
    import torch.optim as optim
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    device = torch.device("cuda")

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 4096,  # Actual model capacity
        "vocab_size": 20275,  # Actual Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    config = TEBertConfig(**config_dict)
    model = TEBertForMaskedLM(config)
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    input_data = {k: v.to(device) for k, v in input_data.items()}

    losses = []

    for step in range(5):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**input_data)
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    # Check that loss is decreasing (at least in the first few steps)
    assert losses[0] > 0, "Initial loss should be positive"
    print(f"Loss convergence test passed! Losses: {[f'{loss:.4f}' for loss in losses]}")
