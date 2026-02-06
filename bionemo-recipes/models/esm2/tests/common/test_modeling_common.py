# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Common test class for BioNeMo models, following HuggingFace transformers patterns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Type

import pytest
import torch
import transformer_engine.pytorch
from torch import nn
from transformer_engine.pytorch import QuantizedTensor
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, set_seed


HAS_DATA_CENTER_GPU = any(
    gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
)


@dataclass
class TestTolerances:
    """Model-specific test tolerances for numerical comparisons."""

    # Golden value test tolerances
    golden_value_loss_atol: float = 1e-2
    golden_value_loss_rtol: float = 1e-3
    golden_value_logits_atol: float = 2.0
    golden_value_logits_rtol: float = 1e-4
    golden_value_hidden_states_atol: float = 0.1
    golden_value_hidden_states_rtol: float = 0.05

    # Context parallel test tolerances
    cp_loss_atol: float = 0.1
    cp_loss_rtol: float = 0.05
    cp_logits_atol: float = 1.0
    cp_logits_rtol: float = 0.1
    cp_gradients_atol: float = 0.1
    cp_gradients_rtol: float = 0.1

    # FP8 test tolerances
    fp8_loss_atol: float = 0.1
    fp8_loss_rtol: float = 0.05
    fp8_logits_atol: float = 5.0
    fp8_logits_rtol: float = 0.1

    # Meta device initialization tolerances
    init_mean_atol: float = 1e-3
    init_mean_rtol: float = 1e-4
    init_std_atol: float = 1e-3
    init_std_rtol: float = 1e-4


class BaseModelTest(ABC):
    """Abstract base class for testing BioNeMo models.

    This class provides common test utilities and defines the interface that
    model-specific testers must implement. It follows the pattern used in
    HuggingFace transformers for model testing.

    Subclasses must implement all abstract methods to provide model-specific
    configuration, data preparation, and conversion functions.

    Example:
        ```python
        class ESM2ModelTester(BioNeMoModelTester):
            def get_model_class(self):
                return NVEsmForMaskedLM

            def get_config_class(self):
                return NVEsmConfig

            def get_upstream_model_id(self):
                return "facebook/esm2_t6_8M_UR50D"

            # ... implement other abstract methods
        ```
    """

    @abstractmethod
    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the TransformerEngine model class to test.

        Returns:
            The model class (e.g., NVEsmForMaskedLM, NVLlamaForCausalLM).
        """
        pass

    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the tokenizer for the model.

        Returns:
            The tokenizer (e.g., AutoTokenizer).
        """
        pass

    @abstractmethod
    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the config class for the model.

        Returns:
            The config class (e.g., NVEsmConfig, NVLlamaConfig).
        """
        pass

    @abstractmethod
    def get_upstream_model_id(self) -> str:
        """Return the HuggingFace model ID for the reference model.

        Returns:
            Model ID string (e.g., "facebook/esm2_t6_8M_UR50D").
        """
        pass

    @abstractmethod
    def get_upstream_model_revision(self) -> str:
        """Return the specific revision/commit hash for the upstream model.

        Returns:
            Revision string or 'main' for latest.
        """
        pass

    @abstractmethod
    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the HuggingFace reference model class.

        Returns:
            The HF model class (e.g., AutoModelForMaskedLM, AutoModelForCausalLM).
        """
        pass

    @abstractmethod
    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers in the model.

        Args:
            model: The model instance.

        Returns:
            List of transformer layer modules.

        Example:
            For ESM2: model.esm.encoder.layers
            For LLaMA3: model.model.layers
        """
        pass

    @abstractmethod
    def get_test_input_data(
        self,
        format: Literal["bshd", "thd"] = "bshd",
        pad_to_multiple_of: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data for the model.

        Args:
            format: Whether to use sequence packing (THD) or bshd format.

        Returns:
            Dictionary of input tensors (input_ids, attention_mask, etc.).
        """
        pass

    @abstractmethod
    def get_hf_to_te_converter(self) -> Callable:
        """Return the function that converts HF model to TE model.

        Returns:
            Conversion function with signature: (hf_model, **kwargs) -> te_model
        """
        pass

    @abstractmethod
    def get_te_to_hf_converter(self) -> Callable:
        """Return the function that converts TE model to HF model.

        Returns:
            Conversion function with signature: (te_model, **kwargs) -> hf_model
        """
        pass

    def get_tolerances(self) -> TestTolerances:
        """Return test tolerances for this model.

        Override this method to provide model-specific tolerances.

        Returns:
            TestTolerances instance with appropriate values.
        """
        return TestTolerances()

    def get_attn_input_formats(self) -> List[str]:
        """Return supported attention input formats.

        Returns:
            List of format strings (e.g., ["bshd", "thd"]).
        """
        return ["bshd"]

    def verify_model_parameters_initialized_correctly(
        self,
        model: PreTrainedModel,
        atol: float | None = None,
        rtol: float | None = None,
        should_be_fp8: bool = False,
    ) -> None:
        """Verify that model parameters are initialized correctly.

        This can be overridden for models that use non-standard weight initialization.

        This checks that:
        1. All parameters are on CUDA device
        2. Embeddings have correct mean and std
        3. Linear layers have correct weight/bias initialization
        4. LayerNorm parameters are initialized correctly
        5. FP8 quantization is applied if requested

        Args:
            model: The model to verify.
            atol: Absolute tolerance for comparisons (uses default if None).
            rtol: Relative tolerance for comparisons (uses default if None).
            should_be_fp8: Whether to expect FP8 quantized weights.
        """
        config = model.config
        tolerances = self.get_tolerances()

        if atol is None:
            atol = tolerances.init_mean_atol
        if rtol is None:
            rtol = tolerances.init_mean_rtol

        # Verify all parameters are on CUDA
        for name, parameter in model.named_parameters():
            assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

        # Verify initialization for each module type
        for name, module in model.named_modules():

            def msg(x):
                return f"Mismatch in module {name}: {x}"

            if isinstance(module, torch.nn.Embedding):
                torch.testing.assert_close(module.weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
                torch.testing.assert_close(
                    module.weight.std().item(),
                    config.initializer_range,
                    atol=tolerances.init_std_atol,
                    rtol=tolerances.init_std_rtol,
                    msg=msg,
                )

            elif isinstance(module, transformer_engine.pytorch.Linear):
                torch.testing.assert_close(module.weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
                torch.testing.assert_close(
                    module.weight.std().item(),
                    config.initializer_range,
                    atol=tolerances.init_std_atol,
                    rtol=tolerances.init_std_rtol,
                    msg=msg,
                )
                if module.bias is not None:
                    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)

                if should_be_fp8:
                    if f"{name}.weight" in set(model._tied_weights_keys):
                        continue  # Skip tied weights
                    elif hasattr(model, "_do_not_quantize") and name in model._do_not_quantize:
                        continue  # Skip weights that should be kept in bf16
                    assert isinstance(module.weight, QuantizedTensor), f"Module {name} weight is not a Float8Tensor"

            elif isinstance(module, transformer_engine.pytorch.LayerNorm):
                torch.testing.assert_close(module.weight, torch.ones_like(module.weight), msg=msg)
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)

            elif isinstance(module, torch.nn.LayerNorm):
                torch.testing.assert_close(module.weight, torch.ones_like(module.weight), msg=msg)
                if module.bias is not None:
                    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        """Create a test configuration with optional overrides.

        Args:
            **kwargs: Configuration parameters to override.

        Returns:
            Configuration instance.
        """
        config_class = self.get_config_class()
        upstream_id = self.get_upstream_model_id()
        revision = self.get_upstream_model_revision()
        return config_class.from_pretrained(upstream_id, revision=revision, **kwargs)

    def get_reference_model(
        self,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ) -> PreTrainedModel:
        """Load the reference HuggingFace model.

        Args:
            dtype: Data type for the model.
            device: Device to load model on.
            attn_implementation: Attention implementation to use.

        Returns:
            The loaded reference model.
        """
        upstream_class = self.get_upstream_model_class()
        upstream_id = self.get_upstream_model_id()
        revision = self.get_upstream_model_revision()

        kwargs = {
            "dtype": dtype,
            "attn_implementation": attn_implementation,
        }
        if revision is not None:
            kwargs["revision"] = revision

        model = upstream_class.from_pretrained(upstream_id, **kwargs)
        model.to("cuda")
        return model

    def get_reference_model_no_weights(self) -> PreTrainedModel:
        """Load the reference HuggingFace model with random weights."""
        return self.get_upstream_model_class()(
            AutoConfig.from_pretrained(
                self.get_upstream_model_id(),
                dtype=torch.float32,
                revision=self.get_upstream_model_revision(),
            )
        )

    def compare_outputs(
        self,
        te_outputs,
        hf_outputs,
        input_data: Dict[str, torch.Tensor],
        compare_loss: bool = True,
        compare_logits: bool = True,
        compare_hidden_states: bool = False,
    ) -> None:
        """Compare outputs from TE and HF models.

        Args:
            te_outputs: Outputs from TransformerEngine model.
            hf_outputs: Outputs from HuggingFace model.
            input_data: Input data dictionary (for attention mask).
            compare_loss: Whether to compare loss values.
            compare_logits: Whether to compare logits.
            compare_hidden_states: Whether to compare hidden states.
        """
        tolerances = self.get_tolerances()

        if compare_loss and hasattr(te_outputs, "loss") and hasattr(hf_outputs, "loss"):
            torch.testing.assert_close(
                te_outputs.loss,
                hf_outputs.loss,
                atol=tolerances.golden_value_loss_atol,
                rtol=tolerances.golden_value_loss_rtol,
                msg=lambda x: f"Loss mismatch between TE and HF models: {x}",
            )

        if compare_logits and hasattr(te_outputs, "logits") and hasattr(hf_outputs, "logits"):
            # Only compare logits where attention mask is True
            if "attention_mask" in input_data:
                mask = input_data["attention_mask"].to(bool)
                torch.testing.assert_close(
                    te_outputs.logits[mask],
                    hf_outputs.logits[mask],
                    atol=tolerances.golden_value_logits_atol,
                    rtol=tolerances.golden_value_logits_rtol,
                    msg=lambda x: f"Logits mismatch between TE and HF models: {x}",
                )
            else:
                torch.testing.assert_close(
                    te_outputs.logits,
                    hf_outputs.logits,
                    atol=tolerances.golden_value_logits_atol,
                    rtol=tolerances.golden_value_logits_rtol,
                    msg=lambda x: f"Logits mismatch between TE and HF models: {x}",
                )

        if compare_hidden_states and hasattr(te_outputs, "hidden_states") and hasattr(hf_outputs, "hidden_states"):
            for i, (te_hidden, hf_hidden) in enumerate(zip(te_outputs.hidden_states, hf_outputs.hidden_states)):
                torch.testing.assert_close(
                    te_hidden,
                    hf_hidden,
                    atol=tolerances.golden_value_hidden_states_atol,
                    rtol=tolerances.golden_value_hidden_states_rtol,
                    msg=lambda x: f"Hidden states mismatch at layer {i}: {x}",
                )

    @pytest.fixture(autouse=True, scope="function")
    def set_seed(self):
        set_seed(42)

    # ==================== Forward and Backward Smoke Tests ====================

    def test_smoke_forward_pass(self, input_format):
        model_class = self.get_model_class()
        config = self.create_test_config(attn_input_format=input_format)

        model = model_class(config)
        model.to(torch.bfloat16)
        model.to("cuda")

        # Prepare input data
        input_data = self.get_test_input_data(input_format)

        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = model(**input_data, output_hidden_states=True)

        # Verify outputs
        assert outputs.logits is not None, "Model should output logits"
        assert outputs.hidden_states is not None, "Model should output hidden states when requested"
        assert len(outputs.hidden_states) == config.num_hidden_layers + 1, (
            f"Expected {config.num_hidden_layers + 1} hidden states, got {len(outputs.hidden_states)}"
        )

    def test_smoke_backward_pass(self, input_format):
        """Smoke test: backward pass."""
        model_class = self.get_model_class()
        config = self.create_test_config(attn_input_format=input_format)

        model = model_class(config)
        model.to(torch.bfloat16)
        model.to("cuda")

        # Prepare input data
        input_data = self.get_test_input_data(input_format)

        # Forward pass
        outputs = model(**input_data, output_hidden_states=True)

        # Backward pass
        outputs.logits.mean().backward()

        # Verify all parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "All trainable parameters should have gradients after backward pass"

    def test_smoke_model_with_loss(self, input_format):
        """Smoke test: model forward pass with labels produces loss."""
        model_class = self.get_model_class()
        config = self.create_test_config(attn_input_format=input_format)

        model = model_class(config)
        model.to(torch.bfloat16)
        model.to("cuda")

        # Prepare input data with labels
        input_data = self.get_test_input_data(input_format)

        # Ensure labels are present
        if "labels" not in input_data:
            input_data["labels"] = input_data["input_ids"].clone()

        # Forward pass
        with torch.no_grad():
            outputs = model(**input_data)

        # Verify loss is computed
        assert outputs.loss is not None, "Model should compute loss when labels are provided"
        assert outputs.loss.item() > 0, "Loss should be positive"

    def test_forward_and_backward(self, input_format):
        """Test that model can perform forward and backward passes."""
        model_class = self.get_model_class()
        config = self.create_test_config(attn_input_format=input_format)

        model = model_class(config)
        model.to(torch.bfloat16)
        model.to("cuda")

        # Prepare input data
        input_data = self.get_test_input_data(input_format)

        # Add labels for loss computation
        if "labels" not in input_data:
            input_data["labels"] = input_data["input_ids"].clone()

        # Forward pass
        outputs = model(**input_data)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    # ==================== Conversion Tests ====================

    def test_convert_hf_to_te(self):
        """Test that HF model can be converted to TE format."""
        # Load reference HF model
        model_hf_original = self.get_reference_model_no_weights()
        # Convert to TE
        convert_fn = self.get_hf_to_te_converter()
        model_te = convert_fn(model_hf_original)

        # Verify model structure
        assert model_te is not None
        assert isinstance(model_te, self.get_model_class())

    def test_convert_te_to_hf(self):
        """Test that TE model can be converted back to HF format."""
        # Load reference HF model
        model_hf_original = self.get_reference_model_no_weights()
        # Convert to TE
        hf_to_te_fn = self.get_hf_to_te_converter()
        model_te = hf_to_te_fn(model_hf_original)

        # Convert back to HF
        te_to_hf_fn = self.get_te_to_hf_converter()
        model_hf_converted = te_to_hf_fn(model_te)

        # Verify model structure
        assert model_hf_converted is not None
        assert isinstance(model_hf_converted, self.get_upstream_model_class())

    def test_convert_te_to_hf_roundtrip(self):
        """Test that HF → TE → HF conversion preserves weights."""
        # Load reference HF model
        model_hf_original = self.get_reference_model_no_weights()
        original_state_dict = model_hf_original.state_dict()

        # Convert to TE and back
        hf_to_te_fn = self.get_hf_to_te_converter()
        te_to_hf_fn = self.get_te_to_hf_converter()

        model_te = hf_to_te_fn(model_hf_original)
        model_hf_converted = te_to_hf_fn(model_te)
        converted_state_dict = model_hf_converted.state_dict()

        # Compare state dicts
        assert set(original_state_dict.keys()) == set(converted_state_dict.keys()), "State dict keys don't match"

        for key in original_state_dict.keys():
            original_param = original_state_dict[key]
            converted_param = converted_state_dict[key]

            # Convert both to the same dtype for comparison (use the original dtype)
            if original_param.dtype != converted_param.dtype:
                converted_param = converted_param.to(original_param.dtype)

            torch.testing.assert_close(
                original_param,
                converted_param,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch in parameter {key} after roundtrip conversion",
            )

    def test_convert_config(self):
        """Test that config can be converted between HF and TE formats."""
        upstream_id = self.get_upstream_model_id()
        revision = self.get_upstream_model_revision()

        # Load HF config
        from transformers import AutoConfig

        kwargs = {}
        if revision is not None:
            kwargs["revision"] = revision
        hf_config = AutoConfig.from_pretrained(upstream_id, **kwargs)

        # Get TE config class
        te_config_class = self.get_config_class()

        # Convert to TE config
        te_config = te_config_class(**hf_config.to_dict())

        # Verify key attributes match
        assert te_config.hidden_size == hf_config.hidden_size
        assert te_config.num_hidden_layers == hf_config.num_hidden_layers
        assert te_config.num_attention_heads == hf_config.num_attention_heads

    @pytest.fixture(scope="class", autouse=True)
    def _set_tmpdir(self, tmp_path_factory):
        """Make sure we can see the saved te checkpoint as a class-scoped fixture."""
        # set on the class, visible as self._tmp_dir
        type(self)._tmp_dir = tmp_path_factory.mktemp(self.__class__.__name__)

    def get_converted_te_model_checkpoint(self) -> Path:
        """Get the path to the converted TE model checkpoint."""
        model_hf = self.get_reference_model()
        convert_fn = self.get_hf_to_te_converter()
        model_te = convert_fn(model_hf)
        model_te.to("cuda")
        checkpoint_path = self._tmp_dir / "converted_te_model"
        model_te.save_pretrained(checkpoint_path)
        return checkpoint_path

    def get_converted_te_model(self, **kwargs) -> PreTrainedModel:
        """Get the converted TE model.

        This shouldn't get called before the checkpoint tests are run in case they're broken.
        """
        checkpoint_path = self.get_converted_te_model_checkpoint()
        model_te = self.get_model_class().from_pretrained(checkpoint_path, **kwargs)
        model_te.to("cuda")
        return model_te

    # ==================== Golden Value Tests ====================

    def test_golden_values(self):
        """Test that TE model outputs match HF reference model."""
        model_hf = self.get_reference_model()
        model_te = self.get_converted_te_model()

        model_hf.eval()
        model_te.eval()

        # Prepare input data
        input_data = self.get_test_input_data("bshd")

        # Run forward pass
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            te_outputs = model_te(**input_data)
            hf_outputs = model_hf(**input_data)

        # Compare outputs
        self.compare_outputs(
            te_outputs,
            hf_outputs,
            input_data,
            compare_loss=True,
            compare_logits=True,
            compare_hidden_states=False,
        )

    def test_golden_values_thd(self, te_attn_backend):
        """Test the model outputs the same results with THD and BSHD input formats."""

        if te_attn_backend == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
            pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")
        elif te_attn_backend == "fused_attn" and torch.cuda.get_device_capability()[0] == 12:
            pytest.xfail("BIONEMO-2840: On sm120, the THD implementation is not available for fused attn.")

        input_data_bshd = self.get_test_input_data(format="bshd")
        input_data_thd = self.get_test_input_data(format="thd")
        tolerances = self.get_tolerances()

        torch.testing.assert_close(
            input_data_bshd["input_ids"][input_data_bshd["attention_mask"].to(bool)],
            input_data_thd["input_ids"].flatten(0),
        )

        # The THD labels will have some extra -100 items due to the separator token, so we need to filter them out.
        labels_bshd = input_data_bshd["labels"][input_data_bshd["attention_mask"].to(bool)]
        labels_thd = input_data_thd["labels"].flatten(0)
        torch.testing.assert_close(labels_bshd[labels_thd != -100], labels_thd[labels_thd != -100])

        model_bshd = self.get_converted_te_model(attn_input_format="bshd", dtype=torch.bfloat16)
        model_thd = self.get_converted_te_model(attn_input_format="thd", dtype=torch.bfloat16)

        model_bshd.eval()
        model_thd.eval()

        with torch.inference_mode():
            outputs_bshd = model_bshd(**input_data_bshd)
            outputs_thd = model_thd(**input_data_thd)

        # Compare logits
        bshd_logits = outputs_bshd.logits[input_data_bshd["attention_mask"].to(bool)]
        torch.testing.assert_close(
            bshd_logits,
            outputs_thd.logits,
            atol=tolerances.golden_value_logits_atol,
            rtol=tolerances.golden_value_logits_rtol,
        )

        # Compare losses
        torch.testing.assert_close(
            outputs_bshd.loss,
            outputs_thd.loss,
            atol=tolerances.golden_value_loss_atol,
            rtol=tolerances.golden_value_loss_rtol,
        )

    def test_thd_padding_input_data_equivalence(self):
        """Test that the THD input data is the same before and after padding."""

        input_data_thd = self.get_test_input_data(format="thd")
        input_data_thd_padded = self.get_test_input_data(format="thd", pad_to_multiple_of=32)

        cu_seq_lens_q = input_data_thd["cu_seq_lens_q"]
        cu_seq_lens_q_padded = input_data_thd_padded["cu_seq_lens_q_padded"]
        cu_num_pads = cu_seq_lens_q_padded - cu_seq_lens_q
        seq_lengths_real = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]

        num_real_tokens = cu_seq_lens_q[-1]

        # How much we need to shift each sequence by.
        offsets = torch.repeat_interleave(cu_num_pads[:-1], seq_lengths_real, dim=0)

        # The indices of the real tokens as appears in the padded logits.
        real_idx = torch.arange(0, num_real_tokens, device="cuda") + offsets

        torch.testing.assert_close(
            input_data_thd["input_ids"],
            input_data_thd_padded["input_ids"].index_select(1, real_idx),
        )

        torch.testing.assert_close(
            input_data_thd["labels"],
            input_data_thd_padded["labels"].index_select(1, real_idx),
        )
        assert input_data_thd_padded["pad_between_seqs"] is True

    @pytest.mark.xfail(
        condition=not HAS_DATA_CENTER_GPU,
        reason="Padded THD sequences are not supported on non-datacenter hardware.",
    )
    def test_golden_values_thd_padded(self):
        """Test that the model outputs the same results with padded input data."""

        input_data_thd = self.get_test_input_data(format="thd")
        input_data_thd_padded = self.get_test_input_data(format="thd", pad_to_multiple_of=32)
        tolerances = self.get_tolerances()

        model_thd = self.get_converted_te_model(attn_input_format="thd", dtype=torch.bfloat16)
        model_thd.eval()

        with torch.inference_mode():
            outputs_thd = model_thd(**input_data_thd)
            outputs_thd_padded = model_thd(**input_data_thd_padded)

        cu_seq_lens_q = input_data_thd["cu_seq_lens_q"]
        cu_seq_lens_q_padded = input_data_thd_padded["cu_seq_lens_q_padded"]
        cu_num_pads = cu_seq_lens_q_padded - cu_seq_lens_q
        seq_lengths_real = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
        num_real_tokens = cu_seq_lens_q[-1]
        offsets = torch.repeat_interleave(cu_num_pads[:-1], seq_lengths_real, dim=0)

        # The indices of the real tokens as appears in the padded logits.
        real_idx = torch.arange(0, num_real_tokens, device="cuda") + offsets
        logits_unpadded = outputs_thd_padded.logits.index_select(0, real_idx.cuda())

        torch.testing.assert_close(
            outputs_thd.logits,
            logits_unpadded,
            atol=tolerances.golden_value_logits_atol,
            rtol=tolerances.golden_value_logits_rtol,
        )

        torch.testing.assert_close(
            outputs_thd.loss,
            outputs_thd_padded.loss,
            atol=tolerances.golden_value_loss_atol,
            rtol=tolerances.golden_value_loss_rtol,
        )

    # ==================== FP8 Tests ====================
    def test_fp8_forward_and_backward_pass(self, fp8_recipe, input_format):
        """Test that model works with FP8 autocast."""
        if input_format == "thd" and not HAS_DATA_CENTER_GPU:
            pytest.xfail("Padded sequences are not supported on non-datacenter hardware for THD.")

        model_class = self.get_model_class()
        config = self.create_test_config(
            dtype=torch.bfloat16, attn_input_format=input_format, self_attn_mask_type="padding_causal"
        )

        model = model_class(config)
        model.to("cuda")
        model.eval()

        # Prepare input data
        input_data = self.get_test_input_data(input_format, pad_to_multiple_of=32)

        # Run without FP8
        with torch.no_grad():
            outputs = model(**input_data)
            loss_bf16 = outputs.loss

        # Run with FP8
        with transformer_engine.pytorch.autocast(recipe=fp8_recipe):
            outputs_fp8 = model(**input_data)
            loss_fp8 = outputs_fp8.loss

        assert torch.isfinite(loss_fp8)

        # Backward pass
        loss_fp8.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient after FP8 backward pass"

        # Compare losses (should be close but not identical due to quantization)
        tolerances = self.get_tolerances()
        torch.testing.assert_close(
            loss_fp8,
            loss_bf16,
            atol=tolerances.fp8_loss_atol,
            rtol=tolerances.fp8_loss_rtol,
            msg=lambda x: f"FP8 loss differs too much from BF16 loss: {x}",
        )

    def test_quantized_model_init_forward_and_backward(self, fp8_recipe, input_format):
        """Test that model initialized with FP8 works correctly."""
        if input_format == "thd" and not HAS_DATA_CENTER_GPU:
            pytest.xfail("Padded sequences are not supported on non-datacenter hardware for THD.")

        model_class = self.get_model_class()
        config = self.create_test_config(attn_input_format=input_format, self_attn_mask_type="padding_causal")

        # Initialize with FP8
        with transformer_engine.pytorch.quantized_model_init(recipe=fp8_recipe):
            model = model_class(config)

        model.to("cuda")
        model.eval()

        # Prepare input data
        input_data = self.get_test_input_data(input_format, pad_to_multiple_of=32)
        if "labels" not in input_data:
            input_data["labels"] = input_data["input_ids"].clone()

        # Forward and backward pass with FP8
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with transformer_engine.pytorch.autocast(recipe=fp8_recipe):
                outputs = model(**input_data)

        loss = outputs.loss
        assert torch.isfinite(loss)

        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient after FP8 backward pass"

    # ==================== Meta Device Initialization Tests ====================

    def test_cuda_init(self):
        """Test that model can be initialized directly on CUDA device."""
        model_class = self.get_model_class()
        config = self.create_test_config()

        model = model_class(config)
        model.to("cuda")

        self.verify_model_parameters_initialized_correctly(model)

    def test_meta_init(self):
        """Test that model can be initialized on meta device and moved to CUDA."""
        model_class = self.get_model_class()
        config = self.create_test_config()

        # Initialize on meta device
        with torch.device("meta"):
            model = model_class(config)

        # Assert parameters are actually on the meta device
        for name, parameter in model.named_parameters():
            assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

        # Move to CUDA (this will materialize the parameters)
        model.init_empty_weights()
        self.verify_model_parameters_initialized_correctly(model)

    def test_cuda_fp8_init(self, fp8_recipe):
        """Test that model can be initialized on CUDA with FP8."""
        model_class = self.get_model_class()
        config = self.create_test_config()

        with transformer_engine.pytorch.quantized_model_init(recipe=fp8_recipe):
            model = model_class(config)

        model.to("cuda")

        self.verify_model_parameters_initialized_correctly(model, should_be_fp8=True)

    def test_meta_fp8_init(self, fp8_recipe):
        """Test that model can be initialized on meta device with FP8 and moved to CUDA."""
        model_class = self.get_model_class()
        config = self.create_test_config()

        # Initialize on meta device with FP8
        with torch.device("meta"):
            with transformer_engine.pytorch.quantized_model_init(recipe=fp8_recipe):
                model = model_class(config)

        # Assert parameters are actually on the meta device
        for name, parameter in model.named_parameters():
            assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

        # Move to CUDA
        model.init_empty_weights()
        self.verify_model_parameters_initialized_correctly(model, should_be_fp8=True)

    # TODO: add multi-GPU tests, e.g., meta-device init after fully_shard, cp tests, etc.
