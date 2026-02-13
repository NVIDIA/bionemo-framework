# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

r"""Text generation (inference) workflow for Evo2 using Megatron Core.

This module provides autoregressive text generation for Evo2 models using the
MCore inference infrastructure (StaticInferenceEngine, TextGenerationController).

Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/examples/inference/gpt/gpt_static_inference.py

Usage (CLI):
    torchrun --nproc_per_node 1 -m bionemo.evo2.run.infer \
        --ckpt-dir /path/to/mbridge/checkpoint \
        --prompt "|d__Bacteria;p__Pseudomonadota|" \
        --max-new-tokens 100

Usage (Python API):
    from bionemo.evo2.run.infer import setup_inference_engine, generate

    # Setup engine (loads model, creates inference components)
    engine, tokenizer = setup_inference_engine(ckpt_dir)

    # Generate text
    results = generate(engine, prompts=["ATCGATCG"], max_new_tokens=100)
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.static_engine import StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH
from bionemo.evo2.models.evo2_provider import HyenaInferenceContext
from bionemo.evo2.run.predict import initialize_inference_distributed, resolve_checkpoint_path


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# Evo2 Model Inference Wrapper
# =============================================================================


class Evo2ModelInferenceWrapper(AbstractModelInferenceWrapper):
    """Inference wrapper for Evo2 models.

    Extends the abstract wrapper to provide Evo2-specific input preparation
    and forward pass handling for autoregressive text generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inference_wrapper_config: InferenceWrapperConfig,
        inference_context: Optional[StaticInferenceContext] = None,
    ):
        """Initialize the Evo2 inference wrapper.

        Args:
            model: The Evo2 model to wrap for inference.
            inference_wrapper_config: Configuration with hidden size, vocab size, etc.
            inference_context: Context for managing state and sequence offsets.
        """
        super().__init__(model, inference_wrapper_config, inference_context)

    def prep_inference_input(self, prompts_tokens: torch.Tensor) -> Dict[str, Any]:
        """Prepare the inference input data.

        Args:
            prompts_tokens: A tensor of shape [batch_size, max_seq_len]

        Returns:
            Dict with tokens, attention_mask, and position_ids
        """
        batch_size, seq_len = prompts_tokens.shape
        device = prompts_tokens.device

        # For Evo2/Hyena models, position_ids are sequential
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Evo2 uses causal attention - for flash attention backend, mask is None
        attention_mask = None

        return {
            "tokens": prompts_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Extract batch for a specific context window.

        Called iteratively during autoregressive generation.

        Args:
            inference_input: Full inference input dict
            context_start_position: Start of context window
            context_end_position: End of context window

        Returns:
            Dict with sliced tokens, positions, and attention mask
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        tokens2use = tokens[:, context_start_position:context_end_position]
        positions2use = position_ids[:, context_start_position:context_end_position]

        if attention_mask is not None:
            attention_mask2use = attention_mask[
                ..., context_start_position:context_end_position, :context_end_position
            ]
        else:
            attention_mask2use = None

        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "attention_mask": attention_mask2use,
        }

    def _forward(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Run a forward pass of the model.

        Override to pass HyenaInferenceContext properly.

        Args:
            inference_input: The input data dict.

        Returns:
            The model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        return self.model(
            tokens,
            position_ids,
            attention_mask,
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )


# =============================================================================
# Inference Components Container
# =============================================================================


@dataclass
class Evo2InferenceComponents:
    """Container for Evo2 inference components.

    This dataclass holds all the components needed for text generation,
    making it easy to pass around and reuse.
    """

    inference_engine: StaticInferenceEngine
    tokenizer: _HuggingFaceTokenizer
    inference_wrapper: Evo2ModelInferenceWrapper
    inference_context: HyenaInferenceContext
    model: torch.nn.Module


# =============================================================================
# Public API: Setup and Generate Functions
# =============================================================================


def setup_inference_engine(
    ckpt_dir: Path,
    *,
    max_seq_length: int = 8192,
    max_batch_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    mixed_precision_recipe: Optional[str] = None,
    random_seed: int = 1234,
) -> Evo2InferenceComponents:
    """Setup the Evo2 inference engine and related components.

    This function loads the model, creates the inference wrapper, and sets up
    all necessary components for text generation.

    Args:
        ckpt_dir: Path to MBridge checkpoint directory.
        max_seq_length: Maximum sequence length for generation.
        max_batch_size: Maximum batch size for inference.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree (must be 1).
        context_parallel_size: Context parallelism degree.
        mixed_precision_recipe: Override mixed precision recipe.
        random_seed: Random seed for reproducibility.

    Returns:
        Evo2InferenceComponents containing all inference components.

    Example:
        >>> components = setup_inference_engine(Path("/path/to/checkpoint"), max_batch_size=4)
        >>> results = generate(components, prompts=["ATCG", "GCTA"], max_new_tokens=100)
    """
    if pipeline_model_parallel_size != 1:
        raise ValueError("Pipeline parallelism > 1 is not supported for inference.")

    # -------------------------------------------------------------------------
    # Step 1: Load configuration from checkpoint
    # -------------------------------------------------------------------------
    resolved_ckpt_dir = resolve_checkpoint_path(ckpt_dir)
    logger.info(f"Loading configuration from checkpoint: {resolved_ckpt_dir}")

    run_config_filename = get_checkpoint_run_config_filename(str(resolved_ckpt_dir))
    if not file_exists(run_config_filename):
        raise FileNotFoundError(f"run_config.yaml not found at {run_config_filename}")

    run_config = read_run_config(run_config_filename)
    model_provider = instantiate(run_config["model"])
    logger.info(f"Instantiated model provider: {type(model_provider).__name__}")

    # -------------------------------------------------------------------------
    # Step 2: Configure parallelism and precision
    # -------------------------------------------------------------------------
    model_provider.tensor_model_parallel_size = tensor_parallel_size
    model_provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_provider.context_parallel_size = context_parallel_size
    # Disable sequence parallelism for inference - Megatron's inference engine
    # does not support it for non-MoE models.
    model_provider.sequence_parallel = False

    # Enable flash decode for inference
    model_provider.flash_decode = True

    # Use bf16_mixed for inference to avoid FP8 issues
    if mixed_precision_recipe is not None:
        mp_config = get_mixed_precision_config(mixed_precision_recipe)
    else:
        mp_config = get_mixed_precision_config("bf16_mixed")

    mp_config.finalize()
    mp_config.setup(model_provider)

    # -------------------------------------------------------------------------
    # Step 3: Load tokenizer
    # -------------------------------------------------------------------------
    tokenizer_dir = resolved_ckpt_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = _HuggingFaceTokenizer(tokenizer_dir)
    else:
        tokenizer = _HuggingFaceTokenizer(DEFAULT_HF_TOKENIZER_MODEL_PATH)

    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # -------------------------------------------------------------------------
    # Step 4: Initialize distributed environment
    # -------------------------------------------------------------------------
    rng_config = instantiate(run_config.get("rng")) if run_config.get("rng") else RNGConfig(seed=random_seed)
    dist_config = instantiate(run_config.get("dist")) if run_config.get("dist") else DistributedInitConfig()

    from megatron.bridge.utils.common_utils import get_world_size_safe

    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    world_size = get_world_size_safe()
    data_parallel_size = world_size // model_parallel_size

    initialize_inference_distributed(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        micro_batch_size=max_batch_size,
        global_batch_size=max_batch_size * data_parallel_size,
        rng_config=rng_config,
        dist_config=dist_config,
    )
    logger.info("Initialized distributed environment")

    # -------------------------------------------------------------------------
    # Step 5: Create model and load weights
    # -------------------------------------------------------------------------
    logger.info("Creating model...")
    model_provider.finalize()

    raw_model = model_provider.provide(pre_process=True, post_process=True).eval().cuda()

    logger.info(f"Loading weights from: {resolved_ckpt_dir}")
    _load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved_ckpt_dir),
        model=[raw_model],
        dist_ckpt_strictness="ignore_all",
    )
    logger.info("Weights loaded successfully")

    # Wrap with Float16Module
    model = Float16Module(model_provider, raw_model)

    # -------------------------------------------------------------------------
    # Step 6: Setup MCore inference infrastructure
    # -------------------------------------------------------------------------
    # Create inference wrapper config
    model_config = get_model_config(raw_model)
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=model_config.hidden_size,
        inference_max_requests=max_batch_size,
        inference_max_seq_length=max_seq_length,
        inference_batch_times_seqlen_threshold=max_seq_length * max_batch_size,
        params_dtype=torch.bfloat16,
        padded_vocab_size=tokenizer.vocab_size,
    )

    # Create Hyena-specific inference context
    inference_context = HyenaInferenceContext(
        max_batch_size=max_batch_size,
        max_sequence_length=max_seq_length,
    )
    # Don't materialize only last token - we need full logits for sampling
    inference_context.materialize_only_last_token_logits = False

    # Create the inference wrapper
    inference_wrapper = Evo2ModelInferenceWrapper(
        model=model,
        inference_wrapper_config=inference_wrapper_config,
        inference_context=inference_context,
    )

    # Create the text generation controller
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapper,
        tokenizer=tokenizer,
    )

    # Create the static inference engine (using legacy mode for simplicity)
    inference_engine = StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=max_batch_size,
        random_seed=random_seed,
        legacy=True,  # Use legacy static engine
    )

    return Evo2InferenceComponents(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        inference_wrapper=inference_wrapper,
        inference_context=inference_context,
        model=model,
    )


def generate(
    components: Evo2InferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    return_log_probs: bool = False,
) -> List[InferenceRequest]:
    """Generate text using the Evo2 inference engine.

    Args:
        components: Inference components from setup_inference_engine.
        prompts: List of prompt strings to generate from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (0 = disabled, 1 = greedy).
        top_p: Nucleus sampling parameter (0 = disabled).
        return_log_probs: Whether to return log probabilities.

    Returns:
        List of InferenceRequest objects containing generated text and metadata.

    Example:
        >>> components = setup_inference_engine(ckpt_dir)
        >>> results = generate(components, ["ATCGATCG"], max_new_tokens=50, top_k=1)
        >>> print(results[0].generated_text)
    """
    # Reset inference context before generation
    components.inference_context.reset()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=max(0, top_k),
        top_p=top_p if top_p > 0 else 0.0,
        num_tokens_to_generate=max_new_tokens,
        return_log_probs=return_log_probs,
    )

    results = components.inference_engine.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Reset context after generation
    components.inference_context.reset()

    return results


# =============================================================================
# CLI: Full Inference Workflow
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Evo2 inference.

    Returns:
        Parsed arguments namespace
    """
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )

    ap = argparse.ArgumentParser(
        description="Generate text with Evo2 models using MCore inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Path to MBridge checkpoint directory",
    )

    # Generation arguments
    ap.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt text for generation",
    )
    ap.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read prompt from a text file (overrides --prompt). Useful for long prompts that exceed shell argument limits.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    ap.add_argument("--top-p", type=float, default=0.0, help="Top-p nucleus sampling (0 = disabled)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    # Parallelism arguments
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism")
    ap.add_argument("--pipeline-model-parallel-size", type=int, choices=[1], default=1, help="Pipeline parallelism")
    ap.add_argument("--context-parallel-size", type=int, default=1, help="Context parallelism")

    # Output arguments
    ap.add_argument("--output-file", type=Path, default=None, help="Save generated text to file")

    # Precision arguments
    ap.add_argument("--mixed-precision-recipe", type=str, default=None, help="Override precision recipe")

    # Model arguments
    ap.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")

    return ap.parse_args()


def infer(
    prompt: str,
    ckpt_dir: Path,
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: Optional[int] = None,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    output_file: Optional[Path] = None,
    mixed_precision_recipe: Optional[str] = None,
    max_seq_length: int = 8192,
) -> str:
    """Run autoregressive text generation with Evo2 using MCore inference.

    This is the main CLI entry point that sets up everything and runs inference.
    For programmatic usage, prefer setup_inference_engine + generate.

    Args:
        prompt: Input text prompt for generation.
        ckpt_dir: Path to MBridge checkpoint directory.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (0 = disabled).
        top_p: Nucleus sampling parameter (0 = disabled).
        seed: Random seed for reproducibility.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree (must be 1).
        context_parallel_size: Context parallelism degree.
        output_file: Optional path to save generated text.
        mixed_precision_recipe: Override mixed precision recipe.
        max_seq_length: Maximum sequence length.

    Returns:
        The generated text string.
    """
    random_seed = seed or 1234

    # Setup inference components
    components = setup_inference_engine(
        ckpt_dir=ckpt_dir,
        max_seq_length=max_seq_length,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        mixed_precision_recipe=mixed_precision_recipe,
        random_seed=random_seed,
    )

    logger.info(f"Generating from prompt: {prompt[:50]}...")

    # Generate
    results = generate(
        components,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Extract generated text
    generated_text = results[0].generated_text if results else ""

    # Output results
    if parallel_state.get_data_parallel_rank() == 0:
        print(f"\n=== Generated Text ===\n{generated_text}\n", file=sys.stdout)

        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(generated_text)
            logger.info(f"Saved generated text to: {output_file}")

    logger.info("Inference complete!")

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return generated_text


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for Evo2 text generation."""
    args = parse_args()

    # Read prompt from file if specified (overrides --prompt)
    prompt = args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()

    infer(
        prompt=prompt,
        ckpt_dir=args.ckpt_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        output_file=args.output_file,
        mixed_precision_recipe=args.mixed_precision_recipe,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
