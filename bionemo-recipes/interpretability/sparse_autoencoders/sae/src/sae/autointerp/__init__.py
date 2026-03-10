"""
Auto-interpretation pipeline for SAE features using LLMs.

Example usage:
    ```python
    from sae.autointerp import (
        AutoInterpreter,
        FeatureSampler,
        AnthropicClient,
        OpenAIClient,
        NIMClient,
    )

    # Define how to format your data for the prompt
    def format_example(data_item, activation, indices):
        return f"Text: {data_item}"

    # Setup sampler with your activations and data
    sampler = FeatureSampler(
        activations=sae_activations,  # [n_samples, hidden_dim]
        data=raw_data,                 # list of data items
        format_fn=format_example,
    )

    # Setup LLM client (uses env vars for API keys)
    client = AnthropicClient(model="claude-sonnet-4-20250514")

    # Run interpretation
    interpreter = AutoInterpreter(llm_client=client)
    results = interpreter.interpret_features(
        sampler=sampler,
        feature_indices=list(range(100)),  # interpret first 100 features
    )

    # Save results (joinable by feature_idx)
    interpreter.save_results(results, "interpretations.json")
    ```
"""

from .llm import (
    LLMClient,
    LLMResponse,
    AnthropicClient,
    OpenAICompatibleClient,
    OpenAIClient,
    NIMClient,
    NVIDIAInternalClient,
)
from .sampler import (
    FeatureSampler,
    FeatureExamples,
)
from .interpreter import (
    AutoInterpreter,
    FeatureInterpretation,
    DEFAULT_PROMPT_TEMPLATE,
    TOKEN_PROMPT_TEMPLATE,
)

__all__ = [
    # LLM clients
    "LLMClient",
    "LLMResponse",
    "AnthropicClient",
    "OpenAICompatibleClient",
    "OpenAIClient",
    "NIMClient",
    "NVIDIAInternalClient",
    # Sampling
    "FeatureSampler",
    "FeatureExamples",
    # Interpreter
    "AutoInterpreter",
    "FeatureInterpretation",
    "DEFAULT_PROMPT_TEMPLATE",
    "TOKEN_PROMPT_TEMPLATE",
]
