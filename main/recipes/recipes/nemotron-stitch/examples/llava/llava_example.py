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

"""All application-owned choices for the LLaVA image example.

Adapt this file when replacing images with another modality. It declares the
encoder and data access, row normalization, base-model
registrations, and projected-token vLLM registration. Cache publication,
datasets, collation, training recipes, projector transport, and rewards use
the modality-neutral package implementations. This is the only Python module
an application adapts.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from PIL.Image import Image

import numpy as np

from nemotron_stitch.automodel.model import build_multimodal_host
from nemotron_stitch.automodel.registry import register_models as _register_model
from nemotron_stitch.features.preparation import (
    InvalidRow,
    Partition,
    prepare_features,
)
from nemotron_stitch.prompt import PlaceholderSpec, render_soft_token_prompt
from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

# Encoder, dataset, projector, and prompt sentinels.
ENCODER_REPO_ID = "openai/clip-vit-base-patch32"
ENCODER_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DATASET_REPO_ID = "lmms-lab/LLaVA-OneVision-Data"
DATASET_REVISION = "7ca5e5bf8b2006d5dfa0549756198474f0897f63"

PROJECTOR_NAME = "image"
CLIP_PATCH_TOKENS = 49
MAX_SCAN_ROWS = 512

PLACEHOLDER_SPEC = PlaceholderSpec(
    PROJECTOR_NAME,
    "<SPECIAL_30>",
    "<SPECIAL_32>",
    "<SPECIAL_31>",
)
CONFIG_ALIGNMENT = "diagram_image_to_text(cauldron)"
CONFIG_CLEVR = "CLEVR-Math(MathV360K)"
PARTITIONS = (
    Partition("alignment", "train", CONFIG_ALIGNMENT, 0, 64),
    Partition("alignment", "validation", CONFIG_ALIGNMENT, 64, 80),
    Partition("sft", "train", CONFIG_CLEVR, 0, 64),
    Partition("sft", "validation", CONFIG_CLEVR, 64, 80),
    Partition("grpo", "train", CONFIG_CLEVR, 80, 144),
    Partition("grpo", "validation", CONFIG_CLEVR, 144, 160),
)

_ANSWER_RE = re.compile(r"The answer is (-?\d+)")
_IMAGE_MARKER_RE = re.compile(r"<image>\s*")


def _decode_rgb(image: Image | dict[str, Any] | bytes | bytearray | None) -> Image:
    """Normalize a streamed image payload to RGB."""
    if image is None:
        raise InvalidRow("image is absent")
    if isinstance(image, dict):
        image = image.get("bytes")
        if image is None:
            raise InvalidRow("image payload is absent")
    if isinstance(image, bytes | bytearray):
        import io

        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(bytes(image)))
    convert = getattr(image, "convert", None)
    if convert is None:
        raise InvalidRow(f"image does not decode: {type(image).__name__}")
    return convert("RGB")


def _conversation(row: dict[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list | tuple) or len(conversations) != 2:
        raise ValueError(f"row {row.get('id')!r}: conversations must be exactly one human and one assistant turn")
    human, assistant = conversations
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError(f"row {row.get('id')!r}: conversations must be [human, assistant]")
    question, answer = human.get("value"), assistant.get("value")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"row {row.get('id')!r}: conversation values must be non-empty strings")
    question = _IMAGE_MARKER_RE.sub("", question).strip()
    if not question:
        raise ValueError(f"row {row.get('id')!r}: human turn is empty after the <image> marker")
    if "<image>" in answer:
        raise ValueError(f"row {row.get('id')!r}: assistant turn carries an <image> marker")
    return question, answer.strip()


def normalize_row(
    row: dict[str, Any],
    *,
    configuration: str,
    source_index: int,
    placeholder_spec: PlaceholderSpec = PLACEHOLDER_SPEC,
) -> tuple[dict[str, Any], Image]:
    """Convert one OneVision row to the shared record plus its encoder input."""
    if not isinstance(row, dict):
        raise ValueError(f"row at scan index {source_index} is not a mapping")
    sample_id = row.get("id")
    if not isinstance(sample_id, str):
        raise ValueError(f"row at scan index {source_index}: id must be a string")
    if not sample_id:
        raise InvalidRow(f"row at scan index {source_index}: empty id")
    question, answer = _conversation(row)
    if configuration == CONFIG_CLEVR:
        match = _ANSWER_RE.fullmatch(answer)
        if match is None:
            raise InvalidRow(f"row {sample_id!r}: CLEVR answer is not the strict form: {answer!r}")
        ground_truth = str(int(match.group(1)))
        target = f"<answer>{ground_truth}</answer>"
    else:
        ground_truth = None
        target = answer
    prompt = render_soft_token_prompt(
        f"{placeholder_spec.marker()}\n{question}",
        [placeholder_spec],
        {placeholder_spec.name: CLIP_PATCH_TOKENS},
    )
    return (
        {
            "sample_id": sample_id,
            "configuration": configuration,
            "source_index": source_index,
            "question": question,
            "prompt": prompt,
            "target": target,
            "ground_truth": ground_truth,
        },
        _decode_rgb(row.get("image")),
    )


def open_dataset_stream(configuration: str) -> Iterable[dict[str, Any]]:
    """Open one pinned OneVision configuration without materializing it."""
    from datasets import load_dataset

    return load_dataset(
        DATASET_REPO_ID,
        configuration,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )


def build_encoder(device: str) -> Callable[[Image], np.ndarray]:
    """Build the frozen CLIP encoder used by both preparation and inference."""
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModel

    processor = CLIPImageProcessor.from_pretrained(ENCODER_REPO_ID, revision=ENCODER_REVISION)
    model = CLIPVisionModel.from_pretrained(ENCODER_REPO_ID, revision=ENCODER_REVISION).to(device).eval()
    model.requires_grad_(False)

    def encode(rgb: Image) -> np.ndarray:
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        return hidden[0, 1:].float().cpu().numpy()

    return encode


def load_input(path: str | Path) -> Image:
    """Load one user-supplied image for inference."""
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        return image.convert("RGB")


def prepare_data(
    streams: dict[str, Iterable[dict[str, Any]]],
    encode: Callable[[Image], np.ndarray],
    *,
    cache_root: str | Path,
    manifest_path: str | Path,
    implementation_revision: str,
    max_scan_rows: int = MAX_SCAN_ROWS,
    placeholder_spec: PlaceholderSpec = PLACEHOLDER_SPEC,
) -> dict[str, Any]:
    """Bind this application's metadata and hooks to package preparation."""
    return prepare_features(
        streams,
        encode,
        partial(normalize_row, placeholder_spec=placeholder_spec),
        partitions=PARTITIONS,
        encoder={
            "repo_id": ENCODER_REPO_ID,
            "revision": ENCODER_REVISION,
            "implementation_revision": implementation_revision,
            "layer": -1,
        },
        dataset={"repo_id": DATASET_REPO_ID, "revision": DATASET_REVISION},
        cache_root=cache_root,
        manifest_path=manifest_path,
        max_scan_rows=max_scan_rows,
    )


def prepare_cli(argv: list[str] | None = None) -> int:
    """Prepare the bounded training snapshot in a temporary datasets cache."""
    parser = argparse.ArgumentParser(description=prepare_cli.__doc__)
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--projector-name", default=PLACEHOLDER_SPEC.name)
    parser.add_argument("--start-token", default=PLACEHOLDER_SPEC.start_token)
    parser.add_argument("--placeholder-token", default=PLACEHOLDER_SPEC.placeholder_token)
    parser.add_argument("--end-token", default=PLACEHOLDER_SPEC.end_token)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--max-scan-rows", type=int, default=MAX_SCAN_ROWS)
    args = parser.parse_args(argv)

    streams: dict[str, Iterable[dict[str, Any]]] = {}
    encode: Callable[[Image], np.ndarray] | None = None
    temporary = tempfile.mkdtemp(prefix="llava-example-prepare-")
    try:
        import os

        os.environ["HF_DATASETS_CACHE"] = temporary
        import transformers

        encode = build_encoder(args.device)
        streams = {
            configuration: open_dataset_stream(configuration)
            for configuration in dict.fromkeys(partition.configuration for partition in PARTITIONS)
        }
        output = Path(args.output_dir)
        summary = prepare_data(
            streams,
            encode,
            cache_root=output / "feature-cache",
            manifest_path=output / "manifest.jsonl",
            implementation_revision=f"transformers=={transformers.__version__}",
            max_scan_rows=args.max_scan_rows,
            placeholder_spec=PlaceholderSpec(
                args.projector_name,
                args.start_token,
                args.placeholder_token,
                args.end_token,
            ),
        )
    finally:
        streams.clear()
        encode = None
        gc.collect()
        shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def prepare_dpo_preferences(
    manifest_path: str | Path = "outputs/data/manifest-grpo-train.jsonl",
    output_path: str | Path = "outputs/data/preferences-llava.jsonl",
    *,
    cache_root: str = "outputs/data/feature-cache",
) -> int:
    """Make a small, deterministic preference demonstration from cached CLEVR rows."""
    rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("DPO demonstration manifest is empty")
    preferences = []
    for row in rows:
        match = re.fullmatch(r"<answer>(-?\d+)</answer>", row["target"])
        if match is None:
            raise ValueError(f"DPO demonstration requires a CLEVR numeric answer: {row['sample_id']}")
        wrong = int(match.group(1)) + 1
        preferences.append(
            {
                "sample_id": row["sample_id"],
                "feature_key": row["feature_key"],
                "adapter": PROJECTOR_NAME,
                "encoder_placeholder_token": PLACEHOLDER_SPEC.placeholder_token,
                "encoder_loader": "nemotron_stitch.nemo_rl.transport.load_cached_policy_features",
                "encoder_loader_kwargs": {"cache_root": cache_root},
                "chat_template_kwargs": {"enable_thinking": False},
                "context": [{"role": "user", "content": row["prompt"]}],
                "completions": [
                    {"rank": 0, "completion": [{"role": "assistant", "content": row["target"]}]},
                    {"rank": 1, "completion": [{"role": "assistant", "content": f"<answer>{wrong}</answer>"}]},
                ],
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in preferences))
    return len(preferences)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass(frozen=True)
class ModelSpec:
    """The values Python needs to construct one registered host class."""

    architecture: str
    repo_id: str
    revision: str
    hidden_size: int
    hub_remote_code: bool


NANO_4B = ModelSpec(
    architecture="LlavaExampleNemotronForCausalLM",
    repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    hidden_size=3136,
    hub_remote_code=True,
)
LIGHTNING_30B = ModelSpec(
    architecture="LlavaExampleNemotronLightningForCausalLM",
    repo_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    revision="b3caaabed0263651a17dc1f2d4ce97e794f76c44",
    hidden_size=2688,
    hub_remote_code=False,
)
MODEL_SPECS = (NANO_4B, LIGHTNING_30B)


def _resolve_base_cls(spec: ModelSpec) -> type:
    if not spec.hub_remote_code:
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        return NemotronHForCausalLM
    # AutoModel 1814c6c9 builds MoE defaults for Nano's dense topology.
    # Automodel#2670 fixes this; use the Hub implementation until the nested
    # AutoModel lock includes commit 33052a2b, then delete this fallback.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(
        "modeling_nemotron_h.NemotronHForCausalLM",
        spec.repo_id,
        revision=spec.revision,
    )


@cache
def build_host_cls(spec: ModelSpec) -> type:
    return build_multimodal_host(_resolve_base_cls(spec), architecture=spec.architecture)


def register_models() -> None:
    """Register each decorated host in the current AutoModel worker."""
    for spec in MODEL_SPECS:
        _register_model(spec.architecture, build_host_cls(spec))


PROJECTED_ARCHITECTURE = "LlavaExampleNemotronProjected"
register_vllm = cast(
    Callable[[], None],
    build_mm_plugin(
        modality=PROJECTOR_NAME,
        architecture=PROJECTED_ARCHITECTURE,
        placeholder_text=f"<{PROJECTOR_NAME}>",
        mode="projected",
        base_model_cls="vllm.model_executor.models.nemotron_h.NemotronHForCausalLM",
        base_processing_info_cls="vllm.multimodal.processing.BaseProcessingInfo",
        base_processor_cls="vllm.multimodal.processing.BaseMultiModalProcessor",
        base_dummy_inputs_cls="vllm.multimodal.processing.BaseDummyInputsBuilder",
        sentinels=SentinelAttrs(
            start="mm_encoder_start",
            placeholder="mm_encoder_placeholder",
            end="mm_encoder_end",
        ),
        num_tokens_attr="mm_encoder_max_tokens",
        geometry="variable",
    ),
)

if __name__ == "__main__":
    status = prepare_cli()
    # A bounded streaming read deliberately abandons parquet downloads
    # mid-file. Some pyarrow/huggingface_hub combinations then wait on dead
    # sockets during interpreter teardown, after every artifact is published.
    # Preserve normal exceptions above, but skip those library atexit hooks.
    import os
    import sys

    sys.stdout.flush()
    os._exit(status)
