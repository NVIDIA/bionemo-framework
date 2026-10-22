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

"""Custom blocks extend native processing without claiming native modalities."""

from __future__ import annotations

import pickle
from copy import deepcopy

import pytest


class _NativeProcessor:
    tokenizer = object()

    def apply_chat_template(self, conversation, *args, **kwargs):
        return conversation, args, kwargs


def test_only_custom_blocks_are_replaced_and_arguments_pass_through(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/chemistry"))
    from chemistry_example import KERMTProcessor, format_molecule

    image = {"type": "image", "url": "https://example.com/a.png", "detail": "high"}
    text = {"type": "text", "text": "Compare: ", "annotation": "keep"}
    audio = {"type": "audio", "path": "example.wav"}
    messages = [
        {"role": "system", "content": "Keep history."},
        {
            "role": "user",
            "name": "alice",
            "content": [
                text,
                image,
                {"type": "molecule", "atom_count": 1, "directed_bond_count": 0, "bond_endpoints": []},
                audio,
            ],
        },
    ]
    original = deepcopy(messages)
    processor = KERMTProcessor(base_processor=_NativeProcessor())
    result, args, kwargs = processor.apply_chat_template(messages, "named-template", tokenize=True, return_dict=True)
    assert result[1]["content"] == [
        text,
        image,
        {"type": "text", "text": format_molecule(messages[1]["content"][2])},
        audio,
    ]
    assert result[1]["content"][0] is text
    assert result[1]["content"][1] is image
    assert result[1]["content"][3] is audio
    assert result[0] == messages[0] and result[1]["name"] == "alice"
    assert messages == original
    assert args == ("named-template",)
    assert kwargs == dict(tokenize=True, return_dict=True)
    batched, _, _ = processor.apply_chat_template([messages, messages])
    assert batched == [result, result]


def test_real_hf_image_processing_is_preserved(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/chemistry"))
    from chemistry_example import KERMTProcessor, format_molecule

    pytest.importorskip("transformers", minversion="5.12.1")
    import torch
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import CLIPImageProcessor, LlavaProcessor, TokenizersBackend

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "<image>": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    kwargs = dict(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}),
        patch_size=2,
        vision_feature_select_strategy="full",
        num_additional_image_tokens=0,
        chat_template="{% for m in messages %}{{ m.role }}:{% if m.content is string %}{{ m.content }}"
        "{% else %}{% for c in m.content %}{% if c.type == 'image' %}<image>"
        "{% elif c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    )
    base = LlavaProcessor(**kwargs)
    processor = KERMTProcessor(base_processor=base)
    image = Image.new("RGB", (4, 4), (128, 64, 32))
    content = [
        {"type": "text", "text": "Compare "},
        {"type": "image", "image": image},
        {"type": "molecule", "atom_count": 1, "directed_bond_count": 0, "bond_endpoints": []},
    ]
    messages = [{"role": "user", "content": content}]
    expected_messages = [
        {"role": "user", "content": [*content[:2], {"type": "text", "text": format_molecule(content[2])}]}
    ]
    options = dict(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    result = processor.apply_chat_template(messages, **options)
    expected = base.apply_chat_template(expected_messages, **options)
    assert processor.tokenizer is base.tokenizer
    assert processor.model_input_names == base.model_input_names
    assert "pixel_values" in result
    assert set(result) == set(expected)
    for key in result:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)

    seen = []

    def encode_molecules(molecules):
        seen.append(molecules)
        return {"mm_features__chem_molecule": torch.ones(1, 3200)}

    processor.molecule_processor = encode_molecules
    result = processor(text="Describe <image>", images=[image], molecules=["C"], return_tensors="pt")
    expected = base(text="Describe <image>", images=[image], return_tensors="pt")
    for key in expected:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(result["mm_features__chem_molecule"], torch.ones(1, 3200))
    assert seen == [["C"]]
    processor.molecule_processor = lambda molecules: {"input_ids": torch.zeros(1, 1)}
    with pytest.raises(ValueError, match="must not overwrite native"):
        processor(text="Describe", molecules=["C"], return_tensors="pt")


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_demo_processor_loads_through_native_from_pretrained(domain, tmp_path, monkeypatch):
    from pathlib import Path

    pytest.importorskip("transformers", minversion="5.12.1")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import TokenizersBackend

    directory = Path(__file__).resolve().parents[1] / "examples" / domain
    monkeypatch.syspath_prepend(str(directory))
    app = __import__(domain + "_example")
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    processor = app.load_processor(str(tmp_path))
    block = (
        dict(type="dna", sequence="A" * 65)
        if domain == "genetics"
        else dict(type="molecule", atom_count=2, directed_bond_count=2, bond_endpoints=[(0, 1)])
    )
    expected = (
        "<SPECIAL_30><SPECIAL_32><SPECIAL_32><SPECIAL_31>"
        if domain == "genetics"
        else (
            "<SPECIAL_30>MOLECULE\n<SPECIAL_33>\nATOMS\n0 <SPECIAL_32>\n"
            "1 <SPECIAL_32>\nBONDS\n0 1 <SPECIAL_34><SPECIAL_31>"
        )
    )
    messages = [dict(role="user", content=[dict(type="text", text="Before "), block, dict(type="text", text=" after")])]
    assert processor.apply_chat_template(messages, tokenize=False) == "user:Before " + expected + " after"

    # Worker transport reconstructs wrappers before their backing fields exist.
    empty = type(processor).__new__(type(processor))
    for name in ("__setstate__", "base_processor", "tokenizer", "missing"):
        with pytest.raises(AttributeError):
            getattr(empty, name)
    for transported in (deepcopy(processor), pickle.loads(pickle.dumps(processor))):
        assert transported.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
        assert transported.apply_chat_template(messages, tokenize=False) == "user:Before " + expected + " after"
        with pytest.raises(AttributeError):
            _ = transported.missing

    # AutoModel checkpoints save the native processor; YAML restores our wrapper.
    checkpoint = tmp_path / "checkpoint"
    processor.save_pretrained(checkpoint)
    restored = app.load_processor(str(checkpoint))
    assert type(restored.base_processor) is type(processor.base_processor)
    assert restored.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    assert restored.apply_chat_template(messages, tokenize=False) == "user:Before " + expected + " after"


def test_collator_uses_native_text_tokenizer_after_chat_rendering(monkeypatch):
    from pathlib import Path

    import torch
    from test_domain_examples import _Tokenizer

    from nemotron_stitch.automodel.data import FeatureCollator

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/genetics"))
    from genetics_example import Evo2Processor

    class NativeProcessor:
        tokenizer = _Tokenizer()

        def __call__(self, **kwargs):
            raise AssertionError("text collation must use processor.tokenizer, not batched native media preprocessing")

        def apply_chat_template(self, messages, **kwargs):
            normalized = [
                {
                    **message,
                    "content": message["content"]
                    if isinstance(message["content"], str)
                    else "".join(block["text"] for block in message["content"]),
                }
                for message in messages
            ]
            return self.tokenizer.apply_chat_template(normalized, **kwargs)

    processor = Evo2Processor(NativeProcessor())
    batch = FeatureCollator(processor, projector_name="dna", placeholder_token_id=32)(
        [
            {
                "messages": [{"role": "user", "content": [{"type": "dna", "sequence": "A" * 65}]}],
                "prompt": "",
                "target": "two",
                "features": torch.ones(2, 3),
            }
        ]
    )
    assert batch["input_ids"].ndim == 2
    indices = batch["mm_token_indices__dna"]
    assert batch["input_ids"].flatten()[indices].tolist() == [32, 32]
