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

"""Raw JSONL → perception HTTP → cache → collation → a small optimizer step."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemotron_stitch.automodel.data import FeatureCollator, ManifestFeatureDataset, PackedManifestFeatureDataset
from nemotron_stitch.automodel.model import build_multimodal_host
from nemotron_stitch.features.cache import open_feature_cache

ROOT = Path(__file__).resolve().parents[1]


def _preparation(domain):
    spec = importlib.util.spec_from_file_location(f"{domain}_prepare", ROOT / "examples" / domain / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "examples" / domain))
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def endpoint():
    pytest.importorskip("bionemo_perception")
    import xxhash

    requests = []
    state = {"bad_width": False}
    chemistry = {
        "C": (1, 0),
        "CC": (2, 2),
        "CO": (2, 2),
        "CCO": (3, 4),
        "CCN": (3, 4),
        "CC(C)O": (4, 6),
        "CCC": (3, 4),
        "CCCC": (4, 6),
        "c1ccccc1": (6, 12),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), request))
            descriptors, chunks, offset = [], [], 0
            for index, output in enumerate(request["outputs"]):
                layer = output["layer"]
                name = output.get("name", layer)
                if "sequence" in request:
                    window = output["pooling"]["window"]
                    shape = (
                        (len(request["sequence"]) + window - 1) // window,
                        1920 if not state["bad_width"] else 4096,
                    )
                else:
                    atoms, bonds = chemistry[request["smiles"]]
                    shape = {
                        "molecule_fingerprint": (1, 3200),
                        "atom_concat": (atoms, 1600),
                        "bond_from_atom": (bonds, 800),
                        "bond_from_bond": (bonds, 800),
                    }[layer]
                identity = request.get("sequence", request.get("smiles"))
                values = np.full(shape, sum(map(ord, identity)) / 256 + index, dtype="<f4")
                chunk = (values.view("<u4") >> 16).astype("<u2").tobytes()
                descriptors.append(
                    dict(
                        name=name + ".output",
                        output_name=name,
                        resolved_layer=layer,
                        pooling={"kind": "windowed", "window": output["pooling"]["window"]}
                        if "sequence" in request
                        else output.get("pooling", {"kind": "none"}),
                        shape=list(shape),
                        byte_offset=offset,
                        byte_length=len(chunk),
                    )
                )
                chunks.append(chunk)
                offset += len(chunk)
            payload = b"".join(chunks)
            header = json.dumps(
                dict(
                    format_version=1,
                    dtype="bfloat16-le",
                    tensors=descriptors,
                    payload_bytes=len(payload),
                    payload_checksum={"algorithm": "xxh64", "digest": xxhash.xxh64_hexdigest(payload)},
                )
            ).encode()
            aligned = (24 + len(header) + 63) // 64 * 64
            body = b"BNMOFWD1" + struct.pack("<HHIQ", 1, 0, len(header), aligned) + header
            body += bytes(aligned - len(body)) + payload
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_raw_examples_through_http_cache_and_training(domain, endpoint, tmp_path, monkeypatch):
    from bionemo_perception import Evo2Client, KERMTClient

    url, requests, _ = endpoint
    client = (Evo2Client if domain == "genetics" else KERMTClient)(url, token="test-secret")
    result = _preparation(domain).prepare(client, ROOT / "examples" / domain / "data/sanity.jsonl", tmp_path)
    assert result["records"] == 24
    assert len(requests) == 8  # Encode each raw input once across all three stages.
    assert all(path == "/v1/forward" and auth == "Bearer test-secret" for path, auth, _ in requests)
    assert "test-secret" not in (tmp_path / "manifest.header.json").read_text()
    _check_prepared_training(domain, tmp_path, monkeypatch)


def _check_prepared_training(domain, tmp_path, monkeypatch):
    from test_domain_examples import _Tokenizer

    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    tokenizer = (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(base_processor=_Tokenizer())
    monkeypatch.setattr(app, "load_processor", lambda path: tokenizer)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert {row["stage"] for row in manifest} == {"alignment", "sft", "grpo"}
    raw = [json.loads(line) for line in (ROOT / "examples" / domain / "data/sanity.jsonl").read_text().splitlines()]
    by_id = {row["sample_id"]: row for row in raw}
    for row in manifest:
        content = by_id[row["sample_id"].split("-", 1)[1]]["messages"][0]["content"]
        prompt = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True)
        blocks = sum(part["type"] != "text" for part in content)
        assert prompt.count("<SPECIAL_30>") == prompt.count("<SPECIAL_31>") == blocks
        position = 0
        for part in content:
            if part["type"] == "text":
                position = prompt.index(part["text"], position) + len(part["text"])
            else:
                position = prompt.index("<SPECIAL_30>", position) + len("<SPECIAL_30>")
        assert prompt.endswith(content[-1]["text"] + "assistant:")
    cache = open_feature_cache(tmp_path / "feature-cache")
    assert all(np.isfinite(cache.get(row["feature_key"])).all() for row in manifest)
    for stage in ("alignment", "sft", "grpo"):
        assert len((tmp_path / f"manifest-{stage}-train.jsonl").read_text().splitlines()) == 8
    data = ManifestFeatureDataset(
        str(tmp_path / "manifest.jsonl"),
        stage="alignment",
        split="train",
        cache_root=str(tmp_path / "feature-cache"),
        extra_fields=("messages", "molecule_count", "atom_count", "directed_bond_count", "bond_representation"),
    )
    if domain == "genetics":
        configs = [dict(name="dna", kind="mlp2x_gelu_norm", mm_hidden_size=1920, hidden_size=16)]
        ids = {"dna": 32}
        batch = FeatureCollator(tokenizer, projector_name="dna", placeholder_token_id=32)([data[0], data[-1]])
    else:
        configs = [
            dict(name=name, kind="mlp2x_gelu", mm_hidden_size=width, hidden_size=16) for name, _, width in app.BRANCHES
        ]
        ids = {name: token for name, token, _ in app.BRANCHES}

        batch = app.ChemistryCollator(tokenizer)([data[0], data[-1]])
    config = SimpleNamespace(
        hidden_size=8, mm_projectors=configs, mm_placeholder_token_ids=ids, mm_projector_ownership="module"
    )
    model = build_multimodal_host(_TinyLM)(config)
    model.requires_grad_(False)
    model.mm_projector.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.mm_projector.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.mm_projector.parameters()]
    logits = model(**batch)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), batch["labels"].reshape(-1), ignore_index=-100)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.mm_projector.parameters(), strict=True))


class _TinyLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = torch.nn.Embedding(512, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, 512)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        # Context mixing makes supervised answer tokens depend on the soft tokens.
        return self.lm_head(inputs_embeds.cumsum(dim=1))


def test_wrong_encoder_width_is_rejected_before_publication(endpoint, tmp_path):
    from bionemo_perception import Evo2Client

    url, _, state = endpoint
    state["bad_width"] = True
    with pytest.raises(ValueError, match="Evo2 features"):
        _preparation("genetics").prepare(Evo2Client(url), ROOT / "examples/genetics/data/sanity.jsonl", tmp_path)
    assert not (tmp_path / "feature-cache").exists()
    assert not (tmp_path / "manifest.jsonl").exists()


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_native_content_passes_through_preparation(domain, tmp_path):
    image = dict(type="image", url="https://example.com/image.png", detail="high")
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(dict(sample_id="native", messages=[dict(role="user", content=[image])], answer="4")) + "\n"
    )
    output = tmp_path / "prepared"
    _preparation(domain).prepare(object(), source, output)
    record = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    assert record["messages"] == [dict(role="user", content=[image])]


@pytest.mark.parametrize("domain,prefix", [("genetics", "EVO2"), ("chemistry", "KERMT")])
def test_live_perception_preparation(domain, prefix, tmp_path, monkeypatch):
    if not os.environ.get(prefix + "_URL"):
        pytest.skip(f"set {prefix}_URL for the live endpoint check")
    client_module = pytest.importorskip("bionemo_perception")
    client_cls = client_module.Evo2Client if domain == "genetics" else client_module.KERMTClient
    client = client_cls(os.environ[prefix + "_URL"], token=os.environ.get(prefix + "_TOKEN") or None, timeout=900)
    result = _preparation(domain).prepare(client, ROOT / "examples" / domain / "data/sanity.jsonl", tmp_path)
    assert result["records"] == 24
    _check_prepared_training(domain, tmp_path, monkeypatch)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_multiple_inputs_preserve_cache_policy_and_rollout_order(domain, endpoint, tmp_path, monkeypatch):
    from bionemo_perception import Evo2Client, KERMTClient
    from test_domain_examples import _Tokenizer

    formatter = pytest.importorskip("nemo_rl.data.llm_message_utils")
    from nemotron_stitch.nemo_rl import transport

    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    tokenizer = (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(base_processor=_Tokenizer())
    monkeypatch.setattr(app, "load_processor", lambda path: tokenizer)
    if domain == "genetics":
        first, second = dict(type="dna", sequence="A" * 130), dict(type="dna", sequence="C" * 64)
    else:
        first, second = dict(type="molecule", smiles="CCO"), dict(type="molecule", smiles="C")
    content = [
        dict(type="text", text="First: "),
        first,
        dict(type="text", text="; second: "),
        second,
        dict(type="text", text="; first again: "),
        first,
        dict(type="text", text=". Count the total."),
    ]
    rows = [
        dict(sample_id=name, messages=[dict(role="user", content=blocks)], answer="7")
        for name, blocks in (("first", [first]), ("second", [second]), ("multiple", content))
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "prepared"
    client = (Evo2Client if domain == "genetics" else KERMTClient)(endpoint[0])
    _preparation(domain).prepare(client, source, output)
    assert len(endpoint[1]) == 2  # Repeated blocks and stages reuse encoding, but retain every occurrence.
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()][:3]
    cache = open_feature_cache(output / "feature-cache")
    packed = [{**row, "features": cache.get(row["feature_key"])} for row in records]
    if domain == "genetics":
        arrays = [(row["features"],) for row in packed]
        branches = (("dna", 32, 1920),)
    else:
        arrays = [app.split_features(row) for row in packed]
        branches = app.BRANCHES
    expected_items = []
    for source_index in (0, 1, 0):
        for array in arrays[source_index]:
            # DNA transports each entire sequence; graph branches transport individual slots.
            expected_items.extend([array[:, :2]] if domain == "genetics" else list(array[:, None, :2]))
    for index, (_name, token, _) in enumerate(branches):
        expected = np.concatenate([arrays[0][index], arrays[1][index], arrays[0][index]])
        np.testing.assert_array_equal(arrays[2][index], expected)
        rendered = tokenizer.apply_chat_template(packed[2]["messages"], add_generation_prompt=True)
        assert rendered.count(f"<SPECIAL_{token}>") == len(expected)

    def formatted(messages, tokenizer, *args, **kwargs):
        return [
            {
                **message,
                "token_ids": torch.tensor(
                    tokenizer(tokenizer.apply_chat_template([message], add_generation_prompt=False)).input_ids
                ),
            }
            for message in messages
        ]

    monkeypatch.setattr(formatter, "get_formatted_message_log", formatted)
    monkeypatch.setattr(transport, "project_features", lambda features, artifact, name: features[:, :2].clone())
    datum = dict(
        records[2],
        encoder_loader_kwargs=dict(cache_root=str(output / "feature-cache"), artifact_dir="unused"),
        task_metadata_callback="nemotron_stitch.nemo_rl.transport.ground_truth_metadata",
        task_name="count",
        system_prompt="Count carefully.",
        chat_template_kwargs={"enable_thinking": False},
    )
    result = app.grpo_processor(datum, None, _Tokenizer(), 4096, 0)
    user = result["message_log"][-1]
    for (name, _, _), array in zip(branches, arrays[2], strict=True):
        np.testing.assert_array_equal(user[f"mm_features__{name}"].as_tensor().numpy(), array)
    modality = "dna" if domain == "genetics" else "chemistry"
    items = result["vllm_multi_modal_data"][modality]
    for item, expected in zip(items, expected_items, strict=True):
        np.testing.assert_array_equal(item.numpy(), expected)
        assert item.untyped_storage().nbytes() == item.numel() * item.element_size()
    rollout = result["vllm_content"]
    for item in items:
        replacement = "<SPECIAL_32>" * len(item)
        if domain == "genetics":
            replacement = "<SPECIAL_30>" + replacement + "<SPECIAL_31>"
        rollout = rollout.replace(f"<{modality}>", replacement, 1)
    policy = tokenizer.apply_chat_template(
        [dict(role="system", content="Count carefully."), *datum["messages"]],
        add_generation_prompt=True,
    )
    assert rollout == policy.replace("<SPECIAL_33>", "<SPECIAL_32>").replace("<SPECIAL_34>", "<SPECIAL_32>")
    assert result["extra_env_info"] == {"ground_truth": "7"}
    first = datum["messages"][0]["content"][1]
    if domain == "genetics":
        first["sequence"] = first["sequence"][:-64]
    else:
        first["atom_count"] -= 1
    with pytest.raises(ValueError, match="policy slots disagree"):
        app.grpo_processor(datum, None, _Tokenizer(), 4096, 0)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
@pytest.mark.parametrize("with_encoder", [False, True])
def test_chat_history_and_text_only_inputs(domain, with_encoder, endpoint, tmp_path, monkeypatch):
    from bionemo_perception import Evo2Client, KERMTClient
    from test_domain_examples import _Tokenizer

    formatter = pytest.importorskip("nemo_rl.data.llm_message_utils")
    from nemotron_stitch.nemo_rl import transport

    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    tokenizer = (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(base_processor=_Tokenizer())
    monkeypatch.setattr(app, "load_processor", lambda path: tokenizer)
    block = dict(type="dna", sequence="A" * 64) if domain == "genetics" else dict(type="molecule", smiles="C")
    messages = [
        dict(role="system", content="Be precise."),
        dict(role="user", content=[dict(type="text", text="First input: ")] + ([block] if with_encoder else [])),
        dict(role="assistant", content="I have read it."),
        dict(
            role="user",
            content=[dict(type="text", text="Read it again: ")]
            + ([block] if with_encoder else [])
            + [dict(type="text", text=". Count.")],
        ),
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(dict(sample_id="chat", messages=messages, answer="2")) + "\n")
    output = tmp_path / "prepared"
    client = (Evo2Client if domain == "genetics" else KERMTClient)(endpoint[0])
    _preparation(domain).prepare(client, source, output)
    assert len(endpoint[1]) == int(with_encoder)
    data = ManifestFeatureDataset(
        str(output / "manifest.jsonl"),
        stage="sft",
        split="train",
        cache_root=str(output / "feature-cache"),
        extra_fields=("messages", "molecule_count", "atom_count", "directed_bond_count", "bond_representation"),
    )
    row = data[0]
    assert [message["role"] for message in row["messages"]] == [message["role"] for message in messages]
    assert row["messages"][0]["content"] == "Be precise."
    if domain == "genetics":
        batch = FeatureCollator(tokenizer, projector_name="dna", placeholder_token_id=32)([row])
    else:
        batch = app.ChemistryCollator(tokenizer)([row])
    full = [*row["messages"], dict(role="assistant", content=row["target"])]
    ids = tokenizer(tokenizer.apply_chat_template(full, add_generation_prompt=False)).input_ids
    assert batch["input_ids"][0].tolist() == ids[:-1]
    prefix = tokenizer(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True)).input_ids
    assert batch["labels"][0, : len(prefix) - 1].eq(-100).all()

    def formatted(messages, tokenizer, *args, **kwargs):
        return [
            {
                **message,
                "token_ids": torch.tensor(
                    tokenizer(tokenizer.apply_chat_template([message], add_generation_prompt=False)).input_ids
                ),
            }
            for message in messages
        ]

    monkeypatch.setattr(formatter, "get_formatted_message_log", formatted)
    projected_calls = []

    def project(features, artifact, name):
        projected_calls.append(name)
        return features[:, :2].clone()

    monkeypatch.setattr(transport, "project_features", project)
    record = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    datum = dict(
        record,
        task_name="count",
        system_prompt="Default used only when the history has no system message.",
        encoder_loader_kwargs=dict(cache_root=str(output / "feature-cache"), artifact_dir="unused"),
    )
    result = app.grpo_processor(datum, None, tokenizer, 4096, 0)
    assert [message["role"] for message in result["message_log"]] == [message["role"] for message in messages]
    modality = "dna" if domain == "genetics" else "chemistry"
    if with_encoder:
        assert len(result["vllm_multi_modal_data"][modality]) == (2 if domain == "genetics" else 4)
        for message in (result["message_log"][1], result["message_log"][3]):
            assert "mm_features__" + ("dna" if domain == "genetics" else "chem_molecule") in message
    else:
        assert result["vllm_multi_modal_data"] == {}
        assert not projected_calls
        assert row["features"].shape[0] == 0
        assert result["vllm_content"] == tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_shared_thd_packing_matches_individual_samples(domain, endpoint, tmp_path, monkeypatch):
    from bionemo_perception import Evo2Client, KERMTClient
    from test_domain_examples import _Tokenizer

    url, _, _ = endpoint
    _preparation(domain).prepare(
        (Evo2Client if domain == "genetics" else KERMTClient)(url),
        ROOT / "examples" / domain / "data/sanity.jsonl",
        tmp_path,
    )
    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    processor = (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(_Tokenizer())
    monkeypatch.setattr(app, "load_processor", lambda path: processor)
    fields = ["messages", "reasoning_content", "chat_template_kwargs"]
    if domain == "chemistry":
        fields += ["molecule_count", "atom_count", "directed_bond_count", "bond_representation"]
    packed = PackedManifestFeatureDataset(
        tokenizer=processor,
        projector_name="dna" if domain == "genetics" else "chem_molecule",
        placeholder_token_id=32 if domain == "genetics" else 33,
        collate_fn=app.ChemistryCollator(processor) if domain == "chemistry" else None,
        manifest_path=str(tmp_path / "manifest.jsonl"),
        stage="sft",
        split="train",
        cache_root=str(tmp_path / "feature-cache"),
        max_tokens=16384,
        max_length=16384,
        extra_fields=fields,
    )
    assert len(packed) == 1
    item = packed[0]
    singles = [packed.collate_fn([packed.dataset[i]]) for i in range(len(packed.dataset))]
    assert item["qkv_format"] == "thd"
    lengths = [single["input_ids"].numel() for single in singles]
    assert item["seq_lens"].tolist() == lengths
    assert item["seq_lens_padded"].tolist() == lengths
    assert torch.equal(item["labels"], torch.cat([single["labels"].flatten() for single in singles]))
    assert torch.equal(item["input_ids"], torch.cat([single["input_ids"].flatten() for single in singles]))
    offsets = np.cumsum([0, *lengths[:-1]])
    for key in (key for key in item if key.startswith("mm_features__")):
        index_key = key.replace("mm_features__", "mm_token_indices__")
        assert torch.equal(item[key], torch.cat([single[key] for single in singles]))
        expected_indices = torch.cat(
            [single[index_key] + int(offset) for single, offset in zip(singles, offsets, strict=True)]
        )
        assert torch.equal(item[index_key], expected_indices)
    assert packed.dataset.rows[0]["reasoning_content"]
    assert packed.dataset.rows[0]["chat_template_kwargs"] == {"enable_thinking": True}
    from nemotron_stitch.automodel.data import packed_collate_fn

    pytest.importorskip("nemo_automodel")
    batch = packed_collate_fn([item])
    assert batch["input_ids"].numel() == sum(lengths)


def test_configured_encoder_width_and_custom_reasoning_target(endpoint, tmp_path):
    from bionemo_perception import Evo2Client

    url, _, state = endpoint
    state["bad_width"] = True  # This fixture returns width 4096 in that mode.
    raw = [
        {
            "sample_id": "custom",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "dna", "sequence": "ACGTN"}, {"type": "text", "text": "Count the bases."}],
                }
            ],
            "answer": "5",
            "target": "Count: <answer>5</answer>",
            "reasoning_content": "There are five positions including the unknown base.",
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(raw[0]) + "\n")
    _preparation("genetics").prepare(Evo2Client(url), source, tmp_path / "prepared", encoder_width=4096)
    record = json.loads((tmp_path / "prepared/manifest-sft-train.jsonl").read_text())
    assert record["target"] == raw[0]["target"]
    assert record["reasoning_content"] == raw[0]["reasoning_content"]
    feature = open_feature_cache(tmp_path / "prepared/feature-cache").get(record["feature_key"])
    assert feature.shape == (1, 4096)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_native_dataset_config_builds_without_example_factory(domain, endpoint, tmp_path, monkeypatch):
    pytest.importorskip("nemo_automodel")
    from bionemo_perception import Evo2Client, KERMTClient
    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.datasets.loader import make_dataset_config
    from nemo_automodel.recipes.llm.train_ft import _build_tokenizer
    from test_domain_examples import _Tokenizer

    url, _, _ = endpoint
    _preparation(domain).prepare(
        (Evo2Client if domain == "genetics" else KERMTClient)(url),
        ROOT / "examples" / domain / "data/sanity.jsonl",
        tmp_path,
    )
    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    processor = (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(_Tokenizer())
    monkeypatch.setattr(app, "load_processor", lambda path, trust_remote_code=True: processor)
    monkeypatch.setenv("STITCH_PACK_TOKENS", "16384")
    config = load_yaml_config(ROOT / "examples" / domain / "configs/sft.yaml")
    _, tokenizer = _build_tokenizer(config.model, config.dataset)
    assert tokenizer is processor
    kwargs = config.dataset.to_dict()
    target = kwargs.pop("_target_")
    kwargs.pop("tokenizer")
    assert kwargs["max_tokens"] == 16384
    kwargs.update(manifest_path=str(tmp_path / "manifest.jsonl"), cache_root=str(tmp_path / "feature-cache"))
    dataset = make_dataset_config(target, kwargs).build(tokenizer=tokenizer)
    assert isinstance(dataset, PackedManifestFeatureDataset)
    packed = dataset[0]
    assert packed["qkv_format"] == "thd"
    expected = ["dna"] if domain == "genetics" else [name for name, _, _ in app.BRANCHES]
    assert all(f"mm_features__{name}" in packed for name in expected)


def test_chemistry_canonical_bond_endpoints_match_feature_order(endpoint, tmp_path, monkeypatch):
    """Ring closure and branching retain the encoder's sorted low/high row order."""
    from bionemo_perception import KERMTClient
    from test_domain_examples import _Tokenizer

    monkeypatch.syspath_prepend(str(ROOT / "examples/chemistry"))
    from chemistry_example import KERMTProcessor

    url, requests, _ = endpoint
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "graphs",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "molecule", "smiles": "OC(C)C"},
                            {"type": "text", "text": " Compare with "},
                            {"type": "molecule", "smiles": "c1ccccc1"},
                            {"type": "text", "text": "\nQUESTION\nWhich is cyclic?"},
                        ],
                    }
                ],
                "answer": "B",
            }
        )
        + "\n"
    )
    output = tmp_path / "prepared"
    _preparation("chemistry").prepare(KERMTClient(url), source, output)
    assert [request[2]["smiles"] for request in requests] == ["CC(C)O", "c1ccccc1"]
    row = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    branch, _, ring, _ = row["messages"][0]["content"]
    assert branch["bond_endpoints"] == [[0, 1], [1, 2], [1, 3]]
    assert ring["bond_endpoints"] == [[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]
    prompt = KERMTProcessor(base_processor=_Tokenizer()).apply_chat_template(
        row["messages"], add_generation_prompt=True
    )
    assert "0 1 <SPECIAL_34>\n0 5 <SPECIAL_34>\n1 2 <SPECIAL_34>" in prompt
    assert "1 2 <SPECIAL_34>\n1 3 <SPECIAL_34>" in prompt
    assert prompt.count("<SPECIAL_30>") == prompt.count("<SPECIAL_31>") == 2
    assert prompt.count("<SPECIAL_34>") == row["directed_bond_count"] // 2 == 9


def test_gene_span_length_comes_from_configured_pooling_not_dataset(endpoint, tmp_path, monkeypatch):
    from bionemo_perception import Evo2Client
    from test_domain_examples import _Tokenizer

    monkeypatch.syspath_prepend(str(ROOT / "examples/genetics"))
    from genetics_example import Evo2Processor

    url, requests, _ = endpoint
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "windowed",
                "messages": [{"role": "user", "content": [{"type": "dna", "sequence": "A" * 65}]}],
                "answer": "A",
            }
        )
        + "\n"
    )
    output = tmp_path / "prepared"
    _preparation("genetics").prepare(Evo2Client(url), source, output, pooling_window=16)
    row = json.loads((output / "manifest.jsonl").read_text().splitlines()[0])
    block = row["messages"][0]["content"][0]
    assert block == {"type": "dna", "sequence": "A" * 65}
    assert requests[0][2]["outputs"][0]["pooling"]["window"] == 16
    data = ManifestFeatureDataset(
        str(output / "manifest.jsonl"),
        stage="alignment",
        split="train",
        cache_root=str(output / "feature-cache"),
        extra_fields=("messages",),
    )
    processor = Evo2Processor(_Tokenizer())
    batch = FeatureCollator(
        processor, projector_name="dna", placeholder_token_id=32, chat_template_kwargs={"pooling_window": 16}
    )([data[0]])
    assert batch["mm_features__dna"].shape == (5, 1920)
    assert batch["mm_token_indices__dna"].numel() == 5
    default_prompt = processor.apply_chat_template(row["messages"], add_generation_prompt=True)
    assert default_prompt.count("<SPECIAL_32>") == 2
    with pytest.raises(ValueError, match="placeholder slots"):
        FeatureCollator(processor, projector_name="dna", placeholder_token_id=32)([data[0]])
