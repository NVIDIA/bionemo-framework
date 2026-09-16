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

"""Tests for recipe configuration files."""

import json
from pathlib import Path

import pytest
import yaml


RECIPE_ROOT = Path(__file__).parents[3]


def test_arc_genome_design_filtering_local_config_is_safe_by_default():
    """The Arc pipeline config should parse and avoid external tools by default."""
    config_path = RECIPE_ROOT / "configs" / "arc_genome_design_filtering_local.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["nucleotide_filtering"] is True
    assert config["orf_filtering"] is False
    assert config["homology_filtering"] is False
    assert config["diversification_filtering"] is False
    assert config["genetic_architecture_visualization_and_synteny_filtering"] is False
    assert config["reference_genome_fasta"].endswith("data/external/arc_evo2/phage_gen/data/NC_001422_1.fna")
    assert config["reference_tropism_protein"].endswith(
        "data/external/arc_evo2/phage_gen/data/NC_001422.1_Gprotein.fasta"
    )
    assert config["genome_length_range"] == [5306, 5730]


def test_required_functions_match_synteny_profile():
    """Completeness and ordered gene content must refer to the same function slots."""
    arc = yaml.safe_load((RECIPE_ROOT / "configs/arc_genome_design_filtering_local.yaml").read_text())
    required_functions = set(arc["required_gene_families"])
    synteny_functions = list(arc["synteny_reference_functions"].values())
    assert required_functions
    assert set(synteny_functions) == required_functions
    assert len(synteny_functions) == len(required_functions)
    assert arc["mmseqs_db_aai_database"] != arc["mmseqs_db_protein_database"]


def test_natural_alpha3_j_is_callable():
    """NC_001330.1 J is 24 aa; ORFipy excludes its stop codon from the length cutoff."""
    orfipy = pytest.importorskip("orfipy_core")
    arc = yaml.safe_load((RECIPE_ROOT / "configs/arc_genome_design_filtering_local.yaml").read_text())
    # Native CDS for NP_039596.1, including its terminal TAA.
    sequence = "ATGAAGAAAGCACGTCGTTCTCCTAGTCGTCGTAAAGGTGCTCGCCTCTGGTATGTAGGCGGTTCTCAGTTTTAA"
    minimum, maximum = arc["orfipy_min_max_orf_lengths"]
    calls = orfipy.orfs(
        sequence,
        minlen=minimum,
        maxlen=maximum,
        strand=arc["orfipy_strand"],
        starts=arc["orfipy_start_codons"].split(","),
        stops=arc["orfipy_stop_codons"].split(","),
    )
    assert [(start, end, strand) for start, end, strand, _description in calls] == [(0, 72, "+")]


def test_rl_backend_avoids_fa4_backward():
    """Both resolved RL policies explicitly select the 26.07-qualified backward path."""
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    for name in ("grpo_phage_megatron.yaml", "gdpo_phage_megatron.yaml"):
        config = _load_config_with_defaults(RECIPE_ROOT / "configs" / name)
        assert config["policy"]["megatron_cfg"].get("attention_backend") == "fused", name


@pytest.mark.parametrize("name", ["grpo_phage_megatron.yaml", "gdpo_phage_megatron.yaml"])
def test_rl_configs_have_consistent_generation_limits(name):
    """Tunable batch sizes and length bounds must still fit the runtime contract."""
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    config = _load_config_with_defaults(RECIPE_ROOT / "configs" / name)
    policy = config["policy"]
    generation = policy["generation"]
    mcore = generation["mcore_generation_config"]
    length = config["env"]["phage_qc"]
    points = [
        length[f"genome_length_reward_{point}"] for point in ("lower_zero", "lower_full", "upper_full", "upper_zero")
    ]
    assert 0 <= points[0] < points[1] <= points[2] < points[3]
    assert 0 < length["genome_length_min"] <= length["genome_length_max"]
    assert policy["max_total_sequence_length"] > generation["max_new_tokens"]
    assert 0 < mcore["prompt_batch_size"] <= mcore["max_requests"]
    assert mcore["prompt_batch_size"] <= policy["generation_batch_size"]
    assert mcore["max_requests"] % policy["megatron_cfg"]["tensor_model_parallel_size"] == 0
    assert policy["megatron_cfg"]["enabled"] and not policy["dtensor_cfg"]["enabled"]
    assert config["checkpointing"]["model_save_format"] is None
    assert config["checkpointing"]["pretrained_checkpoint"]["format"] == "megatron_bridge"
    assert length["zero_reward_without_eod"] is True
    assert config["grpo"]["overlong_filtering"] is False


def test_gdpo_config_uses_registered_objectives_and_mmseqs_diversity(tmp_path):
    """Configured objectives must resolve to independent score columns."""
    from bionemo.evo2_phage_gen.generation import write_rl_prompt_bank
    from bionemo.evo2_phage_gen.reward import REWARD_COMPONENTS
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    config = _load_config_with_defaults(RECIPE_ROOT / "configs/gdpo_phage_megatron.yaml")
    env = config["env"]["phage_qc"]
    objectives = env["gdpo_objectives"]
    registered_columns = {component.score_column for component in REWARD_COMPONENTS}
    assert env["reward_output_mode"] == "gdpo"
    assert objectives
    assert len({objective["name"] for objective in objectives}) == len(objectives)
    assert all(len(objective["columns"]) == 1 for objective in objectives)
    columns = [objective["columns"][0] for objective in objectives]
    assert len(set(columns)) == len(columns)
    assert set(columns) <= registered_columns
    assert env["external_qc"]["fail_on_error"] is True
    assert config["checkpointing"]["save_optimizer"] is True
    assert config["policy"]["train_global_batch_size"] == (
        config["grpo"]["num_prompts_per_step"] * config["grpo"]["num_generations_per_prompt"]
    )
    # Each update samples each actual prefix once, then expands its completions.
    # Repeated identical records would obscure the estimator's true group sizes.
    selection = yaml.safe_load((RECIPE_ROOT / "examples/default-sampling-selection.yaml").read_text())
    bank = write_rl_prompt_bank(tmp_path / "train.jsonl", prompt_lengths=selection["prompt_lengths"], num_records=96)
    prompts = [json.loads(line)["messages"][0]["content"] for line in bank.read_text().splitlines()]
    step_size = config["grpo"]["num_prompts_per_step"]
    assert not config["data"]["shuffle"]
    for start in range(0, len(prompts), step_size):
        update = prompts[start : start + step_size]
        assert len(update) == len(set(update)) == len(set(prompts))
    adapter = config["policy"]["generation"]["mcore_generation_config"]["generation_adapter_config"]
    assert adapter["ignore_eos"] is False
    assert adapter["preserve_eos_token"] is True
    diversity = env["mmseqs_cluster_diversity"]
    assert diversity["enabled"]
    assert diversity["min_seq_id"] == 0.99
    assert diversity["coverage"] == 0.95
    assert diversity["cov_mode"] == 0


@pytest.mark.parametrize("name", ["grpo_phage_megatron.yaml", "gdpo_phage_megatron.yaml"])
def test_equal_scalar_credit(name):
    """Improving any enabled scalar component by the same amount earns equal credit."""
    import pandas as pd

    from bionemo.evo2_phage_gen.reward import REWARD_COMPONENTS, RewardWeights, aggregate_rewards
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    env = _load_config_with_defaults(RECIPE_ROOT / "configs" / name)["env"]["phage_qc"]
    weights = RewardWeights(
        **{
            field: env.get(f"weight_{field}", getattr(RewardWeights(), field))
            for field in RewardWeights.__dataclass_fields__
        }
    )
    columns = [
        component.score_column
        for component in REWARD_COMPONENTS
        if component.weight_attr and getattr(weights, component.weight_attr) > 0
    ]
    scored = pd.DataFrame({column: [float(i == j) for i in range(len(columns))] for j, column in enumerate(columns)})
    scored["safety_gate_pass"] = 1.0
    result = aggregate_rewards(scored, weights)
    assert result["reward"].tolist() == pytest.approx([1.0 / len(columns)] * len(columns))


def test_phix_example_documents_every_gdpo_objective():
    """Every enabled score should have a definition in the worked example."""
    config = yaml.safe_load((RECIPE_ROOT / "configs/gdpo_phage_megatron.yaml").read_text())
    readme = (RECIPE_ROOT / "examples/README.md").read_text()
    for objective in config["env"]["phage_qc"]["gdpo_objectives"]:
        assert f"`{objective['name']}`" in readme


def test_every_inherited_grpo_and_gdpo_config_keeps_mandatory_safety_enabled():
    """Supported GRPO and GDPO configs must keep the mandatory safety gate."""
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    config_dir = RECIPE_ROOT / "configs"
    config_paths = sorted({*config_dir.glob("grpo_phage*.yaml"), *config_dir.glob("gdpo_phage*.yaml")})
    assert config_paths

    for config_path in config_paths:
        resolved = _load_config_with_defaults(config_path)
        safety = resolved["env"]["phage_qc"]["sequence_safety"]
        assert type(safety["enabled"]) is bool and safety["enabled"] is True, config_path.name
        assert safety["host_domain"] in {"BACTERIA", "ARCHAEA", "BACTERIA_AND_ARCHAEA"}, config_path.name
        evidence = safety["host_evidence"]
        assert type(evidence["confirmed"]) is bool and evidence["confirmed"] is True, config_path.name
        assert set(evidence["replication_host_domains"]) <= {
            "BACTERIA",
            "ARCHAEA",
            "BACTERIA_AND_ARCHAEA",
        }, config_path.name
        for path_key in (
            "policy_path",
            "asset_manifest_path",
            "diamond_bin",
            "mmseqs_bin",
            "work_dir",
        ):
            assert isinstance(safety[path_key], str) and safety[path_key], (config_path.name, path_key)

        if config_path.name.startswith("gdpo_"):
            assert resolved["checkpointing"]["metric_name"] == "val:phage_qc/mean_reward", config_path.name
            assert resolved["checkpointing"]["keep_top_k"] >= 3, config_path.name
            objectives = resolved["env"]["phage_qc"]["gdpo_objectives"]
            objective_by_name = {objective["name"]: objective for objective in objectives}
            assert {
                "safety_amr",
                "safety_toxin",
                "safety_lysogeny",
            } <= objective_by_name.keys(), config_path.name
            for name, objective in objective_by_name.items():
                assert type(objective.get("requires_safety_eligibility")) is bool, (config_path.name, name)
                assert objective["requires_safety_eligibility"] is (not name.startswith("safety_")), (
                    config_path.name,
                    name,
                )
