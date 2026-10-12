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

"""render_soft_token_prompt: one renderer for collators and the RL processor (P6)."""

from __future__ import annotations

import pytest

from nemotron_stitch.prompt import PlaceholderSpec, render_soft_token_prompt

SPEC_A = PlaceholderSpec("alpha", "<s_a>", "<p_a>", "<e_a>")
SPEC_B = PlaceholderSpec("beta", "<s_b>", "<p_b>", "<e_b>")


def test_single_projector_is_the_degenerate_case():
    rendered = render_soft_token_prompt("{mm:alpha}\nWhat is shown?", [SPEC_A], {"alpha": 3})
    assert rendered == "<s_a><p_a><p_a><p_a><e_a>\nWhat is shown?"


def test_interleaved_projectors_render_in_template_order():
    rendered = render_soft_token_prompt(
        "Compare {mm:beta} to {mm:alpha}.",
        [SPEC_A, SPEC_B],
        {"alpha": 2, "beta": 1},
    )
    assert rendered == "Compare <s_b><p_b><e_b> to <s_a><p_a><p_a><e_a>."


def test_missing_marker_fails_closed():
    with pytest.raises(ValueError, match="exactly one marker"):
        render_soft_token_prompt("No marker here.", [SPEC_A], {"alpha": 2})


def test_duplicated_marker_fails_closed():
    with pytest.raises(ValueError, match="exactly one marker"):
        render_soft_token_prompt("{mm:alpha} and {mm:alpha}", [SPEC_A], {"alpha": 2})


def test_marker_without_features_fails_closed():
    with pytest.raises(ValueError, match="received no features"):
        render_soft_token_prompt("{mm:alpha} vs {mm:beta}", [SPEC_A, SPEC_B], {"alpha": 1})


def test_marker_for_an_unknown_projector_fails_closed():
    with pytest.raises(ValueError, match="received no features"):
        render_soft_token_prompt("{mm:alpha} vs {mm:gamma}", [SPEC_A], {"alpha": 1})


def test_shared_placeholder_tokens_fail_closed():
    other = PlaceholderSpec("beta", "<s_b>", "<p_a>", "<e_b>")
    with pytest.raises(ValueError, match="distinct"):
        render_soft_token_prompt("{mm:alpha}", [SPEC_A, other], {"alpha": 1})


def test_counts_for_unknown_projectors_fail_closed():
    with pytest.raises(KeyError, match="unknown projectors"):
        render_soft_token_prompt("{mm:alpha}", [SPEC_A], {"alpha": 1, "gamma": 1})


def test_nonpositive_run_length_fails_closed():
    with pytest.raises(ValueError, match="positive soft-token count"):
        render_soft_token_prompt("{mm:alpha}", [SPEC_A], {"alpha": 0})


def test_duplicate_spec_names_fail_closed():
    with pytest.raises(ValueError, match="duplicate projector names"):
        render_soft_token_prompt("{mm:alpha}", [SPEC_A, SPEC_A], {"alpha": 1})


def test_rl_processor_delegates_to_the_public_renderer():
    from nemotron_stitch.nemo_rl.data import _render_policy_content

    assert (
        _render_policy_content(
            "Question?", 3, adapter="image", placeholder_token="<p>", start_token="<s>", end_token="<e>"
        )
        == "<s><p><p><p><e>\nQuestion?"
    )


def test_rl_processor_preserves_supplied_marker_position():
    # Questions may author the soft-token position themselves.
    from nemotron_stitch.nemo_rl.data import _render_policy_content

    assert (
        _render_policy_content(
            "Before?\n{mm:image}\nAfter",
            2,
            adapter="image",
            placeholder_token="<p>",
            start_token="<s>",
            end_token="<e>",
        )
        == "Before?\n<s><p><p><e>\nAfter"
    )


def test_rl_processor_rejects_wrong_projector_marker():
    from nemotron_stitch.nemo_rl.data import _render_policy_content

    with pytest.raises(ValueError, match="found 0"):
        _render_policy_content(
            "{mm:video}\nQuestion?",
            3,
            adapter="image",
            placeholder_token="<p>",
            start_token="<s>",
            end_token="<e>",
        )


def test_rl_processor_rejects_duplicate_supplied_markers():
    from nemotron_stitch.nemo_rl.data import _render_policy_content

    with pytest.raises(ValueError, match="found 2"):
        _render_policy_content(
            "{mm:image}\nQuestion?\n{mm:image}",
            3,
            adapter="image",
            placeholder_token="<p>",
            start_token="<s>",
            end_token="<e>",
        )
