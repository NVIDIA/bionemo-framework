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

from nemotron_stitch.nemo_rl.vllm_worker import ResidentVllmLifecycle


def test_disabled_lifecycle_always_uses_upstream_sleep_and_wake():
    lifecycle = ResidentVllmLifecycle(enabled=False)

    assert lifecycle.should_sleep
    assert lifecycle.should_wake
    lifecycle.mark_wake_complete(["kv_cache"])
    assert lifecycle.should_sleep
    assert lifecycle.should_wake


def test_resident_lifecycle_sleeps_during_model_initialization_only():
    lifecycle = ResidentVllmLifecycle(enabled=True)

    assert lifecycle.should_sleep
    assert lifecycle.should_wake

    lifecycle.mark_wake_complete(["weights"])
    assert lifecycle.should_sleep
    assert lifecycle.should_wake

    lifecycle.mark_wake_complete(["kv_cache"])
    assert not lifecycle.should_sleep
    assert not lifecycle.should_wake


def test_unsplit_wake_marks_resident_lifecycle_ready():
    lifecycle = ResidentVllmLifecycle(enabled=True)

    lifecycle.mark_wake_complete(None)

    assert lifecycle.ready
