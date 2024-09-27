#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -xueo pipefail
export PYTHONDONTWRITEBYTECODE=1

source "$(dirname "$0")/utils.sh"

if ! set_bionemo_home; then
    exit 1
fi

echo "Running pytest tests"
pytest -m "not internal and not needs_fork and not needs_80gb_memory_gpu" -vv --durations=0 --cov=bionemo --cov-report term --cov-report xml:coverage-pytest.xml -k "not test_model_training" -s
