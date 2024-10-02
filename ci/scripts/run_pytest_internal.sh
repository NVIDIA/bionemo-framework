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

source "$(dirname "$0")/utils.sh"
export PYTHONDONTWRITEBYTECODE=1

if ! set_bionemo_home; then
    exit 1
fi

pytest -m "internal" -vv --durations=0 --cov=bionemo --cov-report term --cov-report xml:coverage-internal.xml -k "not test_model_training"
