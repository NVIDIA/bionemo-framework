#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# title: install_third_party..sh
# usage:
#   cd <bionemo repo home>;
#   ./examples/protein/openfold/scripts/install_third_party.sh
#
# description: install two cpp binaries via wget and cmake
#
#   This is updated to address in multi-gpu training jobs on the ORD slurm
#       cluster, where problems arose from multiple downloads and
#       installs to common location.
#
#

set -eo pipefail        # not u

MESSAGE_TEMPLATE='********install_third_party.sh: %s\n'
nanoseconds_since_epoch=$(date +%s%N)
printf "${MESSAGE_TEMPLATE}" "begin at nanoseconds_since_epoch=${nanoseconds_since_epoch}"

# (0) preamble

# constants
FORMAT_DATETIME_STD='%Y-%m-%d %H:%M:%S'

# timer functions
timer_start () {
    datetime_before_task="$(date +"${FORMAT_DATETIME_STD}")"
    seconds_before_task="$(date --date="${datetime_before_task}" "+%s")"
}
timer_end() {
    datetime_after_task="$(date +"${FORMAT_DATETIME_STD}")"
    seconds_after_task="$(date --date="${datetime_after_task}" "+%s")"
    delta_seconds="$((seconds_after_task - seconds_before_task))"
}

# (1) early exit if BIONEMO_HOME not set in environment
if [[ -z "$BIONEMO_HOME" ]]; then

    msg="\$BIONEMO_HOME is unset. Please set the variable and run the
    script again. This variable should be set to the base of the repo path.
    Exiting with exit code 1"
    printf "${MESSAGE_TEMPLATE}" ${msg}
    exit 1
fi

# (2) early exit if LOCAL_RANK is not empty string, and the rank is not zero
#   --> install once per node
if [[ -n "${LOCAL_RANK}" ]]; then

    if [[ ! "${LOCAL_RANK}" == "0" ]]; then
        printf "${MESSAGE_TEMPLATE}" "LOCAL_RANK=${LOCAL_RANK}, do not install, early exit with exit code 0 "
        exit 0
    fi
    printf "${MESSAGE_TEMPLATE}" "LOCAL_RANK=${LOCAL_RANK}, continue to install"
fi

timer_start

# (3) determine install workspace directories
INSTALL_DIR_1="/tmp/install_third_party_1"
INSTALL_DIR_2="/tmp/install_third_party_2"

LABEL_1="ns${nanoseconds_since_epoch}"
LABEL_2="ns${nanoseconds_since_epoch}"

# if LOCAL_RANK is not empty string, add to sub-directory name
if [[ -n "${LOCAL_RANK}" ]]; then

    LABEL_1="lrank${LOCAL_RANK}_${LABEL_1}"
    LABEL_2="lrank${LOCAL_RANK}_${LABEL_2}"
fi

# if GLOBAL_RANK is not empty string
if [[ -n "${GLOBAL_RANK}" ]]; then
    LABEL_1="grank${GLOBAL_RANK}_${LABEL_1}"
    LABEL_2="grank${GLOBAL_RANK}_${LABEL_2}"
fi

INSTALL_DIR_1="${INSTALL_DIR_1}_${LABEL_1}"
INSTALL_DIR_2="${INSTALL_DIR_2}_${LABEL_2}"

# (4) begin installation of first cpp binary
printf "${MESSAGE_TEMPLATE}" "Installing Kalign v3.3.5 to INSTALL_DIR_1=${INSTALL_DIR_1}"
mkdir -p ${INSTALL_DIR_1}/downloads
wget -q -P ${INSTALL_DIR_1}/downloads https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.5.tar.gz
tar -xzf ${INSTALL_DIR_1}/downloads/v3.3.5.tar.gz --directory ${INSTALL_DIR_1}
rm -r ${INSTALL_DIR_1}/downloads
ls ${INSTALL_DIR_1}
cd ${INSTALL_DIR_1}/kalign-3.3.5
mkdir -p build
cd build
cmake ..
make -j
make install
rm -r ${INSTALL_DIR_1}/kalign-3.3.5
printf "${MESSAGE_TEMPLATE}" "Kalign v3.3.5 installed successfuly for LABEL_1=${LABEL_1}"

# (5) begin installation of second cpp binary
printf "${MESSAGE_TEMPLATE}" "Installing HH-suite v.3.3.0 to INSTALL_DIR_2=${INSTALL_DIR_2}"
mkdir -p ${INSTALL_DIR_2}/downloads
wget -q -P ${INSTALL_DIR_2}/downloads https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz
tar -xzf ${INSTALL_DIR_2}/downloads/v3.3.0.tar.gz --directory ${INSTALL_DIR_2}
rm -r ${INSTALL_DIR_2}/downloads
ls ${INSTALL_DIR_2}
cd ${INSTALL_DIR_2}/hh-suite-3.3.0
mkdir -p build
cd build
cmake ..
make -j
make install
rm -r ${INSTALL_DIR_2}/hh-suite-3.3.0
printf "${MESSAGE_TEMPLATE}" "HH-suite v.3.3.0 installed successfuly for LABEL_2=${LABEL_2}"

timer_end

printf "${MESSAGE_TEMPLATE}" "seconds for download and install steps: ${delta_seconds}"
printf "${MESSAGE_TEMPLATE}" "end with exit code 0"
exit 0
