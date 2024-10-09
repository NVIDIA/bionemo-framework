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


from nemo.utils import logging

from bionemo.model.protein.openfold.inductor import enable as enable_inductor
from bionemo.model.protein.openfold.triton.mha import enable as enable_mha


class OptimHub:
    __conf = {
        "mha_fused_gemm": False,
        "mha_triton": False,
        "layernorm_triton": False,  # takes precedence before inductor if both enabled
        "layernorm_inductor": False,
        "inductor_global": False,
        "dataloader_pq": False,
    }

    @staticmethod
    def config(name):
        return OptimHub.__conf[name]

    @staticmethod
    def set(name, value):
        if name in OptimHub.__conf:
            OptimHub.__conf[name] = value
            logging.info(f"Enabled {name} mlperf optimisation.")
        else:
            raise Exception(
                f"Optimisation name {name} not recognised. Available optimisations: {list(OptimHub.__conf.keys())}. "
                ' You can also provide "all" and all available optimisations will be turned on. '
            )

    @staticmethod
    def enable_multiple(optims):
        if "all" in optims:
            optims = OptimHub.__conf.keys()
        for optim in optims:
            # TODO: [optim-hub] this assumes optims is a set, enforce in hydra configs or make it work with dict as well
            OptimHub.set(optim, True)

        if OptimHub.config("layernorm_triton") and OptimHub.config("layernorm_inductor"):
            logging.info("Both Triton and inductor layernorm has been turned on. Triton takes precedence.")

    @staticmethod
    def disable_all():
        for optim in OptimHub.__conf.keys():
            OptimHub.set(optim, False)


def enable_mlperf_optim(cfg):
    optimisations = cfg.get("optimisations", None)
    if optimisations is None:
        logging.warning("No optimisations were provided through model.optimisations. Skipping")
        return

    OptimHub.enable_multiple(optimisations)

    # optimizations on module-level
    if OptimHub.config("mha_triton"):
        enable_mha()

    if OptimHub.config("inductor_global"):
        enable_inductor()
