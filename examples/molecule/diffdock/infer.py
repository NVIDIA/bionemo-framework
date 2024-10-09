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


"""
Entry point to DiffDock inference: generating ligand poses.

modify parameters from conf/*.yaml
"""

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch_geometric.seed import seed_everything

from bionemo.data.diffdock.inference import build_inference_datasets
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference, do_inference_sampling


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(seed)

    # process input and build inference datasets for score model, and confidence model, build dataloader
    complex_name_list, test_dataset, confidence_test_dataset, test_loader = build_inference_datasets(cfg)

    # load score model
    model = DiffDockModelInference(cfg.score_infer)

    # load confidence model
    if "confidence_infer" in cfg:
        confidence_model = DiffDockModelInference(cfg.confidence_infer)
    else:
        confidence_model = None

    # Perform the inference, which as following steps
    # 1. randomize the initial ligand positions.
    # 2. Doing reverse diffusion sampling with the score model and get confidence scores for the generated ligand poses.
    # 3. Write out results.
    failures, skipped = do_inference_sampling(
        cfg, model, confidence_model, complex_name_list, test_loader, test_dataset, confidence_test_dataset
    )

    logging.info(f"Failed for {failures} complexes")
    logging.info(f"Skipped {skipped} complexes")
    logging.info(f"Results are in {cfg.out_dir}")


if __name__ == "__main__":
    main()
