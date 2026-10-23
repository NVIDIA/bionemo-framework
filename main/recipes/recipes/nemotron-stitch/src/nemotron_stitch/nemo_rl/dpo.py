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

"""Select an external preference processor for NeMo RL's unchanged DPO trainer.

NeMo RL 4d969c932 offers ``setup_preference_data(processor_fn=...)`` but its
stock launcher fixes that argument to the text processor (U-54). Delete this
entry point when the upstream launcher accepts a processor FQN in configuration.
"""

from __future__ import annotations

import argparse
import pprint


def main(argv: list[str] | None = None) -> None:
    """Run DPO with a validated public processor callback and stock collation."""
    from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_preference_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from nemo_rl.utils.logger import get_next_experiment_dir
    from omegaconf import OmegaConf

    from nemotron_stitch.nemo_rl.runner import install_extensions
    from nemotron_stitch.nemo_rl.transport import resolve_callback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--processor", required=True, help="Fully qualified preference processor")
    args, overrides = parser.parse_known_args(argv)
    processor = resolve_callback(args.processor)
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    pprint.pprint(config)

    install_extensions()
    init_telemetry_driver(config, algorithm="dpo")
    try:
        init_ray()
        tokenizer = get_tokenizer(config.policy["tokenizer"])
        dataset, val_dataset = setup_preference_data(tokenizer, config.data, processor_fn=processor)
        (
            policy,
            _,
            train_loader,
            val_loader,
            loss_fn,
            logger,
            checkpointer,
            save_state,
            config,
        ) = setup(config, tokenizer, dataset, val_dataset)
        with checkpointer:
            dpo_train(
                policy,
                train_loader,
                val_loader,
                tokenizer,
                loss_fn,
                config,
                logger,
                checkpointer,
                save_state,
            )
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
