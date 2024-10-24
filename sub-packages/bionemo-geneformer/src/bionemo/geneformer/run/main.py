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


from typing import Optional, Union

import nemo_run as run
import pydantic

from bionemo.geneformer.run.factories import (
    DataConfig,
    ExperimentConfig,
    ExposedModelConfig,
    ModelConfigT,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
    pretrain_partial,
)


class NeMoRunConfig(pydantic.BaseModel):
    # These are all mutually exclusive, I think thats important to capture.
    # NOTE havent figured out how to use this config yet.
    #       could pass this into the entrypoint and do a branch based on the config
    new_experiment_title: Optional[str]
    resume_from_id: Optional[str]
    resume_from_title: Optional[str]

    def __post_init__(self):
        if not any([self.new_experiment_title, self.resume_from_id, self.resume_from_title]):
            raise ValueError(
                "Exactly one of new_experiment_title, resume_from_id, resume_from_title must be set. None are set."
            )

        if sum([bool(self.new_experiment_title), bool(self.resume_from_id), bool(self.resume_from_title)]) > 1:
            raise ValueError(
                "Exactly one of new_experiment_title, resume_from_id, resume_from_title must be set. More than one field was set."
            )


@run.cli.entrypoint
def run_again(
    resume_from_id: Optional[
        str
    ],  # Note, in these cases we dont actually need the rest of the configs. Maybe these deserve distinct entrypoints.
    resume_from_title: Optional[str],
):
    """Example entrypoint of how to re-run an existing job."""
    assert (
        resume_from_id or resume_from_title
    ), "Exactly one of resume_from_id or resume_from_title must be set to rerun an experiment."
    assert not (
        resume_from_id and resume_from_title
    ), "Exactly one of resume_from_id or resume_from_title must be set to rerun an experiment."

    # Setup the context manager with the correct entrypoint, expect these to be mutually exclusive
    with run.Experiment.from_title(resume_from_title) if resume_from_title is not None else run.Experiment.from_id(
        resume_from_id
    ) as exp:
        exp.executor = run.LocalExecutor()  # Can we mutate?
        exp.reset()
        exp.run(direct=True, tail_logs=True, sequential=True)


@run.cli.entrypoint
def simple_example(this_or_that: Union[str, int]):
    print(this_or_that)


@run.cli.entrypoint
def run_firsttime(
    # NeMo Run controls.
    experiment_title: str,
    # Pretrain configuration requirements.
    model_config: ExposedModelConfig[ModelConfigT],
    data_config: DataConfig,
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    # Remaining are things that live outside a config
    resume_if_exists: bool = True,
    # WANDB
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = True,
    launcher: Optional[str] = None,
):
    # Set launcher='torchrun' to execute on the cluster
    # local_executor = run.LocalExecutor(launcher=launcher)

    local_executor = run.LocalExecutor(ntasks_per_node=parallel_config.num_devices, launcher=launcher)
    with run.Experiment(title=experiment_title, executor=local_executor) as e:
        # Input has to be a partial wrapper of pretrain?
        e.add(
            pretrain_partial(
                model_config,
                data_config,
                parallel_config,
                training_config,
                optim_config,
                experiment_config,
                resume_if_exists,
                wandb_entity,
                wandb_project,
                wandb_offline,
            )
        )
        e.run()


if __name__ == "__main__":
    run.cli.main(run_firsttime)

    """
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        run.cli.main(run_firsttime)
    else:
    """
    # run.cli.main(simple_example)
    # run.cli.main(run_again)
