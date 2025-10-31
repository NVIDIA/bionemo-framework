# Lepton CI

This directory holds code required for triggering automated partial-convergence/performance benchmarking runs in Lepton.

The dashboards may be viewd at the (internal only) url: [nv/bionemo-dashboards](https://nv/bionemo-dashboards).

## Overview

Currently, there are two ongoing benchmark runs, each triggered nightly:

- **model_convergence**: Partial convergence runs for `bionemo-recipes`. Use GPU resources on Lepton.
- **scdl_performance**: Performance benchmarking runs for `bionemo-scdl`. Use CPU resources on Lepton.

The code is organized as follows:

```
ci/lepton
├── core
│   ├── __init__.py
│   ├── launch_job.py
│   ├── lepton_utils.py
│   └── utils.py
├── model_convergence
│   ├── configs
│   └── launchers
├── README.md
├── requirements.txt
└── scdl_performance
    ├── configs
    └── launchers
```

- `core/`: Holds the core logic for triggering jobs to Lepton. It makes use of hydra configs.
- `model_convergence/`:
  - `configs/`: model-specific configs.
  - `launchers/`: Logic to grab job-specific data and upload it to kratos.
- `scdl_performance/`
  - `configs/`: Hydra configs detailing performance benchmarking.
  - `launchers/`: Logic to grab job-specific data and upload it to kratos.

## Triggering Jobs

Each type of benchmark may run as follows:

- Triggered locally from Python.
- Triggered manually from GitHub Actions.

In addition, each job runs each morning at 1am PST on a schedule.

_Note - if running the Python code locally, you will have to edit the secrets reference in the `configs/base.yaml` files. For this reason, creating a branch and triggering it from the Github Action is the preferred method of development._

### Model Convergence

#### Python Trigger

To run locally, call `core/launch_job.py`, providing the path to the config directory and the config name:

```
# call launch_job with specified config
python ci/lepton/core/launch_job.py \
    --config-path="../model_convergence/configs" \
    --config-name="recipes/codonfm_ptl_te"
```

#### Github Actions Trigger

The GH Action is defined in [.github/workflows/convergence-tests.yml](.github/workflows/convergence-tests.yml).
To trigger the GH Actions, you may [trigger the action manually from github](https://github.com/NVIDIA/bionemo-framework/actions/workflows/convergence-tests.yml) and supply the provided information.

If you are developing a new config, simply create the new config (following the structures of the others), and provide that `branch` to the GitHub action. If you created a new config file, you will also have to add that as an option in the `convergence-tests.yml` dropdown. _(If you do edit `convergence-tests.yml`, make sure to use that as the branch for the **Use workflow from** option)._

The job also runs every night on a [schedule](https://github.com/NVIDIA/bionemo-framework/blob/main/.github/workflows/convergence-tests.yml#L33).

### SCDL Performance

#### Python Trigger

To run locally, call `core/launch_job.py`, providing the path to the config directory and the config name:

```
# call launch_job with specified config
python ci/lepton/core/launch_job.py \
    --config-path="../scdl_performance/configs" \
    --config-name="scdl"
```

#### Github Actions Trigger

The GH Action is defined in [.github/workflows/scdl-performance-tests.yml](.github/workflows/scdl-performance-tests.yml).
To trigger the GH Actions, you may [trigger the action manually from github](https://github.com/NVIDIA/bionemo-framework/actions/workflows/scdl-performance-tests.yml) and supply the provided information.

If you are developing a new config, simply create the new config (following the structures of the others), and provide that `branch` to the GitHub action. If you created a new config file, you will also have to add that as an option in the `scdl-performance-tests.yml` dropdown. _(If you do edit `scdl-performance-tests.yml`, make sure to use that as the branch for the **Use workflow from** option)._

The job also runs every night on a [schedule](https://github.com/NVIDIA/bionemo-framework/blob/main/.github/workflows/convergence-tests.yml#L33).
