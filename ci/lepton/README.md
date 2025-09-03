# Model Convergence Tests with DGX Lepton

This sub-directory contains logic for launching model convergence tests with DGX Lepton.

## How It Works

The `launch_job.py` script works as follows:

1. Lepton job and script-specific values are read from the corresponding `yaml` file in `configs/`.

2. A Lepton [Batch Job](https://docs.nvidia.com/dgx-cloud/lepton/get-started/batch-job/) is launched via Lepton's [Python SDK](https://docs.nvidia.com/dgx-cloud/lepton/reference/api/) populated with the config values above.
   - Note, after launching, this job is viewable in Lepton's [job list UI](https://dashboard.dgxc-lepton.nvidia.com/workspace/vfco61g2/compute/jobs/list).
3. On job complete (_currently only on job success, job failure tbd_), job-specific logs are collected:
   - **wandb logs:** the wandb last run's `wandb-metadata.json` and `wandb-summary.json`.
   - **lepton job info:** From running `lep log get --job "$JOB_NAME"` and collecting the result.
4. These logs are sent up as a single payload to [Kratos explorer](https://explorer.kratos.nvidia.com/), where they can be queried further.

   - [Link to explore our data](https://explorer.kratos.nvidia.com/queries/442840/source).

5. Finally, our [model convergence dashboard](https://nv/bionemo-dashboards) update is triggered, so the data should be visible within ~5 minutes.
   - In particular, the [`query_lepton_data.py`]() script from our [dashboards](https://gitlab-master.nvidia.com/clara-discovery/dashboards) GitLab repo is triggered, which reads the data from Kratos and populates the dashboard.

## File Layout

The layout of the relevant files is:

```
├── model_convergence
│   ├── configs
│   │   ├── evo2_config.yaml
│   │   ├── evo2_finetune_lora.yaml
│   │   ├── evo2_finetune.yaml
│   │   └── recipe_config.yaml
│   └── scripts
│       ├── launch_job.py
```

where

- `configs/`: holds the configs for all models to be run on a schedule.
- `scripts/launch_job.py`: Uses Lepton's [Python SDK](https://docs.nvidia.com/dgx-cloud/lepton/reference/api/) to launch a Lepton batch job.

## Dashboard

As mentioned above, the responsibility of the code in this directory is to trigger a Lepton Batch Job and store the resulting logs to Kratos. The logic for visualizing this data is in a separate repository in GitLab: [dashboards](https://gitlab-master.nvidia.com/clara-discovery/dashboards).

Consult the [README]() there for more information.

## Triggering Build

To trigger a new build, you can use the Jenkins [build_lepton_job pipeline]().

This pipeline will trigger a job with the given arguments.
