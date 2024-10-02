# Initialization Guide

**Note:** Prior to beginning this section, you must confirm that your computing platform meets or exceeds the prerequisites outlined in the [Hardware and Software Prerequisites](./pre-reqs.md) page.

Now that you have successfully launched the Docker container and entered it, this section will guide you through the container, initial steps to take within the container (such as configuration, downloading pre-trained model weights, etc.), and where to find tutorials.

## NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC. If NGC is not already installed in the container, download it as per the instructions [here](https://org.ngc.nvidia.com/setup/installers/cli) (note that within the container, the AMD64 Linux version should be installed).

Once installed, run `ngc config set` to establish NGC credentials within the container.

## First-Time Setup

First, invoke the following launch script. The first time, it will create a .env file and exit:

```bash
./launch.sh
```

Next, edit the .env file with the correct NGC parameters for your organization and team:

```bash
    NGC_CLI_API_KEY=<YOUR_API_KEY>
    NGC_CLI_ORG=<YOUR_ORG>
    NGC_CLI_TEAM=<YOUR_TEAM>
```

## Download Model Weights

You may now download all pre-trained model checkpoints from NGC through the following command:

```bash
./launch.sh download
```
This will download all models to the `workspace/bionemo/models` directory. Optionally, you may persist the models by copying them to your mounted workspace, so that they need not be redownloaded each time.

## Directory Structure

Note that `workspace/bionemo` is the home directory for the container. Below are a few key components:
* `bionemo`: Contains the core BioNeMo package, which includes base classes for BioNeMo data modules, tokenizers, models, etc.
* `examples`: Contains example scripts, datasets, YAML files, and notebooks
* `models`: Contains all pre-trained models checkpoints in .nemo format.

## Weights and Biases Setup (Optional)

[Weights and Biases (W&B)](https://wandb.ai/) is a Machine Learning Operations (MLOps) platform that provides tools and services to help machine learning practitioners and teams build, train, and deploy models more efficiently. Their products are particularly useful in the life sciences domain, where machine learning is increasingly being used to analyze complex biological data and drive discoveries. BioNeMo is built to work with W&B and requires only simple setup steps to start tracking your experiments. To set up W&B tracking, following the steps below:

1. Setup your [API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) with W&B to enable logging.
2. Set the `WANDB_API_KEY` variable in your `.env` in the same way as you set the previous environment variable in the First-Time Setup instructions above.
3. Use one of the following strategies to enable W&B logging during training:
    - for command line script-based training, add the following override to your command line arguments: `++exp_manager.create_wandb_logger=True`
    - for interactive training (for example, in a Jupyter notebook), add the following line to your script, where `cfg` is the variable holding your parsed Hydra config: `cfg.exp_manager.create_wandb_logger = True`

Additional [properties of the generated W&B run](https://docs.wandb.ai/guides/track/environment-variables) can also be confiugured by providing overrides using the `exp_manager.wandb_logger_kwargs.<property>` syntax. For example to provide the name "training_expt_0" for your run, you would provide the following overrides, depending on your environment:

- for command line script-based training, add the following override to your command line arguments: `++exp_manager.wandb_logger_kwargs.name=training_expt_0`

- for interactive training (for example, in a Jupyter notebook), add the following line to your script, where `cfg` is the variable holding your parsed Hydra config: `cfg.exp_manager.wandb_logger_kwargs.name = "training_expt_0"`
