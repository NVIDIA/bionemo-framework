# Training Models

## Pydantic Configuration

BioNeMo 2 provides two entrypoints for models with both argparse and pydantic. Both documented in the `Models` section below.
Pydantic based configuration is designed to accept a configuration yaml file as input, along with context-specific
arguments (e.g., should we resume from existing checkpoints?). These YAML configs go through a Pydantic Validator, in
this case referred to as `MainConfig`. This Config is composed of several other Pydantic models, see the class
definition for details. To pre-populate a config with reasonable defaults for various standard models, we provide
'recipes.' These are simple methods that instantiate the config object and then serialize it to a YAML configuration
file. From this file, you may either submit it directly, or modify the various parameters to meet your use case. For
example, Weights and biases, devices, precision, and dataset options are all extremely useful to modify. Then, you would
submit this config for training.

!!! note "5D Parallel Training Moved to bionemo-recipes"
The 5D parallel training implementations for ESM-2 and Geneformer have been moved to [bionemo-recipes](https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes). For training these models, please refer to the recipes in `bionemo-recipes/recipes/` (e.g., `esm2_native_te`, `geneformer_native_te_mfsdp_fp8`).

The following workflows are packaged as executables when esm2 is installed with pip. These commands will appear as:

```bash
bionemo-esm2-recipe
bionemo-esm2-train
```

## ESM-2

### Running

First off, we have a utility function for downloading full/test data and model checkpoints called `download_bionemo_data` that our following examples currently use. This will download the object if it is not already on your local system, and then return the path either way. For example if you run this twice in a row, you should expect the second time you run it to return the path almost instantly.

**NOTE**: NVIDIA employees should use `pbss` rather than `ngc` for the data source.

```bash
export MY_DATA_SOURCE="ngc"
```

or for NVIDIA internal employees with new data etc:

```bash
export MY_DATA_SOURCE="pbss"
```

```bash
# The fastest transformer engine environment variables in testing were the following two
TEST_DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source $MY_DATA_SOURCE); \
ESM2_650M_CKPT=$(download_bionemo_data esm2/650m:2.0 --source $MY_DATA_SOURCE); \

train_esm2     \
    --train-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet     \
    --train-database-path ${TEST_DATA_DIR}/2024_03_sanity/train_sanity.db     \
    --valid-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/valid_clusters.parquet     \
    --valid-database-path ${TEST_DATA_DIR}/2024_03_sanity/validation.db     \
    --result-dir ./results     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 1 \
    --num-steps 10 \
    --max-seq-length 1024 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --restore-from-checkpoint-path ${ESM2_650M_CKPT}
```

### Running with Pydantic configs

Alternatively, we provide a validated and serialized configuration file entrypoint for executing the same workflow. These can be generated using the `bionemo-esm2-recipe` entrypoints. Recipes
are available for 8m, 650m, and 3b ESM2 models. You may select which preset config to use by setting the `--recipe` parameter.
The output is then a serialized configuration file that may be used in the associated `bionemo-esm2-train` commands.

```bash
# The fastest transformer engine environment variables in testing were the following two
TEST_DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source $MY_DATA_SOURCE); \
bionemo-esm2-recipe \
--train-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet     \
--train-database-path ${TEST_DATA_DIR}/2024_03_sanity/train_sanity.db     \
--valid-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/valid_clusters.parquet     \
--valid-database-path ${TEST_DATA_DIR}/2024_03_sanity/validation.db     \
--result-dir ./results     \
--dest my_config.yaml\
--recipe esm2_8m_recipe
```

> ⚠️ **IMPORTANT:** Inspect and edit the contents of the outputted my_config.yaml as you see fit

> NOTE: To continue training from an existing checkpoint, simply pass in the path --initial-ckpt-path to the recipe command. This will populate the YAML with the correct field to ensure pretraining is initialized from an existing checkpoint.

To submit a training job with the passed config, first update the yaml file with any additional execution parameters
of your choosing: number of devices, workers, steps, etc. Second, invoke our training entrypoint. To do this, we need
three things:

- Configuration file, the YAML produced by the previous step
- Model config type, in this case the pretraining config. This will validate the arguments in the config YAML against
  those required for pretraining. Alternatively, things like fine-tuning with custom task heads may be specified here.
  This allows for mixing/matching Data Modules with various tasks.
- Data Config type, this specifies how to parse, validate, and prepare the DataModule. This may change depending on task,
  for example, pretraining ESM2 uses a protein cluster oriented sampling method. In the case of inference or fine-tuning
  a pretrained model, a simple fasta file may be sufficient. There is a one-to-one relationship between DataConfig types
  and DataModule types.

> ⚠️ **Warning:** This setup does NO configuration of Weights and Biases. Edit your config YAML and populate it with your WandB details.

```
bionemo-esm2-train \
--data-config-cls bionemo.esm2.run.config_models.ESM2DataConfig \
--model-config-cls bionemo.esm2.run.config_models.ExposedESM2PretrainConfig \
--config my_config.yaml
```

> NOTE: both data-config-cls and model-config-cls have default values corresponding to ESM2DataConfig and ExposedESM2PretrainingConfig

DataConfigCls and ModelConfigCls can also refer to locally defined types by the user. As long as python knows how to import
the specified path, they may be configured. For example, you may have a custom Dataset/DataModule that you would like to
mix with an existing recipe. In this case, you define a DataConfig object with the generic specified as your DataModule
type, and then pass in the config type to the training recipe.

## Geneformer

!!! note "Geneformer Training Moved to bionemo-recipes"
The 5D parallel training implementation for Geneformer has been moved to [bionemo-recipes](https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes/recipes/geneformer_native_te_mfsdp_fp8). For training Geneformer models, please refer to the recipe in `bionemo-recipes/recipes/geneformer_native_te_mfsdp_fp8/`.

Model checkpoints and benchmark information remain available in the [Geneformer model card](../models/geneformer.md).

DataConfigCls and ModelConfigCls can also refer to locally defined types by the user. As long as python knows how to import
the specified path, they may be configured. For example, you may have a custom Dataset/DataModule that you would like to
mix with an existing recipe. In this case, you define a DataConfig object with the generic specified as your DataModule
type, and then pass in the config type to the training recipe.

#### Weights and Biases Tricks and Tips

##### Trainer/Global Step

At some point you may encounter some funny plots inside the Weights and Biases charts having to do with `trainer/global_steps`. An oscillation pattern like this might be present.
![Trainer Global Step Oscillation](../assets/images/wandb_tips_tricks/trainer_global_step.png). This is actually due to an interaction between Pytorch Lightning and Weights and Biases.
The issue is that during validation, the `validation_step` will be used as `trainer.global_step`. It will not impact model performance, accuracy, or the learning rate scheduler. Moreover, there is also another column called `global_step` that will reflect the accurate step counts over time.
