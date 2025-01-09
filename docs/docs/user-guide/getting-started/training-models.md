# Training Models

## Pydantic Configuration

BioNeMo 2 provides two entrypoints for models with both argparse and pydantic. Both documented in the `Models` section below.
Pydantic based configuration is designed to accept a configuration yaml file as input, along with context specific
arguments (e.g., should we resume from existing checkpoints?). These YAML configs go through a Pydantic Validator, in
this case referred to as `MainConfig`. This Config is composed of several other Pydantic models, see the class
definition for details. To pre-populate a config with reasonable defaults for various standard models, we provide
'recipes.' These are simple methods that instantiate the config object and then serialize it to a YAML configuration
file. From this file, you may either submit it directly, or modify the various parameters to meet your usecase. For
example, Weights and biases, devices, precision, and dataset options are all extremely useful to modify. Then, you would
submit this config for training.

These two workflows are packaged as executables when esm2 or geneformer are installed with pip. These commands will appear as:

```bash
bionemo-geneformer-recipe
bionemo-esm2-recipe
bionemo-geneformer-train
bionemo-esm2-train
```

## ESM-2

### Running

First off, we have a utility function for downloading full/test data and model checkpoints called `download_bionemo_data` that our following examples currently use. This will download the object if it is not already on your local system,  and then return the path either way. For example if you run this twice in a row, you should expect the second time you run it to return the path almost instantly.

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

### Running

Similar to ESM-2, you can download the dataset and checkpoint through our utility function.

```bash
TEST_DATA_DIR=$(download_bionemo_data single_cell/testdata-20241203 --source $MY_DATA_SOURCE); \
GENEFORMER_10M_CKPT=$(download_bionemo_data geneformer/10M_240530:2.0 --source $MY_DATA_SOURCE); \
train_geneformer     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small_processed_scdl    \
    --result-dir ./results     \
    --restore-from-checkpoint-path ${GENEFORMER_10M_CKPT} \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2
```

To fine-tune, you to specify a different combination of model and loss. Pass the path to the outputted config file from the previous step as the `--restore-from-checkpoint-path`, and also change
`--training-model-config-class` to the newly created model-config-class.

While no CLI option currently exists to hot swap in different data modules and processing functions _now_, you could
copy the `sub-projects/bionemo-geneformer/geneformer/scripts/train_geneformer.py` and modify the DataModule class that gets initialized.

Simple fine-tuning example (**NOTE**: please change `--restore-from-checkpoint-path` to be the checkpoint directory path that was output last
by the previous train run)

```bash
TEST_DATA_DIR=$(download_bionemo_data single_cell/testdata-20241203 --source $MY_DATA_SOURCE); \
train_geneformer     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small_processed_scdl    \
    --result-dir ./results     \
    --experiment-name test_finettune_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --training-model-config-class FineTuneSeqLenBioBertConfig \
    --restore-from-checkpoint-path results/test_experiment/dev/checkpoints/test_experiment--val_loss=4.3506-epoch=1-last
```

### Running with Pydantic configs

Alternatively, we provide a validated and serialized configuration file entrypoint for executing the same workflow. Recipes
are available for 10m, and 106m geneformer models. Additionally we provide an example recipe of finetuning, where the objective
is to 'regress' on token IDs rather than the traditional masked language model approach. In practice, you will likely
need to implement your own DataModule, DataConfig, and Finetuning model. You can use the same overall approach, but with
customizations for your task.

```bash
TEST_DATA_DIR=$(download_bionemo_data single_cell/testdata-20241203 --source $MY_DATA_SOURCE); \
bionemo-geneformer-recipe \
    --recipe 10m-pretrain \
    --dest my_config.json \
    --data-path ${TEST_DATA_DIR}/cellxgene_2023-12-15_small_processed_scdl \
    --result-dir ./results
```

> ⚠️ **IMPORTANT:** Inspect and edit the contents of the outputted my_config.yaml as you see fit

> NOTE: To pretrain from an existing checkpoint, simply pass in the path --initial-ckpt-path to the recipe command. This will populate the YAML with the correct field to ensure pretraining is initialized from an existing checkpoint.

To submit a training job with the passed config, first update the yaml file with any additional execution parameters
of your choosing: number of devices, workers, steps, etc. Second, invoke our training entrypoint. To do this, we need
three things:

- Configuration file, the YAML produced by the previous step
- Model config type, in this case the pretraining config. This will validate the arguments in the config YAML against
    those required for pretraining. Alternatively, things like fine-tuning with custom task heads may be specified here.
    This allows for mixing/matching Data Modules with various tasks.
- Data Config type, this specifies how to parse, validate, and prepare the DataModule. This may change depending on task,
for example, while fine-tuning you may want to use a custom Dataset/DataModule that includes PERTURB-seq. In this case,
the default pretraining DataConfig and DataModule will be insufficient. See ESM2 for additional example usecases.

> ⚠️ **Warning:** This setup does NO configuration of Weights and Biases. Edit your config YAML and populate it with your WandB details.

```bash
bionemo-geneformer-train \
--data-config-cls bionemo.geneformer.run.config_models.GeneformerPretrainingDataConfig \
--model-config-cls bionemo.geneformer.run.config_models.ExposedGeneformerPretrainConfig \
--config my_config.yaml
```

> NOTE: both data-config-cls and model-config-cls have default values corresponding to GeneformerPretrainingDataConfig and ExposedGeneformerPretrainConfig

DataConfigCls and ModelConfigCls can also refer to locally defined types by the user. As long as python knows how to import
the specified path, they may be configured. For example, you may have a custom Dataset/DataModule that you would like to
mix with an existing recipe. In this case, you define a DataConfig object with the generic specified as your DataModule
type, and then pass in the config type to the training recipe.
