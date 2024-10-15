# Development with BioNeMo

On this page, we will cover the organization of the codebase and the setup necessary for the two primary development
workflows for users of the BioNeMo Framework: training and fine-tuning models. For both of these workflows, we recommend
setting the `NGC_CLI_API_KEY` environment variable as discussed in the [Initialization Guide](./initialization-guide.md).
This environment variable is required by the script that will be used to download both model checkpoints and data from NGC
to be used in these workflows.

## BioNeMo Code Overview

The BioNeMo codebase is structured as a meta-package that collects together many Python packages. We designed BioNeMo
this way with the expectation that users will import and use BioNeMo in their own projects. By structuring code this way,
we ensure that BioNeMo developers follow similar patterns to those we expect of our end users.

Each model is stored in its own subdirectory of `sub-packages`. Some examples of models include:

* `bionemo-esm2`: The ESM2 model
* `bionemo-geneformer`: The Geneformer model
* `bionemo-example_model`: A minimal example MNIST model that demonstrates how you can write a lightweight
    Megatron model that does not actually support any megatron parallelism, but should run fine as long as you only use
    data parallelism to train.

We also include useful utility packages, for example:

* `bionemo-scdl`: Single Cell Dataloader (SCDL) provides a dataset implementation that can be used by
    downstream single-cell models in the bionemo package.
* `bionemo-testing`: A suite of utilities that are useful in testing, think `torch.testing` or `np.testing`.

Finally some of the packages represent common functions and abstract base classes that expose APIs that are useful for
interacting with `NeMo2`. Some examples of these include:

* `bionemo-core`: High-level APIs
* `bionemo-llm`: Abstract base classes for code that multiple large language models (eg BERT variants) share.

### Package Structure

Within each of the Bionemo packages, a consistent structure is employed to facilitate organization and maintainability.
The following components are present in each package:

* **`pyproject.toml`**: Defines package metadata, including version, package name, and executable scripts to be installed.
* **`src`**: Contains all source code for the package. Each package features a top-level `bionemo` folder, which serves
    as the primary namespace for imports. During the build process, all `bionemo/*` source files are combined into a
    single package, with unique subdirectory names appended to the `bionemo` directory.
* **`tests`**: Houses all package tests. The convention for test files is to locate them in the same path as the
    corresponding `src` file, but within the `tests` directory, with a `test_` prefix added to the test file name. For
    example, to test a module-level file `src/bionemo/my_module`, a test file `tests/bionemo/test_my_module.py` should
    be created. Similarly, to test a specific file `src/bionemo/my_module/my_file.py`, the test file should be named
    `tests/bionemo/my_module/test_my_file.py`. Running `py.test sub-packages/my_package` will execute all tests within
    the `tests` directory.
* **`examples`**: Some packages include an `examples` directory containing Jupyter Notebook (`.ipynb`) files, which are
    aggregated into the main documentation.
* **`README.md`**: The core package README file serves as the primary documentation for each sub-package when uploaded
    to PyPI.
* **`LICENSE`**: For consistency, all Bionemo packages should utilize the Apache-2.0 license. By contributing code to
    BioNeMo, you acknowledge permission for the code to be re-released under an Apache v2 license.

## Model Training Process

The process for pretraining models from BioNeMo involves running scripts located in the `scripts` directory. Each script
exposes a Command-Line Interface (CLI) that contains and documents the options available for that model.

To pretrain a model, you need to run the corresponding script with the required parameters. For example, to pretrain the
ESM-2 model, you would run the `esm2_pretrain.py` script located in `scripts/protein/esm2`. Similarly, to pretrain the
Geneformer model, you would run the `train.py` script located in `scripts/singlecell/geneformer`.

The scripts provide various options that can be customized for pretraining, such as:

* Data directories and paths
* Model checkpoint paths
* Experiment names for tracking
* Number of GPUs and nodes
* Validation check intervals
* Number of dataset workers
* Number of steps
* Sequence lengths
* Micro-batch sizes
* Limit on validation batches

You can specify these options when running the script using command-line arguments. For each of the available scripts,
you can use the `--help` option for an explanation of the available options for that model.

For more information on pretraining a model, refer to the [ESM2 Pretraining
Tutorial](../examples/bionemo-esm2/pretrain.md).

## Fine-Tuning

The model fine-tuning process involves downloading the required model checkpoints using the `download_bionemo_data`
script. This script takes in the model name and version as arguments, along with the data source, which can be either
`ngc` (default) or `pbss` for NVIDIA employees.

To view a list of available resources (both model checkpoints and datasets), you can use the following command:

```bash
download_bionemo_data --list-resources
```

### Step 1: Download Data and Checkpoints

To download the data and checkpoints, use the following command:

```bash
export DATA_SOURCE="ngc"
MODEL_CKPT=$(download_bionemo_data <model_name>/<checkpoint_name>:<version> --source $DATA_SOURCE);
```

Replace `<model_name>` with the desired model (for example, `esm2` or `geneformer`), `<version>` with the desired
version, and `<checkpoint_name>` with the desired checkpoint name.

Additionally, you can download available datasets from NGC using the following command, making similar substitutions as
with the model checkpoint download command above:

```bash
TEST_DATA_DIR=$(download_bionemo_data <model_name>/testdata:<version> --source $DATA_SOURCE);
```

Alternatively, you can use your own data by configuring your container run with volume mounts as discussed in the
[Initialization Guide](./initialization-guide.md).

### Step 2: Adapt the Training Process

Fine-tuning may involve specifying a different combination of model and loss than was used to train the initial version
of the model. The fine-tuning steps will be application-specific, but a general set of steps include:

1. **Prepare your dataset**: Collect and prepare your dataset, including the sequence data and target values. This step is
    crucial to ensure that your dataset is in a format that can be used for training.
2. **Create a custom dataset class**: Define a custom dataset class that can handle your specific data format. This class should
    be able to initialize the dataset and retrieve individual data points.
3. **Create a datamodule**: Define a datamodule that prepares the data for training. This includes tasks such as data loading,
    tokenization, and batching.
4. **Fine-tune the model**: Use a pre-trained model as a starting point and fine-tune it on your dataset. This involves
    adjusting the model's parameters to fit your specific task and dataset.
5. **Configure the fine-tuning**: Set various hyperparameters for the fine-tuning process, such as the batch size, number of
    training steps, and learning rate. These hyperparameters can significantly affect the performance of the fine-tuned
    model.
6. **Run inference**: Once the model is fine-tuned, use it to make predictions on new, unseen data.

For more information on fine-tuning a model, refer to the [ESM2 Fine-tuning
Tutorial](../examples/bionemo-esm2/finetune.md).

## Advanced Developer Documentation

For advanced development information (for example, developing the source code of BioNeMo), refer to the [README]({{ github_url }}) found on the main page of the BioNeMo GitHub Repository.
