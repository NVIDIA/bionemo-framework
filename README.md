# BioNeMo2 Repo
To get started, please build the docker container using
```bash
./launch.sh build
```

Launch a container from the build image by executing
```bash
./launch.sh dev
```

All `bionemo2` code is partitioned into independently installable namespace packages. These live under the `sub-packages/` directory.


## Downloading artifacts
Set the AWS access info in your `.env` in the host container prior to running docker:

```bash
AWS_ACCESS_KEY_ID="team-bionemo"
AWS_SECRET_ACCESS_KEY=$(grep aws_secret_access_key ~/.aws/config | cut -d' ' -f 3)
AWS_REGION="us-east-1"
AWS_ENDPOINT_URL="https://pbss.s8k.io"
```
then, running tests should download the test data to a cache location when first invoked.

For more information on adding new test artifacts, see the documentation in [bionemo.testing.data.load](sub-packages/bionemo-testing/src/bionemo/testing/data/README.md)


## Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-fw-ea.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

Different branches of the repo can have different pinned versions of these third-party submodules. To update submodules
after switching branches (or pulling recent changes), run

```bash
git submodule update
```

To configure git to automatically update submodules when switching branches, run

```bash
git config submodule.recurse true
```

### Updating pinned versions of NeMo / Megatron-LM

To update the pinned commits of NeMo or Megatron-LM, checkout that commit in the submodule folder, and then commit the
result in the top-level bionemo repository.

```bash
cd 3rdparty/NeMo/
git fetch
git checkout <desired_sha>
cd ../..
git add '3rdparty/NeMo/'
git commit -m "updating NeMo commit"
```


## Testing Locally
Inside the development container, run `./ci/scripts/static_checks.sh` to validate that code changes will pass the code
formatting and license checks run during CI. In addition, run the longer `./ci/scripts/pr_test.sh` script to run unit
tests for all sub-packages.


## Publishing Packages

*Note*: Once we have a pypi deployment strategy, we should automate the following commands to run automatically via
github actions on new git tags. We can therefore trigger wheel building and pypi deployment by minting new releases as
part of the github.com CI.

### Add a new git tag

We use [setuptools-scm](https://setuptools-scm.readthedocs.io/en/latest/) to dynamically determine the library version
from git tags. As an example:

```bash
$ git tag 2.0.0a1
$ docker build . -t bionemo-uv
$ docker run --rm -it bionemo-uv:latest python -c "from importlib.metadata import version; print(version('bionemo.esm2'))"
2.0.0a1
```

Bionemo packages follow [semantic versioning 2.0](https://semver.org/) rules: API-breaking changes are `MAJOR`, new
features are `MINOR`, and bug-fixes and refactors are `PATCH` in `MAJOR.MINOR.PATCH` version string format.

If subsequent commits are added after a git tag, the version string will reflect the additional commits (e.g.
`2.0.0a1.post1`). Note, we don't consider uncommitted changes in determining the version string.

### Building a python wheel

An overview for publishing packages with `uv` can be found here: https://docs.astral.sh/uv/guides/publish/

Build the bionemo sub-package project by executing the following for the desired package:
```shell
uv build sub-packages/bionemo-core/
```

This will produce a wheel file for the sub-package's code and its dependencies:
```shell
$ ls sub-packages/bionemo-core/dist/
bionemo_core-2.0.0a1.post0-py3-none-any.whl  bionemo_core-2.0.0a1.post0.tar.gz
```

### Uploading a python wheel

After building, the wheel file can be uploaded to PyPI (or a compatible package registry) by executing
`uvx twine upload sub-packages/bionemo-core/dist/*`.

### All steps together

Assumes we're building a wheel for `bionemo-core`.
```bash
git tag MY-VERSION-TAG
uv build /sub-packages/bionemo-core
TWINE_PASSWORD="<pypi pass>" TWINE_USERNAME="<pypi user>" uvx twine upload /sub-packages/bionemo-core/dist/*
```


## Models
### ESM-2
#### Running
First off, we have a utility function for downloading full/test data and model checkpoints called `download_bionemo_data` that our following examples currently use. This will download the object if it is not already on your local system,  and then return the path either way. For example if you run this twice in a row, you should expect the second time you run it to return the path almost instantly.

Note NVIDIA employees should use `pbss` rather than `ngc` for the data source.

```bash
export MY_DATA_SOURCE="ngc"
```
or for NVIDIA internal employees with new data etc:
```bash
export MY_DATA_SOURCE="pbss"
```

```bash
TEST_DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source $MY_DATA_SOURCE); \
ESM2_650M_CKPT=$(download_bionemo_data esm2/650m:2.0 --source $MY_DATA_SOURCE); \
python  \
    scripts/protein/esm2/esm2_pretrain.py     \
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
    --max-seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --restore-from-checkpoint-path ${ESM2_650M_CKPT}
```

### Geneformer
#### Running

Similar to ESM-2, you can download the dataset and checkpoint through our utility function.

```bash
TEST_DATA_DIR=$(download_bionemo_data single_cell/testdata-20240506 --source $MY_DATA_SOURCE); \
GENEFORMER_10M_CKPT=$(download_bionemo_data geneformer/10M_240530:2.0 --source $MY_DATA_SOURCE); \
python  \
    scripts/singlecell/geneformer/train.py     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data    \
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

To fine-tune, you just need to specify a different combination of model and loss (TODO also data class). To do that you
pass the path to the config output by the previous step as the `--restore-from-checkpoint-path`, and also change the
`--training-model-config-class` to the new one.

Eventually we will also add CLI options to hot swap in different data modules and processing functions so you could
pass new information into your model for fine-tuning or new targets, but if you want that functionality _now_ you could
copy the `scripts/singlecell/geneformer/train.py` and modify the DataModule class that gets initialized.

Simple fine-tuning example (NOTE: please change `--restore-from-checkpoint-path` to be the one that was output last
by the previous train run)
```bash
TEST_DATA_DIR=$(download_bionemo_data single_cell/testdata-20240506 --source $MY_DATA_SOURCE); \
python  \
    scripts/singlecell/geneformer/train.py     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data    \
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


## Updating License Header on Python Files
Make sure you have installed [`license-check`](https://gitlab-master.nvidia.com/clara-discovery/infra-bionemo),
which is defined in the development dependencies. If you add new Python (`.py`) files, be sure to run as:
```bash
license-check --license-header ./license_header --check . --modify --replace
```


# UV-based python packaging

We've begun migrating to use `uv` (https://docs.astral.sh/uv/) to handle python packaging inside our docker containers.
In addition to streamlining how we specify intra-repo dependencies, it will allow us to create a uv lockfile to pin our
dependencies for our bionemo docker container.

We'll likely maintain two images going forward:

1. An image that derives from `nvcr.io/nvidia/pytorch` that will be our performance baseline. The advantage of this
   image base is that the performance of pytorch is validated by the NVIDIA pytorch team, but the downsides are that (1)
   the overall image size is quite large, and (2) using `uv sync` to install a pinned virtual environment is not
   possible with the existing python environment in the ngc image.

2. An image that derives from `nvcr.io/nvidia/cuda`, where we use uv to create the python environment from scratch. This
   image uses pytorch wheels from https://download.pytorch.org.

Currently, the devcontainer derives from the cuda-based image above, while the release image derives from the pytorch
image.

## Generating uv.lock

The current `uv.lock` file was generated by running

```bash
uv lock --refresh --no-cache
```

For cuda 12.4, we can run

```bash
uv lock --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match --refresh --no-cache
```

(to match https://pytorch.org/get-started/locally/#start-locally)

## Building the CUDA image

```bash
docker build -f Dockerfile.uv . -t bionemo-uv
```

## Runnings tests inside the CUDA image.

```bash
docker run --rm -it \
    -v ${HOME}/.aws:/home/bionemo/.aws \
    -v ${HOME}/.ngc:/home/bionemo/.ngc \
    -v ${PWD}:/home/bionemo/ \
    -v ${HOME}/.cache:/home/bionemo/.cache \
    -e HOST_UID=$(id -u) \
    -e HOST_GID=$(id -g) \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    bionemo-uv:latest \
    py.test sub-packages/ scripts/
```
