# BioNeMo2 Repo
To get started, please build the docker container using
```bash
./launch.sh build
```

All `bionemo2` code is partitioned into independently installable namespace packages. These live under the `sub-packages/` directory.


# TODO: Finish this.

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

## Running
The following command runs a very small example of geneformer pretraining, as well as using our test data loading
mechanism to grab the example data files and return the local path.

```bash
TEST_DATA_DIR=$(bionemo_test_data_path single_cell/testdata-20240506 --source pbss); \
python  \
    scripts/singlecell/geneformer/train.py     \
    --data-dir ${TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data    \
    --result-dir ./results     \
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
TEST_DATA_DIR=$(bionemo_test_data_path single_cell/testdata-20240506 --source pbss); \
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
    --restore-from-checkpoint-path results/test_experiment/dev/checkpoints/test_experiment--val_loss=10.2042-epoch=0
```

## Updating License Header on Python Files
Make sure you have installed [`license-check`](https://gitlab-master.nvidia.com/clara-discovery/infra-bionemo),
which is defined in the development dependencies. If you add new Python (`.py`) files, be sure to run as:
```bash
license-check --license-header ./license_header --check . --modify --replace
```


# UV notes

## Generating uv.lock

The current `uv.lock` file was generated by running

```bash
uv lock --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match --refresh --no-cache
```

For cuda 12.1, we can just do

```bash
uv lock --refresh --no-cache
```

(to match https://pytorch.org/get-started/locally/#start-locally)

Updating dependency locks can be done via

```bash
uv lock --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
```

## Building the image

```bash
docker build -f Dockerfile.uv . -t bionemo-uv
```

## Runnings tests

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

## Setting library versions

We use [setuptools-scm](https://setuptools-scm.readthedocs.io/en/latest/) to dynamically determine the library version
from git tags. As an example:

```bash
$ git tag 2.0.0a1
$ docker build . -t bionemo-uv
$ docker run --rm -it bionemo-uv:latest python -c "from importlib.metadata import version; print(version('bionemo.esm2'))"
2.0.0a1
```

If subsequent commits are added after a git tag, the version string will reflect the additional commits (e.g.
`2.0.0a2.dev1+g4d62638a9`). Note, we don't consider uncommitted changes in determining the version string.
