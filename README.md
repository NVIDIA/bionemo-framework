# BioNeMo Framework (v2.0)

NVIDIA BioNeMo Framework is a collection of programming tools, libraries, and models for computational drug discovery. It accelerates the most time-consuming and costly stages of building and adapting biomolecular AI models by providing domain-specific, optimized models and tooling that are easily integrated into GPU-based computational resources for the fastest performance on the market. You can access BioNeMo Framework as a free community resource here in this repository or learn more at https://www.nvidia.com/en-us/clara/bionemo/ about getting an enterprise license for improved expert-level support.


`bionemo2` code is partitioned into independently installable namespace packages.
These are located under the `sub-packages/` directory. Please refer to [PEP 420 – Implicit Namespace Packages](https://peps.python.org/pep-0420/) for details.

## Developing and Developer Certificate of Origin (DCO)
By contributing to this repo you acknowledge that either this is your original work, or have the right to submit the work
under our license, which as of this writing is Apache v2. See [license](LICENSE/license.txt) for the current license,
and the [contributing document](CONTRIBUTING.md) for more information.

If you find yourself having made a number of commits in a PR, and need to sign them all, a useful tool is the following:
1. Find your first unsigned commit, say it is `mYcmtShrtHash`.
2. Run `git rebase --signoff mYcmtShrtHash^` to sign that commit and all future commits (in your branch please).
3. Push the updated commits `git push -f`.


## Initializing 3rd-party dependencies as git submodules

The NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules for development purposes. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-framework.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

Different branches of the repo can have different pinned versions of these third-party submodules. Make sure you
update submodules after switching branches or pulling recent changes!

To configure git to automatically update submodules when switching branches, run
```bash
git config submodule.recurse true
```
**NOTE**: this setting will not download **new** or remove **old** submodules with the branch's changes.
You will have to run the full `git submodule update --init --recursive` command in these situations.

## First Time Setup
After cloning the repository, you need to run the setup script **first**:
```bash
./internal/scripts/setup_env_file.sh
```
This will return an exit code of 1 on a first time run.

## Release Image Building
To build the release image, run the following script:
```bash
DOCKER_BUILDKIT=1 ./ci/scripts/build_docker_image.sh \
  -regular-docker-builder \
  -image-name "nvcr.io/nvidian/cvai_bnmo_trng/bionemo:bionemo2-$(git rev-parse HEAD)"
```

## Development Image Building
To build the development image, run the following script:
```bash
./internal/scripts/build_dev_image.sh
```

## Interactive Shell in Development Image
After building the development image, you can start a container from it and open a bash shell in it by executing:
```bash
./internal/scripts/run_dev.sh
```

## Downloading artifacts (For NVIDIA Employees)
Set the AWS access info in environment prior to running the dev-container launch script:

```bash
AWS_ACCESS_KEY_ID="team-bionemo"
AWS_SECRET_ACCESS_KEY=$(grep aws_secret_access_key ~/.aws/config | cut -d' ' -f 3)
AWS_REGION="us-east-1"
AWS_ENDPOINT_URL="https://pbss.s8k.io"
```

Running tests downloads the test data to a cache location when first invoked.

For more information on adding new test artifacts, see the documentation in
[`bionemo.testing.data.load`](sub-packages/bionemo-testing/src/bionemo/testing/data/README.md).

## Updating pinned versions of NeMo / Megatron-LM

Pinned commits are bumped by depend-a-bot. To update the pinned commits of NeMo or Megatron-LM manually, checkout the
commit of interest in the submodule folder, and then commit the result in the top-level bionemo repository.

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
`2.0.0a1.post1`). **NOTE**: we don't consider uncommitted changes in determining the version string.

### Building a python wheel

An overview for publishing packages with `uv` can be found here: https://docs.astral.sh/uv/guides/publish/

Build the bionemo sub-package project by executing the following for the desired package:
```shell
uv build sub-packages/bionemo-core/
```

Produce a wheel file for the sub-package's code and its dependencies:
```shell
$ ls sub-packages/bionemo-core/dist/
bionemo_core-2.0.0a1.post0-py3-none-any.whl  bionemo_core-2.0.0a1.post0.tar.gz
```

### Uploading a python wheel

After building, the wheel file may be uploaded to PyPI (or a compatible package registry) by executing
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
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0

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
    --max-seq-length 1024 \
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

To fine-tune, you just need to specify a different combination of model and loss. Pass the path to the outputted config file from the previous step as the `--restore-from-checkpoint-path`, and also change
`--training-model-config-class` to the newly created model-config-class.

While no CLI option currently exists to hot swap in different data modules and processing functions _now_, you could
copy the `scripts/singlecell/geneformer/train.py` and modify the DataModule class that gets initialized.

Simple fine-tuning example (**NOTE**: please change `--restore-from-checkpoint-path` to be the checkpoint directory path that was output last
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
If you add new Python (`.py`) files, be sure to run our license-check. If you have not already done sone, please install
the dev-requirements.txt. If you are working directly inside a release container, you may need to manually install these.
We recommend using the developer container for contributions.

```bash
pip install -r dev-requirements.txt --user
python ./scripts/license_check.py --modify --replace --license-header ./license_header -c sub-packages/ -c docs/ -c scripts/ -c ci/ -c internal/
```

## Updating the secrets baseline file

If false-positives are raised by the [detect-secrets](https://github.com/Yelp/detect-secrets) pre-commit hook, they can
be added to the baseline files by running the following commands:

```bash
detect-secrets scan --baseline .secrets.baseline --exclude-files '(.*\.ipynb|.*\.baseline)$'
detect-secrets scan --baseline .secrets-nb.baseline --exclude-files '^.(?!.*\.ipynb)' --exclude-lines '"(hash|id|image/\w+)":.*'
```

The resulting altered baseline files should then be committed.

# UV-based python packaging

BioNeMo FW is migrating to use `uv` (https://docs.astral.sh/uv/) for handling python packaging inside our docker containers.
In addition to streamlining how we specify intra-repo dependencies, it allows us to create a uv lockfile to pin our
dependencies for our bionemo docker container.

We'll maintain two images going forward:

2. An image that derives from `nvcr.io/nvidia/pytorch` that will be our performance baseline. The advantage of this
   image base is that the performance of pytorch is validated by the NVIDIA pytorch team, but the downsides are that (1)
   the overall image size is quite large, and (2) using `uv sync` to install a pinned virtual environment is not
   possible with the existing python environment in the ngc image.

2. An image that derives from `nvcr.io/nvidia/cuda`, where we use uv to create the python environment from scratch. This
   image uses pytorch wheels from https://download.pytorch.org.

Currently, the devcontainer derives from the cuda-based image above, while the release image derives from the pytorch
image.


## Runnings tests inside the CUDA container.

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
