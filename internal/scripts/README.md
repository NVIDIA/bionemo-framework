# Scripts for commonly performed bionemo-framework actions.

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

## Testing Locally

Inside the development container, run `./ci/scripts/static_checks.sh` to validate that code changes will pass the code
formatting and license checks run during CI. In addition, run the longer `./ci/scripts/run_pytest.sh` script to run unit
tests for all sub-packages.
