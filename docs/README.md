# BioNeMo2 Documentation

## Viewing the current documentation on github pages

The documentation should be viewable at [https://nvidia.github.io/bionemo-framework/](https://nvidia.github.io/bionemo-framework/).

## Previewing the documentation locally

From the repository root:

```bash
# Build the Docker image
docker build -t nvcr.io/nvidian/cvai_bnmo_trng/bionemo2-docs -f docs/Dockerfile .

# Run the Docker container
docker run --rm -it -p 8000:8000 \
    -v ${PWD}/docs:/docs -v ${PWD}/sub-packages:/sub-packages \
    nvcr.io/nvidian/cvai_bnmo_trng/bionemo2-docs:latest
```

And then navigate to [`http://0.0.0.0:8000`](http://0.0.0.0:8000) on your local
machine.
