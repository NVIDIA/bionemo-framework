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

## Hiding/collapsing `.ipynb` cells
To remove cells from the rendered `mkdocs` html you can add a `remove-cell` tag to the cell. Note that `remove-output` is also an option to hide outputs but not the code cell. Unfortunately
`remove-input` does not seem to be supported.

To collapse jupyter-lab rendered code cells, for example in a `brev.dev` or user ran `jupyter lab` session, you can add a special `jupyter` block to the `metadata` block for that cell in the
json representation of your `.ipynb` file. You can do this in vscode by clicking the `...` above the cell and selecting `Edit cell tags (JSON)`.

A metadata field with both changes, (removed from the rendered docs and collapsed in jupyter) would look like the following:

```json
"metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": [
     "remove-cell"
    ]
   },
```

aliases for these options can be found in the `- mkdocs-jupyter:` section of `mkdocs.yml` in this folder.
