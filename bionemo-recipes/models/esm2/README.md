# ESM-2 Implemented with TE layers

Running tests:

```bash
docker build -t esm2 .
docker run --rm -it --gpus all esm2 pytest tests/
```

Generating converted ESM-2 checkpoints:

```bash
docker run --rm -it --gpus all \
  -v /path/to/checkpoint_export/:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  esm2 python export.py
```

## Local development with vscode

To get vscode to run these tests, you can to add the following to your `.vscode/settings.json`:

```json
{
    "python.testing.pytestArgs": [
        "models/esm2/tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

Additionally, run the following command to install the dependencies:

```bash
cd models/esm2
PIP_CONSTRAINT= pip install -e .[convert,test]
```
