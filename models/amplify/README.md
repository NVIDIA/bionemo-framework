# AMPLIFY Implemented with TE layers

Running tests:

```bash
docker build -t amplify .
docker run --rm -it --gpus all amplify pytest tests/
```

Generating converted AMPLIFY checkpoints:

```bash
mkdir checkpoint_export
docker run --rm -it --gpus all \
  -v $PWD/checkpoint_export:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  amplify python export.py
```
