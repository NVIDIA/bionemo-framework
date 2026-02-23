# vLLM inference for BioNeMo Models

To build the image:

```bash
docker build -t vllm .
```

Set `HF_TOKEN` in your environment to avoid getting rate limited.

To launch a container:

```bash
docker run -it --gpus all --network host --ipc=host -e HF_TOKEN --rm -v ${PWD}:/workspace/bionemo vllm /bin/bash
```

or use `launch.sh`.

To test ESM2 inference using vLLM inside the container:

```python
python test_esm2_golden_values.py
```
