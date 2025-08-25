# ESM-2 training with megatron-fsdp and custom pytorch training loop

Build the docker image with the following command:

```bash
docker build -t my_image .
```

## Running training

Run training with

```bash
docker run --rm -it --gpus all my_image torchrun train.py --config-name L0_sanity
```
