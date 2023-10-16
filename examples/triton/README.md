This example shows how you can easily deploy a BioNeMo model for inference.
For this purpose you will use [PyTriton](https://github.com/triton-inference-server/pytriton) -- a Flask/FastAPI-like interface that simplifies Triton's deployment in Python environments. The library allows serving Machine Learning models directly from Python through NVIDIA's [Triton Inference Server](https://github.com/triton-inference-server).


# Starting the Server
To start the Triton Server (and send it to the background) run:
```
python examples/triton/<embeddings,sampling>_server.py --config-path </path/to/dir/with/inference/conf> &
```

You can start server for generating embeddings with MegaMolBART:

```
python examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/molecule/megamolbart/conf &
```

...ESM1nv:

```
python examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/protein/esm1nv/conf/ &
```

...or ProtT5nv:
```
python examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/protein/prott5nv/conf/ &
```

Alternatively you can start server for sampling new sequences with MegaMolBART:
```
python examples/triton/sampling_server.py --config-path /workspace/bionemo/examples/molecule/megamolbart/conf &
```

Under the hood, the scripts use `hydra` and load model configuration from `infer.yaml` present in the specified config directory,
so you can provide custom configuration by specifying a different yaml or overriding particular arguments.

# Querying the Server

Regardless of the model and server type selected, you can send a query with the client script:
```
python examples/triton/client.py --sequences SEQUENCES [SEQUENCES ...] [--output_path OUTPUT_PATH]
```

For example:
```
python examples/triton/client.py --sequences "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "c1ccccc1CC(O)=O"
```
if you loaded MegaMolBART in the server script, or:

```
python examples/triton/client.py --sequences "MTADAHWIPVPTNVAYDALNPGAPGTLAFAAANGWQHHPLVTVQPLPGVVFRDAAGRSRFTQRAGD"
```
if you're serving one of the protein models, ESM1nv or ProtT5.

# Closing the Server:

When you're done, you can turn the server off by bringing it from the background (`fg`) and killing it with `ctrl+c`.


# Notes

You can also use only one of the provided components (server or client) - they are fully compatible with bare-metal Triton.
* You can interact with the server like you would do with any other Triton instance
* You can use the client to interact with any Triton server
