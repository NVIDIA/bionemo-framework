# Inference with NVIDIA Triton Inference Server

This section will use the pre-trained BioNeMo checkpoints to demonstrate NVIDIA [Triton Inference Server](https://github.com/triton-inference-server).
[PyTriton](https://github.com/triton-inference-server/pytriton), which is a Flask/FastAPI-like interface that simplifies Triton deployment in Python environments will be used.

## Prerequisites

* Linux OS
* Pascal, Volta, Turing, or an NVIDIA Ampere architecture-based GPU.
* NVIDIA Driver
* Docker

## Detailed Example with ESM-1nv

PyTriton provides a light-weight wrapper that allows you to set up the Triton Inference Server based on existing inference code. The only requirement is that inference is done by a function, that takes as an input and returns numpy arrays of supported types (numerical types or bytes).

In this example, the **Sequence to Embedding** task for ESM1 will be used as an example. The solution will consist of two components - server that performs the inference, and a client that queries this server.

The `bionemo.model.protein.esm1nv.infer.ESM1nvInference` class provides `seq_to_embeddings` method that can be used for this purpose. This method requires a list of FASTA sequences as input (list of strings) and returns a torch Tensor object as a result, so a converter must be implemented.

On the client side, a function that takes list of sequences as input and converts it into a numpy bytes array must be implemented:

```python
import numpy as np

sequences = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVL', 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLA']

sequences = np.array([[seq] for seq in sequences])
sequences = np.char.encode(sequences, "utf-8")
```

On the server side, an inference callable that performs the following must be implemented:

* accepted input in a supported format (numpy bytes array)
* decodes it to a list of strings
* runs inference with the pre-trained BioNeMo model (for example, ESM1)
* converts output to a supported format
* and sends it back to the client

Mark this callable with the `@batch` decorator from PyTriton. This decorator converts the input request into a more suitable format that can be directly passed to the model (refer to more details on batch decorator in the [PyTrtion documentation](https://github.com/triton-inference-server/pytriton/blob/main/docs/decorators.md#batch)).

An example inference callable is provided below:

```python
import numpy as np
from pytriton.decorators import batch

@batch
def _infer_fn(sequences: np.ndarray):

    sequences = np.char.decode(sequences.astype("bytes"), "utf-8")
    sequences = sequences.squeeze(1).tolist()

    embedding = MODEL.seq_to_embeddings(sequences)

    response = {
        "embedding":  embedding.cpu().numpy(),
    }

    return response
```

where `MODEL` is an instance of `ESM1nvInference` class.

Now, define and start the server:

```python
from pytriton.model_config import Tensor
from pytriton.triton import Triton

with Triton() as triton:
    triton.bind(
        model_name="ESM1",
        infer_func=_infer_fn,
        inputs=[
                Tensor(name="sequences", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embedding", dtype=np.float32, shape=(-1,)),
            ],
    )

    triton.serve()
```

:::{note}
The expected shapes for the inputs and outputs are defined in `_infer_fn` (without the batch dimension), where -1 denotes a dynamic size.
:::

When the server is running, use the client to perform a query:

```python
from pytriton.client import ModelClient

with ModelClient("localhost", "ESM1") as client:
    result_dict = client.infer_batch(sequences)
```

## Predefined Server-Client Scripts

BioNeMo comes with a set of example scripts for inference with PyTriton.

These scripts utilize hydra configs available in the `bionemo/examples/` directory to set up the model for inference. The three BioNeMo models are supported: MegaMolBART, ESM1 and ProtT5; and two inference modes: **Sequence to Embedding** (for all 3 models) and **Sampling** (for MegaMolBART)

### Starting the Server

To start the Triton Server (and send it to the background) run:

```bash
python bionemo/examples/triton/<embeddings,sampling>_server.py --config-path </path/to/dir/with/inference/config> &
```

You can start server for generating embeddings with MegaMolBART:

```bash
python bionemo/examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/molecule/megamolbart/conf &
```

For ESM1:

```bash
python bionemo/examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/protein/esm1nv/conf/ &
```

And for ProtT5:

```bash
python bionemo/examples/triton/embeddings_server.py --config-path /workspace/bionemo/examples/protein/prott5nv/conf/ &
```

Alternatively start server for sampling new sequences with MegaMolBART:

```bash
python bionemo/examples/triton/sampling_server.py --config-path /workspace/bionemo/examples/molecule/megamolbart/conf &
```

The scripts use `hydra` and load model configuration from `infer.yaml` present in the specified config directory, so custom configurations can be provided by specifying a different yaml file or overriding particular arguments.

### Querying the Server

Regardless of the model and server type selected, you can send a query with the general-purpose client script:

```bash
python bionemo/examples/triton/client.py --sequences SEQUENCES [SEQUENCES ...] [--output_path OUTPUT_PATH]
```

The only required argument is `--sequences`, which should be a whitespace separated list of SMILES (for molecules) or FASTA (for protein) sequences.
Optionally, you can also provide `--output_path` flag to save the pickled results in a specified file. If this argument is not given, the results will be printed.

For example:

```bash
python bionemo/examples/triton/client.py --sequences "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "c1ccccc1CC(O)=O"
```

if MegaMolBART was loaded by the server script, or:

```bash
python bionemo/examples/triton/client.py --sequences "MTADAHWIPVPTNVAYDALNPGAPGTLAFAAANGWQHHPLVTVQPLPGVVFRDAAGRSRFTQRAGD"
```
for one of the protein models, ESM1 or ProtT5.


To interact with a server set up in a different way (refer to [Extending These Examples](#extending-these-examples)), provide the url with `--url` flag (defaults to `"localhost"`).


### Closing the Server

The server can be distbled by bringing it from the background (`fg`) and killing it with `ctrl+c`.


## Extending These Examples

1. Inference callable can contain any Python code. Extend the existing example with a custom post-processing or implement a more complex, multi-step inference pipeline.

2. For more control over inference parameters (for example, sampling strategy for MegaMolBART), they can be exposed to the user. Remember to represent all inputs and outputs to the inference callable as numpy arrays.

3. Use one of the provided components (server or client) alone - they are fully compatible with native solutions for Triton Inference Server.
* Query the server with a different tool, like you would do with any other Triton instance
* Yse the client to interact with any Triton server, not necessarily set up with PyTriton

1. Finally, PyTriton provides variety of options to customize the server. Refer to the [PyTriton documentation](https://triton-inference-server.github.io/pytriton/0.1.5/).
