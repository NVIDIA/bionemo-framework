# Example of PyTriton Inference

PyTriton provides a light-weight wrapper that allows you to set up the Triton Inference Server based on existing inference code. The only requirement is that inference is done by a function, that takes as an input and returns numpy arrays of supported types (numerical types or bytes).

Note that you should follow the instructions from [the general BioNeMo PyTriton inference documentation](./inference-triton-md) first before proceeding with this detailed example.

## Detailed Example with ESM-1nv

In this example, the **Sequence to Embedding** task for ESM1 will be used as an example. The solution will consist of two components - server that performs the inference, and a client that queries this server.

The `bionemo.model.protein.esm1nv.infer.ESM1nvInference` class provides `seq_to_embeddings` method that can be used for this purpose. This method requires a list of FASTA sequences as input (list of strings) and returns a torch Tensor object as a result, so a converter must be implemented.

On the client side, a function that takes list of sequences as input and converts it into a numpy bytes array must be implemented:

```python
import numpy as np

from bionemo.triton.utils import encode_str_batch


seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVL', 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLA']
sequences: np.ndarray = encode_str_batch(seqs)
```

On the server side, an inference callable that performs the following must be implemented:

- accepted input in a supported format (numpy bytes array)
- decodes it to a list of strings
- runs inference with the pre-trained BioNeMo model (for example, ESM1)
- converts output to a supported format
- and sends it back to the client

Mark this callable with the `@batch` decorator from PyTriton. This decorator converts the input request into a more suitable format that can be directly passed to the model (refer to more details on batch decorator in the [PyTrtion documentation](https://github.com/triton-inference-server/pytriton/blob/main/docs/decorators.md#batch)).

An example inference callable is provided below:

```python
from typing import Dict

import numpy as np
from pytriton.decorators import batch

from bionemo.model.protein.esm1.infer import ESM1nvInference
from bionemo.triton.utils import decode_str_batch


# have this loaded in-memory already
# or load from a config with load_model_config and and use load_model_for_inference
# from the bionemo.triton.utils module to load the model
model: ESM1nvInference = ...

@batch
def infer_fn(sequences: np.ndarray) -> Dict[str, np.ndarray]:
    seqs = decode_str_batch(sequences)
    embedding = model.seq_to_embeddings(seqs)
    response = {"embeddings": embedding.detach().cpu().numpy()}
    return response
```

NOTE: This function is alreadty defined as `triton_embedding_infer_fn` in the `bionemo.triton.embeddings` module.

Now, define and start the server:

```python
from pytriton.model_config import Tensor
from pytriton.triton import Triton


with Triton() as triton:
    triton.bind(
        model_name="ESM1",
        infer_func=infer_fn,
        inputs=[
            Tensor(name="sequences", dtype=bytes, shape=(1,)),
        ],
        outputs=[
            Tensor(name="embeddings", dtype=np.float32, shape=(-1,)),
        ],
    )
    triton.serve()
```

NOTE: See the `bionemo.triton.serve_bionemo_model` script for more serious use. This `bind` and `serve` action is perfomed in the `main` function.

The expected shapes for the inputs and outputs are defined in `infer_fn` (without the batch dimension), where -1 denotes a dynamic size.

If your shapes are incorrect, then Triton will fail to perform inference!

When using the `@batch` decorator, it is **vital** that the `infer_fn` parmaeter names align exactly with what is
deinfed for `inputs` to the `.bind()` call. These names are how PyTriton ensures that the right tensors are passed
along. Similiarly, the keys in the returned dictionary must align 1:1 with the names defined in the output tensors.

When the server is running, use the client to perform a query:

```python
from pytriton.client import ModelClient


# you may use http://localhost:8000" or just "localhost" to use the HTTP interface
# this example uses the gRPC interface, which is faster and more network-efficient
with ModelClient("grpc://localhost:8001", "ESM1") as client:
    result_dict = client.infer_batch(sequences)

embeddings = result_dict["embeddings"]
print(f"{embeddings.shape=}\n{embeddings}")"
```

## Extending These Examples

1. Inference callable can contain any Python code. Extend the existing example with a custom post-processing or implement a more complex, multi-step inference pipeline.

2. For more control over inference parameters (for example, sampling strategy for MegaMolBART), they can be exposed to the user. Remember to represent all inputs and outputs to the inference callable as numpy arrays.

3. Use one of the provided components (server or client) alone - they are fully compatible with native solutions for Triton Inference Server.

- Query the server with a different tool, like you would do with any other Triton instance
- Use the client to interact with any Triton server, not necessarily set up with PyTriton

4. Finally, PyTriton provides variety of options to customize the server. Refer to the [PyTriton documentation](https://triton-inference-server.github.io/pytriton/latest).
