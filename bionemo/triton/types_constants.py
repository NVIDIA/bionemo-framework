# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Dict, List, Literal, Sequence, Tuple, TypeVar, Union

import numpy as np


__all__: Sequence[str] = (
    # types
    "StrArray",
    "NamedArrays",
    "StrInferFn",
    "SeqsOrBatch",
    # constants
    "SEQUENCES",
    "EMBEDDINGS",
    "DECODES",
    "HIDDENS",
    "SAMPLINGS",
    "BIONEMO_MODEL",
    "GENERATED",
    "MASK",
)

SEQUENCES: Literal["sequences"] = "sequences"

EMBEDDINGS: Literal["embeddings"] = "embeddings"

DECODES: Literal["decodes"] = "decodes"

HIDDENS: Literal["hiddens"] = "hiddens"

SAMPLINGS: Literal["samplings"] = "samplings"

BIONEMO_MODEL: Literal["bionemo_model"] = "bionemo_model"

GENERATED: Literal["generated"] = "generated"

MASK: Literal["mask"] = "mask"

StrArray = np.ndarray[Tuple[Literal[1]], bytes]
"""A string of characters, as a UTF-8 encoded bytestring, implemented as a single-dimension NumPy array.

NOTE: Only useful for Triton inference functions use the `@batch` decorator.
"""

NamedArrays = TypeVar("NamedArrays", bound=Dict[str, np.ndarray])
"""Format of I/O for Triton inference functions: a collection of possibly batched arrays, identified by names.

Note that for a Triton model service, all of the underlying model's output tensors must be named and provided
at bind/model load time in the `outputs` configuration.

Note that this is the same for inputs: Triton model services accept a (again, possibly batched) set of named tensors
that shall be input directly into the model.

 The `@batch` decorator from PyTriton provides unwrapping support into arguments directly. I.e. for a model
 that accepts input:
 ```json
 {
   "sequences": [...],
   "mask": [...]
 }
 ```

 This is a valid inference function:
 ```python
 def infer_fn(tensors: Dict[str, np.ndarray]):
   # both will be present
   sequence: np.ndarray = tensors['sequence']
   mask: np.ndarray = tensors['mask']
   # and the values will be the input teneors
   # they will *ALL* either have a batch dimension or not,
   # depending on the way the model config was set up (i.e. was there a batch dimension in the model spec?)
   return real_infer_fn(sequence, mask)
```

Using `@batch` lets us write the following
```python
@batch
def infer_batched(sequence: np.ndarray, mask: np.ndarray):
  # sequence and mask are equivalent to our dict accesses above
  # implementation is now *EXACTLY EQUIVALENT* to `infer_fn` after this point
  return real_infer_fn(sequence, mask)
```

Thus, `@batch` is simply a convenience wrapper for not having to manually obtain the named input tensors.
"""

StrInferFn = Callable[[StrArray], NamedArrays]
"""A Triton inference function that accepts a batch of strings as input and produces named tensors as output.
"""

SeqsOrBatch = Union[List[str], List[List[str]]]
"""A batch of string-sequences.
"""
