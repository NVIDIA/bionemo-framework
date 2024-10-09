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


from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import model_navigator
import numpy as np
import torch
from model_navigator.package.package import Package
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from bionemo.model.core.infer import M
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.triton.types_constants import (
    EMBEDDINGS,
    HIDDENS,
    MASK,
    SEQUENCES,
    NamedArrays,
    SeqsOrBatch,
    StrInferFn,
)


__all__: Sequence[str] = (
    # binding inference functions to Triton
    "register_str_embedding_infer_fn",
    "register_masked_decode_infer_fn",
    "register_controlled_generation_infer_fn",
    # loading a model's configuration object
    # loading a base encode-decode model for inference, w/ or w/o model navigator support
    "load_model_for_inference",
    "load_nav_package_for_model",
    "load_navigated_model_for_inference",
    "model_navigator_filepath",
    # encoding & decoding strings as numpy arrays
    "decode_str_batch",
    "encode_str_batch",
)


def register_str_embedding_infer_fn(
    triton: Triton,
    infer_fn: StrInferFn,
    triton_model_name: str,
    *,
    output_masks: bool,
    in_name: str = SEQUENCES,
    out: str = EMBEDDINGS,
    out_dtype: np.dtype = np.float32,
    out_shape: Tuple[int] = (-1,),
    max_batch_size: int = 10,
    verbose: bool = True,
) -> None:
    """Binds a string-to-embedding batch inference function to Triton.

    The inference function's input is assumed to be a batch of strings, encoded as utf-8 bytes in a dense numpy array.
    The output will be a single named dense numpy array of floats, corresponding 1:1 to the batch indicies.

    The `in_name` and `out` names correspond to these input and output tensors, respectively. The output embeddings
    may be customized on their underlying datatype and shape. The input, however, is fixed.
    """
    if verbose:
        logging.info(
            f"Binding new inference function in Triton under '{triton_model_name}'\n"
            f"Inference fn. docs: {infer_fn.__doc__} ({type(infer_fn)=})\n"
            f"Input, batched, utf8 text, named:  {in_name}\n"
            f"Output, batched, f32 embeddings:   {out}"
        )
    if any((len(x) == 0 for x in [out, in_name, triton_model_name])):
        raise ValueError(f"Need non-empty values, not {in_name=}, {out=}, {triton_model_name=}")
    if max_batch_size < 1:
        raise ValueError(f"Need positive {max_batch_size=}")

    outputs = [
        Tensor(name=out, dtype=out_dtype, shape=out_shape),
    ]
    if output_masks:
        outputs.append(Tensor(name=MASK, dtype=np.bool_, shape=out_shape))

    triton.bind(
        model_name=triton_model_name,
        infer_func=infer_fn,
        inputs=[Tensor(name=in_name, dtype=bytes, shape=(1,))],
        outputs=outputs,
        config=ModelConfig(max_batch_size=max_batch_size),
    )


def register_masked_decode_infer_fn(
    triton: Triton,
    infer_fn: Callable[[NamedArrays], NamedArrays],
    triton_model_name: str,
    *,
    in_name: str = HIDDENS,
    in_dtype: np.dtype = np.float32,
    in_shape: Tuple[int] = (-1,),
    out: str = SEQUENCES,
    max_batch_size: int = 10,
    verbose: bool = True,
) -> None:
    """Binds a Triton inference function that decodes embeddings, with an input mask, to a batch of strings.

    The inference function's input is a dense batch of embeddings and a mask. While the mask's tensor name is fixed,
    the embeddings input tensor name can be customized via the `in_name` parameter. Additionally, the datatype for
    this tensor is provided by `in_dtype`. The `in_shape` applies to both input tensors: the mask and the embedding.
    Note that the "mask" is a binary array: its datatype is `bool`.

    The expected output of the inference function is a batch of strings, encoded as utf-8 bytes in a dense numpy array.
    The `out` parameter customizes the name of this output tensor.
    """
    if verbose:
        logging.info(
            f"Binding new inference function in Triton under '{triton_model_name}'\n"
            f"Inference fn. docs: {infer_fn.__doc__} ({type(infer_fn)=})\n"
            f"Input, batched, f32 embedding, named:  {in_name}\n"
            f"Output, batched, utf8 text, named:     {out}"
        )
    if any((len(x) == 0 for x in [out, in_name, triton_model_name])):
        raise ValueError(f"Need non-empty values, not {in_name=}, {out=}, {triton_model_name=}")
    if max_batch_size < 1:
        raise ValueError(f"Need positive {max_batch_size=}")

    triton.bind(
        model_name=triton_model_name,
        infer_func=infer_fn,
        inputs=[
            Tensor(name=in_name, dtype=in_dtype, shape=in_shape),
            Tensor(name=MASK, dtype=np.bool_, shape=(-1,)),
        ],
        outputs=[
            Tensor(name=out, dtype=bytes, shape=(1,)),
        ],
        config=ModelConfig(max_batch_size=max_batch_size),
    )


def register_controlled_generation_infer_fn(
    triton: Triton,
    infer_fn: Callable[[NamedArrays], NamedArrays],
    triton_model_name: str,
    *,
    # intput
    in_smi: str = "smi",
    in_algorithm: str = "algorithm",
    in_num_molecules: str = "num_molecules",
    in_property_name: str = "property_name",
    in_minimize: str = "minimize",
    in_min_similarity: str = "min_similarity",
    in_particles: str = "particles",
    in_iterations: str = "iterations",
    in_radius: str = "radius",
    # output
    out_samples: str = "samples",
    out_scores: str = "scores",
    out_score_type: str = "score_type",
    # triton
    max_batch_size: int = 1,
    verbose: bool = True,
) -> None:
    if verbose:
        logging.info(
            f"Binding new inference function in Triton under '{triton_model_name}'\n"
            f"Inference fn. docs: {infer_fn.__doc__} ({type(infer_fn)=})\n"
            f"Input, batched, single utf8 text, named:    {in_smi}\n"
            f"Input, single, utf8 text:                   {in_algorithm}\n"
            f"Input, single, int:                         {in_num_molecules}\n"
            f"Input, single, utf8 text:                   {in_property_name}\n"
            f"Input, single, bool:                        {in_minimize}\n"
            f"Input, single, float:                       {in_min_similarity}\n"
            f"Input, single, int:                         {in_particles}\n"
            f"Input, single, int:                         {in_iterations}\n"
            f"Input, single, float:                       {in_radius}\n"
            f"Output, batched, single utf8 text, named:   {out_score_type}\n"
            f"Output, batched, floats, named:             {out_scores}\n"
            f"Output, batched, multiple utf8 text, named: {out_samples}"
        )
    if any((len(x) == 0 for x in [out_samples, in_smi, triton_model_name])):
        raise ValueError(f"Need non-empty values, not {in_smi=}, {out_samples=}, {triton_model_name=}")
    if max_batch_size < 1:
        raise ValueError(f"Need positive {max_batch_size=}")

    triton.bind(
        model_name=triton_model_name,
        infer_func=infer_fn,
        inputs=[
            # Tensor(name=in_smi, dtype=bytes, shape=(-1,)),
            Tensor(name=in_smi, dtype=bytes, shape=(1,)),
            Tensor(name=in_algorithm, dtype=bytes, shape=(1,)),
            Tensor(name=in_num_molecules, dtype=np.int32, shape=(1,)),
            Tensor(name=in_property_name, dtype=bytes, shape=(1,)),
            Tensor(name=in_minimize, dtype=np.bool_, shape=(1,)),
            Tensor(name=in_min_similarity, dtype=np.float32, shape=(1,)),
            Tensor(name=in_particles, dtype=np.int32, shape=(1,)),
            Tensor(name=in_iterations, dtype=np.int32, shape=(1,)),
            Tensor(name=in_radius, dtype=np.float32, shape=(1,)),
        ],
        outputs=[
            # Tensor(name=out_samples, dtype=bytes, shape=(-1,)),
            # Tensor(name=out_scores, dtype=np.float32, shape=(-1,)),
            Tensor(name=out_samples, dtype=bytes, shape=(1,)),
            Tensor(name=out_scores, dtype=np.float32, shape=(1,)),
            Tensor(name=out_score_type, dtype=bytes, shape=(1,)),
        ],
        config=ModelConfig(max_batch_size=1, batching=True),
    )


def load_model_for_inference(cfg: DictConfig, *, interactive: bool = False, **kwargs) -> M:
    """Loads a bionemo encoder-decoder model from a complete configuration, preparing it only for inference."""
    if not hasattr(cfg, "infer_target"):
        raise ValueError(f"Expecting configuration to have an `infer_target` attribute. Invalid config: {cfg=}")
    infer_class = import_class_by_path(cfg.infer_target)

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
        interactive=interactive,
    )

    model = infer_class(cfg, interactive=interactive, **kwargs)
    model.freeze()

    return model


def load_navigated_model_for_inference(cfg: DictConfig, strategy: RuntimeSearchStrategy) -> Tuple[M, Package]:
    """Like `load_model_for_inference`, but also uses `load_nav_package_for_model` to load model's optimized runtime."""
    nav_path = model_navigator_filepath(cfg)

    model = load_model_for_inference(cfg)

    runner = load_nav_package_for_model(model, nav_path, strategy)
    runner.activate()

    return model, runner


def model_navigator_filepath(cfg: DictConfig) -> str:
    """Convention for obtaining the Model Navigator artifact path from a large config object."""
    if hasattr(cfg, "nav_path"):
        return cfg.nav_path
    else:
        return f"{cfg.model.downstream_task.restore_from_path[: -len('.nemo')]}.nav"


class NavWrapper(torch.nn.Module, Generic[M]):
    def __init__(self, inferer: M) -> None:
        """WARNING: side effects: moves the inferer to the cuda device and calls its _prepare_for_export method."""
        super().__init__()
        self.model = inferer.model.cuda()
        self.prepare_for_export()

    def prepare_for_export(self) -> None:
        self.model._prepare_for_export()

    def forward(self, tokens_enc: torch.Tensor, enc_mask: torch.Tensor):
        """Alias for the model's encoder forward-pass."""
        return self.model.encode(tokens_enc=tokens_enc, enc_mask=enc_mask)


def load_nav_package_for_model(
    model: M,
    nav_path: str,
    strategy: RuntimeSearchStrategy,
) -> Package:
    """Loads the previously exported model navigator optimized runtime for the given model."""
    nav_model = NavWrapper(model)

    package = model_navigator.package.load(nav_path)
    package.load_source_model(nav_model)

    runner = package.get_runner(strategy=strategy)

    return runner


def decode_str_batch(sequences: np.ndarray) -> List[str]:
    """Decodes a utf8 byte array into a batch of string sequences."""
    seqs = np.char.decode(sequences.astype("bytes"), "utf-8")  # N x 1
    seqs = seqs.squeeze(1)  # N
    return seqs.tolist()


def encode_str_batch(sequences: SeqsOrBatch) -> np.ndarray:
    """Encodes a batch of string sequences as a dense utf8 byte array."""
    if len(sequences) == 0:
        raise ValueError("Need at least one sequence to encode")
    if isinstance(sequences[0], str):
        # List[str] case
        seqs: List[List[str]] = [[s] for s in sequences]
    else:
        # assume List[List[str]] case
        seqs = sequences
    return np.char.encode(np.array(seqs), encoding="utf-8")


def decode_str_batch_rows(sequences: np.ndarray) -> List[str]:
    """Decodes a utf8 byte array into a batch of string sequences."""
    seqs = np.char.decode(sequences.astype("bytes"), "utf-8")  # 1 x N
    seqs = seqs.squeeze(0)
    return seqs.tolist()


def encode_str_batch_rows(sequences: SeqsOrBatch) -> np.ndarray:
    """Encodes a batch of string sequences as a dense utf8 byte array."""
    encoded_colwise = encode_str_batch(sequences)
    return encoded_colwise.reshape(1, -1)


def decode_str_single(single_string: np.ndarray) -> str:
    x = np.char.decode(single_string.astype("bytes"), encoding="utf-8")
    assert len(x) == 1
    return x.reshape(-1)[0]


def encode_str_single(single_string: str) -> np.ndarray:
    return np.char.encode(np.array([single_string]), encoding="utf-8").reshape(1, 1)


def encode_single(val: float | int | bool) -> np.ndarray:
    if isinstance(val, bool):
        dtype = np.bool_
    elif isinstance(val, int):
        dtype = np.int32
    elif isinstance(val, float):
        dtype = np.float32
    else:
        raise ValueError(f"Unknown type for value: {type(val)} ({val})")
    return np.array([[val]], dtype=dtype)


N = TypeVar("N", float, int, bool)


def decode_single(val: np.ndarray, t: type[N]) -> N:
    assert len(val) == 1
    assert val.shape == (1, 1)

    x = val[0][0]

    if issubclass(t, bool):
        return bool(x)
    elif issubclass(t, int):
        return int(x)
    elif issubclass(t, float):
        return float(x)
    else:
        raise ValueError(f"Unexpected type: {t}")


def read_bytes_from_filepath(filepath: Union[str, Path]) -> bytes:
    """Reads file content in bytes"""
    with open(str(filepath), "rb") as rb:
        return rb.read()


def read_bytes_from_filepaths(*filepaths) -> List[bytes]:
    """Reads file contents in bytes from all filepaths"""
    return [read_bytes_from_filepath(f) for f in filepaths]


def write_tempfiles_from_str_list(
    strings: List[str], exit_stack: Optional[ExitStack] = None, **tempfile_kwargs
) -> List[NamedTemporaryFile]:
    """Writes and returns list of strings to temporary files

    Args:
        strings (List[str]): list of strings to be written to NamedTemporaryFile
        exit_stack (contextlib.ExitStack): exit stack for automated NamedTemporaryFile clean up
        **tempfile_kwargs: kwargs for NamedTemporaryFile

    Returns:
        List[NamedTemporaryFile]: list of temporary file objects for each input string
    """
    temp_files = []
    for string in strings:
        if exit_stack is None:
            temp_file = exit_stack.enter_context(NamedTemporaryFile(**tempfile_kwargs))
        else:
            temp_file = NamedTemporaryFile(**tempfile_kwargs)

        with open(temp_file.name, "w") as fopen:
            fopen.write(string)

        temp_files.append(temp_file)

    return temp_files
