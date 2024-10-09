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


from pathlib import Path
from typing import Callable, List, Optional, Sequence

import click
import model_navigator
from model_navigator.package.package import Package
from nemo.utils import logging
from omegaconf import DictConfig
from pytriton.triton import Triton

from bionemo.model.core.infer import M
from bionemo.triton import decodes
from bionemo.triton.controlled_generations import triton_controlled_generation_infer_fn
from bionemo.triton.embeddings import nav_triton_embedding_infer_fn, triton_embedding_infer_fn
from bionemo.triton.hiddens import triton_hidden_infer_fn
from bionemo.triton.samplings import triton_sampling_infer_fn
from bionemo.triton.types_constants import (
    BIONEMO_MODEL,
    DECODES,
    EMBEDDINGS,
    GENERATED,
    HIDDENS,
    SAMPLINGS,
    SEQUENCES,
    StrInferFn,
)
from bionemo.triton.utils import (
    load_model_for_inference,
    load_nav_package_for_model,
    load_navigated_model_for_inference,
    model_navigator_filepath,
    register_controlled_generation_infer_fn,
    register_masked_decode_infer_fn,
    register_str_embedding_infer_fn,
)
from bionemo.utils.hydra import load_model_config


__all_: Sequence[str] = (
    "main",
    "bind_embedding",
    "bind_sampling",
    "bind_decode",
    "bind_hidden",
)


@click.command()
@click.option("--config-path", required=True, help="Path to Hydra config directory where configuration date lives.")
@click.option(
    "--config-name",
    default="infer.yaml",
    show_default=True,
    required=True,
    help="Name of YAML config file in --config-path to load from.",
)
@click.option(
    "--nav", is_flag=True, help="If present, load runtime optimized with model navigator. Requires export beforehand."
)
@click.option(
    "--embedding",
    type=str,
    help="Starts Triton model name for sequence -> embedding inference. Only active is present.",
)
@click.option(
    "--sampling", type=str, help="Triton model name for sampling inference encoding. Only active is present."
)
@click.option(
    "--decode",
    type=str,
    help="Triton model name for hidden state -> original sequence decoding. Only active is present.",
)
@click.option(
    "--hidden", type=str, help="Triton model name for sequence -> hidden state inference. Only active is present."
)
@click.option(
    "--controlled-generation",
    type=str,
    help="Triton model name for controlled generation inference. Only active is present.",
)
@click.option(
    "--override-name",
    is_flag=True,
    help="If present, must also only have one Triton model specified: will override the name to 'bionemo_moddel'.",
)
def entrypoint(
    config_path: str,
    config_name: str,
    nav: bool,
    embedding: Optional[str],
    sampling: Optional[str],
    decode: Optional[str],
    hidden: Optional[str],
    controlled_generation: Optional[str],
    override_name: bool = False,
) -> None:  # pragma: no cover
    def take(x: Optional[str], default: str) -> Optional[str]:
        if x is not None:
            if len(x) == 0:
                return default
            return x
        return None

    if override_name:
        main(
            config_path,
            config_name,
            nav,
            embedding=take(embedding, EMBEDDINGS),
            sampling=take(sampling, SAMPLINGS),
            decode=take(decode, DECODES),
            hidden=take(hidden, HIDDENS),
            controlled_generation=take(controlled_generation, "generated"),
            allow_override_name=True,
        )
    else:
        main(
            config_path,
            config_name,
            nav,
            embedding=embedding,
            sampling=sampling,
            decode=decode,
            hidden=hidden,
            controlled_generation=controlled_generation,
            allow_override_name=False,
        )


def main(
    config_path: str,
    config_name: str,
    nav: bool,
    embedding: Optional[str] = None,
    sampling: Optional[str] = None,
    decode: Optional[str] = None,
    hidden: Optional[str] = None,
    controlled_generation: Optional[str] = None,
    allow_override_name: bool = False,
) -> None:
    config_file = Path(config_path) / config_name
    print(f"Loading config from:                        {str(config_file)}")
    print(f"Using model navigator runtimes?:            {nav}")
    print(f"Starting embedding inference?:              {embedding}")
    print(f"Starting sampling inference?:               {sampling}")
    print(f"Starting decode inference?:                 {decode}")
    print(f"Starting hidden inference?:                 {hidden}")
    print(f"Starting controlled generation inference?:  {controlled_generation}")
    print(f"Override name?:                             {allow_override_name}")
    print("-" * 80)

    models_to_enable: List[bool] = [
        x is not None and len(x) > 0 for x in [embedding, sampling, decode, hidden, controlled_generation]
    ]
    if not any(models_to_enable):
        raise ValueError(
            f"Need at least one of --{embedding=}, --{sampling=}, --{decode=}, --{hidden=}, --{controlled_generation=}"
        )

    override_model_name = BIONEMO_MODEL if allow_override_name and sum(models_to_enable) == 1 else None

    if not config_file.is_file():
        raise ValueError(
            f"--config-path={config_path} and --config-name={config_name} do not exist at {config_file.absolute()}"
        )

    cfg: DictConfig = load_model_config(config_name=config_name, config_path=config_path, logger=logging)

    if sum(models_to_enable) > 1:
        print(f"Loading model once and re-using for the {sum(models_to_enable)} .bind() calls.")
        maybe_model: Optional[M] = load_model_for_inference(cfg)
    else:
        maybe_model = None

    with Triton() as triton:
        for maybe_triton_model_name, bind_fn in [
            (embedding, bind_embedding),
            (sampling, bind_sampling),
            (decode, bind_decode),
            (hidden, bind_hidden),
            (controlled_generation, bind_controlled_generation),
        ]:
            if maybe_triton_model_name is not None:
                bind_fn(triton, cfg, maybe_model, nav, override_model_name or maybe_triton_model_name)
        triton.serve()


def bind_embedding(
    triton: Triton, cfg: DictConfig, preloaded_model: Optional[M], nav: bool, triton_model_name: str
) -> None:
    in_name = SEQUENCES
    out = EMBEDDINGS
    print(f"Binding Triton **EMBEDDING** inference under '{triton_model_name}' ({in_name} -> {out})")
    infer_fn = _make_infer_fn(nav, cfg, preloaded_model, triton_embedding_infer_fn, nav_triton_embedding_infer_fn)
    register_str_embedding_infer_fn(
        triton,
        infer_fn,
        triton_model_name,
        output_masks=False,
        in_name=in_name,
        out=out,
        verbose=True,
    )


def bind_sampling(
    triton: Triton, cfg: DictConfig, preloaded_model: Optional[M], nav: bool, triton_model_name: str
) -> None:
    in_name = SEQUENCES
    out = GENERATED
    print(f"Binding Triton **SAMPLING** inference under '{triton_model_name}' ({in_name} -> {out})")
    infer_fn = _make_infer_fn(nav, cfg, preloaded_model, triton_sampling_infer_fn, None)
    register_str_embedding_infer_fn(
        triton,
        infer_fn,
        triton_model_name,
        output_masks=False,
        in_name=in_name,
        out=out,
        out_dtype=bytes,
        out_shape=(1,),
        verbose=True,
    )


def bind_decode(
    triton: Triton,
    cfg: DictConfig,
    preloaded_model: Optional[M],
    nav: bool,
    triton_model_name: str,
    in_shape=(-1, 512),
) -> None:
    in_name = HIDDENS
    out = SEQUENCES
    print(f"Binding Triton **DECODE** inference under '{triton_model_name}' ({in_name} -> {out})")
    infer_fn = _make_infer_fn(nav, cfg, preloaded_model, decodes.triton_decode_infer_fn, None)
    register_masked_decode_infer_fn(
        triton,
        infer_fn,
        triton_model_name,
        in_name=in_name,
        in_shape=in_shape,
        out=out,
        verbose=True,
    )


def bind_hidden(
    triton: Triton, cfg: DictConfig, preloaded_model: Optional[M], nav: bool, triton_model_name: str
) -> None:
    in_name = SEQUENCES
    out = HIDDENS
    print(f"Binding Triton **HIDDEN** inference under '{triton_model_name}' ({in_name} -> {out})")
    infer_fn = _make_infer_fn(nav, cfg, preloaded_model, triton_hidden_infer_fn, None)
    register_str_embedding_infer_fn(
        triton,
        infer_fn,
        triton_model_name,
        output_masks=True,
        in_name=in_name,
        out=out,
        verbose=True,
    )


def bind_controlled_generation(
    triton: Triton, cfg: DictConfig, preloaded_model: Optional[M], ignore_nav: bool, triton_model_name: str
) -> None:
    print(f"Binding Triton **CONTROLLED GENERATION** inference under '{triton_model_name}' (smi -> samples)")
    if ignore_nav is True:
        print("WARNING: ignoring nav! No model navigator function for controlled generation!")

    infer_fn = _make_infer_fn(
        nav=False,
        cfg=cfg,
        preloaded_model=preloaded_model,
        dev_triton_infer=triton_controlled_generation_infer_fn,
        nav_triton_infer=None,
    )

    register_controlled_generation_infer_fn(
        triton,
        infer_fn,
        triton_model_name,
        verbose=True,
    )


def _make_infer_fn(
    nav: bool,
    cfg: DictConfig,
    preloaded_model: Optional[M],
    dev_triton_infer: Callable[[M], StrInferFn],
    nav_triton_infer: Optional[Callable[[M, Package], StrInferFn]],
) -> callable:
    try:
        name: str = f"{dev_triton_infer.__module__.__name__}.{dev_triton_infer.__name__}"
    except AttributeError:
        name = dev_triton_infer.__doc__

    if nav_triton_infer is None:
        if nav:
            print(f"ERROR: model navigator support for not available for '{name}'")
        using_nav = False
    else:
        using_nav = nav

    if using_nav:
        strategy = model_navigator.MaxThroughputStrategy()
        print(f"Loading optimized runtime with model navigator for triton model: '{name}'")
        if preloaded_model is not None:
            model = preloaded_model
            runner = load_nav_package_for_model(model, model_navigator_filepath(cfg), strategy)
            runner.activate()
        else:
            model, runner = load_navigated_model_for_inference(cfg, strategy)
        infer_fn = nav_triton_infer(model, runner)
    else:
        if preloaded_model is not None:
            model = preloaded_model
        else:
            print(f"Loading base version for triton model: '{name}'")
            model = load_model_for_inference(cfg)
        infer_fn = dev_triton_infer(model)

    return infer_fn


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
