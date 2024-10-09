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


# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, List, Mapping, Optional, Sequence, TypedDict, TypeVar, cast

import numpy as np
import torch
from guided_molecule_gen.optimizer import MoleculeGenerationOptimizer, OracleCallbackSignature
from guided_molecule_gen.oracles import penalized_logp, qed, tanimoto_similarity
from pytriton.decorators import batch

from bionemo.model.core.controlled_generation import ControlledGenerationPerceiverEncoderInferenceWrapper
from bionemo.model.core.infer import M
from bionemo.triton.utils import (
    decode_single,
    decode_str_batch,
    decode_str_single,
    encode_str_batch_rows,
    encode_str_single,
)


__all__: Sequence[str] = (
    "ControlledGenerationInferFn",
    "ControlledGenerationResponse",
    "triton_controlled_generation_infer_fn",
)


T = TypeVar("T")


def add_jitter(embedding: torch.Tensor, radius: float, cnt: int) -> list[torch.Tensor]:
    permuted_emb = embedding
    distorteds = []
    for _ in range(cnt):
        noise = torch.normal(0, radius, permuted_emb.shape).to(permuted_emb.device)
        distorted = noise + permuted_emb
        distorteds.append(distorted)

    return distorteds


PROPERTIES: Mapping[str, OracleCallbackSignature] = {
    "QED": qed,
    "plogP": penalized_logp,
}

SCALING_FACTORS: Mapping[str, float] = {
    "QED": 0.9,
    "plogP": 20.0,
}

SCALED_RADIUS: float = 1.0


class ControlledGenerationRequestBase(TypedDict):
    algorithm: np.ndarray  # 1x1, str
    smi: np.ndarray  # B x <string length>, str
    num_molecules: np.ndarray  # 1x1, int


class ControlledGenerationRequest(ControlledGenerationRequestBase, total=False):
    property_name: np.ndarray  # 1x1, str
    minimize: np.ndarray  # 1x1, bool
    min_similarity: np.ndarray  # 1x1, float
    particles: np.ndarray  # 1x1, int
    iterations: np.ndarray  # 1x1, int
    radius: np.ndarray  # 1x1, float


class ControlledGenerationResponse(TypedDict):
    samples: np.ndarray  # B x N x <string length>, list[str] (N=iterations)
    scores: np.ndarray  # B x N x 1, list[float] (N=iterations)
    score_type: np.ndarray  # 1x1, str


ControlledGenerationInferFn = Callable[[ControlledGenerationRequest], ControlledGenerationResponse]


def create_oracle(
    scoring_fun: OracleCallbackSignature,
    scaling_factor: float,
    sim_threshold: float = 0.4,
    minimize: bool = False,
) -> OracleCallbackSignature:
    def oracle(smis: List[str], reference: str, **_) -> np.ndarray:
        similarities = tanimoto_similarity(smis, reference)
        similarities = np.clip(similarities / sim_threshold, a_min=None, a_max=1)
        scores = scoring_fun(smis)
        scores = scores / scaling_factor
        if minimize:
            scores = -scores
        return -1 * (similarities + scores)

    return oracle  # type: ignore


def _safe_triton_controlled_generation_infer_fn(
    model: M,
    scaled_radius: float = SCALED_RADIUS,
    properties: Optional[Mapping[str, OracleCallbackSignature]] = None,
    scaling_factors: Optional[Mapping[str, float]] = None,
) -> ControlledGenerationInferFn:
    """Produces a PyTriton-compatible inference function that uses a bionemo encoder-decoder model to generate new
    sequences given an input sequence as a starting point."""

    properties, scaling_factors = _init_properties_scaling_factors(properties, scaling_factors)

    # assumes input is single paramater (i.e. 1x1 array) -> for str, int, float, bool
    # if list, assume list[str] *only*
    def _get(
        request: dict[str, np.ndarray],
        fieldname: str,
        t: type[list | str | float | int | bool],
        test: tuple[Callable[[T], bool], str] | None = None,
    ) -> T:
        try:
            a = request[fieldname]
        except KeyError:
            raise ValueError(f"Missing required field '{fieldname}'")

        if issubclass(t, str):
            x = np.char.decode(a, encoding="utf-8")
            y = cast(t, next(iter(x)))

        elif issubclass(t, list):
            x = decode_str_batch(a)
            y = cast(t, x)

        else:
            try:
                x = next(iter(a))
            except Exception as error:
                raise ValueError(f"Could not extract. Expecting 1x1 array but got {a.shape=}") from error
            try:
                y = cast(T, t(x))
            except Exception as error:
                raise ValueError(f"Could not convert {x} into {t}") from error
        if test is not None:
            tester, message = test
            assert tester(y), f"Property '{fieldname}' failed test with value {x}: {message}"
        return y

    class FinalResult(TypedDict):
        sample: str
        score: float

    # def infer_fn(request: ControlledGenerationRequest) -> ControlledGenerationResponse:
    def infer_fn(**request) -> ControlledGenerationResponse:
        # algorithm = _get(request, 'algorithm', str)

        print(f"{request['algorithm']=}")
        algorithm = decode_str_single(request["algorithm"])

        # TODO: can actually accept a batch of smiles strings!
        # smi = _get(request, 'smi', str)

        print(f"{request['smi']=}")
        # smis = decode_str_batch(request['smi'])
        # smis = decode_str_batch_rows(request['smi'])
        # print(f"{smis=} | {type(smis)=}")
        # smi = smis[0]
        smi = decode_str_single(request["smi"])
        print(f"{smi=}")

        num_molecules = _get(request, "num_molecules", int, test=(lambda x: x > 0, "Positive"))

        if algorithm == "CMA-ES":
            score_type: str = "cma-es"

            print(f"{request['property_name']=}")
            print(f"{request['particles']=}")
            print(f"{request['iterations']=}")
            print(f"{request['minimize']=}")
            print(f"{request['min_similarity']=}")

            property_name = decode_str_single(request["property_name"])
            particles = decode_single(request["particles"], int)
            iterations = decode_single(request["iterations"], int)
            minimize = decode_single(request["minimize"], bool)
            min_similarity = decode_single(request["min_similarity"], float)
            print(f"{property_name=}")
            print(f"{particles=}")
            print(f"{iterations=}")
            print(f"{minimize=}")
            print(f"{min_similarity=}")

            # particles = _get(request, 'particles', int)
            # iterations = _get(request, 'iterations', int, test=(lambda x: x > 0, "Positive"))
            # minimize = _get(request, 'minimize', bool)
            # min_similarity = _get(request, 'min_similarity', float, test=(lambda x: 0 <= x <= 1.0, "In [0,1]"))

            assert particles >= num_molecules, f"{particles=} must be greater than or equal to {num_molecules=}"

            prop_fn, scale_factor = _get_property_scaling_factor(property_name, properties, scaling_factors)

            oracle = create_oracle(
                prop_fn,
                scale_factor,
                sim_threshold=min_similarity,
                minimize=minimize,
            )

            # Init the controlled generation keyword args
            controlled_gen_kwargs = {
                "sampling_method": "beam-search",
                "sampling_kwarg_overrides": {"beam_size": 1, "keep_only_best_tokens": True, "return_scores": False},
            }

            # Wrap the model
            model_wrapped = ControlledGenerationPerceiverEncoderInferenceWrapper(
                model, enforce_perceiver=True, hidden_steps=1, **controlled_gen_kwargs
            )

            # Set up the CMA-ES Optimizer
            optimizer = MoleculeGenerationOptimizer(
                model_wrapped,
                oracle,
                [smi],
                popsize=particles,  # larger values will be slower but more thorough
                optimizer_args={"sigma": 0.75},
            )

            # Run the optimizer
            optimizer.optimize(iterations)

            # Collect the results
            generated = optimizer.generated_smis[0]
            scores = oracle(generated, reference=smi)

            # Subset to max sample num
            selected_mols: list[str] = [mol for _, mol in sorted(zip(scores, generated))[:num_molecules]]

            # Get the property values
            prop_values: np.ndarray = prop_fn(selected_mols)

            # Collect final generated samples & scores
            final_result: list[FinalResult] = []
            for sample, score in zip(selected_mols, prop_values):
                final_result.append({"sample": sample, "score": score})

        else:
            score_type = "tanimoto_similarity"
            radius = _get(request, "radius", float, test=(lambda x: 0 <= x <= 2, "Must be in [0,2]"))

            embedding = model.seq_to_embeddings([smi]).unsqueeze(1)

            # Add noise to embeddings
            distorted_embeddings = torch.stack(add_jitter(embedding, scaled_radius * radius, num_molecules)).squeeze(1)

            # Create mask for decoding
            mask = torch.ones(distorted_embeddings.shape[:2], dtype=bool)
            sequences = model.hiddens_to_seq(distorted_embeddings, mask)
            similarities = tanimoto_similarity(sequences, smi)

            # Collect final generated samples & scores
            final_result = []
            for sample, score in zip(sequences, similarities):
                final_result.append({"sample": sample, "score": score})

        # Sort the final results
        final_result.sort(key=lambda d: d["score"], reverse=True)
        # And turn back into tensors in name-oriented batch form
        final_samples: list[str] = []
        final_scores: list[float] = []
        for d in final_result:
            final_samples.append(d["sample"])
            final_scores.append(d["score"])

        # formulate & return response
        response: ControlledGenerationResponse = {
            "samples": encode_str_batch_rows(final_samples),
            "scores": np.array(final_scores, dtype=np.float32).reshape(1, -1),
            "score_type": encode_str_single(score_type),
        }
        for k, v in response.items():
            print(f"{k}:{v.shape} : {v}")

        # return response, len(final_result)
        return response

    return infer_fn


def triton_controlled_generation_infer_fn(
    model: M,
    scaled_radius: float = SCALED_RADIUS,
    properties: Optional[Mapping[str, OracleCallbackSignature]] = None,
    scaling_factors: Optional[Mapping[str, float]] = None,
) -> ControlledGenerationInferFn:
    return batch(_safe_triton_controlled_generation_infer_fn(model, scaled_radius, properties, scaling_factors))


def _init_properties_scaling_factors(
    properties: Optional[Mapping[str, OracleCallbackSignature]] = None,
    scaling_factors: Optional[Mapping[str, float]] = None,
) -> tuple[Mapping[str, OracleCallbackSignature], Mapping[str, float]]:
    if properties is None:
        properties = PROPERTIES

    if scaling_factors is None:
        scaling_factors = SCALING_FACTORS

    assert len(properties) > 0 and len(scaling_factors) > 0, "Must have non-empty properties & scaling factors!"
    assert (
        len(set(properties.keys()).intersection(set(scaling_factors.keys())))
        == len(properties.keys())
        == len(scaling_factors.keys())
    ), "Scaling Factors & Properties Keys don't match 1:1 !!!"

    return properties, scaling_factors


def _get_property_scaling_factor(
    property_name: str,
    properties: Mapping[str, OracleCallbackSignature],
    scaling_factors: Mapping[str, float],
) -> tuple[OracleCallbackSignature, float]:
    if property_name not in properties:
        raise ValueError(f"Unrecognized property name: '{property_name}'. Only know about {list(properties.keys())}")
    if property_name not in scaling_factors:
        raise ValueError(
            f"Unrecognized property name: '{property_name}'. Only know about {list(scaling_factors.keys())}"
        )

    return properties[property_name], scaling_factors[property_name]
