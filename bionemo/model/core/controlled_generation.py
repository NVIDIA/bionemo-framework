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


from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from bionemo.model.core.infer import BaseEncoderDecoderInference


class ControlledGenerationPerceiverEncoderInferenceWrapper:
    def __init__(
        self,
        inference_model: BaseEncoderDecoderInference,
        enforce_perceiver: bool = True,
        hidden_steps: Optional[int] = None,
        to_cpu: bool = True,
        batch_second: bool = True,
        additional_decode_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwarg_overrides: Optional[Dict[str, Any]] = None,
        sampling_method: str = "greedy-search",
    ):
        """A wrapper that you can put over your inference model (if it uses a perceiver encoder) to get a new model that's compatible with
        the IO expectations of the bionemo-controlled-generation package.

        Args:
            inference_model (BaseEncoderDecoderInference): An inference model that has the `seq_to_hiddens` and `hiddens_to_seq` functions defined.
            enforce_perceiver (bool, optional): Generally this method supports perceivers with a fixed size latent encoding. As a hack for testing/research,
                you can take a regular encoder and flatten the output. If you want to do this hack, set this to False and supply the number of
                position encodings in the batch as `hidden_steps`. Defaults to True.
            hidden_steps (Optional[int], optional): As mentioned in the previous arg, only set this if you are testing a model that does not
                have a perceiver encoder. Defaults to None.
            to_cpu (bool, optional): The controlled-generation code runs on CPU, so set this to True if you are using that regular example. Defaults to True.
            batch_second (bool, optional): controlled-generation currently uses (Position, Batch, Model_D) io rather than the Framework default of
                (Batch, Position, Model_D). Set this to False if you test with a different package that conforms to the same shape definitions as the framework.
                Otherwise leave this as True if you want compatibility with the controlled-generation package. Defaults to True.
            additional_decode_kwargs (Optional[Dict[str, Any]], optional): Any additional args to pass along to the decoder, for example limiting sampled molecules
                to a specific length, etc. Defaults to None.
            sampling_kwarg_overrides (_type_, optional): Any sampler kwarg overrides to the NeMo `decode` function.
                Defaults to {"beam_size": 5, "keep_only_best_tokens": True, "return_scores": False}.
            sampling_method (str, optional): Which sampling strategy NeMo `decode` should use. Defaults to "beam-search".
        """
        super().__init__()
        self.inference_model = inference_model
        self.encoder_cfg = self.inference_model.model.enc_dec_model.encoder_cfg
        self.decoder_cfg = self.inference_model.model.enc_dec_model.decoder_cfg
        self.to_cpu = to_cpu
        self.batch_second = batch_second
        self.additional_decode_kwargs = additional_decode_kwargs
        self.sampling_kwarg_overrides = sampling_kwarg_overrides
        self.sampling_method = sampling_method
        if enforce_perceiver:
            assert self.encoder_cfg.arch == "perceiver"
        # First set hidden_steps. this is either the K term of the perceiver architecture in MolMIM ()
        if hidden_steps is not None:
            self.hidden_steps: int = hidden_steps
            if self.encoder_cfg.arch == "perceiver":
                assert "hidden_steps" in self.encoder_cfg
            if "hidden_steps" in self.encoder_cfg:
                assert self.encoder_cfg.hidden_steps == hidden_steps
        else:
            assert "hidden_steps" in self.encoder_cfg
            self.hidden_steps = self.encoder_cfg.hidden_steps
        self.hidden_size: int = int(self.encoder_cfg.hidden_size)  # sometimes called D
        # This is what we need to flatten the embedding dimension to for compatibility with bionemo-controlled-generation
        self.flattened_hidden_dimension: int = self.hidden_steps * self.hidden_size

    @property
    def device(self) -> torch.device:
        """Return the device of the wrapped inference model."""
        return self.inference_model.device

    def encode(self, sequences: List[str]) -> torch.Tensor:
        """Encodes a list of input sequences, returning the encoder hiddens output for each input sequence. See the
            init options `batch_second` and `to_cpu` for controlling the device and shape of these outputs.

        Args:
            sequences (List[str]): Input sequences, for example SMILE strings.

        Returns:
            torch.Tensor: hiddens representation from `seq_to_hiddens`
        """
        with torch.no_grad():
            hiddens, _ = self.inference_model.seq_to_hiddens(sequences=sequences)

        if self.hidden_steps > 1:
            bs = hiddens.shape[0]
            d_model = hiddens.shape[2]
            hidden_steps = hiddens.shape[1]
            assert hidden_steps == self.hidden_steps
            hiddens = hiddens.reshape(bs, 1, d_model * hidden_steps)
        else:
            # Should be batch x hidden_steps x hidden_size
            assert hiddens.shape[1] == 1
        if self.to_cpu:
            hiddens = hiddens.cpu()
        if self.batch_second:
            hiddens = hiddens.transpose(0, 1)
        return hiddens

    def decode(
        self,
        hiddens: Union[torch.Tensor, np.array],
    ) -> List[str]:
        """Samples sequences from the provided hiddens. See the init options `sampling_kwarg_overrides`,
            `additional_decode_kwargs` and `sampling_method` for controlling this decoding process.

        Args:
            hiddens (Union[torch.Tensor, np.array]): Hiddens to decode and sample from into sequences.
        Returns:
            List[str]: List of sampled sesquences.
        """
        with torch.no_grad():
            hiddens = torch.tensor(hiddens, device=self.device)
            if self.batch_second:
                # Put back to batch first
                hiddens = hiddens.transpose(0, 1)
            assert hiddens.shape[0] >= 1
            assert hiddens.shape[1] == 1  # batch first again
            assert hiddens.shape[2] == self.flattened_hidden_dimension
            if self.hidden_steps > 1:
                bs = hiddens.shape[0]
                hiddens = hiddens.reshape(bs, self.hidden_steps, self.hidden_size)
            dummy_mask = torch.zeros(size=hiddens.shape[:-1], dtype=torch.bool, device=hiddens.device)
            additional_kwargs: Dict[str, Any] = {}
            if self.additional_decode_kwargs is not None:
                additional_kwargs.update(**self.additional_decode_kwargs)
            if self.sampling_kwarg_overrides is not None:
                additional_kwargs["sampling_kwargs"] = self.sampling_kwarg_overrides
            return self.inference_model.hiddens_to_seq(
                hiddens, enc_mask=dummy_mask, sampling_method=self.sampling_method, **additional_kwargs
            )

    def num_latent_dims(self) -> int:
        # This should be the entire flattened hiddens size. For example if you use a perceiver it would be
        #   encoder.hidden_steps * encoder.hidden_size
        return self.flattened_hidden_dimension
