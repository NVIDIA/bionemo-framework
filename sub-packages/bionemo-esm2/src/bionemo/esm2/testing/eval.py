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


import gc

import torch
from megatron.core.transformer.module import Float16Module
from torch import Tensor

from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.model import ESM2Config


class ESM2ModelEvaluator:
    """evaluator class for ESM2 models."""

    def __init__(self, config: ESM2Config, tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer()):
        """Initialize the ESM2ModelEvaluator with the given configuration and tokenizer.

        Args:
            config (ESM2Config): Path to the pre-trained ESM2 model checkpoint.
            tokenizer (BioNeMoESMTokenizer): The tokenizer to use for tokenization.

        Example usage:
            evaluator = ESM2ModelEvaluator(config)

            # Set up the model
            evaluator.setup()

            # Perform inference iteratively
            for batch in data_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                output = evaluator.eval(input_ids, attention_mask)
                # Process output...

            # Clean up resources
            evaluator.teardown()
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = None

    def setup(self):
        """Set up the ESM2 model for inference."""
        self.model = self.config.configure_model(self.tokenizer).to("cuda").eval()
        if self.config.bf16 or self.config.fp16:
            self.model = Float16Module(self.config, self.model)

    def teardown(self):
        """Clean up resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def eval(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        """Perform inference using the ESM2 model.

        Args:
            input_ids (Tensor): Tensor of tokenized sequences.
            attention_mask (Tensor): Tensor of attention mask required for ESM2 forward.

        Returns:
            dict: A dictionary containing the outputs of the ESM2 model.
        """
        if self.model is None:
            raise RuntimeError("Model is not set up. Call setup() before evaluate().")

        with torch.no_grad():
            output = self.model(attention_mask, input_ids)
        return output
