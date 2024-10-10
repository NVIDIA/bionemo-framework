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


import copy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

import torch
from nemo.utils import logging
from omegaconf import OmegaConf
from pandas import DataFrame, Series
from pytorch_lightning.core import LightningModule

from bionemo.data.utils import pad_token_ids
from bionemo.model.utils import initialize_model_parallel, restore_model


try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )

    HAVE_APEX: bool = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__: Sequence[str] = (
    "SamplingMethods",
    "BaseEncoderInference",
    "BaseEncoderDecoderInference",
    "M",
    "SeqsOrBatch",
    "BatchOfSamples",
    "Forward",
    "Inference",
    "to_sequences_ids",
    "HAVE_APEX",
)

SeqsOrBatch = Union[List[str], List[List[str]]]
BatchOfSamples = List[SeqsOrBatch]
SamplingMethods = Literal["greedy-perturbate", "topkp-perturbate", "beam-search-perturbate"]


class IdedSeqs(TypedDict):
    id: List[int]
    sequences: List[str]


class SI(TypedDict):
    sequence_ids: torch.Tensor


class Hidden(SI):
    hiddens: torch.Tensor
    mask: torch.Tensor


class Embedding(SI):
    embeddings: torch.Tensor


class Forward(Embedding, Hidden, total=False):
    """Output of BaseEncoderDecoderInference's forward()"""


class InferenceBase(TypedDict):
    hiddens: torch.Tensor
    mask: torch.Tensor
    embeddings: torch.Tensor
    sequence: List[str]


class Inference(InferenceBase, total=False):
    id: List[int]


def centered_linspace(
    n_points: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
):
    """Like linspace, but return centered points. For example if requesting one point it will return [0.5] rather than [0]. This version
        always returns points on [0,1].

    Args:
        n_points (int): number of points
        dtype: dtype for output tensor
        device: device to send output tensor to
        requires_grad: does the output require grad.

    Returns:
        1d tensor of points from the middle of the range.
    """
    # Handle the case where only one point is requested.
    if n_points == 1:
        return torch.tensor([0.5], dtype=dtype, device=device, requires_grad=requires_grad)

    # Calculate the step size.
    step = 1 / (n_points + 1)

    # Generate points starting from 1 step to n_points steps.
    points = torch.linspace(step, 1 - step, n_points, dtype=dtype, device=device, requires_grad=requires_grad)
    return points


# FIXME: add mask for all non-special tokens (add hiddens_tokens_only)
# TODO: add model-specific prepare_for_inference and release_from_inference methods
class BaseEncoderInference(LightningModule):
    """
    Base class for inference of models with encoder.
    """

    def __init__(
        self,
        cfg,
        model: Optional[Any] = None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
        inference_batch_size_for_warmup: Optional[int] = None,
        strict_restore_from_path: bool = True,
        needs_warmup: bool = True,
    ):
        """
        Parameters:
            strict_restore_from_path: When the underlying model is restored, requires that all keys are present in the checkpoint.
            Users may want to set this to false when they are modifying the underlying model upon restoring. For example, this is done
            to use the embeddingtoken_types field for Fine tuning on a downstream task.
        """
        super().__init__()
        self.needs_warmup = needs_warmup
        self.cfg = cfg
        self._freeze_model = freeze
        self.adjust_config = adjust_config
        self.training = training
        self.interactive = interactive
        self.model = self.load_model(cfg, model=model, restore_path=restore_path, strict=strict_restore_from_path)
        self._trainer = self.model.trainer
        self.tokenizer = self.model.tokenizer

        try:
            self.k_sequence: Optional[str] = cfg.model.data.data_fields_map.sequence
        except AttributeError:
            print(
                "WARNING: Missing key for extracting the sequence in batches! The forward method call will fail! "
                "Provide a valid configuration that has this value under model.data.data_fields_map.sequence."
            )
            self.k_sequence = None

        try:
            self.k_id: Optional[str] = cfg.model.data.data_fields_map.id
        except AttributeError:
            print(
                "WARNING: Missing key for extracting the sequence IDs in batches! The forward method call will fail! "
                "Provide a valid configuration that has this value under model.data.data_fields_map.id."
            )
            self.k_id = None
        if inference_batch_size_for_warmup is None:
            inference_batch_size_for_warmup = self.model.cfg.get(
                "micro_batch_size", 2
            )  # Default value of 2, or what the user trained with if we can find it
        # FIXME: remove the need for the following line in our idempotency tests (see CDISCOVERY-2878)
        #   for example, remove it and see if you can still get tests/test_molecule_inference.py to pass.
        if self.needs_warmup:
            self.warmup(max_bs=inference_batch_size_for_warmup)

    def get_example_input_sequence(self) -> str:
        raise NotImplementedError("Please return a short example input sequence to be used for warmup.")

    def warmup(self, max_bs: int):
        self.seq_to_hiddens(sequences=[self.get_example_input_sequence()] * max_bs)  # warmup

    def load_model(
        self, cfg, model: Optional[Any] = None, restore_path: Optional[str] = None, strict: bool = True
    ) -> Any:
        """
        Loads a model from a given configuration and restore path.

        Args:
            cfg (Any): The configuration object.
            model (Optional[Any]): The model object to load. If None, a new model will be restored .
            restore_path (Optional[str]): The path to restore the model from. If None, the restore path will be obtained from the configuration.
            strict (bool): Requires all keys in the restored checkpoint to match those in the target model class.

        Returns:
            Any: The loaded model.

        Raises:
            Any: Any exceptions that occur during model loading.

        """
        # load model class from config which is required to load the .nemo file

        if model is None:
            if restore_path is None:
                restore_path = cfg.model.downstream_task.restore_from_path

            if cfg.trainer.get("accumulate_grad_batches", 1) != 1:
                logging.warning(
                    f"Reset cfg.trainer.accumulate_grad_batches from current value of {cfg.trainer.accumulate_grad_batches} to 1 to initialize encoder model"
                )  # restore NeMo model must have accumulate_grad_batches = 1
                cfg = copy.deepcopy(cfg)  # update cfg without affecting anything upstream
                cfg.trainer.accumulate_grad_batches = 1

            model = restore_model(
                restore_path=restore_path,
                cfg=cfg,
                adjust_config=self.adjust_config,
                interactive=self.interactive,
                strict=strict,
            )
        # move self to same device as loaded model
        self.to(model.device)

        # check whether the DDP is initialized
        initialize_model_parallel(model, interactive=self.interactive)

        if not self.interactive:
            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )

        # Check for PEFT flag
        if not OmegaConf.select(cfg, "model.peft.enabled"):  # skipped if peft.enabled is True
            if (
                OmegaConf.select(cfg, "model.freeze_layers") is not None
            ):  # Check whether to freeze individual layers of the model.
                num_layers = len(model.model.language_model.encoder.layers)
                freeze_layers = OmegaConf.select(cfg, "model.freeze_layers")
                if freeze_layers < 0:  # If freeze_layers < 0, freeze everything but the last `freeze_layers`
                    freeze_layers = num_layers + freeze_layers
                for i in range(freeze_layers):
                    for param in model.model.language_model.encoder.layers[i].parameters():
                        param.requires_grad = False
            elif self._freeze_model:  # only use encoder_frozen flag if not doing peft
                model.freeze()

        self.model = model

        return model

    def forward(self, sequences: Union[Series, Dict, List[str]]) -> torch.Tensor:
        """
        Computes embeddings for a list of sequences.
        Embeddings are detached from model.

        Params
            sequences: Pandas Series containing a list of strings or or a list of strings (e.g., SMILES, AA sequences, etc)

        Returns
            embeddings
        """
        ids = None
        if isinstance(sequences, Series):
            sequences = sequences.tolist()
        if isinstance(sequences, Dict):
            ids = sequences["id"]
            sequences = sequences["sequence"]
        result_dict = {}
        hiddens, enc_mask = self.seq_to_hiddens(sequences)
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)
        result_dict["embeddings"] = embeddings.float().detach().clone()
        result_dict["hiddens"] = hiddens.float().detach().clone()
        result_dict["mask"] = enc_mask.detach().clone()
        result_dict["sequence"] = sequences
        if ids is not None:
            result_dict["id"] = ids

        return result_dict

    def _tokenize(self, sequences: List[str]) -> torch.Tensor:
        """
        Model specific tokenization.
        Here <BOS> and <EOS> tokens are added for instance.

        Returns:
            token_ids (torch.Tensor, long): token ids
        """
        raise NotImplementedError("Please implement in child class")

    def tokenize(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize sequences.
        Returns:
            token_ids (torch.Tensor, long): token ids
            mask (torch.Tensor, long, float): boolean mask for padded sections
        """
        token_ids = self._tokenize(sequences=sequences)

        # Validate input sequences length
        if any(len(t) > self.model.cfg.seq_length for t in token_ids):
            raise ValueError(f"One or more sequence exceeds max length({self.model.cfg.seq_length}).")

        # Set fixed padding when dynamic padding is disabled
        if self.model.cfg.data.get("dynamic_padding") is False:
            padding_length = self.model.cfg.seq_length
        else:
            padding_length = None
        # Pad token ids (1/True = Active, 0/False = Inactive)
        token_ids, mask = pad_token_ids(
            token_ids,
            padding_value=self.tokenizer.pad_id,
            padding_len=padding_length,
            device=self.device,
        )

        return token_ids, mask

    def _detokenize(self, tokens_ids: List[List[str]]) -> List[str]:
        """
        Helper method for detokenize method
        Args:
            tokens_ids (list[list[str]]): a list with sequence of str stored as list

        Returns:
            sequences (list[str]): list of str corresponding to sequences
        """
        for i, cur_tokens_id in enumerate(tokens_ids):
            if self.tokenizer.eos_id in cur_tokens_id:
                idx = cur_tokens_id.index(self.tokenizer.eos_id)
                tokens_ids[i] = cur_tokens_id[:idx]
            else:
                tokens_ids[i] = [id for id in cur_tokens_id if id != self.tokenizer.pad_id]

        sequences = self.tokenizer.ids_to_text(tokens_ids)
        return sequences

    def detokenize(self, tokens_ids: torch.Tensor) -> SeqsOrBatch:
        """
        Detokenize a matrix of tokens into a list or nested list of sequences (i.e., strings).

        Args:
            tokens_ids (torch.Tensor, long): a matrix of token ids

        Returns:
            sequences (list[str] or list[list[str]]): list of sequences
        """
        tensor_dim = len(tokens_ids.size())
        supported_dims = [2, 3]

        tokens_ids = tokens_ids.cpu().detach().numpy().tolist()
        if tensor_dim == 2:
            # For instance, can correspond to the greedy search or topkp sampling where tensors with
            # predicted tokens ids have shape [batch_size, num_tokens_to_generate]
            return self._detokenize(tokens_ids=tokens_ids)
        elif tensor_dim == 3:
            # For instance, can correspond to the beam search with beam_size >1 where tensors with predicted tokens ids
            # have shape [batch_size, beam_size, num_tokens_to_generate]
            sequences = []
            for tokens_ids_i in tokens_ids:
                sequences.append(self._detokenize(tokens_ids=tokens_ids_i))
            return sequences
        else:
            raise ValueError(
                f"The shape of the tensor with token_ids is not supported. "
                f"Supported numbers of dims: {supported_dims}"
            )

    def seq_to_hiddens(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms Sequences into hidden state.
        This class should be implemented in a child class, since it is model specific.
        This class should return only the hidden states, without the special tokens such as
         <BOS> and <EOS> tokens, for example.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        """
        raise NotImplementedError("Please implement in child class")

    def hiddens_to_embedding(self, hidden_states: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """
        Transforms hiddens into embedding.

        Args:
            hiddens (torch.Tensor, float): hidden states
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            embeddings (torch.Tensor, float):
        """
        # compute average on active hiddens
        lengths = enc_mask.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        embeddings = torch.sum(hidden_states * enc_mask.unsqueeze(-1), dim=1) / lengths

        return embeddings

    def seq_to_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """Compute hidden-state and padding mask for sequences.

        Params
            sequences: strings, input sequences

        Returns
            embedding array
        """
        # get hiddens and mask
        hiddens, enc_mask = self.seq_to_hiddens(sequences)
        # compute embeddings from hiddens
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)

        return embeddings

    def hiddens_to_seq(self, hiddens: torch.Tensor, enc_mask: torch.Tensor, **kwargs) -> SeqsOrBatch:
        """
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            sequences (list[str] or list[list[str]]): list of sequences
        """
        raise NotImplementedError("Encoder only models do not support decoding")

    def sample(
        self,
        num_samples: Optional[int] = 10,
        return_embedding: bool = False,
        sampling_method: str = "greedy-perturbate",
        **sampling_kwarg,
    ) -> BatchOfSamples:
        """
        Sample from the model given sampling_method.

        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.
        """
        raise NotImplementedError("Encoder only models do not support decoding")


class BaseEncoderDecoderInference(BaseEncoderInference):
    """
    Base class for inference of models with encoder and decoder.
    """

    def __init__(
        self,
        cfg,
        model=None,
        freeze=True,
        restore_path=None,
        training=False,
        adjust_config=True,
        interactive: bool = False,
        inference_batch_size_for_warmup: Optional[int] = None,
    ):
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
            inference_batch_size_for_warmup=inference_batch_size_for_warmup,
        )

    def warmup(self, max_bs: int):
        # Override encoder only warmup to also warmup the decoder
        hiddens, mask = self.seq_to_hiddens(sequences=[self.get_example_input_sequence()] * max_bs)  # warmup encoder
        self.hiddens_to_seq(hiddens, mask)  # warmup decoder as well

    def hiddens_to_seq(self, hiddens: torch.Tensor, enc_mask: torch.Tensor, **kwargs) -> SeqsOrBatch:
        """
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            sequences (list[str] or list[list[str]]): list of sequences
        """
        raise NotImplementedError("Please implement in child class")

    def interpolate_samples(
        self,
        sample1: torch.Tensor,
        sample2: torch.Tensor,
        num_interpolations: int = 11,
        num_samples: int = 1,
        return_embedding: bool = False,
        hiddens_to_seq_kwargs: Dict[str, Any] = {},
        sampling_method: str = "greedy-perturbate",
        sampling_kwarg: Dict[str, Any] = {"scaled_radius": 0},
    ) -> BatchOfSamples:
        """
        Interpolate between samples.
        Args:
            sample1, sample2 (str): starting and ending samples
            num_interpolations (int): number of interpolations between each pair of samples. The mixture ratios between sample1 and sample2
                are drawn by taking evenly spaced points in [0,1]. For example if you request 1 interpolation
                then the one interpolative result would be 0.5. For two you would get [1/3, 2/3] and for 3 you would get
                [1/4, 1/2, 3/4], etc. In general odd lengths include 0.5 while even lengths do not.
            # below are sampling parameters
            num_samples (int): number of samples to generate (depends on sampling method). If you want to draw multiple
                samples from each interpolative point, then increase "scaled_radius" to something greater than zero to
                get non-deterministic draws, and increase num_samples.
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class, if desired.
                Recommended options to try are `greedy-perturbate` which should be fastest, or `beam-search-perturbate` which should
                be more accurate.
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.

        Returns:
            interpolations (list[List[str]]): list of samples from each interpolative mixture (or [(samples, embs)] if return_embedding is True)
        """
        # encode samples
        hiddens, enc_mask = self.seq_to_hiddens([sample1, sample2])
        # TODO: enable by default + warning
        # NOTE: do we want to expose this method to the user for non-bottleneck models?
        # validate that the samples are of the same length
        if enc_mask.sum(dim=1).unique().shape[0] != 1 and self.model.cfg.encoder.arch != "perceiver":
            logging.warning(
                "Interpolation may have unexpected behavior when samples have different length, unless you use a Perceiver encoder"
            )

        # interpolate between hiddens
        alpha = centered_linspace(num_interpolations, device=hiddens.device, dtype=hiddens.dtype)
        interp_hiddens = torch.lerp(hiddens[[0]], hiddens[[1]], alpha[:, None, None])

        # store interpolated hiddens in sampling_kwarg
        sampling_kwarg = sampling_kwarg.copy()
        sampling_kwarg["hiddens"] = interp_hiddens
        sampling_kwarg["enc_masks"] = enc_mask[0:1].repeat_interleave(num_interpolations, 0)

        # return samples
        return self.sample(
            num_samples=num_samples,
            return_embedding=return_embedding,
            sampling_method=sampling_method,
            hiddens_to_seq_kwargs=hiddens_to_seq_kwargs,
            **sampling_kwarg,
        )

    @property
    def supported_sampling_methods(self) -> List[str]:
        """
        Returns a list of supported sampling methods.
        Example:
            ["greedy-perturbate", "beam-search"]
        """
        return list(self.default_sampling_kwargs.keys())

    @property
    def default_sampling_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict of default sampling kwargs per sampling method.
        """
        return {
            # seqs - a list of Sequence strings to perturbate num_samples times each (could be SMILE, protein AA, etc)
            "greedy-perturbate": {"scaled_radius": 1, "seqs": [], "hiddens": None, "enc_masks": None},
            # top-k limits maximum number of token candidtaes, top-p can further reduce to accumulate top-p probability mass
            "topkp-perturbate": {
                "scaled_radius": 1,
                "seqs": [],
                "top_k": 0,
                "top_p": 0.9,
                "temperature": 1.0,
                "hiddens": None,
                "enc_masks": None,
            },
            # Beam search perturbate works the same way as `greedy-perturbate` but is more likely to find the global optima
            #  of argmax_sequence P(sequence|hiddens). The best result is returned greedily, but the search is less biased
            #  by the best tokens at prior states.
            "beam-search-perturbate": {
                "scaled_radius": 1,
                "seqs": [],
                "beam_size": 5,  # This strategy uses the same perturb hiddens then decode process, so beam_size is independent now
                "keep_only_best_tokens": True,  # Now we only return the best result, greedily, from beam-search.
                "beam_alpha": 0,
                "hiddens": None,
                "enc_masks": None,
            },
            # beam search single sample, "beam_size" is number of the best sequences at each decode iteration to be left per target
            # and "beam_alpha" is the parameter of length penalty applied to predicted sequences.
            # NOTE with this method we only draw one gaussian sample of the hidden state, and then use beam search to return the
            #  top num_samples results. You probably want to use `beam-search-perturbate`.
            "beam-search-single-sample": {
                "scaled_radius": 1,
                "seqs": [],
                "beam_size": 1,  # will be set to num_samples in code. If left as 1 this is basically greedy-search
                "beam_alpha": 0,
                "keep_only_best_tokens": False,  # this strategy returns all of the beam search internal top_k results
                "hiddens": None,
                "enc_masks": None,
            },
            # beam search perturbate sample, "beam_size" is number of the best sequences at each decode iteration to be left per target
            # and "beam_alpha" is the parameter of length penalty applied to predicted sequences.
            # NOTE with this method we only draw one gaussian sample of the hidden state, and then use beam search to return the
            #  top num_samples results. You probably want to use `beam-search-perturbate`.
            "beam-search-perturbate-sample": {
                "scaled_radius": 1,
                "seqs": [],
                "beam_size": 5,  # this is the number of top samples to return for each num_sample hidden.
                "beam_alpha": 0,
                "keep_only_best_tokens": False,  # this strategy returns all of the beam search internal top_k results
                "hiddens": None,
                "enc_masks": None,
            },
        }

    def sample(
        self,
        num_samples: int = 1,
        return_embedding: bool = False,
        sampling_method: Optional[str] = None,
        hiddens_to_seq_kwargs: Dict[str, Any] = {},
        scaled_radius: float = 1.0,
        **sampling_kwarg,
    ) -> Union[BatchOfSamples, Tuple[BatchOfSamples, torch.Tensor]]:
        """
        Sample from the model given sampling_method.

        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Options:
                - "greedy-perturbate": Sample the best sequence for each of our perturbed hiddens, using greedy-search (per token)
                    to find the best result.
                - "topkp-perturbate": Sample the best sequence for each of our perturbed hiddens, using `topkp-sampling` to
                    find the best result.
                - "beam-search-perturbate": Sample the best sequence for each of our perturbed hiddens,
                    using beam-search to find the best result.
                - "beam-search-single-sample": Sample the top num_samples sequences using beam-search (rather than single best) given our
                    single perturbed hidden.
                - "beam-search-perturbate-sample": Sample the top beam_size (default 5) sequences using beam-search
                    for each of our `num_samples` purturbed hiddens. This will return a `beam_size * num_samples` set of results.
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method. Defaults are defined in
                this class per sampling method name.
        Returns:
            Returns a structured list of sequences, the first dimension is the input batch (either hiddens or seqs that were provided). The second dimension are the
                samples per item in the batch. Note that in the case of "beam-search-perturbate-sample" there is a third dimension which are the beam_size samples
                per num_samples gaussian perturbations.
        """
        # get sampling kwargs
        default_sampling_kwarg = self.default_sampling_kwargs
        if sampling_method not in default_sampling_kwarg:
            raise ValueError(
                f"Invalid samping method {sampling_method}, supported sampling methods are {default_sampling_kwarg.keys()}"
            )

        cur_sampling_kwarg = default_sampling_kwarg[sampling_method].copy()
        cur_sampling_kwarg.update(sampling_kwarg)
        sampling_kwarg = cur_sampling_kwarg

        # execute selected sampling method
        assert (
            sampling_method in default_sampling_kwarg.keys()
        ), f"Invalid sampling method {sampling_method}, supported sampling methods are {list(default_sampling_kwarg.keys())}"

        # accept hidden states directly or via sequences
        hiddens = sampling_kwarg.pop("hiddens")
        enc_masks = sampling_kwarg.pop("enc_masks")
        seqs = sampling_kwarg.pop("seqs")
        if hiddens is not None or enc_masks is not None:
            if len(seqs):
                raise ValueError(
                    'Both hiddens and Seqs strings provided for sampling via "seqs" argument, please provide only one'
                )

            # we enforce providing both hidden states and enc_masks or neither
            if hiddens is None or enc_masks is None:
                raise ValueError("Either both hiddens and enc_masks should be provided or neither")
        else:
            if not len(seqs):
                raise ValueError('No sequences provided for sampling via "seqs" argument')

            hiddens, enc_masks = self.seq_to_hiddens(seqs)
        batch_size: int = len(hiddens)  # Make sure to do this before we explode into num_samples

        if sampling_method == "beam-search-single-sample":
            # With this strategy, the top num_samples results are returned from beam search rather than the single best one.
            sample_masks = enc_masks.clone()
            perturbed_hiddens = hiddens.clone()
        else:
            # Our normal case, we use gaussian sampling to grab `num_samples` different points in hidden space
            #  so repeat over the batch axis.
            sample_masks = enc_masks.repeat_interleave(num_samples, 0)
            perturbed_hiddens = hiddens.repeat_interleave(num_samples, 0)

        # Apply gaussian noise of the desired `scaled_radius` to the hidden states.
        perturbed_hiddens = perturbed_hiddens + (
            scaled_radius * torch.randn(perturbed_hiddens.shape).to(perturbed_hiddens.device)
        )

        # Get the sequences from our various search method options.
        if sampling_method == "greedy-perturbate":
            # Sample the best sequence for each of our perturbed hiddens, using greedy-search (per token) to find the best result.
            samples = self.hiddens_to_seq(
                perturbed_hiddens,
                sample_masks,
                sampling_method="greedy-search",
                sampling_kwargs={},
                **hiddens_to_seq_kwargs,
            )
        elif sampling_method == "topkp-perturbate":
            # Sample the best sequence for each of our perturbed hiddens, using `topkp-sampling` to find the best result.
            samples = self.hiddens_to_seq(
                perturbed_hiddens,
                sample_masks,
                sampling_method="topkp-sampling",
                sampling_kwargs=sampling_kwarg,
                **hiddens_to_seq_kwargs,
            )
        elif sampling_method == "beam-search-perturbate":
            # Sample the best sequence for each of our perturbed hiddens, using beam-search to find the best result.
            assert sampling_kwarg[
                "keep_only_best_tokens"
            ], "`beam-search-perturbate` is incompatible with returning top k results from beam search. Maybe you meant to use `beam-search-sample`?"
            samples = self.hiddens_to_seq(
                perturbed_hiddens,
                sample_masks,
                sampling_method="beam-search",
                sampling_kwargs=sampling_kwarg,
                **hiddens_to_seq_kwargs,
            )
        elif sampling_method == "beam-search-single-sample":
            # Sample the top num_hiddens sequences given our single perturbed hidden.
            if num_samples is not None:
                sampling_kwarg["beam_size"] = num_samples
            samples = self.hiddens_to_seq(
                perturbed_hiddens,
                sample_masks,
                sampling_method="beam-search",
                sampling_kwargs=sampling_kwarg,
                **hiddens_to_seq_kwargs,
            )
        elif sampling_method == "beam-search-perturbate-sample":
            # Sample the top beam_size sequences given each of our perturbed hiddens
            samples = self.hiddens_to_seq(
                perturbed_hiddens,
                sample_masks,
                sampling_method="beam-search",
                sampling_kwargs=sampling_kwarg,
                **hiddens_to_seq_kwargs,
            )
        else:
            raise NotImplementedError(f"Sampling method {sampling_method} has not been implemented.")

        if sampling_method != "beam-search-single-sample":
            # "beam-search-single-sample" already returns batch x num_samples samples, so nothing to be done here.

            # Reshape interleaved samples, putting the original batch first
            samples_tmp: List[List[Union[List[str], str]]] = []
            for i in range(batch_size):
                samples_tmp.append([])
                for j in range(num_samples):
                    idx = i * num_samples + j
                    samples_tmp[i].append(samples[idx])
            samples = samples_tmp

        if return_embedding:
            embs = self.hiddens_to_embedding(perturbed_hiddens, sample_masks)
            return samples, embs
        else:
            return samples

    # TODO: we might want to return embeddings only in some cases, why always return hiddens + embeddings? (commented for now, need to fix ir just use parent implementation)
    def forward(self, sequences: Union[Series, IdedSeqs, List[str]]) -> Inference:
        """
        Computes embeddings for a list of sequences.
        Embeddings are detached from model.

        Params
            sequences: Pandas Series containing a list of strings or or a list of strings (e.g., SMILES, AA sequences, etc)

        Returns
            embeddings
        """
        seqs, ids = to_sequences_ids(sequences)

        hiddens, enc_mask = self.seq_to_hiddens(seqs)
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)

        result_dict: Inference = {
            "embeddings": embeddings.float().detach().clone(),
            "hiddens": hiddens.float().detach().clone(),
            "mask": enc_mask.detach().clone(),
            "sequence": seqs,
        }
        if ids is not None:
            result_dict["id"] = ids

        return result_dict


M = TypeVar("M", bound=BaseEncoderDecoderInference)
"""Generic type for any model that implements :class:`BaseEncoderDecoderInference`.
"""


def to_sequences_ids(
    sequences: Union[DataFrame, Series, IdedSeqs, List[str]],
) -> Tuple[List[str], Optional[List[int]]]:
    """Converts supported sequence data formats into a simple list of sequences and their ids.

    The input must be one of:
        - Pandas Series containing a list of strings
        - A dictionary with `id` and `sequence`, int IDs along with each string sequence, respectively
        - A DataFrame with columns `id` and `sequence`
        - a list of strings (e.g., SMILES, AA sequences, etc)

    The returned output will be a copy of the sequences and their accompanying IDs. If there are no IDs,
    then the returned IDs value is `None`.
    """
    ids: Optional[List[int]] = None
    if isinstance(sequences, Series):
        seqs: List[str] = sequences.tolist()
    elif isinstance(sequences, Dict):
        ids = sequences["id"]
        seqs = sequences["sequence"]
    elif isinstance(sequences, DataFrame):
        ids = sequences.loc["id"].values.tolist()
        seqs = sequences.loc["sequence"].values.tolist()
    else:
        seqs = list(sequences)
    return seqs, ids
