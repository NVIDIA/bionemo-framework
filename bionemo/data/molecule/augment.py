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


import math
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from nemo.collections.common.tokenizers.char_tokenizer import TokenizerSpec
from nemo.utils import logging
from rdkit import Chem


MAX_HEXADECIMAL = 2_147_483_647

__all__ = ["MoleculeEnumeration", "MoleculeInputTargetEnumeration"]


# FIXME: apply masking on ids instead of tokens
class MoleculeEnumeration:
    """
    Provides collate_fn for pretraining of MegaMolBART based on batches from molecule datasets.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        seq_length: int,
        encoder_augment: bool,
        encoder_mask: bool,
        decoder_independent_augment: bool,
        decoder_mask: bool,
        canonicalize_target_smile: bool,
        canonicalize_decoder_output: bool,
        canonicalize_encoder_input: bool,
        pad_size_divisible_by_8: bool,
        mask_prob: Optional[float] = None,
        span_lambda: Optional[float] = None,
        augmentation_prob: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            tokenizer: tokenizer used to tokenize smiles to ids
            seq_length: maximum sequence length of SMILES in a batch
            encoder_augment: should smiles for the encoder input be augmented?
            encoder_mask: should tokens for the encoder input be masked?
            decoder_mask: should tokens for the decoder input be masked?
            decoder_independent_augment: should smiles for the decoder input be augmented independently of the encoder input? If false, the encoder augmentation is used for the decoder.
            canonicalize_encoder_input: should encoder input smiles be canonicalized? This would be for models that always operate in the sapce of canonicalized molecules.
            canonicalize_target_smile: should target smiles be canonicalized? This should be True for the molecular accuracy computation. Set to False to
                instead use the unprocessed input dataset.
            canonicalize_decoder_output: should we ask the decoder to predict canonicalized output?
            pad_size_divisible_by_8: should sequences of ids be padded to be divisible by 8?
            mask_prob: a probability of masking single token, should be between 0 and 1
            augmentation_prob: a probability of augmenting single token, should be between 0 and 1. Only applies if one of encoder_augment/decoder_independent_augment is True
            span_lambda: masking parameter, used in mask_scheme="span"
            kwargs: other kwargs

        Notes:
            Common use cases and their respective config settings are as follows, to be set under the model.data block:
                1. canonicalized input and output to the model.
                    canonicalize_encoder_input: True
                    canonicalize_decoder_output: True
                    canonicalize_target_smile: True
                    encoder_augment: False
                    decoder_independent_augment: False
                2. augmented input with matched output from the model.
                    canonicalize_encoder_input: False
                    canonicalize_decoder_output: False
                    canonicalize_target_smile: True
                    encoder_augment: True
                    decoder_independent_augment: False
                3. augmented input with canonicalized output from the model.
                    canonicalize_encoder_input: False
                    canonicalize_decoder_output: True
                    canonicalize_target_smile: True
                    encoder_augment: True
                    decoder_independent_augment: False
                4. Uncommon option: independent augmentation of input and output from the model.
                    canonicalize_encoder_input: False
                    canonicalize_decoder_output: False
                    canonicalize_target_smile: True
                    encoder_augment: True
                    decoder_independent_augment: True
        """
        assert 0 <= augmentation_prob <= 1, "Augmentation probability should belong to [0, 1] "
        self.augmentation_prob = augmentation_prob
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_size_divisible_by_8 = pad_size_divisible_by_8  # workaround for CUDA alignment bug
        self.encoder_augment = encoder_augment
        self.decoder_independent_augment = decoder_independent_augment
        if canonicalize_encoder_input:
            assert (
                not encoder_augment
            ), "Augmentation of encoder input and canonicalization of encoder input are mutually exclusive"
        if canonicalize_decoder_output:
            assert not decoder_independent_augment, "Independent augmentation of decoder output and canonicalization of decoder output are mutually exclusive"
        self.canonicalize_decoder_input = canonicalize_decoder_output
        self.canonicalize_encoder_input = canonicalize_encoder_input
        self.canonicalize_target_smile = canonicalize_target_smile

        self.encoder_mask = encoder_mask
        self.decoder_mask = decoder_mask
        self.mask_prob = mask_prob
        self.span_lambda = span_lambda

        assert ~(
            (self.encoder_mask or self.decoder_mask) ^ (self.mask_prob is not None and self.span_lambda is not None)
        ), "If masking is enabled, parameters mask_prob and span_lambda must be specified!"

        if self.encoder_mask or self.decoder_mask:
            assert 0 <= self.mask_prob <= 1, "Masking probability should belong to [0, 1] "

    def _smiles_augmeter_func(self, smiles: str, augment_data: bool, canonicalize_input: bool) -> Tuple[str, str]:
        """
        Regularize SMILES by converting to RDKit mol objects and back. In addition, outputs canonicalized SMILES if requested
        otherwise input SMILES

        Args:
            smiles: str representation of molecule to possibly be canonicalized or augmented
            augment_data: should smiles be augmented/randomised in x[0]? Otherwise x[0] has the input_smiles as they appeared in the dataset
            canonicalize_input: should canonicalized smiles be returned in x[1]? Otherwise x[1] has the input_smiles as they appeared in the dataset
        Returns:
            (str, str) Augmented smiles and canonicalized smiles if requested, otherwise x[0] and x[1] could just be copies of smiles.
        """

        mol = Chem.MolFromSmiles(smiles)
        canon_smiles = Chem.MolToSmiles(mol, canonical=True) if canonicalize_input else smiles
        # If we want to augment the data, and additionally if the augmentation probability is greater than 1 or the random number is less than the augmentation probability
        #  then we augment the data. Otherwise we just return the canonicalized smiles. The or clause is to skip the random check if the augmentation probability is 1.
        if augment_data and (self.augmentation_prob >= 1 or self.augmentation_prob >= torch.rand(1).item()):
            try:
                # See example in https://www.rdkit.org/docs/Cookbook.html#enumerate-smiles
                #  set the global torch seed if you want to reproducible sequences
                rand_int = torch.randint(low=0, high=MAX_HEXADECIMAL, size=(1,)).item()
                aug_smiles = Chem.MolToRandomSmilesVect(mol, 1, randomSeed=rand_int)[0]
            except RuntimeError:
                logging.info(f"Could not generate smiles for {smiles} after augmenting. Forcing canonicalization")
                aug_smiles = canon_smiles if canonicalize_input else Chem.MolToSmiles(mol, canonical=True)
        else:
            aug_smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=False)

        assert len(aug_smiles) > 0, AssertionError("Augmented SMILES string is empty")
        assert len(canon_smiles) > 0, AssertionError("Canonical SMILES string is empty")
        return aug_smiles, canon_smiles

    def _check_seq_len(
        self, tokens: List[List[str]], mask: List[List[int]]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Warns user and shortens sequence of tokens if the sequence is too long and exceeds seq_length
        Args:
            tokens: list of with sequences of tokens represented as list of str
            mask: list of with sequences of mask corresponding to tokens represented as list of int
        Returns:
            tuple, list of token sequences (shortened, if necessary) and corresponding mask sequences
        """
        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.seq_length:
            tokens_short = [ts[: self.seq_length] for ts in tokens]
            mask_short = [ms[: self.seq_length] for ms in mask]
            return (tokens_short, mask_short)
        return (tokens, mask)

    def _prepare_tokens(
        self, batch: List[str], mask_data: bool = False
    ) -> Dict[str, Union[List[List[str]], List[List[int]]]]:
        """
        Prepares tokens for encoder or decoder from list of SMILES strings. Firstly, tokens are tokenized and,
        if requested, masked. Finally, each sequence length is verified and shortened, if necessary.
        Args:
            batch: list of SMILES
            mask_data:  should the tokens be masked?
        Returns:
            dict with original tokens with all True mask or masked tokens with corresponding mask
                or True or False (indicating masked tokens)
        """
        # Tokenize with optional masking, padding is done later due to differences in encoder/decoder bos/eos tokens
        token_output = self.tokenize(batch, mask_data=mask_data)

        if mask_data:
            tokens = token_output["masked_tokens"]
            mask = token_output["token_masks"]
        else:
            tokens = token_output["original_tokens"]
            mask = [[True] * len(ts) for ts in tokens]  # 1/True = Active, 0/False = Inactive

        # Verify sequence length
        tokens, mask = self._check_seq_len(tokens, mask)

        return {"tokens": tokens, "mask": mask}

    def _pad_seqs(self, seqs: List[List[int]], pad_token_id: int) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Pads sequences to have the same length and, if requested, divisible by 8
        Args:
            seqs: list of sequences represented as list of ids
            pad_token: id of pad_token
        Returns:
            padded list of sequences with corresponding masks indicating padding by False
        """
        pad_length = max([len(seq) for seq in seqs])
        if self.pad_size_divisible_by_8:
            pad_length = int(math.ceil(pad_length / 8) * 8)

        padded = [seq + ([pad_token_id] * (pad_length - len(seq))) for seq in seqs]
        masks = [
            ([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in seqs
        ]  # 1/True = Active, 0/False = Inactive
        return padded, masks

    def tokenize(self, smi: str, mask_data=False) -> Dict[str, Union[List[List[str]], List[List[int]]]]:
        """
        Tokenizes SMILES and masks tokens, if requested.
        Args:
            smi: string representation of molecules
            mask_data: boolean, should the tokens be masked?
        Returns:
            dict with sequences of tokenized and masked SMILES
        """
        # TODO this function needs cleanup
        tokens = [self.tokenizer.text_to_tokens(s) for s in smi]
        m_tokens, token_masks = self.mask_tokens(tokens, empty_mask=not mask_data)

        output = {}
        output["original_tokens"] = tokens

        if mask_data:
            output["masked_tokens"] = m_tokens
            output["token_masks"] = token_masks

        return output

    def mask_tokens(self, tokens, empty_mask=False):
        if empty_mask:
            mask = [[True] * len(ts) for ts in tokens]
            return tokens, mask

        masked_tokens = []
        token_masks = []

        for ts in tokens:
            masked, token_mask = self._mask_span(ts)
            masked_tokens.append(masked)
            token_masks.append(token_mask)

        return masked_tokens, token_masks

    def _mask_span(self, ts):
        curr_token = 0
        masked = []
        token_mask = []

        mask_bools = [True, False]
        weights = [self.mask_prob, 1 - self.mask_prob]
        sampled_mask = random.choices(mask_bools, weights=weights, k=len(ts))

        while curr_token < len(ts):
            # If mask, sample from a poisson dist to get length of mask
            if sampled_mask[curr_token]:
                mask_len = torch.poisson(torch.tensor(self.span_lambda)).long().item()
                masked.append(self.tokenizer.mask_token)
                token_mask.append(True)
                curr_token += mask_len

            # Otherwise don't mask
            else:
                masked.append(ts[curr_token])
                token_mask.append(False)
                curr_token += 1

        return masked, token_mask

    def _prepare_input(
        self,
        smiles: List[str],
        mask_data: bool,
        append_bos_token: bool = False,
        get_labels: bool = False,
        label_pad: Optional[int] = None,
    ) -> dict:
        """
        Prepare tokens for encoder or decoder input from list with molecules smiles,

        Args:
            smiles_list: list with SMILES (augmented and canonised if requested)
            get_canon_smiles: should canonical SMILES be returned as well?
            mask_data: should the tokens be masked?
            get_labels: should the labels for loss function be returned as well?
        Returns:
            Input to encoder or decoder as a dict
        """
        input_dict = {}

        # TODO masks from masked tokens are never used
        tokens_dict = self._prepare_tokens(smiles, mask_data=mask_data)
        token_ids = [self.tokenizer.token_to_ids(t) for t in tokens_dict["tokens"]]

        if get_labels:
            label_ids = [sample + [self.tokenizer.eos_id] for sample in token_ids]
            label_token_ids, loss_mask = self._pad_seqs(label_ids, self.tokenizer.pad_id)
            label_token_ids = torch.tensor(label_token_ids, dtype=torch.int64)
            loss_mask = torch.tensor(loss_mask, dtype=torch.int64)
            label_token_ids[~loss_mask.to(torch.bool)] = label_pad
            input_dict["label_ids"] = label_token_ids
            input_dict["loss_mask"] = loss_mask

        if append_bos_token:
            token_ids = [[self.tokenizer.bos_id] + sample for sample in token_ids]
        token_ids, mask = self._pad_seqs(token_ids, self.tokenizer.pad_id)

        input_dict["token_ids"] = torch.tensor(token_ids, dtype=torch.int64)
        input_dict["mask"] = torch.tensor(mask, dtype=torch.int64)
        return input_dict

    def _get_encoder_decoder_input_smiles(self, batch: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Helper method to prepare input smiles to be tokenized for encoder and decoder layers.
        Args:
            batch: list with SMILES
        Returns:
            lists with SMILES that are used and encoder and decoder inputs and target smiles, respectively
        """
        encoder_smiles_list = [
            self._smiles_augmeter_func(
                smiles,
                augment_data=self.encoder_augment,
                canonicalize_input=self.canonicalize_encoder_input
                or self.canonicalize_target_smile
                or self.canonicalize_decoder_input,
            )
            for smiles in batch
        ]
        if self.canonicalize_encoder_input:
            # If we want to canonicalize the input to the encoder, then we use the x[1] position from encoder_smiles_list
            encoder_idx = 1
        else:
            encoder_idx = 0
        if self.canonicalize_target_smile:
            target_idx = 1
        else:
            target_idx = 0
        encoder_smiles = [x[encoder_idx] for x in encoder_smiles_list]
        # target smiles are canonised or not encoder input smiles, they are used for visual inspection and debugging. The decoder
        #  smiles/labels are what are fed into the loss etc as the target.
        target_smiles = [x[target_idx] for x in encoder_smiles_list]
        if self.decoder_independent_augment:
            decoder_smiles_list = [
                self._smiles_augmeter_func(
                    smiles,
                    augment_data=self.decoder_independent_augment,
                    canonicalize_input=False,  # Skip canonicalization since this idx is not used
                )
                for smiles in encoder_smiles
            ]
            # We want the independently augmented decoder smiles, rather than the encoder matched smiles
            decoder_smiles = [x[0] for x in decoder_smiles_list]
        else:
            # Which idx do we want?
            if self.canonicalize_decoder_input:
                decoder_idx = 1
            else:
                decoder_idx = 0
            decoder_smiles = [x[decoder_idx] for x in encoder_smiles_list]
        return encoder_smiles, decoder_smiles, target_smiles

    def collate_fn(self, batch: List[str], label_pad: int = -1) -> dict:
        """
        Collate function for NeMo MegaMolBART model
        Args:
            batch: batch with SMILES
            label_pad: padding token for labels
        Returns:
            Input to encoder and decoder of a model as a dict
        """
        encoder_smiles, decoder_smiles, target_smiles = self._get_encoder_decoder_input_smiles(batch)

        encoder_input = self._prepare_input(encoder_smiles, mask_data=self.encoder_mask)

        decoder_input = self._prepare_input(
            decoder_smiles, mask_data=self.decoder_mask, append_bos_token=True, get_labels=True, label_pad=label_pad
        )

        collate_output = {
            "text_enc": encoder_input["token_ids"],
            "enc_mask": encoder_input["mask"],
            "text_dec": decoder_input["token_ids"],
            "dec_mask": decoder_input["mask"],
            "labels": decoder_input["label_ids"],
            "loss_mask": decoder_input["loss_mask"],
            "target_smiles": target_smiles,
        }  # target smiles strings

        return collate_output


class MoleculeInputTargetEnumeration(MoleculeEnumeration):
    """
    Provides collate_fn for MegaMolBARTRetro based on batches ie from a datasets that outputs a dict with
    keys input_name and target_name with corresponding SMILES.

    For example, for retrosynthesis input_name should correspond to the key name in batch related to product and
    target_name - to reactants. For forward synthesis - otherwise.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        seq_length: int,
        encoder_augment: bool,
        encoder_mask: bool,
        decoder_independent_augment: bool,
        decoder_mask: bool,
        canonicalize_target_smile: bool,
        canonicalize_decoder_output: bool,
        canonicalize_encoder_input: bool,
        pad_size_divisible_by_8: bool,
        input_name: str,
        target_name: str,
        mask_prob: Optional[float] = None,
        span_lambda: Optional[float] = None,
        augmentation_prob: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            seq_length=seq_length,
            encoder_augment=encoder_augment,
            encoder_mask=encoder_mask,
            decoder_independent_augment=decoder_independent_augment,
            decoder_mask=decoder_mask,
            canonicalize_target_smile=canonicalize_target_smile,
            canonicalize_decoder_output=canonicalize_decoder_output,
            canonicalize_encoder_input=canonicalize_encoder_input,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            mask_prob=mask_prob,
            span_lambda=span_lambda,
            augmentation_prob=augmentation_prob,
            **kwargs,
        )
        """
        Args:
            input_name: the key name in a batch from a reaction dataset to encoder input
            target_name: the key name in a batch from a reaction dataset to target and decoder input
                           to calculate the loss
        """

        self.input_name = input_name
        self.target_name = target_name
        self.get_canonicalized_decoder_input = canonicalize_encoder_input
        self.get_canonicalized_encoder_input = False

    def _get_encoder_decoder_input_smiles(self, batch: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Helper method to prepare input smiles to be tokenized for encoder and decoder layers
        Args:
            batch: batch from reaction dataset, list of dicts with SMILES of products and reactants
        Returns:
            lists with SMILES that are used and encoder and decoder inputs and target smiles, respectively
        """
        encoder_smiles_list = [
            self._smiles_augmeter_func(
                react[self.input_name],
                augment_data=self.encoder_augment,
                canonicalize_input=self.canonicalize_encoder_input,
            )
            for react in batch
        ]

        decoder_smiles_list = [
            self._smiles_augmeter_func(
                react[self.target_name],
                augment_data=self.decoder_independent_augment,
                canonicalize_input=self.canonicalize_target_smile or self.canonicalize_decoder_input,
            )
            for react in batch
        ]
        encoder_idx = 1 if self.canonicalize_encoder_input else 0
        decoder_idx = 1 if self.canonicalize_decoder_input else 0
        target_idx = 1 if self.canonicalize_target_smile else 0
        encoder_smiles = [x[encoder_idx] for x in encoder_smiles_list]
        decoder_smiles = [x[decoder_idx] for x in decoder_smiles_list]
        target_smiles = [x[target_idx] for x in decoder_smiles_list]
        return encoder_smiles, decoder_smiles, target_smiles
