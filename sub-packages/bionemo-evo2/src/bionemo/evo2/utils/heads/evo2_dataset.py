# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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
from typing import ClassVar, Dict, Optional

import torch
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.indexed_dataset import IndexedDataset
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case
from nemo.utils import logging


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=data.device)).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class Evo2Dataset(GPTDataset):
    """Dataset for training Evo2."""

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    TO_UPPER_TOKENS: bool = True  # If set, do an in-place transform to make all tokens capital letters
    RESET_PAD_EOD_MASK: bool = True  # If set, unset the mask for [pad] and [eod] tokens (matches Evo2 paper).

    def _get_gpt_batch(self, idx: Optional[int]) -> dict[str, torch.Tensor]:
        return super().__getitem__(idx)

    def _modify_gpt_batch(self, databatch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss_mask = databatch.get("loss_mask", None)
        if self.RESET_PAD_EOD_MASK and loss_mask is not None:
            # Reset the mask for 'pad', '[eod]', '[pad token]', which will lower the loss, but matches Evo2 pub.
            loss_mask = torch.ones_like(loss_mask)
        labels = databatch.get("labels", None)
        if labels is None or loss_mask is None:
            # No next-token labels or loss to mask.
            return databatch

        # Mask special label tags in loss.
        control_mask = torch.isin(labels, torch.tensor(self.CONTROL_TAGS, device=labels.device))
        loss_mask[control_mask] = 0
        phylotag_mask = self.mask_phylogenetic_tags(
            labels,
            self.TAG_BOUNDS,
            self.TAG_CHARS,
            self.config.tokenizer.eod if self.config.tokenizer is not None else self.DEFAULT_EOD,  # type: ignore
        )
        databatch["loss_mask"] = loss_mask * phylotag_mask
        if self.TO_UPPER_TOKENS:
            # When making tokens uppercase, make sure this is done after the mask_phylogenetic_tags function which
            #  relies in part on the original case of the tag tokens.
            databatch["tokens"], _ = make_upper_case(databatch["tokens"])
        return databatch

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get data at the specified index."""
        # 1. Call the default gpt dataset object
        databatch: dict = self._get_gpt_batch(idx)
        # 2. Modify loss tokens and upper-case as configured.
        return self._modify_gpt_batch(databatch)

    @staticmethod
    def mask_phylogenetic_tags(
        tokenized_sequence: torch.Tensor,
        terminal_tag_char: int,  # e.g. ASCII for '|'
        other_tag_chars: set[int],  # e.g. {95, 59, 32} for '_', ';', space
        eod_token_id: int,  # e.g. 0
    ) -> torch.Tensor:
        """Phylogenetic tags.

        Creates a binary mask for sequences containing phylogenetic tags and DNA.
        The rules are as follows (applied per contiguous sub-sequence between EOD tokens):

          - Any token equal to the terminal_tag_char (the pipe, '|') is masked.
          - For the region *before* the first pipe (the “prefix”):
              * If the first token is in taxonomy_prefixes (d, p, c, o, f, g, s),
                or if the prefix is exactly one lowercase letter,
                or if any token in the prefix is one of other_tag_chars,
                or if not every token is a valid DNA base,
                then mask the entire prefix.
          - For the region between pipes:
              * If any token is in other_tag_chars or not all tokens are valid DNA, mask that region.
          - For the region *after* the last pipe (the “suffix”):
              * If the first token is the letter 'd' (ASCII 100) or if the region contains
                any other tag characters or any EOD tokens or non-DNA, mask the suffix.

        Finally, any token equal to eod_token_id is forced to remain unmasked.
        (EOD tokens “break” a sequence so that tags never span across them.)

        Args:
            tokenized_sequence (torch.Tensor): shape (seq_len,) or (batch_size, seq_len)
              containing ASCII values.
            terminal_tag_char (int): ASCII value for the pipe character.
            other_tag_chars (set[int]): Set of ASCII values that appear only in tags.
            eod_token_id (int): The token ID for EOD.

        Notes:
        - The tag token is constructed as follows: So note that one way to know you have a tag is if you look
         at the first token after the pipe and it is a 'd' character. Make sure implementation handles this.
            ```
            return (
                "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                    lineage.domain if random.random() >= dropout else None,
                    lineage.phylum if random.random() >= dropout else None,
                    lineage.clazz if random.random() >= dropout else None,
                    lineage.order if random.random() >= dropout else None,
                    lineage.family if random.random() >= dropout else None,
                    lineage.genus if random.random() >= dropout else None,
                    lineage.species if random.random() >= dropout else None,
                )
                if lineage is not None
                else None
            )
            ```
        Returns:
            torch.Tensor: A mask of the same shape as input where 1 = keep (DNA) and 0 = mask (tag).
        """
        device = tokenized_sequence.device
        dtype = tokenized_sequence.dtype
        # Handle empty or single-token sequences.
        if tokenized_sequence.numel() == 0:
            return torch.ones(0, device=device, dtype=torch.int)
        if tokenized_sequence.numel() == 1:
            mask = torch.ones(1, device=device, dtype=torch.int)
            token = tokenized_sequence.item()
            if token == terminal_tag_char or token in other_tag_chars:
                mask[0] = 0
            return mask

        # Ensure input is 2D (batch, seq_len)
        batched = tokenized_sequence.ndim == 2
        if not batched:
            tokenized_sequence = tokenized_sequence.unsqueeze(0)
        batch_size, seq_len = tokenized_sequence.shape
        first_taxonomy_prefix_token: int = 100

        # Valid DNA tokens: A, C, G, T, N (both uppercase and lowercase)
        valid_dna = {65, 67, 71, 84, 78, 97, 99, 103, 116, 110}
        valid_dna_or_control_tensor = torch.tensor(
            list(valid_dna | set(Evo2Dataset.CONTROL_TAGS)), device=device, dtype=dtype
        )

        # Initialize output mask to all ones.
        out_mask = torch.ones_like(tokenized_sequence, dtype=torch.int)

        # Helper: Check if all tokens in a region are valid DNA.
        def region_all_valid_or_control(region: torch.Tensor) -> bool:
            if region.numel() == 0:
                return True
            # Using torch's all() over the token values.
            return bool(torch.all(torch.isin(region, valid_dna_or_control_tensor)).cpu().item())

        # Process one EOD-free segment using the O1 logic.
        def process_segment(seg_seq: torch.Tensor) -> torch.Tensor:
            seg_len = seg_seq.size(0)
            seg_mask = torch.ones(seg_len, device=device, dtype=torch.int)
            # Identify positions of terminal tag (pipe)
            pipe_pos = (seg_seq == terminal_tag_char).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(pipe_pos) == 0:
                # If no pipe exists and any token is a known tag char or not valid DNA,
                # mask the entire segment.
                if not region_all_valid_or_control(seg_seq):
                    seg_mask.zero_()
                return seg_mask

            # Always mask the pipe positions.
            seg_mask[pipe_pos] = 0

            # Does tag start before the first pipe? This determines the starting state of our state machine.
            first_pipe = pipe_pos[0]
            if first_pipe >= 0 and first_pipe < seg_len - 1:
                # fastest check is to look at the first token after the pipe, if it is a 'd' then the
                # tag starts _after_ the pipe, otherwise it starts before.
                next_tok = seg_seq[first_pipe + 1].item()
                if next_tok == first_taxonomy_prefix_token:
                    # 'd' character for domain, which is the first part of a phylo tag.
                    # tag starts after the pipe.
                    is_tag = False
                else:
                    # tag starts before the pipe.
                    is_tag = True
            else:
                # The sequence ends with a pipe, so just check everything before the pipe and return the seg mask
                assert first_pipe == seg_len - 1
                # The sequence ends with a pipe, so just check everything before the pipe.
                if region_all_valid_or_control(seg_seq[:first_pipe]):
                    return seg_mask  # Pipe pos has already been masked
                else:
                    seg_mask[:first_pipe] = 0
                    return seg_mask
            start = 0
            for end in pipe_pos:
                if is_tag:
                    seg_mask[start:end] = 0
                else:
                    pass
                is_tag = not is_tag  # Flip the state machine.
                start = end + 1  # position after the pipe
            # Process the last segment after the last pipe.
            if is_tag:
                seg_mask[start:] = 0
            return seg_mask

        # Process each row by splitting on EOD tokens.
        for b in range(batch_size):
            row = tokenized_sequence[b]
            # Get indices of EOD tokens.
            eod_positions = (row == eod_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
            start_idx = 0
            for pos in eod_positions:
                if pos > start_idx:
                    seg = row[start_idx:pos]
                    seg_mask = process_segment(seg)
                    out_mask[b, start_idx:pos] = seg_mask
                # Leave the EOD token itself unmasked.
                start_idx = pos + 1
            # Process any remaining tokens after the last EOD.
            if start_idx < seq_len:
                seg = row[start_idx:]
                seg_mask = process_segment(seg)
                out_mask[b, start_idx:] = seg_mask

        # Just to make sure we do not allow any non-DNA tokens to be unmasked, even if something went wrong with our
        #  mask logic.
        out_mask[~torch.isin(tokenized_sequence, valid_dna_or_control_tensor)] = 0
        # Finally, force every EOD token to be unmasked. User decides outside of this function if they want EOD mask.
        out_mask[tokenized_sequence == eod_token_id] = 1

        if not batched:
            out_mask = out_mask.squeeze(0)
        return out_mask


class Evo2DatasetPadEodLossMask(Evo2Dataset):
    """Dataset for training Evo2 with pad and eod loss mask (more standard approach than the Evo2 paper)."""

    TO_UPPER_TOKENS: bool = True
    RESET_PAD_EOD_MASK: bool = False


class Evo2RNASeqDataset(GPTDataset):
    """Extended Evo2Dataset that loads both tokens and RNA-seq data."""

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    TO_UPPER_TOKENS: bool = True
    RESET_PAD_EOD_MASK: bool = True

    def __init__(self, *args, **kwargs):
        """Initialize the Evo2RNASeqDataset."""
        super().__init__(*args, **kwargs)

        # Try to load parallel RNA-seq dataset
        self.rna_seq_dataset = None
        self._load_rna_seq_dataset()

    def _load_rna_seq_dataset(self):
        """Load the parallel RNA-seq indexed dataset if it exists.

        The RNA-seq dataset is expected to have the same prefix as the DNA dataset,
        but with the suffix '_rna_seq.bin'. This is part of the `preprocess.py` script.
        """
        base_path = self.dataset.path_prefix  # type: ignore # Original DNA dataset path prefix
        rna_seq_path = base_path + "_rna_seq.bin"  # Append suffix for RNA-seq dataset

        if Path(rna_seq_path).exists():
            # Remove .bin extension for IndexedDataset
            rna_seq_prefix = rna_seq_path.replace(".bin", "")
            self.rna_seq_dataset = IndexedDataset(rna_seq_prefix)
            logging.info(f"  ✅ Loaded RNA-seq dataset: {rna_seq_prefix}")
            logging.info(f"  Number of documents: {len(self.rna_seq_dataset)}")
        else:
            logging.warning(f"  ⚠️ No RNA-seq dataset found at: {rna_seq_path}")
            self.rna_seq_dataset = None

    def _get_gpt_batch(self, idx: Optional[int]) -> dict[str, torch.Tensor]:
        """Get the standard GPT batch with document IDs."""
        if idx is None:
            text, document_ids = self._query_document_sample_shuffle_indices(0)
        else:
            text, document_ids = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:  # type: ignore
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if not self.masks_and_position_ids_are_cacheable or not self.masks_and_position_ids_are_cached:
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,  # type: ignore
                self.config.reset_position_ids,  # type: ignore
                self.config.reset_attention_mask,  # type: ignore
                self.config.eod_mask_loss,  # type: ignore
                self.config.create_attention_mask,  # type: ignore
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        # ✅ Add document IDs to the batch
        databatch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "document_ids": document_ids,  # ← Store document IDs
        }

        if self.config.create_attention_mask:  # type: ignore
            databatch["attention_mask"] = attention_mask

        return databatch

    # def _add_rna_seq_data(self, databatch: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    #     """Add RNA-seq targets to the batch."""
    #     # Load RNA-seq data for this document
    #     rna_seq_data = self.rna_seq_dataset[idx] # type: ignore
    #     rna_seq_tensor = torch.tensor(rna_seq_data, dtype=torch.float32)

    #     # Match sequence length with tokens
    #     tokens = databatch.get("tokens", None)
    #     if tokens is not None:
    #         target_length = len(tokens)

    #         if len(rna_seq_tensor) > target_length:
    #             # Truncate to match token length
    #             rna_seq_tensor = rna_seq_tensor[:target_length]
    #         elif len(rna_seq_tensor) < target_length:
    #             # Pad with zeros to match token length
    #             padding = torch.zeros(target_length - len(rna_seq_tensor), dtype=torch.float32)
    #             rna_seq_tensor = torch.cat([rna_seq_tensor, padding])

    #     databatch["rna_seq_targets"] = rna_seq_tensor
    #     return databatch

    def _add_rna_seq_data(self, databatch: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
        """Add RNA-seq targets to the batch using document IDs for correct alignment.

        Args:
            databatch: Dictionary containing batch data including tokens and document_ids
            idx: Sample index (not used directly, kept for signature compatibility)

        Returns:
            Updated databatch with rna_seq_targets added
        """
        if self.rna_seq_dataset is not None:
            try:
                # ✅ Use document_ids for correct alignment
                document_ids = databatch.get("document_ids", None)

                if document_ids is not None and len(document_ids) > 0:
                    # Load RNA-seq for all documents in this sample
                    rna_seq_parts = []
                    for doc_id in document_ids:
                        rna_seq_data = self.rna_seq_dataset[int(doc_id)]
                        rna_seq_parts.append(torch.tensor(rna_seq_data, dtype=torch.float32))

                    # Concatenate if sample spans multiple documents
                    rna_seq_tensor = torch.cat(rna_seq_parts) if len(rna_seq_parts) > 1 else rna_seq_parts[0]

                    # Match sequence length with tokens
                    tokens = databatch.get("tokens", None)
                    if tokens is not None:
                        target_length = len(tokens)

                        if len(rna_seq_tensor) > target_length:
                            rna_seq_tensor = rna_seq_tensor[:target_length]
                        elif len(rna_seq_tensor) < target_length:
                            padding = torch.zeros(target_length - len(rna_seq_tensor), dtype=torch.float32)
                            rna_seq_tensor = torch.cat([rna_seq_tensor, padding])

                    databatch["rna_seq_targets"] = rna_seq_tensor
                else:
                    # No document IDs available - fall back to zeros
                    tokens = databatch.get("tokens", torch.tensor([]))
                    databatch["rna_seq_targets"] = torch.zeros(len(tokens), dtype=torch.float32)

            except Exception as e:
                logging.warning(f"Failed to load RNA-seq data: {e}")
                tokens = databatch.get("tokens", torch.tensor([]))
                databatch["rna_seq_targets"] = torch.zeros(len(tokens), dtype=torch.float32)
        else:
            tokens = databatch.get("tokens", torch.tensor([]))
            databatch["rna_seq_targets"] = torch.zeros(len(tokens), dtype=torch.float32)

        return databatch

    def _modify_gpt_batch(self, databatch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply Evo2-specific modifications (same as original)."""
        loss_mask = databatch.get("loss_mask", None)
        if self.RESET_PAD_EOD_MASK and loss_mask is not None:
            loss_mask = torch.ones_like(loss_mask)

        labels = databatch.get("labels", None)
        if labels is None or loss_mask is None:
            return databatch

        # Mask special label tags in loss
        control_mask = torch.isin(labels, torch.tensor(self.CONTROL_TAGS, device=labels.device))
        loss_mask[control_mask] = 0
        phylotag_mask = self.mask_phylogenetic_tags(
            labels,
            self.TAG_BOUNDS,
            self.TAG_CHARS,
            self.config.tokenizer.eod if self.config.tokenizer is not None else self.DEFAULT_EOD,  # type: ignore
        )
        databatch["loss_mask"] = loss_mask * phylotag_mask

        # ✅ ADD THIS: Calculate num_valid_tokens_in_ub
        if "loss_mask" in databatch:
            databatch["num_valid_tokens_in_ub"] = databatch["loss_mask"].sum()

        if self.TO_UPPER_TOKENS:
            databatch["tokens"], _ = make_upper_case(databatch["tokens"])

        return databatch

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get data with both tokens and RNA-seq targets."""
        # 1. Get standard GPT batch (now includes document_ids)
        databatch = self._get_gpt_batch(idx)

        # 2. Add RNA-seq data using document_ids
        databatch = self._add_rna_seq_data(databatch, idx)  # type: ignore

        # 3. Apply Evo2 modifications
        databatch = self._modify_gpt_batch(databatch)  # type: ignore

        # 4. Remove document_ids from final batch (not needed downstream)
        databatch.pop("document_ids", None)

        return databatch

    @staticmethod
    def mask_phylogenetic_tags(
        tokenized_sequence: torch.Tensor,
        terminal_tag_char: int,  # e.g. ASCII for '|'
        other_tag_chars: set[int],  # e.g. {95, 59, 32} for '_', ';', space
        eod_token_id: int,  # e.g. 0
    ) -> torch.Tensor:
        """Phylogenetic tags.

        Creates a binary mask for sequences containing phylogenetic tags and DNA.
        The rules are as follows (applied per contiguous sub-sequence between EOD tokens):

          - Any token equal to the terminal_tag_char (the pipe, '|') is masked.
          - For the region *before* the first pipe (the “prefix”):
              * If the first token is in taxonomy_prefixes (d, p, c, o, f, g, s),
                or if the prefix is exactly one lowercase letter,
                or if any token in the prefix is one of other_tag_chars,
                or if not every token is a valid DNA base,
                then mask the entire prefix.
          - For the region between pipes:
              * If any token is in other_tag_chars or not all tokens are valid DNA, mask that region.
          - For the region *after* the last pipe (the “suffix”):
              * If the first token is the letter 'd' (ASCII 100) or if the region contains
                any other tag characters or any EOD tokens or non-DNA, mask the suffix.

        Finally, any token equal to eod_token_id is forced to remain unmasked.
        (EOD tokens “break” a sequence so that tags never span across them.)

        Args:
            tokenized_sequence (torch.Tensor): shape (seq_len,) or (batch_size, seq_len)
              containing ASCII values.
            terminal_tag_char (int): ASCII value for the pipe character.
            other_tag_chars (set[int]): Set of ASCII values that appear only in tags.
            eod_token_id (int): The token ID for EOD.

        Notes:
        - The tag token is constructed as follows: So note that one way to know you have a tag is if you look
         at the first token after the pipe and it is a 'd' character. Make sure implementation handles this.
            ```
            return (
                "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                    lineage.domain if random.random() >= dropout else None,
                    lineage.phylum if random.random() >= dropout else None,
                    lineage.clazz if random.random() >= dropout else None,
                    lineage.order if random.random() >= dropout else None,
                    lineage.family if random.random() >= dropout else None,
                    lineage.genus if random.random() >= dropout else None,
                    lineage.species if random.random() >= dropout else None,
                )
                if lineage is not None
                else None
            )
            ```
        Returns:
            torch.Tensor: A mask of the same shape as input where 1 = keep (DNA) and 0 = mask (tag).
        """
        device = tokenized_sequence.device
        dtype = tokenized_sequence.dtype
        # Handle empty or single-token sequences.
        if tokenized_sequence.numel() == 0:
            return torch.ones(0, device=device, dtype=torch.int)
        if tokenized_sequence.numel() == 1:
            mask = torch.ones(1, device=device, dtype=torch.int)
            token = tokenized_sequence.item()
            if token == terminal_tag_char or token in other_tag_chars:
                mask[0] = 0
            return mask

        # Ensure input is 2D (batch, seq_len)
        batched = tokenized_sequence.ndim == 2
        if not batched:
            tokenized_sequence = tokenized_sequence.unsqueeze(0)
        batch_size, seq_len = tokenized_sequence.shape
        first_taxonomy_prefix_token: int = 100

        # Valid DNA tokens: A, C, G, T, N (both uppercase and lowercase)
        valid_dna = {65, 67, 71, 84, 78, 97, 99, 103, 116, 110}
        valid_dna_or_control_tensor = torch.tensor(
            list(valid_dna | set(Evo2Dataset.CONTROL_TAGS)), device=device, dtype=dtype
        )

        # Initialize output mask to all ones.
        out_mask = torch.ones_like(tokenized_sequence, dtype=torch.int)

        # Helper: Check if all tokens in a region are valid DNA.
        def region_all_valid_or_control(region: torch.Tensor) -> bool:
            if region.numel() == 0:
                return True
            # Using torch's all() over the token values.
            return bool(torch.all(torch.isin(region, valid_dna_or_control_tensor)).cpu().item())

        # Process one EOD-free segment using the O1 logic.
        def process_segment(seg_seq: torch.Tensor) -> torch.Tensor:
            seg_len = seg_seq.size(0)
            seg_mask = torch.ones(seg_len, device=device, dtype=torch.int)
            # Identify positions of terminal tag (pipe)
            pipe_pos = (seg_seq == terminal_tag_char).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(pipe_pos) == 0:
                # If no pipe exists and any token is a known tag char or not valid DNA,
                # mask the entire segment.
                if not region_all_valid_or_control(seg_seq):
                    seg_mask.zero_()
                return seg_mask

            # Always mask the pipe positions.
            seg_mask[pipe_pos] = 0

            # Does tag start before the first pipe? This determines the starting state of our state machine.
            first_pipe = pipe_pos[0]
            if first_pipe >= 0 and first_pipe < seg_len - 1:
                # fastest check is to look at the first token after the pipe, if it is a 'd' then the
                # tag starts _after_ the pipe, otherwise it starts before.
                next_tok = seg_seq[first_pipe + 1].item()
                if next_tok == first_taxonomy_prefix_token:
                    # 'd' character for domain, which is the first part of a phylo tag.
                    # tag starts after the pipe.
                    is_tag = False
                else:
                    # tag starts before the pipe.
                    is_tag = True
            else:
                # The sequence ends with a pipe, so just check everything before the pipe and return the seg mask
                assert first_pipe == seg_len - 1
                # The sequence ends with a pipe, so just check everything before the pipe.
                if region_all_valid_or_control(seg_seq[:first_pipe]):
                    return seg_mask  # Pipe pos has already been masked
                else:
                    seg_mask[:first_pipe] = 0
                    return seg_mask
            start = 0
            for end in pipe_pos:
                if is_tag:
                    seg_mask[start:end] = 0
                else:
                    pass
                is_tag = not is_tag  # Flip the state machine.
                start = end + 1  # position after the pipe
            # Process the last segment after the last pipe.
            if is_tag:
                seg_mask[start:] = 0
            return seg_mask

        # Process each row by splitting on EOD tokens.
        for b in range(batch_size):
            row = tokenized_sequence[b]
            # Get indices of EOD tokens.
            eod_positions = (row == eod_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
            start_idx = 0
            for pos in eod_positions:
                if pos > start_idx:
                    seg = row[start_idx:pos]
                    seg_mask = process_segment(seg)
                    out_mask[b, start_idx:pos] = seg_mask
                # Leave the EOD token itself unmasked.
                start_idx = pos + 1
            # Process any remaining tokens after the last EOD.
            if start_idx < seq_len:
                seg = row[start_idx:]
                seg_mask = process_segment(seg)
                out_mask[b, start_idx:] = seg_mask

        # Just to make sure we do not allow any non-DNA tokens to be unmasked, even if something went wrong with our
        #  mask logic.
        out_mask[~torch.isin(tokenized_sequence, valid_dna_or_control_tensor)] = 0
        # Finally, force every EOD token to be unmasked. User decides outside of this function if they want EOD mask.
        out_mask[tokenized_sequence == eod_token_id] = 1

        if not batched:
            out_mask = out_mask.squeeze(0)
        return out_mask


class Evo2RNASeqDatasetPadEodLossMask(Evo2RNASeqDataset):
    """RNA-seq dataset with pad and eod loss mask."""

    TO_UPPER_TOKENS: bool = True
    RESET_PAD_EOD_MASK: bool = False
