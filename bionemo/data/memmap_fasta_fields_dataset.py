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


from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset


class FASTAFieldsMemmapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    def __init__(
        self,
        dataset_paths,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        data_sep="\n",
        data_fields={"data": 0},
        strip_first_element=True,
        collpase_sequence_elements=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_paths: list of paths to text files
            workers: number of workers to use for parallel data indexing (on first run)
            tokenizer: tokenizer to use for tokenization
            sort_dataset_paths: whether to sort dataset paths by name
            data_sep: separator between data fields (within a sample)
            data_fields: dictionary of field names and their indices
            strip_first_element: whether to strip the first element of the sequence from all text past first space
            collpase_sequence_elements: whether to collapse all sequence elements into a single string (all but first)
            index_mapping_dir: directory to store index mapping cached files
        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1,  # skip first line since it is not an empty sequence
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        self._data_fields = data_fields
        self._data_sep = data_sep
        self._strip_first_element = strip_first_element
        self._collpase_sequence_elements = collpase_sequence_elements

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # convert text into data
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        data = {}
        text_fields = text.split(self._data_sep)

        # remove trailing words in first element
        if self._strip_first_element:
            elements = text_fields[0].split(" ")
            text_fields[0] = elements[0].strip()
            # add new fields for all elements with "=" past first element
            for e in elements[1:]:
                if "=" in e:
                    key, val = e.split("=")
                    data[key] = val

        if self._collpase_sequence_elements:
            # collapse all sequence elements into a single string (all but first)
            seq_element = "".join(text_fields[1:])
            text_fields = [text_fields[0], seq_element]

        # map text fields to data fields by index
        for field_name, field_idx in self._data_fields.items():
            data[field_name] = _build_data_from_text(text_fields[field_idx])

        return data
