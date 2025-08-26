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

import logging
import os
import pickle
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset


_logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")  # singleton, kept public for bwd compat use
_IMG_EXTENSIONS_SET = set(IMG_EXTENSIONS)  # set version, private, kept in sync


def infinite_dataloader(dataloader, sampler):
    """Create an infinite iterator that automatically restarts at the end of each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def get_img_extensions(as_set=False):
    return deepcopy(_IMG_EXTENSIONS_SET if as_set else IMG_EXTENSIONS)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def load_class_map(map_or_filename: str):
    """
    Parse a TSV or PKL file that contains a list of class IDs. Then create a class-to-index mapping
    where the enumerated index will represent the class index when computing the cross-entropy loss.

    Args:
        map_or_filename (str): Path to a TSV or PKL file that contains a list of class IDs.

    Returns:
        class_to_idx: dict mapping class IDs to class indices.
    """
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == ".txt" or class_map_ext == ".tsv":
        with open(map_or_filename) as f:
            class_to_idx = {}
            for idx, line in enumerate(f):
                if "\t" in line:
                    # Parse a TSV labeling format: (Class ID, Class Semantic Labels)
                    class_id, _ = line.strip().split("\t")
                    class_to_idx[class_id] = idx
                else:
                    # Parse a plain text labeling format: (Class ID)
                    class_to_idx[line.strip()] = idx
    elif class_map_ext == ".pkl":
        with open(map_or_filename, "rb") as f:
            class_to_idx = pickle.load(f)
    else:
        assert False, f"Unsupported class map file extension ({class_map_ext})."
    return class_to_idx


def load_image_labels(map_or_filename: str):
    """
    Parse a TSV or PKL file that maps image filenames to class IDs.

    Args:
        map_or_filename (str): Path to a TSV or PKL file that maps image filenames to class IDs.

    Returns:
        image_to_label: dict mapping image filenames to class IDs.
    """
    image_label_ext = os.path.splitext(map_or_filename)[-1].lower()
    if image_label_ext == ".txt" or image_label_ext == ".tsv":
        with open(map_or_filename) as f:
            image_to_label = {}
            for line in f:
                # Parse a TSV format: (Image Filename, Class ID, etc.)
                image_filename, class_id, *_ = line.strip().split("\t")
                image_to_label[image_filename] = class_id
    elif image_label_ext == ".pkl":
        with open(map_or_filename, "rb") as f:
            image_to_label = pickle.load(f)
    else:
        assert False, f"Unsupported image label file extension ({image_label_ext})."
    return image_to_label


def find_images_and_targets(
    folder: str,
    types: Optional[Union[List, Tuple, Set]] = None,
    class_to_idx: Optional[Dict] = None,
    image_to_label: Optional[Dict] = None,
    sort: bool = True,
    class_filter: Optional[List[Any]] = None,
):
    """
    Walk folder recursively to discover images and map them to classes by folder names.

    Args:
        folder: root of folder to recursively search
        types: types (file extensions) to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        image_to_label: specify mapping for image filename to class index if set
        sort: re-sort found images by name (for consistent ordering)
        class_filter: filter out images with class labels not in this list

    Returns:
        A list of image and target tuples, class_to_idx mapping
    """
    types = get_img_extensions(as_set=True) if not types else set(types)
    labels = []
    filenames = []
    for root, _, files in os.walk(folder, topdown=False, followlinks=True):
        for f in files:
            file_basename, ext = os.path.splitext(f)
            if ext.lower() in types:
                # Check for labels from either the filename or a label dictionary.
                if image_to_label is not None and f in image_to_label:
                    label = image_to_label[f]
                else:
                    # Get the class ID from the filename, and remove _N suffixes.
                    label = file_basename.split("_")[0]
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [
        (f, class_to_idx[l])
        for f, l in zip(filenames, labels)
        if l in class_to_idx and (class_filter is None or l in class_filter)
    ]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ImageNetReader:
    def __init__(
        self,
        root: str,
        class_map: Optional[str | dict] = None,
        label_map: Optional[str | dict] = None,
        class_filter: Optional[List[Any]] = None,
    ):
        super().__init__()

        self.root = root
        class_to_idx = None
        image_to_label = None
        if isinstance(class_map, str):
            class_to_idx = load_class_map(class_map)
        elif isinstance(class_map, dict):
            assert dict, "Class-to-Index mapping dict must be non-empty."
            class_to_idx = class_map
        if isinstance(label_map, str):
            image_to_label = load_image_labels(label_map)
        elif isinstance(label_map, dict):
            assert dict, "Image-to-Label mapping dict must be non-empty."
            image_to_label = label_map
        self.samples, self.class_to_idx = find_images_and_targets(
            root,
            class_to_idx=class_to_idx,
            image_to_label=image_to_label,
            types=get_img_extensions(as_set=True),
            class_filter=class_filter if class_filter is not None else None,
        )
        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {root}. "
                f"Supported image extensions are {', '.join(get_img_extensions())}"
            )

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [
            self._filename(index, basename=basename, absolute=absolute)
            for index in range(len(self))
        ]


class ImageNetDataset(Dataset):
    """
    ImageDataset class for loading image datasets from a root directory.

    Expects the following directory structure:

    tiny-imagenet-200
    ├── test
    │   └── images                      # JPEG Images
    ├── train
    │   ├── n01443537                   # Class ID
    │   │   └── images                  # JPEG Images
    |   |       └── n01443537_0.JPEG
    |   |       └── n01443537_1.JPEG
    |   |       └── ...
    │   ├── n01629819                   # Class ID
    │   │   └── images                  # JPEG Images
    |   |       └── n01629819_0.JPEG
    |   |       └── n01629819_1.JPEG
    |   |       └── ...
    |   ...
    ├── val
    |   ├── images                      # JPEG Images
    |   └── val_annotations.txt         # JPEG Filename + Class ID TSV (One Sample Per Line)
    ├── wnids.txt                       # Class ID List (One Class Per Line)
    └── words.txt                       # Class ID + Semantic Label TSV (One Class Per Line)
    """

    def __init__(
        self,
        root,
        reader=None,
        class_map: Optional[str | dict] = None,
        label_map: Optional[str | dict] = None,
        load_bytes=False,
        input_img_mode="RGB",
        transform=None,
        target_transform=None,
        class_filter: Optional[List[Any]] = None,
        **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = ImageNetReader(
                root=root,
                class_map=class_map,
                label_map=label_map,
                class_filter=class_filter if class_filter is not None else None,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(
                f"Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}"
            )
            self._consecutive_errors += 1
            if self._consecutive_errors < 50:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)
