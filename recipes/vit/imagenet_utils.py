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
import math
import numbers
import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


try:
    from torchvision.transforms.functional import InterpolationMode

    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = "center"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)

_logger = logging.getLogger(__name__)


class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:
    """ToTensor with no rescaling of values"""

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        return F.pil_to_tensor(pil_img).to(dtype=self.dtype)


class MaybeToTensor(transforms.ToTensor):
    """Convert a PIL Image or ndarray to tensor if it's not already one."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic) -> torch.Tensor:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaybePILToTensor:
    """Convert a PIL Image to a tensor of the same type - this does not scale values."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        """
        Note: A deep copy of the underlying array is performed.

        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.pil_to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Pillow is deprecating the top-level resampling attributes (e.g., Image.BILINEAR) in
# favor of the Image.Resampling enum. The top-level resampling attributes will be
# removed in Pillow 10.
if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: "nearest",
        Image.Resampling.BILINEAR: "bilinear",
        Image.Resampling.BICUBIC: "bicubic",
        Image.Resampling.BOX: "box",
        Image.Resampling.HAMMING: "hamming",
        Image.Resampling.LANCZOS: "lanczos",
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: "nearest",
        Image.BILINEAR: "bilinear",
        Image.BICUBIC: "bicubic",
        Image.BOX: "box",
        Image.HAMMING: "hamming",
        Image.LANCZOS: "lanczos",
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: "nearest",
        InterpolationMode.BILINEAR: "bilinear",
        InterpolationMode.BICUBIC: "bicubic",
        InterpolationMode.BOX: "box",
        InterpolationMode.HAMMING: "hamming",
        InterpolationMode.LANCZOS: "lanczos",
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


_RANDOM_INTERPOLATION = (str_to_interp_mode("bilinear"), str_to_interp_mode("bicubic"))


class ResizeKeepRatio:
    """Resize and Keep Aspect Ratio"""

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation="bilinear",
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_scale_area=False,
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        """

        Args:
            size:
            longest:
            interpolation:
            random_scale_prob:
            random_scale_range:
            random_scale_area:
            random_aspect_prob:
            random_aspect_range:
        """
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.longest = float(longest)
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
        img,
        target_size,
        longest,
        random_scale_prob=0.0,
        random_scale_range=(1.0, 1.33),
        random_scale_area=False,
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        """Get parameters"""
        img_h, img_w = img_size = F.get_dimensions(img)[1:]
        target_h, target_w = target_size
        ratio_h = img_h / target_h
        ratio_w = img_w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )

        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            if random_scale_area:
                # make ratio factor equivalent to RRC area crop where < 1.0 = area zoom,
                # otherwise like affine scale where < 1.0 = linear zoom out
                ratio_factor = 1.0 / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1.0, 1.0)

        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            log_aspect = (
                math.log(random_aspect_range[0]),
                math.log(random_aspect_range[1]),
            )
            aspect_factor = math.exp(random.uniform(*log_aspect))
            aspect_factor = math.sqrt(aspect_factor)
            # currently applying random aspect adjustment equally to both dims,
            # could change to keep output sizes above their target where possible
            ratio_factor = (
                ratio_factor[0] / aspect_factor,
                ratio_factor[1] * aspect_factor,
            )

        size = [round(x * f / ratio) for x, f in zip(img_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img,
            self.size,
            self.longest,
            self.random_scale_prob,
            self.random_scale_range,
            self.random_scale_area,
            self.random_aspect_prob,
            self.random_aspect_range,
        )
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = F.resize(img, size, interpolation)
        return img

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [interp_mode_to_str(x) for x in self.interpolation]
            )
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += f", interpolation={interpolate_str}"
        format_string += f", longest={self.longest:.3f}"
        format_string += f", random_scale_prob={self.random_scale_prob:.3f}"
        format_string += f", random_scale_range=({self.random_scale_range[0]:.3f}, {self.random_aspect_range[1]:.3f})"
        format_string += f", random_aspect_prob={self.random_aspect_prob:.3f}"
        format_string += f", random_aspect_range=({self.random_aspect_range[0]:.3f}, {self.random_aspect_range[1]:.3f}))"
        return format_string


def _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            _logger.warning("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        img_w, img_h = F.get_image_size(img)
        area = img_w * img_h

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if target_w <= img_w and target_h <= img_h:
                i = random.randint(0, img_h - target_h)
                j = random.randint(0, img_w - target_w)
                return i, j, target_h, target_w

        # Fallback to central crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            target_w = img_w
            target_h = int(round(target_w / min(ratio)))
        elif in_ratio > max(ratio):
            target_h = img_h
            target_w = int(round(target_h * max(ratio)))
        else:  # whole image
            target_w = img_w
            target_h = img_h
        i = (img_h - target_h) // 2
        j = (img_w - target_w) // 2
        return i, j, target_h, target_w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [interp_mode_to_str(x) for x in self.interpolation]
            )
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def center_crop_or_pad(
    img: torch.Tensor,
    output_size: Union[int, List[int]],
    fill: Union[int, Tuple[int, int, int]] = 0,
    padding_mode: str = "constant",
) -> torch.Tensor:
    """Center crops and/or pads the given image.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    output_size = _setup_size(output_size)
    crop_height, crop_width = output_size
    _, image_height, image_width = F.get_dimensions(img)

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(
        self,
        size: Union[int, List[int]],
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = "constant",
    ):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(
            img, self.size, fill=self.fill, padding_mode=self.padding_mode
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def crop_or_pad(
    img: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    fill: Union[int, Tuple[int, int, int]] = 0,
    padding_mode: str = "constant",
) -> torch.Tensor:
    """Crops and/or pads image to meet target size, with control over fill and padding_mode."""
    _, image_height, image_width = F.get_dimensions(img)
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(image_width, left), 0),
            max(bottom - max(image_height, top), 0),
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)

    top = max(top, 0)
    left = max(left, 0)
    return F.crop(img, top, left, height, width)


class RandomCropOrPad(torch.nn.Module):
    """Crop and/or pad image with random placement within the crop or pad margin."""

    def __init__(
        self,
        size: Union[int, List[int]],
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = "constant",
    ):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, size):
        _, image_height, image_width = F.get_dimensions(img)
        delta_height = image_height - size[0]
        delta_width = image_width - size[1]
        top = int(math.copysign(random.randint(0, abs(delta_height)), delta_height))
        left = int(math.copysign(random.randint(0, abs(delta_width)), delta_width))
        return top, left

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        top, left = self.get_params(img, self.size)
        return crop_or_pad(
            img,
            top=top,
            left=left,
            height=self.size[0],
            width=self.size[1],
            fill=self.fill,
            padding_mode=self.padding_mode,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device="cuda",
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == "rand":
            self.rand_color = True  # per block random normal
        elif self.mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for attempt in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f"(p={self.probability}, mode={self.mode}"
        fs += f", count=({self.min_count}, {self.max_count}))"
        return fs


def patchify_image(
    img: torch.Tensor,
    patch_size: Tuple[int, int],
    pad: bool = True,
    include_info: bool = True,
    flatten_patches: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    c, h, w = img.shape
    ph, pw = patch_size

    # Ensure the image is divisible by patch size
    if pad and (h % ph != 0 or w % pw != 0):
        pad_h = (ph - h % ph) % ph  # amount to add on bottom
        pad_w = (pw - w % pw) % pw  # amount to add on right
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        c, h, w = img.shape

    # Calculate number of patches in each dimension
    nh, nw = h // ph, w // pw
    # Reshape image to patches
    patches = img.view(c, nh, ph, nw, pw).permute(1, 3, 2, 4, 0)
    # [nh, nw, ph, pw, c] -> [nh * nw, ph * pw * c] or [nh * nw, ph, pw, c]
    patches = (
        patches.reshape(-1, ph * pw * c)
        if flatten_patches
        else patches.reshape(-1, ph, pw, c)
    )

    if include_info:
        # Create coordinate indices
        y_idx, x_idx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
        # Stack into a single coords tensor [N, 2] with (y, x) order
        coord = torch.stack([y_idx.reshape(-1), x_idx.reshape(-1)], dim=1)
        # Create type indicators (all 1s for regular patches)
        valid = torch.ones(nh * nw, dtype=torch.bool)
        return patches, coord, valid

    return patches


class Patchify(torch.nn.Module):
    """Transform an image into patches with corresponding coordinates and type indicators."""

    def __init__(
        self, patch_size: Union[int, Tuple[int, int]], flatten_patches: bool = True
    ):
        super().__init__()
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.flatten_patches = flatten_patches

    def forward(self, img):
        """
        Args:
            img: A PIL Image or tensor of shape [C, H, W]

        Returns:
            A dictionary containing:
                - patches: Tensor of shape [N, P*P*C] if flatten_patches=True,
                          or [N, Ph, Pw, C] if flatten_patches=False
                - patch_coord: Tensor of shape [N, 2] with (y, x) coordinates
                - patch_valid: Valid indicator (all 1s for non-padding patches)
        """
        if isinstance(img, Image.Image):
            # Convert PIL Image to tensor [C, H, W]
            img = transforms.functional.to_tensor(img)

        patches, coord, valid = patchify_image(
            img, self.patch_size, flatten_patches=self.flatten_patches
        )

        return {
            "patches": patches,
            "patch_coord": coord,
            "patch_valid": valid,
        }


class TrimBorder(torch.nn.Module):
    def __init__(
        self,
        border_size: int,
    ):
        super().__init__()
        self.border_size = border_size

    def forward(self, img):
        w, h = F.get_image_size(img)
        top = left = self.border_size
        top = min(top, h)
        left = min(left, h)
        height = max(0, h - 2 * self.border_size)
        width = max(0, w - 2 * self.border_size)
        return F.crop(img, top, left, height, width)


def transforms_imagenet_train(
    img_size: Union[int, Tuple[int, int]] = 224,
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    train_crop_mode: Optional[str] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter: Union[float, Tuple[float, ...]] = 0.4,
    color_jitter_prob: Optional[float] = None,
    grayscale_prob: float = 0.0,
    gaussian_blur_prob: float = 0.0,
    interpolation: str = "random",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    re_prob: float = 0.0,
    re_mode: str = "const",
    re_count: int = 1,
    re_num_splits: int = 0,
    normalize: bool = True,
    separate: bool = False,
    patch_size: Union[int, Tuple[int, int]] = 16,
    patchify: bool = False,
):
    """ImageNet-oriented image transforms for training.

    Args:
        img_size: Target image size.
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        force_color_jitter: Force color jitter where it is normally disabled (ie with RandAugment on).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalize tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.
        naflex: Enable NaFlex mode, sequence constrained patch output
        patch_size: Patch size for NaFlex mode.
        max_seq_len: Max sequence length for NaFlex mode.

    Returns:
        If separate==True, the transforms are returned as a tuple of 3 separate transforms
          for use in a mixing dataset that passes
            * all data through the first (primary) transform, called the 'clean' data
            * a portion of the data through the secondary transform
            * normalizes and converts the branches above with the third, final transform
    """
    train_crop_mode = train_crop_mode or "rrc"
    assert train_crop_mode in {"rrc", "rkrc", "rkrr"}

    primary_tfl = []
    if train_crop_mode in ("rkrc", "rkrr"):
        # FIXME integration of RKR is a WIP
        scale = tuple(scale or (0.8, 1.00))
        ratio = tuple(ratio or (0.9, 1 / 0.9))
        primary_tfl += [
            ResizeKeepRatio(
                img_size,
                interpolation=interpolation,
                random_scale_prob=0.5,
                random_scale_range=scale,
                random_scale_area=True,  # scale compatible with RRC
                random_aspect_prob=0.5,
                random_aspect_range=ratio,
            ),
            CenterCropOrPad(img_size, padding_mode="reflect")
            if train_crop_mode == "rkrc"
            else RandomCropOrPad(img_size, padding_mode="reflect"),
        ]
    else:
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
        primary_tfl += [
            RandomResizedCropAndInterpolation(
                img_size,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
            )
        ]

    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if color_jitter is not None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        if color_jitter_prob is not None:
            secondary_tfl += [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(*color_jitter),
                    ],
                    p=color_jitter_prob,
                )
            ]
        else:
            secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    if grayscale_prob:
        secondary_tfl += [transforms.RandomGrayscale(p=grayscale_prob)]

    if gaussian_blur_prob:
        secondary_tfl += [
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=23),  # hardcoded for now
                ],
                p=gaussian_blur_prob,
            )
        ]

    final_tfl = []
    if not normalize:
        # when normalize disable, converted to tensor without scaling, keeps original dtype
        final_tfl += [MaybePILToTensor()]
    else:
        final_tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]
        if re_prob > 0.0:
            final_tfl += [
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cpu",
                )
            ]

    if patchify:
        final_tfl += [Patchify(patch_size=patch_size)]

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
    img_size: Union[int, Tuple[int, int]] = 224,
    crop_pct: Optional[float] = None,
    crop_mode: Optional[str] = None,
    crop_border_pixels: Optional[int] = None,
    interpolation: str = "bilinear",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    normalize: bool = True,
    patch_size: Union[int, Tuple[int, int]] = 16,
    patchify: bool = False,
):
    """ImageNet-oriented image transform for evaluation and inference.

    Args:
        img_size: Target image size.
        crop_pct: Crop percentage. Defaults to 0.875 when None.
        crop_mode: Crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Trim a border of specified # pixels around edge of original image.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalize tensor output w/ provided mean/std (if prefetcher not used).
        naflex: Enable NaFlex mode, sequence constrained patch output
        patch_size: Patch size for NaFlex mode.
        max_seq_len: Max sequence length for NaFlex mode.
        patchify: Patchify the output instead of relying on prefetcher

    Returns:
        Composed transform pipeline
    """
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    tfl = []

    if crop_border_pixels:
        tfl += [TrimBorder(crop_border_pixels)]

    if crop_mode == "squash":
        # squash mode scales each edge to 1/pct of target, then crops
        # aspect ratio is not preserved, no img lost if crop_pct == 1.0
        tfl += [
            transforms.Resize(
                scale_size, interpolation=str_to_interp_mode(interpolation)
            ),
            transforms.CenterCrop(img_size),
        ]
    elif crop_mode == "border":
        # scale the longest edge of image to 1/pct of target edge, add borders to pad, then crop
        # no image lost if crop_pct == 1.0
        fill = [round(255 * v) for v in mean]
        tfl += [
            ResizeKeepRatio(scale_size, interpolation=interpolation, longest=1.0),
            CenterCropOrPad(img_size, fill=fill),
        ]
    else:
        # default crop model is center
        # aspect ratio is preserved, crops center within image, no borders are added, image is lost
        if scale_size[0] == scale_size[1]:
            # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
            tfl += [
                transforms.Resize(
                    scale_size[0], interpolation=str_to_interp_mode(interpolation)
                )
            ]
        else:
            # resize the shortest edge to matching target dim for non-square target
            tfl += [ResizeKeepRatio(scale_size)]
        tfl += [transforms.CenterCrop(img_size)]

    if not normalize:
        # when normalize disabled, converted to tensor without scaling, keeps original dtype
        tfl += [MaybePILToTensor()]
    else:
        tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]

    if patchify:
        tfl += [Patchify(patch_size=patch_size)]

    return transforms.Compose(tfl)
