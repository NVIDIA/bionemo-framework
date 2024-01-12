# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from bionemo.utils.remote import RemoteResource


__all__ = ["ResourcePreprocessor"]


@dataclass
class ResourcePreprocessor(ABC):
    """Interface defining a ResourcePreprocessor. Implementors promise to provide both a complete RemoteResource and a freeform
    preprocess method. This interface can be used to generically define a workflow from a config file.

        remote -> prepare -> prepared data.
    """

    root_directory: Optional[str] = RemoteResource.get_env_tmpdir()
    dest_directory: str = 'data'

    def get_checksums(self) -> List[str]:
        return [resource.checksum for resource in self.get_remote_resources()]

    def get_urls(self) -> List[str]:
        return [resource.url for resource in self.get_remote_resources()]

    @abstractmethod
    def get_remote_resources(self) -> List[RemoteResource]:
        """Gets the remote resources associated with this preparor."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> List:
        """Returns a list of prepared filenames."""
        raise NotImplementedError
