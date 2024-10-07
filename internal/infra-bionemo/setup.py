# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from setuptools import setup


if __name__ == "__main__":

    def has_requirement(x: str) -> bool:
        x = x.strip()
        if x.startswith("#"):
            return False
        return True

    def strip_version(x: str) -> str:
        x = x.strip()
        for s in (
            ">=",
            "<=",
            "<",
            ">",
            "==",
        ):
            if s in x:
                return x.split(s)[0]
        return x

    with open("./requirements.txt", "rt") as rt:
        requirements_no_versions = [strip_version(x) for x in rt if has_requirement(x)]

    setup(install_requires=requirements_no_versions)
