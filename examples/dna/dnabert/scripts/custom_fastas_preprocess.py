# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os

from bionemo.utils.fasta import FastaSplitNs


def create_output_dirs(base_output_dir):
    """
    Creates train, val, and test directories if they do not exist.

    Args:
        base_output_dir (str): The base directory where the train, val, and test directories will be created.
    """
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created directory: {split_dir}")


def split_fasta_ns(fasta_files, output_dir):
    """
    Splits FASTA sequences on 'N's and creates new FASTA files in train, val, and test directories.

    Args:
        fasta_files (list): List of FASTA file paths to process.
        output_dir (str): Base directory to save the processed files.
    """
    # Creating train, val, and test directories
    create_output_dirs(output_dir)

    # Processing each FASTA file
    for fasta_file in fasta_files:
        fasta_file = os.path.abspath(fasta_file)
        print(f"Processing file: {fasta_file}")

        try:
            # Split on Ns
            ns_splitter = FastaSplitNs([fasta_file])
            new_filename = ns_splitter.apply(fasta_file)  # Applying the split on 'N's
            print(f"Generated new file: {new_filename}")

            if "train" in fasta_file:
                split_dir = os.path.join(output_dir, "train")
            elif "val" in fasta_file:
                split_dir = os.path.join(output_dir, "val")
            elif "test" in fasta_file:
                split_dir = os.path.join(output_dir, "test")
            else:
                raise ValueError("Filename should contain 'train', 'val', or 'test' to determine the split.")

            new_path = os.path.join(split_dir, os.path.basename(new_filename))
            os.rename(new_filename, new_path)
            print(f"Processed file saved to {new_path}")

        except Exception as e:
            print(f"Error processing {fasta_file}: {e}")


if __name__ == "__main__":
    # Argparse setup for command-line arguments
    parser = argparse.ArgumentParser(
        description="Split FASTA files on 'N's and generate new files in train/val/test directories."
    )

    # Accepting a list of FASTA files and an output directory
    parser.add_argument(
        "fasta_files",
        nargs="+",
        help="List of FASTA files to process (should contain 'train', 'val', or 'test' in the filename)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Base directory to save processed files in train/val/test directories"
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    # Running the function to process files
    split_fasta_ns(args.fasta_files, output_dir)
