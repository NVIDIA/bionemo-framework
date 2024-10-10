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


"""
These GFFs are used exclusively for training. This assumption makes aggressive specific parsers a valid approach:
    - GFF is a very general format
    - We mostly care about reproducing an existing project
    - Future datasets that do not conform may want to use their own parsing method anyway.

Provide a set of solutions for GFF files and leave room for future improvement:
    - A generic GFF entry validation tool
    - namespaced parsing tools
    - Leave room for extension based on assumptions about what is contained in the attributes column.

"""

from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, validator


# NOTE: This class may be more well suited for a MixinPattern. Where we are providing mixins that operate on the attribute column. If a need occurs, we will need to redesign this class a little.
class GFFEntry(BaseModel):
    """Class for validating GFF3 entries. https://uswest.ensembl.org/info/website/upload/gff3.html"""

    seqid: str
    source: str
    type: str
    start: int
    end: int
    score: Union[float, Literal["."]]
    strand: Literal["+", "-", "."]
    phase: Literal["0", "1", "2", "."]
    attributes: str  # Using a blob to store these for now

    @validator("type")
    def lower_type(t):
        return t.lower()

    def attributes_as_dict(self) -> Dict[str, str]:
        """Conforms to the GFF3 spec. Returns the attributes column as a dictionary. Key value pairs are separated with semicolon (;) and specified with equals (=).
            Ex: Key1=Value1;Key2=Value2;

        Returns:
            Dict[str, str]: Dictionary mapping keys to values from the attribute field.
        """
        stuff = self.attributes.split(";")
        attrs_as_dict = {}
        for items in stuff:
            key, value = items.split("=")
            attrs_as_dict[key] = value
        return attrs_as_dict


class SpliceSiteEntry(GFFEntry):
    """Small extension to GFFEntry that adds the ability to parse the Parent field from the attributes column. This is necessary for constructing datasets for the downstream tasks."""

    def get_parent(self) -> Optional[str]:
        """Returns the value of the Parent field in the attribute column. That is, the value of Parent={value}.

        Returns:
            Optional[str]: The value of the Parent field in the attribute column, None if no such field exists.
        """
        # We might want to propagate the KeyError on failure.
        return self.attributes_as_dict()["Parent"]


T = TypeVar("T", bound=GFFEntry)


def parse_gff3(filename: str, cls: T = SpliceSiteEntry) -> List[T]:
    """Parses a gff3 file and validates all the lines. Class T is provided so users can

    Args:
        filename (str): filename of the gff3 file to parse.
        cls (T): The kind of GFFEntry (T) class to construct.

    Returns:
        List[T]: List of entries for each row in the passed gff3 file, parsed as the specific GFFEntry (T) object.
    """
    entries = []
    with open(filename, "r") as fd:
        for line in fd:
            if line[0] == "#":
                continue
            line_parts = line.strip().split("\t")
            entry = cls(
                seqid=line_parts[0],
                source=line_parts[1],
                type=line_parts[2],
                start=line_parts[3],
                end=line_parts[4],
                score=line_parts[5],
                strand=line_parts[6],
                phase=line_parts[7],
                attributes=line_parts[8],
            )
            entries.append(entry)
    return entries


def _get_exons_by_parent(entries: List[SpliceSiteEntry]) -> Dict[str, List[SpliceSiteEntry]]:
    """Internal method for creating a dictionary of seqid:parent_id -> Exon list.

    Args:
        entries (List[SpliceSiteEntry]): List of parsed entries.

    Returns:
        Dict[str, List[SpliceSiteEntry]]: Dictionary that maps seqid:parent_id to child exon rows.
    """
    exon_groups = defaultdict(list)
    for entry in entries:
        if entry.type == "exon" and (parent := entry.get_parent()) is not None:
            key = f"{entry.seqid}:{parent}"
            exon_groups[key].append(entry)
    return exon_groups


def get_seqid_from_key(key):
    """Given a key in our parent dictionary, returns the parsed sequence id."""
    seqid, _ = key.split(":")
    return seqid


def _build_donor_acceptors_midpoints(
    exon_groups: Dict[str, List[SpliceSiteEntry]],
) -> Dict[str, Tuple[List[SpliceSiteEntry], List[SpliceSiteEntry], List[SpliceSiteEntry]]]:
    """Internal method used for going from exon groupings to donor, acceptor, and midpoint lists.

    Args:
        exon_groups (Dict[str, List[SpliceSiteEntry]]): _description_

    Returns:
        Dict[str, Tuple[List[SpliceSiteEntry], List[SpliceSiteEntry], List[SpliceSiteEntry]]]: _description_
    """

    # Might want to break this into another function
    by_parent = {}
    for parent, exons in exon_groups.items():
        if len(exons) == 1:
            continue

        donors, acceptors = _get_donors_acceptors(exons)
        midpoints = _get_intron_midpoints(exons)
        by_parent[parent] = donors, acceptors, midpoints
    return by_parent


def build_donor_acceptors_midpoints(
    gff_entries: List[SpliceSiteEntry],
) -> Dict[str, Tuple[List[SpliceSiteEntry], List[SpliceSiteEntry], List[SpliceSiteEntry]]]:
    """Creates a dictionary mapping locus IDs to donor, acceptor, and midpoint lists from a set of GFF lines.

    Args:
        gff_entries: list of parsed and validated SpliceSiteEntries.

    Returns:
        Dictionary mapping transcript IDs to donor, acceptor, and midpoint lists.
    """
    parent_lookup = _get_exons_by_parent(gff_entries)
    donor_acceptors_midpoints = _build_donor_acceptors_midpoints(parent_lookup)
    return donor_acceptors_midpoints


def _get_intron_midpoints(exons: List[SpliceSiteEntry]) -> List[int]:
    midpoints = []
    for i, j in zip(range(0, len(exons)), range(1, len(exons))):
        midpoint = exons[i].end + (exons[j].start - exons[i].end) // 2
        midpoints.append(midpoint)
    return midpoints


def _is_exon_start_order_consistent(exons: List[SpliceSiteEntry]) -> bool:
    """Checks that for all consecutive pairs of exons, the start position of the first exon in the pair occurs first.

    Args:
        exons (List[SpliceSiteEntry]): List of exons within a single gene body.

    Returns:
        bool: returns true if the exons follow an ascending order.
    """
    return all(exons[i].start < exons[j].start for i, j in zip(range(0, len(exons)), range(1, len(exons))))


def _get_donors_acceptors(exons: List[SpliceSiteEntry]) -> Tuple[List[int], List[int]]:
    """Internal method for getting lists of donors and acceptors of this set of exons.

    Args:
        exons (List[SpliceSiteEntry]): exons to fetch donors and acceptors from.

    Returns:
        Tuple[List[int], List[int]]: Tuple containing two lists, a list of donors and a list of acceptors.
    """
    donors, acceptors = [], []
    for i, exon in enumerate(exons):
        # Python 3.10 please, this would be so much cleaner in a `match` statement

        # Ignore the first exon for acceptors on the forward strand, and for donors on the reverse strand
        if i > 0:
            if exon.strand == "+":
                acceptors.append(exon.start)
            elif exon.strand == "-":
                donors.append(exon.start)
            else:
                raise ValueError(f"Unexpected strand detected: {exon.strand}")

        # Ignore the last exon for donors on the forward strand, and for acceptors on the reverse strand
        if i < len(exons) - 1:
            if exon.strand == "+":
                donors.append(exon.end)
            elif exon.strand == "-":
                acceptors.append(exon.end)
            else:
                raise ValueError(f"Unexpected strand detected: {exon.strand}")

    return donors, acceptors
