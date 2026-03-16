"""Resolve gene/sequence IDs to UniProt accessions and AlphaFold IDs."""

import re
import requests
from typing import Dict, List, Optional


def _extract_gene_name(seq_id: str) -> str:
    """Extract gene name from various ID formats.

    Examples:
        "FLT3_2438_GCC-GCA" -> "FLT3"
        "ABL1_ref" -> "ABL1"
        "sp|P36888|FLT3_HUMAN" -> "FLT3"
        "BRCA1" -> "BRCA1"
    """
    if "|" in seq_id:
        # SwissProt format: sp|P36888|FLT3_HUMAN
        parts = seq_id.split("|")
        if len(parts) >= 3:
            return parts[2].split("_")[0]
        return parts[1]
    if "_" in seq_id:
        return seq_id.split("_")[0]
    return seq_id


def _query_uniprot(gene_name: str) -> Optional[str]:
    """Query UniProt API for a gene name, return the best accession or None."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'gene_exact:"{gene_name}" AND reviewed:true',
        "fields": "accession,organism_id",
        "size": "5",
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0]["primaryAccession"]
    except Exception:
        pass
    return None


def resolve_gene_to_alphafold(sequence_ids: List[str]) -> Dict[str, str]:
    """Resolve a list of sequence IDs to AlphaFold IDs.

    Extracts gene names, queries UniProt for accessions, and constructs
    AlphaFold IDs (e.g., "AF-P36888-F1").

    Caches by gene name to avoid duplicate API calls.

    Args:
        sequence_ids: List of sequence identifiers from the dataset.

    Returns:
        Dict mapping sequence_id -> alphafold_id (empty string if unresolved).
    """
    # Extract unique gene names
    id_to_gene = {sid: _extract_gene_name(sid) for sid in sequence_ids}
    unique_genes = list(set(id_to_gene.values()))

    # Query UniProt for each unique gene
    gene_to_accession = {}
    for gene in unique_genes:
        acc = _query_uniprot(gene)
        gene_to_accession[gene] = acc

    # Build the mapping
    result = {}
    for sid in sequence_ids:
        gene = id_to_gene[sid]
        acc = gene_to_accession.get(gene)
        if acc:
            result[sid] = f"AF-{acc}-F1"
        else:
            result[sid] = ""

    return result
