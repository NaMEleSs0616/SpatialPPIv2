"""Prey protein retrieval from UniProt."""

from __future__ import annotations

import time
from pathlib import Path

import requests


UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
UNIPROT_SEARCH_URL = (
    "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene}+AND+organism_id:9606"
    "&fields=accession,gene_names&format=json&size=1"
)


def gene_to_uniprot(gene: str) -> str | None:
    """Resolve a human gene name to a UniProt accession. Returns None on failure."""
    try:
        r = requests.get(UNIPROT_SEARCH_URL.format(gene=gene), timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["primaryAccession"] if results else None
    except Exception:
        return None


def download_fasta(gene: str, out_dir: Path, force: bool = False) -> bool:
    """Fetch FASTA from UniProt for *gene*. Returns True on success."""
    out = out_dir / f"{gene}.fasta"
    if out.exists() and not force:
        return True
    acc = gene_to_uniprot(gene)
    if not acc:
        return False
    try:
        r = requests.get(UNIPROT_FASTA_URL.format(acc=acc), timeout=15)
        if r.status_code == 200:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(r.text)
            return True
    except requests.RequestException:
        pass
    return False


def bulk_download(genes: list[str], out_dir: Path, delay: float = 0.2) -> dict[str, bool]:
    """Download FASTA files for a list of gene names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for gene in genes:
        ok = download_fasta(gene, out_dir)
        results[gene] = ok
        print(f"  {gene}: {'ok' if ok else 'failed'}")
        time.sleep(delay)
    return results
