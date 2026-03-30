"""
Structure downloader: RCSB first, AlphaFold DB fallback.

Usage (CLI)
-----------
    sppi-pdbs                   # download missing structures from cleaned_node.csv
    sppi-pdbs --all             # force re-download everything
    sppi-pdbs --gene TP53       # fetch a single gene
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from spatialppiv2.utils.config import get_config

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


# ---------------------------------------------------------------------------
# Public helpers (also imported by demo.ipynb)
# ---------------------------------------------------------------------------


def gene_to_pdb_id(gene: str) -> str | None:
    """
    Try to resolve a gene name to a RCSB PDB ID via the RCSB search API.
    Returns the first matching entry, or None.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_gene_name.value",
                        "operator": "exact_match",
                        "value": gene,
                    },
                }
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1}},
    }
    try:
        r = requests.post(url, json=query, timeout=10)
        r.raise_for_status()
        hits = r.json().get("result_set", [])
        return hits[0]["identifier"] if hits else None
    except Exception:
        return None


def download_rcsb(pdb_id: str, out_path: Path) -> bool:
    """Download a PDB file from RCSB. Returns True on success."""
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_alphafold(uniprot_id: str, out_path: Path) -> bool:
    """Download an AlphaFold structure. Returns True on success."""
    url = ALPHAFOLD_URL.format(uniprot_id=uniprot_id.upper())
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_gene(gene: str, pdb_dir: Path, force: bool = False) -> str:
    """
    Download structure for *gene*. Returns status: "ok" | "failed" | "skipped".
    Strategy: RCSB → AlphaFold fallback.
    """
    out = pdb_dir / f"{gene}.pdb"
    if out.exists() and not force:
        return "skipped"

    pdb_id = gene_to_pdb_id(gene)
    if pdb_id and download_rcsb(pdb_id, out):
        return "ok"

    # AlphaFold fallback requires UniProt ID — here we use the gene name
    # as a rough heuristic; a production system would look up via UniProt API.
    if download_alphafold(gene, out):
        return "ok"

    return "failed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Download PDB structures.")
    parser.add_argument("--node-csv", default=cfg["data"]["node_csv"])
    parser.add_argument("--pdb-dir", default=cfg["data"]["pdb_dir"])
    parser.add_argument("--gene", default=None, help="Download a single gene.")
    parser.add_argument("--all", action="store_true", help="Force re-download.")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    if args.gene:
        status = download_gene(args.gene, pdb_dir, force=args.all)
        print(f"{args.gene}: {status}")
        return

    node_csv = Path(args.node_csv)
    if not node_csv.exists():
        print(f"Node CSV not found: {node_csv}")
        return

    genes = pd.read_csv(node_csv)["name"].tolist()
    print(f"Downloading {len(genes)} structures to {pdb_dir}/")

    results: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}
    for gene in genes:
        status = download_gene(gene, pdb_dir, force=args.all)
        results[status] += 1
        print(f"  {gene}: {status}")
        time.sleep(0.1)  # be polite to RCSB

    print(f"\nDone — ok={results['ok']}, skipped={results['skipped']}, failed={results['failed']}")


if __name__ == "__main__":
    main()
