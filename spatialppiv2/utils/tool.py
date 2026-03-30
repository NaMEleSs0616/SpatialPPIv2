"""Core utility helpers: protein embedding, PDB extraction, FASTA reading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def getConfig(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dict."""
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# PDB extraction
# ---------------------------------------------------------------------------


def extractPDB(pdb_path: str | Path, chain: str = "first") -> tuple[str, np.ndarray]:
    """
    Parse a PDB file and return (amino_acid_sequence, Cα_coordinates).

    Args:
        pdb_path: Path to a .pdb or .cif file.
        chain:    Chain ID to extract, or "first" to use the first chain found.

    Returns:
        sequence : one-letter amino acid string (length L)
        coords   : float32 array of shape (L, 3) — Cα coordinates in Å
    """
    _3TO1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "MSE": "M",
        "SEC": "U",
        "PYL": "O",
    }

    pdb_path = Path(pdb_path)
    seq_residues: list[tuple[int, str]] = []  # (res_seq, one_letter)
    ca_coords: dict[int, np.ndarray] = {}
    target_chain: str | None = None

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21]
            res_name = line[17:20].strip()
            res_seq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if target_chain is None:
                target_chain = chain if chain != "first" else chain_id
            if chain_id != target_chain:
                continue
            if atom_name == "CA" and res_name in _3TO1:
                if res_seq not in ca_coords:
                    ca_coords[res_seq] = np.array([x, y, z], dtype=np.float32)
                    seq_residues.append((res_seq, _3TO1[res_name]))

    seq_residues.sort(key=lambda t: t[0])
    sequence = "".join(r for _, r in seq_residues)
    coords = np.stack([ca_coords[s] for s, _ in seq_residues], axis=0)
    return sequence, coords


def read_fasta(fasta_path: str | Path) -> tuple[str, str]:
    """Return (header, sequence) from a single-record FASTA file."""
    fasta_path = Path(fasta_path)
    header, chunks = "", []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                header = line[1:]
            else:
                chunks.append(line)
    return header, "".join(chunks)


# ---------------------------------------------------------------------------
# ProtT5-XL embedder
# ---------------------------------------------------------------------------


class Embed:
    """
    Lazy-loaded ProtT5-XL (or ESM-2) sequence embedder.

    Usage::

        embedder = Embed("Rostlab/prot_t5_xl_uniref50", device)
        emb = embedder.encode("MKTAYIAKQR...")   # Tensor (L, D)
    """

    def __init__(self, model_name: str, device: torch.device | str) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ lazy
    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer

        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self._model = T5EncoderModel.from_pretrained(self.model_name)
        self._model.eval().to(self.device)

    # --------------------------------------------------------------- features
    @property
    def featureLen(self) -> int:
        return 1024  # ProtT5-XL hidden dim

    # ----------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encode a protein sequence into per-residue embeddings.

        Args:
            sequence: amino acid string (standard one-letter codes)

        Returns:
            Tensor of shape (L, featureLen) on CPU.
        """
        self._load()
        # ProtT5 expects space-separated residues with 'U' mapped to 'X'
        seq_spaced = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        ids = self._tokenizer(
            seq_spaced,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids["attention_mask"].to(self.device)
        emb = self._model(input_ids=input_ids, attention_mask=attention_mask)
        # shape: (1, L+1, D)  — drop EOS token
        return emb.last_hidden_state[0, : len(sequence), :].cpu()
