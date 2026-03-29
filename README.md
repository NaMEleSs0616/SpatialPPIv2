# SpatialPPIv2

**Protein-protein interaction scoring via ProtT5-XL embeddings and Graph Neural Networks.**

[![CI](https://github.com/NaMEleSs0616/SpatialPPIv2/actions/workflows/ci.yml/badge.svg)](https://github.com/NaMEleSs0616/SpatialPPIv2/actions)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

SpatialPPIv2 predicts whether two proteins physically interact by combining **sequence-level representations** from a 650M-parameter protein language model with **structural context** encoded as a graph.

The pipeline:

1. **FASTA retrieval** — fetch protein sequences from UniProt / RCSB
2. **Structure download** — RCSB first, AlphaFold DB fallback
3. **Embedding extraction** — encode each sequence with ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`), producing per-residue feature vectors
4. **Graph construction** — build a residue contact graph from Cα–Cα distances (≤ 8 Å threshold); ProtT5 embeddings serve as node features
5. **GNN inference** — a multi-layer Graph Neural Network reads the interaction graph and outputs an interaction probability in [0, 1]

A lighter-weight variant (`ESM-2+ac`) substitutes ESM-2 attention maps for structural coordinates, enabling inference directly from sequence when no structure is available.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/NaMEleSs0616/SpatialPPIv2.git
cd SpatialPPIv2

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install (CPU)
pip install -e .

# Install with GPU support (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Install dev extras (linting, tests)
pip install -e ".[dev]"
```

> **Note:** Model weights are not included in this repository.  
> Download checkpoints from [Releases](https://github.com/NaMEleSs0616/SpatialPPIv2/releases) and place them in `checkpoint/`.

---

## Quick Start

### Single-pair inference

```bash
# Structure-based (ProtT5 — recommended)
sppi-infer --A data/raw/proteinA.pdb --B data/raw/proteinB.pdb

# Sequence-only (ESM-2+ac — no structure required)
sppi-infer --A data/raw/proteinA.fasta --B data/raw/proteinB.fasta

# Specify chain and device
sppi-infer --A data/raw/proteinA.pdb --chain_A B \
           --B data/raw/proteinB.pdb --chain_B A \
           --device cuda
```

### Batch scoring

```bash
# Download all PDB structures listed in cleaned_node.csv
sppi-pdbs

# Re-attempt only previously failed downloads
sppi-pdbs  # (default behaviour — retries failed genes)
sppi-pdbs --all  # force re-download of everything

# Score all pairs in cleaned_edge.csv
sppi-score

# Override paths and device
sppi-score --edge-csv data/raw/cleaned_edge.csv \
           --pdb-dir  data/processed/pdbs \
           --out-csv  results/ppi_scores.csv \
           --device   cpu
```

Batch scoring automatically **resumes** from a partial `ppi_scores.csv` — safe to interrupt and restart.

---

## Project Structure

```
SpatialPPIv2/
├── spatialppiv2/               # Installable Python package
│   ├── data/
│   │   ├── download_pdbs.py    # RCSB + AlphaFold structure downloader
│   │   └── download_preys.py   # Prey protein retrieval
│   ├── models/                 # GNN architecture (getModel)
│   ├── scoring/
│   │   ├── inference.py        # Single-pair prediction
│   │   └── run_scoring.py      # Batch scoring with resume support
│   └── utils/
│       ├── config.py           # YAML config loader
│       ├── dataset.py          # Graph construction (build_data, build_data_from_adj)
│       ├── model.py            # Model factory (getModel)
│       └── tool.py             # Embed, extractPDB, read_fasta, getConfig
│
├── config/
│   └── default.yaml            # All hyperparameters and paths
│
├── checkpoint/                 # Model weights (not tracked by git)
│   ├── SpatialPPIv2_ProtT5.ckpt
│   └── SpatialPPIv2_ESM.ckpt
│
├── data/
│   ├── raw/                    # Input CSVs (cleaned_node.csv, cleaned_edge.csv)
│   └── processed/              # Downloaded PDBs, manifest
│
├── tests/                      # Pytest test suite
├── .github/workflows/ci.yml    # GitHub Actions CI
└── pyproject.toml              # Package metadata and dependencies
```

---

## Configuration

All paths and hyperparameters live in `config/default.yaml`:

```yaml
basic:
  hidden_dim: 256
  num_layers: 4
  dropout: 0.2

training:
  epochs: 100
  learning_rate: 1e-4
  early_stopping_patience: 10

data:
  pdb_dir: data/processed/pdbs
  edge_csv: data/raw/cleaned_edge.csv
  contact_threshold_angstrom: 8.0
```

Pass `--config path/to/custom.yaml` to any CLI command to override.

---

## Input Format

| File | Required columns | Description |
|------|-----------------|-------------|
| `cleaned_node.csv` | `name`, `type` | One row per protein; `type` is `bait` or `prey` |
| `cleaned_edge.csv` | `source`, `target` | Protein pairs to score; extra columns are preserved in output |

---

## Output

`ppi_scores.csv` — all columns from `cleaned_edge.csv` plus:

| Column | Description |
|--------|-------------|
| `spatial_score` | Predicted interaction probability [0, 1] |
| `status` | `ok`, `missing_bait_pdb`, `missing_prey_pdb`, or `error: <msg>` |

---

## Models

| Model | Input | Backbone | Notes |
|-------|-------|----------|-------|
| ProtT5 | PDB / CIF structure | `Rostlab/prot_t5_xl_uniref50` (650M) | Recommended — uses 3D coordinates |
| ESM-2+ac | FASTA sequence | `esm2_t33_650M_UR50D` | Structure-free; attention maps replace coordinates |

---

## Graph Attention Network (GATv2)

The core encoder uses **GATv2Conv** layers instead of a standard GCN:

- Multi-head attention (default: 4 heads) over residue contact graphs
- Residual connections + LayerNorm between every layer
- Edge features (Cα–Cα distances or ESM-2 contact weights) fed directly into attention
- Flexible readout: `mean`, `max`, or `both` (mean ∥ max → 2× hidden dim)
- Pair representation: `[h_A ∥ h_B ∥ |h_A−h_B| ∥ h_A⊙h_B]` — borrowed from NLI, well-suited to symmetric interaction tasks

```yaml
# config/default.yaml
basic:
  hidden_dim: 256
  num_layers: 4
  heads: 4
  readout: "mean"
```

---

## AlphaFold pLDDT Features

When PDB files originate from AlphaFold DB, **per-residue pLDDT confidence scores** are automatically extracted from the B-factor column and appended as an extra node feature (normalised to [0, 1]).

This gives the GNN a direct signal about which regions of a predicted structure are reliable, allowing it to down-weight low-confidence loops during attention. RCSB structures with crystallographic B-factors outside [0, 100] are automatically detected and the feature is silently skipped — no config needed.

---

## Contrastive Pre-training (SimCLR)

Before supervised fine-tuning on labelled PPI pairs, the GATv2 encoder can be pre-trained with **NT-Xent contrastive loss** — no interaction labels required.

**How it works:**
1. Each protein graph is augmented twice → two "views" of the same protein
2. The encoder + a non-linear projection head maps each view to a 128-d hypersphere
3. NT-Xent pulls the two views of the same protein together and pushes all other proteins in the batch apart
4. After pre-training, the projection head is discarded; only the encoder weights carry forward

**Augmentations applied stochastically:**

| Augmentation | What it does |
|---|---|
| Node dropout | Zeros out random residue features (p=0.1) |
| Edge dropout | Removes random contact edges (p=0.2) |
| Gaussian noise | Adds small noise to node embeddings (σ=0.02) |
| Subgraph crop | Retains a contiguous 70–100% subsequence |

```bash
# Pre-train then fine-tune in one command
sppi-train --pretrain --pretrain-epochs 20 --epochs 100 --device cuda

# Fine-tuning only
sppi-train --epochs 100 --device cuda
```

---

## Embedding API

SpatialPPIv2 ships a **Cohere-compatible REST API** for serving protein embeddings. Graph-level embeddings are L2-normalised and suitable for cosine similarity search, clustering, and retrieval pipelines.

```bash
sppi-api --port 8000 --device cpu
```

**`POST /embed`** — encode one or more protein sequences into graph embeddings:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"]}'
```

**`POST /score`** — predict interaction probability for a protein pair:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"protein_a": "data/processed/pdbs/TP53.pdb", "protein_b": "data/processed/pdbs/MDM2.pdb", "input_type": "pdb_path"}'
```

**`GET /health`** — liveness check with model metadata.

Interactive Swagger docs at `http://localhost:8000/docs`.

---

## Development

```bash
# Run tests
pytest

# Lint
ruff check spatialppiv2/

# Format
ruff format spatialppiv2/

# Type-check
mypy spatialppiv2/
```

CI runs automatically on every push and pull request via GitHub Actions (see `.github/workflows/ci.yml`).

---

## Relevance to Representation Learning

SpatialPPIv2 directly demonstrates:

- **Embedding extraction at scale** — ProtT5-XL produces per-residue representations that encode evolutionary and physicochemical context; the same pattern applies to token/sentence embeddings in NLP
- **Graph construction from embeddings** — node features from a language model are fed into a GNN, mirroring retrieval-augmented and structured representation pipelines
- **Fine-tuning pre-trained representations** — the GNN head is trained end-to-end on top of frozen or partially-frozen LM features, the standard paradigm for downstream adaptation of large models
- **Modular, reproducible pipelines** — each stage (retrieval → embedding → graph → inference) is independently testable and configurable, reflecting production ML system design

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use SpatialPPIv2 in your research, please cite the original SpatialPPI paper:

```bibtex
@article{spatialppi2024,
  title   = {SpatialPPI: Three-dimensional Space Protein-Protein Interaction Prediction with AlphaFold Multimer},
  journal = {Computational and Structural Biotechnology Journal},
  year    = {2024},
}
```
