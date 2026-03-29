# Changelog

All notable changes to SpatialPPIv2 are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Full project scaffold: `pyproject.toml`, installable CLI entry points
- `GATv2Encoder` with multi-head attention, residual connections, LayerNorm
- `SpatialPPIv2Model`: NLI-style pair representation `[h_A ∥ h_B ∥ |h_A−h_B| ∥ h_A⊙h_B]`
- SimCLR contrastive pre-training: NT-Xent loss, node/edge dropout, Gaussian noise augmentations
- AlphaFold pLDDT feature extraction from B-factor column
- Batch scoring with resume support (`sppi-score`)
- FastAPI REST server: `/embed`, `/score`, `/health` (`sppi-api`)
- Evaluation suite: AUC-ROC, AUPR, threshold sweep, publication-ready plots (`sppi-eval`)
- RCSB + AlphaFold DB structure downloader with gene-name resolution (`sppi-pdbs`)
- Docker (CPU) and Docker GPU (CUDA 12.1) images
- GitHub Actions CI: lint, pytest (Python 3.9/3.10/3.11), Docker build
- `demo.ipynb`: end-to-end walkthrough with TP53 × MDM2
- Sample `cleaned_node.csv` and `cleaned_edge.csv`

---

## [0.1.0] — Initial release

- Based on: *SpatialPPI: Three-dimensional Space Protein-Protein Interaction Prediction with AlphaFold Multimer*, CSBJ 2024
- ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`) as sequence encoder
- ESM-2+ac variant for structure-free inference
- Cα–Cα contact graph (8 Å threshold)
