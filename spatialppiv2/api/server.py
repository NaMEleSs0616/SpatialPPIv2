"""
SpatialPPIv2 REST API — Cohere-compatible embedding endpoint + PPI scoring.

Endpoints
---------
GET  /health             liveness + model metadata
POST /embed              encode protein sequences → graph embeddings
POST /score              predict interaction probability for a pair

Start with:
    sppi-api --host 0.0.0.0 --port 8000 --device cpu
    # or: uvicorn spatialppiv2.api.server:app --port 8000
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spatialppiv2.utils.config import get_config
from spatialppiv2.utils.model import getModel
from spatialppiv2.utils.tool import Embed, extractPDB
from spatialppiv2.utils.dataset import build_data


# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpatialPPIv2",
    description="Protein-protein interaction scoring via ProtT5-XL and GATv2 GNN.",
    version="0.1.0",
)

_state: dict[str, Any] = {}


def _load(device: str = "cpu", cfg_path: str | None = None) -> None:
    """Load model and embedder into module-level state (called once at startup)."""
    cfg = get_config(cfg_path)
    dev = torch.device(device)

    embedder = Embed(cfg["models"]["prott5_name"], dev)
    cfg["basic"]["num_features"] = embedder.featureLen

    ckpt = cfg["checkpoints"]["prott5"]
    model = getModel(cfg, ckpt=ckpt if Path(ckpt).exists() else None).to(dev)
    model.eval()

    _state.update({"embedder": embedder, "model": model, "device": dev, "cfg": cfg})


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    meta: dict


class ScoreRequest(BaseModel):
    protein_a: str
    protein_b: str
    input_type: str = "pdb_path"   # "pdb_path" | "sequence"
    chain_a: str = "first"
    chain_b: str = "first"


class ScoreResponse(BaseModel):
    interaction_probability: float
    prediction: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    num_parameters: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = _state.get("model")
    return HealthResponse(
        status="ok",
        model="SpatialPPIv2-ProtT5",
        device=str(_state.get("device", "unknown")),
        num_parameters=sum(p.numel() for p in model.parameters()) if model else 0,
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    embedder: Embed = _state["embedder"]
    model = _state["model"]
    dev   = _state["device"]
    t0 = time.perf_counter()

    results: list[list[float]] = []
    for seq in req.texts:
        emb = embedder.encode(seq)           # (L, D)
        # Graph-level: mean pooling over residues
        h = emb.mean(0).to(dev)
        if req.normalize:
            h = F.normalize(h, dim=0)
        results.append(h.tolist())

    latency_ms = (time.perf_counter() - t0) * 1000
    return EmbedResponse(
        embeddings=results,
        meta={"latency_ms": round(latency_ms, 2), "n": len(results)},
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    embedder: Embed = _state["embedder"]
    model = _state["model"]
    dev   = _state["device"]
    t0 = time.perf_counter()

    if req.input_type == "pdb_path":
        try:
            seq_a, coords_a = extractPDB(req.protein_a, req.chain_a)
            seq_b, coords_b = extractPDB(req.protein_b, req.chain_b)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"PDB parse error: {e}")
    else:
        raise HTTPException(status_code=422, detail="Only 'pdb_path' input_type is supported.")

    emb_a = embedder.encode(seq_a)
    emb_b = embedder.encode(seq_b)
    data = build_data(
        torch.cat([emb_a, emb_b]),
        [coords_a, coords_b],
        [req.protein_a, req.protein_b],
    ).to(dev)

    with torch.no_grad():
        prob = model(data).cpu().item()

    latency_ms = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        interaction_probability=round(prob, 6),
        prediction="INTERACTING" if prob >= 0.5 else "NON-INTERACTING",
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the SpatialPPIv2 API server.")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    _load(device=args.device, cfg_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
