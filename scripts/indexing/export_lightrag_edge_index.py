"""Export edge-friendly retrieval artifacts from a lightrag-hku working_dir.

Goal: keep edge runtime light by avoiding lightrag-hku dependency.

Inputs (from lightrag-hku working_dir):
- kv_store_entity_chunks.json (entity_name -> {chunk_ids: [...]})
- kv_store_text_chunks.json   (chunk_id -> {content: "..."})

Outputs (edge index dir):
- meta.json
- entity_names.json
- entity_embeddings.npy
- entity_chunks.json
- text_chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_entity_chunks(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[str]] = {}
    for entity_name, payload in raw.items():
        if isinstance(payload, dict):
            chunk_ids = payload.get("chunk_ids", [])
        else:
            chunk_ids = payload
        if not isinstance(chunk_ids, list):
            continue
        out[str(entity_name)] = [str(x) for x in chunk_ids]
    return out


def _load_text_chunks(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for chunk_id, payload in raw.items():
        if isinstance(payload, dict):
            content = payload.get("content", "")
        else:
            content = payload
        if not isinstance(content, str):
            continue
        out[str(chunk_id)] = content
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", required=True, help="lightrag-hku working_dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for edge artifacts")
    parser.add_argument(
        "--embed-model-id",
        default="minilm",
        help="Embedding model id or alias (e.g., minilm/mpnet/distiluse or full HF id)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = parser.parse_args()

    ingest_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ingest_root / "src"))
    from dataset_pipeline.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Services.Retrieval.Embeddings import EmbeddingRetriever

    working_dir = Path(args.working_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_chunks_path = working_dir / "kv_store_entity_chunks.json"
    text_chunks_path = working_dir / "kv_store_text_chunks.json"
    if not entity_chunks_path.exists() or not text_chunks_path.exists():
        raise FileNotFoundError(
            f"Missing expected files in working_dir: {entity_chunks_path.name}, {text_chunks_path.name}"
        )

    entity_to_chunk_ids = _load_entity_chunks(entity_chunks_path)
    chunk_texts = _load_text_chunks(text_chunks_path)

    entity_names = sorted(entity_to_chunk_ids.keys())

    embedder = EmbeddingRetriever(model_id=args.embed_model_id)
    embeddings: List[np.ndarray] = []
    for i in range(0, len(entity_names), args.batch_size):
        batch = entity_names[i : i + args.batch_size]
        vecs = embedder.encode(batch, use_cache=False)
        embeddings.append(vecs)
    entity_embeddings = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, embedder.dim), dtype=np.float32)

    meta = {
        "embed_model_id": str(embedder.model_id),
        "embedding_dim": int(entity_embeddings.shape[1]) if entity_embeddings.ndim == 2 else int(embedder.dim),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_working_dir": str(working_dir),
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(output_dir / "entity_names.json", "w", encoding="utf-8") as f:
        json.dump(entity_names, f, ensure_ascii=False)

    np.save(output_dir / "entity_embeddings.npy", entity_embeddings)

    with open(output_dir / "entity_chunks.json", "w", encoding="utf-8") as f:
        json.dump(entity_to_chunk_ids, f, ensure_ascii=False)

    with open(output_dir / "text_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_texts, f, ensure_ascii=False)

    print(f"[export] entities={len(entity_names)} chunks={len(chunk_texts)} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
