"""Build a lightrag-hku working_dir from an input corpus (offline/server).

This script intentionally depends on lightrag-hku and an LLM backend.
After building the working_dir, run:
  scripts/indexing/export_lightrag_edge_index.py
to generate edge-friendly artifacts (no lightrag-hku dependency on device).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


def _iter_jsonl_docs(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _get_text(item: dict) -> str:
    for key in ("text", "content", "document", "body"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Corpus JSONL with a text/content field")
    parser.add_argument("--working-dir", required=True, help="Output lightrag-hku working_dir")
    parser.add_argument("--limit", type=int, default=None, help="Optional max docs")
    args = parser.parse_args()

    ingest_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ingest_root / "src"))
    from dataset_pipeline.bootstrap import ensure_search_on_path

    ensure_search_on_path()

    from core.Models.Schemas import SourceDoc
    from core.Services.Retrieval.LightRAG import create_lightrag_retriever

    input_path = Path(args.input_jsonl)
    working_dir = Path(args.working_dir)

    docs: List[SourceDoc] = []
    for item in _iter_jsonl_docs(input_path, limit=args.limit):
        text = _get_text(item)
        if not text:
            continue
        doc_id = str(item.get("id") or item.get("_id") or f"doc{len(docs)}")
        docs.append(SourceDoc(id=doc_id, text=text, metadata={}))

    if not docs:
        raise ValueError(f"No valid documents found in {input_path}")

    # This will call lightrag-hku to build storages under working_dir.
    _ = create_lightrag_retriever(working_dir=working_dir, docs=docs, query_mode="hybrid")
    print(f"[build] inserted={len(docs)} working_dir={working_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
