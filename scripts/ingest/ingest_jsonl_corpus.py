#!/usr/bin/env python3
"""
JSONL 코퍼스를 FastAPI /ingest 로 일괄 인게스트.

사용 예:
  python scripts/ingest/ingest_jsonl_corpus.py \
    --input /path/to/corpus.jsonl \
    --host http://127.0.0.1:41177 \
    --lang ko
  # Or use environment variable: CORPUS_PATH=/data/corpus.jsonl

JSONL 형식:
  {"id": "...", "text_en": "...", "text_ko": "...", "metadata": {...}}
  또는
  {"id": "...", "text": "...", "metadata": {...}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import requests

ingest_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ingest_root / "src"))
from dataset_pipeline.bootstrap import ensure_search_on_path

ensure_search_on_path()
from core.Config.Constants import CorpusFields


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_text(row: Dict[str, Any], lang: str) -> str:
    """Extract text from corpus row based on language."""
    if lang == "ko":
        return row.get(CorpusFields.TEXT_KO) or row.get(CorpusFields.TEXT, "")
    elif lang == "en":
        return row.get(CorpusFields.TEXT_EN) or row.get(CorpusFields.TEXT, "")
    else:
        # Both - concatenate
        text_en = row.get(CorpusFields.TEXT_EN) or row.get(CorpusFields.TEXT, "")
        text_ko = row.get(CorpusFields.TEXT_KO, "")
        if text_ko:
            return f"{text_en}\n\n[한국어]\n{text_ko}"
        return text_en


def ingest_batch(host: str, docs: List[Dict[str, Any]], batch_size: int = 10) -> int:
    """Ingest documents in batches.
    
    Uses /ingest endpoint for individual docs.
    """
    total_added = 0
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        for doc in batch:
            try:
                r = requests.post(
                    f"{host}/ingest",
                    json=doc,
                    timeout=60,
                )
                r.raise_for_status()
                added = r.json().get("added", 0)
                total_added += added
            except Exception as e:
                print(f"  Error ingesting {doc.get('id', '?')}: {e}")
        
        print(f"  Batch {i//batch_size + 1}: {len(batch)} docs processed")
    
    return total_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL corpus to RAG index.")
    parser.add_argument("--input", required=True, help="Path to JSONL corpus file.")
    parser.add_argument("--host", default="http://127.0.0.1:41177", help="RAG API host.")
    parser.add_argument(
        "--lang",
        choices=["ko", "en", "both"],
        default="ko",
        help="Language to ingest (ko, en, both).",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for ingestion.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs to ingest.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Load and transform documents
    docs = []
    for i, row in enumerate(iter_jsonl(input_path)):
        if args.limit and i >= args.limit:
            break
        
        text = get_text(row, args.lang)
        if not text.strip():
            continue
        
        doc_id = row.get("id", f"doc_{i}")
        metadata = row.get("metadata", {})
        metadata["original_id"] = doc_id
        metadata["source_file"] = str(input_path.name)
        metadata["lang"] = args.lang
        
        docs.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata,
        })

    print(f"[ingest_jsonl_corpus] Loaded {len(docs)} documents from {input_path.name}")
    print(f"  Language: {args.lang}")
    print(f"  Host: {args.host}")

    if not docs:
        print("No documents to ingest.")
        return

    # Check server health
    try:
        r = requests.get(f"{args.host}/health", timeout=5)
        r.raise_for_status()
        print(f"  Server health: OK")
    except Exception as e:
        print(f"  Warning: Server health check failed: {e}")
        print("  Proceeding anyway...")

    # Ingest
    total = ingest_batch(args.host, docs, args.batch_size)
    print(f"[ingest_jsonl_corpus] Total chunks added: {total}")


if __name__ == "__main__":
    main()
