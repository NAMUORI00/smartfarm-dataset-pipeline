#!/usr/bin/env python3
"""Batch ingest JSONL corpus with progress tracking and resume support."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import requests

ingest_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ingest_root / "src"))
from dataset_pipeline.bootstrap import ensure_search_on_path

ensure_search_on_path()
from core.Config.Constants import CorpusFields


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_progress(progress_file: Path) -> set:
    """Load already ingested IDs."""
    if not progress_file.exists():
        return set()
    with open(progress_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(progress_file: Path, doc_id: str) -> None:
    """Append ingested ID to progress file."""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(doc_id + "\n")


def get_text(row: dict, lang: str) -> str:
    """Extract text from corpus row based on language."""
    if lang == "ko":
        return row.get(CorpusFields.TEXT_KO) or row.get(CorpusFields.TEXT, "")
    elif lang == "en":
        return row.get(CorpusFields.TEXT_EN) or row.get(CorpusFields.TEXT, "")
    else:
        return row.get(CorpusFields.TEXT) or row.get(CorpusFields.TEXT_KO) or row.get(CorpusFields.TEXT_EN, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:41177")
    parser.add_argument("--lang", default="ko", choices=["ko", "en", "both"])
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--progress-file", default="ingest_progress.txt")
    args = parser.parse_args()

    input_path = Path(args.input)
    progress_file = Path(args.progress_file)
    
    ingested = load_progress(progress_file)
    print(f"Already ingested: {len(ingested)} docs")

    success = 0
    errors = 0
    
    for i, row in enumerate(iter_jsonl(input_path)):
        doc_id = row.get(CorpusFields.ID, f"doc_{i}")

        if doc_id in ingested:
            continue

        # Get text using helper function
        text = get_text(row, args.lang)

        if not text.strip():
            continue
        
        payload = {
            CorpusFields.ID: doc_id,
            CorpusFields.TEXT: text,
            CorpusFields.METADATA: row.get(CorpusFields.METADATA, {}),
        }
        
        try:
            r = requests.post(f"{args.host}/ingest", json=payload, timeout=60)
            r.raise_for_status()
            save_progress(progress_file, doc_id)
            success += 1
            
            if success % 20 == 0:
                print(f"Progress: {success} ingested, {errors} errors")
                
        except Exception as e:
            print(f"Error {doc_id}: {e}")
            errors += 1
        
        time.sleep(args.delay)
    
    print(f"\nComplete! Success: {success}, Errors: {errors}")


if __name__ == "__main__":
    main()
