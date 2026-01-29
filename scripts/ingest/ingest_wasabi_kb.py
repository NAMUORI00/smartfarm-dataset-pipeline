#!/usr/bin/env python3
"""
와사비 도메인 문서 폴더를 FastAPI /ingest 로 일괄 인게스트.

사용 예:
  python scripts/ingest/ingest_wasabi_kb.py --input data/wasabi_docs --host http://127.0.0.1:41177

폴더 내의 .txt/.md/.json/.csv 파일을 UTF-8로 읽어
파일명 기반 id로 /ingest 요청을 보낸다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import requests


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json", ".csv"}:
            yield p


def ingest_one(host: str, path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    doc_id = path.stem
    payload = {"id": doc_id, "text": text, "metadata": {"source": str(path)}}
    r = requests.post(f"{host}/ingest", json=payload, timeout=60)
    r.raise_for_status()
    return int(r.json().get("added", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest wasabi KB folder.")
    parser.add_argument("--input", required=True, help="Folder containing wasabi documents.")
    parser.add_argument("--host", default="http://127.0.0.1:41177", help="RAG API host.")
    args = parser.parse_args()

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")

    total = 0
    files = list(iter_files(root))
    print(f"[ingest_wasabi_kb] Found {len(files)} files under {root}")
    for f in files:
        try:
            n = ingest_one(args.host, f)
            total += n
            print(f"✓ {f.name} -> {n} chunks")
        except Exception as e:
            print(f"✗ {f.name} -> ERROR: {e}")

    print(f"[ingest_wasabi_kb] Total chunks added: {total}")


if __name__ == "__main__":
    main()
