#!/usr/bin/env python3
"""데이터셋 파이프라인에서 생성된 JSONL을 RAG API에 벌크 로드하는 스크립트.

Usage:
    python scripts/ingest/load_dataset.py --file /path/to/corpus.jsonl
    python scripts/ingest/load_dataset.py --file /path/to/corpus.jsonl --lang ko
    # Or: CORPUS_PATH=/data/corpus.jsonl python scripts/ingest/load_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import requests

ingest_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ingest_root / "src"))
from dataset_pipeline.bootstrap import ensure_search_on_path

ensure_search_on_path()
from core.Config.Constants import CorpusFields


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """JSONL 파일 로드."""
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def transform_to_rag_format(
    docs: List[Dict[str, Any]],
    lang: str = "ko",
    use_bilingual: bool = False
) -> List[Dict[str, Any]]:
    """파이프라인 포맷 → RAG API 포맷 변환.

    Args:
        docs: 파이프라인에서 생성된 문서 리스트
        lang: 사용할 언어 (ko, en)
        use_bilingual: True면 영어+한글 병렬 텍스트 사용

    Returns:
        RAG API /ingest 엔드포인트용 포맷
    """
    rag_docs = []

    for doc in docs:
        doc_id = doc.get("id", "")

        # 텍스트 선택
        if use_bilingual:
            text = f"[EN] {doc.get(CorpusFields.TEXT_EN, '')}\n\n[KO] {doc.get(CorpusFields.TEXT_KO, '')}"
        elif lang == "ko":
            text = doc.get(CorpusFields.TEXT_KO, doc.get(CorpusFields.TEXT, ""))
        else:
            text = doc.get(CorpusFields.TEXT_EN, doc.get(CorpusFields.TEXT, ""))

        if not text:
            continue

        # 메타데이터 추출
        metadata = doc.get("metadata", {})

        rag_doc = {
            "id": doc_id,
            "text": text,
            "metadata": {
                "source": metadata.get("source", "smartfarm-ingest"),
                "category": metadata.get("category", ""),
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "작물": "와사비",  # 와사비 데이터셋
                "lang": lang if not use_bilingual else "bilingual",
            }
        }
        rag_docs.append(rag_doc)

    return rag_docs


def upload_to_rag(
    docs: List[Dict[str, Any]],
    base_url: str = "http://localhost:41177",
    batch_size: int = 10
) -> Dict[str, Any]:
    """RAG API에 문서 업로드.

    Args:
        docs: RAG 포맷 문서 리스트
        base_url: RAG API 주소
        batch_size: 배치 크기

    Returns:
        업로드 결과 통계
    """
    stats = {
        "total": len(docs),
        "success": 0,
        "failed": 0,
        "chunks_added": 0,
        "errors": []
    }

    print(f"\n[Upload] Starting {len(docs)} documents...")
    print(f"   API: {base_url}")
    print(f"   Batch size: {batch_size}\n")

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        for doc in batch:
            try:
                resp = requests.post(
                    f"{base_url}/ingest",
                    json=doc,
                    timeout=60
                )

                if resp.status_code == 200:
                    result = resp.json()
                    stats["success"] += 1
                    stats["chunks_added"] += result.get("added", 0)
                else:
                    stats["failed"] += 1
                    error_msg = f"{doc['id']}: {resp.status_code} - {resp.text[:100]}"
                    stats["errors"].append(error_msg)

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"{doc['id']}: {str(e)}")

        # 진행률 표시
        progress = min(i + batch_size, len(docs))
        pct = progress / len(docs) * 100
        print(f"   진행: {progress}/{len(docs)} ({pct:.1f}%) - 성공: {stats['success']}, 실패: {stats['failed']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="데이터셋 → RAG 벌크 로드")
    parser.add_argument("--file", "-f", required=True, help="JSONL 파일 경로")
    parser.add_argument("--lang", "-l", default="ko", choices=["ko", "en"], help="사용할 언어")
    parser.add_argument("--bilingual", "-b", action="store_true", help="영-한 병렬 텍스트 사용")
    parser.add_argument("--host", default="http://localhost:41177", help="RAG API 주소")
    parser.add_argument("--batch-size", type=int, default=10, help="배치 크기")
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드 없이 변환만 테스트")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 문서 수 (0=전체)")

    args = parser.parse_args()

    # 파일 확인
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"[Error] File not found: {filepath}")
        sys.exit(1)

    print(f"[File] Loading: {filepath}")

    # JSONL 로드
    docs = load_jsonl(str(filepath))
    print(f"   Total {len(docs)} documents found")

    # 제한 적용
    if args.limit > 0:
        docs = docs[:args.limit]
        print(f"   Limit applied: {len(docs)}")

    # RAG 포맷 변환
    rag_docs = transform_to_rag_format(docs, lang=args.lang, use_bilingual=args.bilingual)
    print(f"   RAG format conversion done: {len(rag_docs)}")

    # 샘플 출력
    if rag_docs:
        print(f"\n[Sample]")
        sample = rag_docs[0]
        print(f"   ID: {sample['id']}")
        print(f"   Text: {sample['text'][:80]}...")
        print(f"   Meta: {sample['metadata']}")

    # Dry run 모드
    if args.dry_run:
        print(f"\n[Dry run] Skipping upload")
        return

    # 업로드
    stats = upload_to_rag(rag_docs, base_url=args.host, batch_size=args.batch_size)

    # 결과 출력
    print(f"\n[Result]")
    print(f"   Total: {stats['total']}")
    print(f"   Success: {stats['success']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Chunks added: {stats['chunks_added']}")

    if stats["errors"]:
        print(f"\n[Errors] (max 5):")
        for err in stats["errors"][:5]:
            print(f"   - {err}")


if __name__ == "__main__":
    main()
