from __future__ import annotations

import json
import sqlite3
import sys
import tarfile
from pathlib import Path


def _add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def test_compile_base_kb_writes_sqlite_and_bundle(tmp_path: Path) -> None:
    """
    Compile a tiny public corpus into base.sqlite using a stub LLM.

    Given: JSONL corpus with one document
    When: compile_base_kb() runs
    Then: base.sqlite schema + rows exist, and tar bundle contains base.sqlite + manifest.json
    """
    _add_src_to_path()

    from dataset_pipeline.base_kb_compiler import compile_base_kb

    class StubLLM:
        def generate(self, prompt: str, role: str = "judge", system_prompt: str | None = None, **kwargs) -> str:
            # Intentionally return a non-normalized canonical_id to verify normalization.
            return json.dumps(
                {
                    "entities": [
                        {"text": "고온", "type": "cause", "canonical_id": "High Temperature", "confidence": 0.95},
                        {"text": "낙화", "type": "symptom", "canonical_id": "flower_drop", "confidence": 0.90},
                    ],
                    "relations": [
                        {
                            "source": "고온",
                            "target": "낙화",
                            "type": "causes",
                            "confidence": 0.85,
                            "evidence": "고온 환경에서 낙화 발생",
                        }
                    ],
                },
                ensure_ascii=False,
            )

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "doc1",
                        "text": "고온이다. 낙화가 발생한다. 환기한다. 온도를 낮춘다. 증상이 줄어든다. 관리가 필요하다.",
                        "metadata": {"source": "unit_test"},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_sqlite = tmp_path / "base.sqlite"
    out_bundle = tmp_path / "base_bundle.tar.gz"
    stats = compile_base_kb(
        input_jsonl=corpus,
        output_sqlite=out_sqlite,
        llm=StubLLM(),
        bundle_out=out_bundle,
        limit_docs=1,
    )

    assert Path(stats.output_sqlite).exists()
    # Distribution invariant: base.sqlite should be self-contained (no -wal/-shm sidecars required).
    assert not (out_sqlite.parent / f"{out_sqlite.name}-wal").exists()
    assert not (out_sqlite.parent / f"{out_sqlite.name}-shm").exists()
    assert Path(stats.manifest_path).exists()
    assert Path(stats.bundle_path or "").exists()
    assert stats.docs == 1
    assert stats.chunks > 0
    assert stats.extractions_ok == stats.chunks
    assert stats.extractions_error == 0

    # Verify SQLite contents.
    conn = sqlite3.connect(str(out_sqlite))
    try:
        conn.row_factory = sqlite3.Row
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        assert {"chunks", "extractions", "entities", "relations"}.issubset(tables)

        chunk_count = conn.execute("SELECT COUNT(1) AS n FROM chunks;").fetchone()["n"]
        assert chunk_count == stats.chunks

        # Public-only invariants.
        row = conn.execute("SELECT sensitivity, owner_id FROM chunks LIMIT 1;").fetchone()
        assert str(row["sensitivity"] or "") == "public"
        assert row["owner_id"] is None

        # canonical_id normalization check.
        cids = {r["canonical_id"] for r in conn.execute("SELECT canonical_id FROM entities;").fetchall()}
        assert "high_temperature" in cids
        assert "flower_drop" in cids

        rel_count = conn.execute("SELECT COUNT(1) AS n FROM relations;").fetchone()["n"]
        assert rel_count == stats.relations
    finally:
        conn.close()

    # Verify bundle contents.
    with tarfile.open(out_bundle, "r:gz") as tf:
        names = set(tf.getnames())
        assert "base.sqlite" in names
        assert "manifest.json" in names
