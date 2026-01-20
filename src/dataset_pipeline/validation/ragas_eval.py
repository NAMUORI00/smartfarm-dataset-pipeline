"""
RAGAS evaluation utilities for QA datasets.

Supports evaluating dataset quality (question/answer/context alignment) and
writing summary metrics for comparison between dataset versions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import ConfigManager, get_config
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

try:  # Optional dependency
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    HAS_RAGAS = True
except Exception:  # pragma: no cover - optional dependency at runtime
    HAS_RAGAS = False


DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

_METRIC_MAP = {
    "faithfulness": faithfulness if HAS_RAGAS else None,
    "answer_relevancy": answer_relevancy if HAS_RAGAS else None,
    "context_precision": context_precision if HAS_RAGAS else None,
    "context_recall": context_recall if HAS_RAGAS else None,
    "answer_correctness": answer_correctness if HAS_RAGAS else None,
}


@dataclass
class LLMSettings:
    model: str
    base_url: str
    api_key: str


def _ensure_ragas_available() -> None:
    if HAS_RAGAS:
        return
    raise ImportError(
        "ragas가 설치되어 있지 않습니다. "
        "dataset-pipeline/requirements.txt 또는 pyproject.toml에 "
        "ragas, datasets, langchain-openai를 추가 설치하세요."
    )


def _normalize_metric_names(names: Sequence[str]) -> List[str]:
    return [n.strip().lower() for n in names if n and n.strip()]


def resolve_metrics(metric_names: Optional[Sequence[str]] = None) -> List[Any]:
    _ensure_ragas_available()
    names = _normalize_metric_names(metric_names or DEFAULT_METRICS)
    metrics = []
    for name in names:
        metric = _METRIC_MAP.get(name)
        if metric is None:
            raise ValueError(f"지원하지 않는 RAGAS metric: {name}")
        metrics.append(metric)
    return metrics


def _resolve_llm_settings(
    config: ConfigManager,
    role: str,
    model_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> LLMSettings:
    llm_cfg = config.get_llm_config(role)
    return LLMSettings(
        model=model_override or llm_cfg.get("model", "gpt-4o-mini"),
        base_url=base_url_override or llm_cfg.get("base_url", "https://api.openai.com/v1"),
        api_key=api_key_override or llm_cfg.get("api_key", ""),
    )


def _get_contexts(item: Dict[str, Any]) -> List[str]:
    contexts = (
        item.get("contexts")
        or item.get("context")
        or item.get("retrieved_contexts")
        or item.get("source_texts")
    )
    if isinstance(contexts, list):
        return [str(c) for c in contexts if c]
    if contexts:
        return [str(contexts)]
    return []


def _get_answer(item: Dict[str, Any]) -> str:
    return str(item.get("answer") or item.get("response") or "")


def _get_reference(item: Dict[str, Any], answer: str) -> str:
    return str(item.get("ground_truth") or item.get("reference") or answer)


def load_qa_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def build_ragas_dataset(items: Sequence[Dict[str, Any]]) -> "Dataset":
    _ensure_ragas_available()
    data = {
        "id": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "reference": [],
    }
    for item in items:
        question = str(item.get("question") or item.get("query") or "").strip()
        answer = _get_answer(item)
        contexts = _get_contexts(item)
        reference = _get_reference(item, answer)
        data["id"].append(item.get("id", ""))
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(reference)
        data["reference"].append(reference)
    return Dataset.from_dict(data)


def _aggregate_scores(scores: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in scores:
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k)}


def _extract_summary(result: Any) -> Dict[str, float]:
    if hasattr(result, "scores"):
        return _aggregate_scores(result.scores)
    try:
        return dict(result)
    except Exception:
        return {}


def run_ragas_eval(
    items: Sequence[Dict[str, Any]],
    *,
    config: Optional[ConfigManager] = None,
    llm_role: str = "judge",
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    emb_model: Optional[str] = None,
    emb_device: Optional[str] = None,
    embeddings: Optional[Any] = None,
    metric_names: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    _ensure_ragas_available()
    config = config or get_config()
    llm_settings = _resolve_llm_settings(
        config,
        role=llm_role,
        model_override=llm_model,
        base_url_override=llm_base_url,
        api_key_override=llm_api_key,
    )

    if llm_settings.api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = llm_settings.api_key
    if llm_settings.base_url and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = llm_settings.base_url

    client = OpenAI(
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key,
    )
    llm = llm_factory(model=llm_settings.model, client=client)
    if embeddings is None:
        embedding_model = emb_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        device = emb_device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
        )
    dataset = build_ragas_dataset(items)
    metrics = resolve_metrics(metric_names)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=show_progress,
        batch_size=batch_size,
    )
    summary = _extract_summary(result)
    per_sample = list(result.scores) if hasattr(result, "scores") else []
    return summary, per_sample


def run_ragas_compare(
    baseline_items: Sequence[Dict[str, Any]],
    improved_items: Sequence[Dict[str, Any]],
    *,
    config: Optional[ConfigManager] = None,
    llm_role: str = "judge",
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    emb_model: Optional[str] = None,
    emb_device: Optional[str] = None,
    metric_names: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    embedding_model = emb_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device = emb_device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
    )

    baseline_summary, _ = run_ragas_eval(
        baseline_items,
        config=config,
        llm_role=llm_role,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        emb_model=emb_model,
        emb_device=emb_device,
        embeddings=embeddings,
        metric_names=metric_names,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    improved_summary, _ = run_ragas_eval(
        improved_items,
        config=config,
        llm_role=llm_role,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        emb_model=emb_model,
        emb_device=emb_device,
        embeddings=embeddings,
        metric_names=metric_names,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    delta = {
        key: improved_summary.get(key, 0.0) - baseline_summary.get(key, 0.0)
        for key in set(baseline_summary) | set(improved_summary)
    }
    return {
        "baseline": baseline_summary,
        "improved": improved_summary,
        "delta": delta,
    }
