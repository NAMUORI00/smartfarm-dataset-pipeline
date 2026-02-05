"""
RAG 기반 LLM-as-a-Judge 데이터셋 구축 시스템

참고 연구:
- Self-Instruct (Wang et al., 2023, ACL)
- Evol-Instruct (Xu et al., 2023, ICLR)
- RAFT (Zhang et al., 2024, COLM)
- LLM-as-a-Judge (Zheng et al., 2024, NeurIPS)
- Prometheus (Kim et al., 2024, NeurIPS)

주의:
이 패키지는 CLI/파이프라인/검증 등 다양한 모듈을 포함하며, 일부 의존성(transformers 등)이 무겁습니다.
import 비용을 줄이기 위해 **lazy import**를 사용합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "Smart Farm RAG Team"

__all__ = [
    "ConfigManager",
    "get_config",
    "DatasetPipeline",
    "LLMConnector",
    "RAGConnector",
    "QuestionGenerator",
    "LLMJudge",
    "AnswerRefiner",
]

if TYPE_CHECKING:
    from .config import ConfigManager
    from .generator import QuestionGenerator
    from .judge import LLMJudge
    from .llm_connector import LLMConnector
    from .main import DatasetPipeline
    from .rag_connector import RAGConnector
    from .refiner import AnswerRefiner


def __getattr__(name: str) -> Any:
    if name == "ConfigManager":
        from .config import ConfigManager as v

        return v
    if name == "get_config":
        from .config import get_config as v

        return v
    if name == "DatasetPipeline":
        from .main import DatasetPipeline as v

        return v
    if name == "LLMConnector":
        from .llm_connector import LLMConnector as v

        return v
    if name == "RAGConnector":
        from .rag_connector import RAGConnector as v

        return v
    if name == "QuestionGenerator":
        from .generator import QuestionGenerator as v

        return v
    if name == "LLMJudge":
        from .judge import LLMJudge as v

        return v
    if name == "AnswerRefiner":
        from .refiner import AnswerRefiner as v

        return v
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
