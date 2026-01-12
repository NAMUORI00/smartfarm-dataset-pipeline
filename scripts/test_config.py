#!/usr/bin/env python3
"""
ConfigManager 스모크 테스트

Usage:
    python scripts/test_config.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_pipeline.config import ConfigManager


def test_basic():
    """기본 설정 로딩 테스트."""
    print("=" * 60)
    print("ConfigManager 스모크 테스트")
    print("=" * 60)

    # 테스트용 환경 변수 설정 (없으면 기본값 사용)
    os.environ.setdefault("API_KEY", "test-api-key")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")

    try:
        config = ConfigManager()
        print("\n[PASS] ConfigManager 초기화 성공")

        # 기본 접근 테스트
        model = config.get("llm.generator.model")
        print(f"[PASS] Generator 모델: {model}")

        # LLM 설정 테스트
        llm_config = config.get_llm_config("generator")
        print(f"[PASS] Generator 설정 로드: model={llm_config.get('model')}")

        judge_config = config.get_llm_config("judge")
        print(f"[PASS] Judge 설정 로드: model={judge_config.get('model')}")

        # 기본값 테스트
        missing = config.get("missing.key.path", "default-value")
        assert missing == "default-value", "기본값 테스트 실패"
        print("[PASS] 기본값 반환 테스트")

        # HuggingFace 토큰 테스트
        hf_token = config.get_huggingface_token()
        print(f"[PASS] HuggingFace 토큰: {'설정됨' if hf_token else '미설정'}")

        # MQM judges 테스트
        judges = config.get_mqm_judges()
        print(f"[PASS] MQM Judges: {len(judges)}개")

        # 디버그 덤프 (마스킹)
        print("\n" + "=" * 60)
        print("설정 출력 (시크릿 마스킹됨):")
        print("=" * 60)
        dump = config.debug_dump(mask_secrets=True)
        
        # 마스킹 확인
        if "***MASKED***" in dump:
            print("[PASS] 시크릿 마스킹 작동")
        
        # 처음 50줄만 출력
        lines = dump.split("\n")[:50]
        print("\n".join(lines))
        if len(dump.split("\n")) > 50:
            print("... (생략)")

        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 테스트 통과!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] 오류: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_basic()
