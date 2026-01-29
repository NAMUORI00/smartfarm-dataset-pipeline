#!/usr/bin/env python3
"""
스마트팜 기본 지식 베이스 추가 (로컬 인게스트)
- 온도, 양액, 병해충, 환경제어 등 다양한 도메인 커버
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import requests

DEFAULT_HOST = os.getenv("ERA_RAG_HOST", "http://127.0.0.1:41177")

# 스마트팜 기본 지식 문서 (작물별, 주제별)
KNOWLEDGE_BASE = [
    {
        "id": "kb_tomato_temp",
        "text": "토마토 생육 최적 온도는 주간 20-25℃, 야간 15-18℃입니다. 생육 초기에는 다소 높은 온도(주간 25-28℃)가 유리하며, 과실 비대기에는 주간 23-25℃를 유지합니다. 10℃ 이하로 떨어지면 생육이 정지되고 저온 장해가 발생할 수 있습니다.",
        "metadata": {"작물": "토마토", "주제": "환경-온도", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_tomato_nutrient",
        "text": "토마토 양액 EC는 생육 단계에 따라 2.0-3.0 dS/m 범위를 권장합니다. pH는 5.5-6.5가 적정하며, pH가 5.0 이하로 낮아지면 철, 망간 과잉 흡수 우려가 있으므로 pH 상향제(KOH, NaOH)로 조절합니다.",
        "metadata": {"작물": "토마토", "주제": "양액", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_paprika_nutrient",
        "text": "파프리카 양액 EC는 생육 초기 1.8-2.2 dS/m, 착과기 2.2-2.5 dS/m, 수확기 2.5-3.0 dS/m로 단계적으로 높입니다. pH는 5.8-6.2 범위를 유지하며, 칼슘 결핍 방지를 위해 Ca:Mg 비율을 3:1로 관리합니다.",
        "metadata": {"작물": "파프리카", "주제": "양액", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_strawberry_disease",
        "text": "딸기 흰가루병 초기 증상은 잎 표면에 흰색 가루 형태의 곰팡이가 나타나며, 잎이 위로 말립니다. 예방을 위해 온실 습도를 60% 이하로 유지하고, 야간 온도를 8℃ 이상으로 관리합니다. 발병 시 유황훈증제 또는 친환경 미생물제(바실러스) 살포를 권장합니다.",
        "metadata": {"작물": "딸기", "주제": "병해충-흰가루병", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_lettuce_schedule",
        "text": "상추 재배 일정은 파종→발아(3-5일)→육묘(15-20일)→정식→생육(20-30일)→수확 순서입니다. 총 재배 기간은 약 40-55일이며, 품종과 환경에 따라 변동됩니다. 육묘기 주간 온도 18-20℃, 야간 12-15℃를 유지하고, 생육기에는 주간 15-20℃가 적정합니다.",
        "metadata": {"작물": "상추", "주제": "재배일정", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_humidity_control",
        "text": "온실 습도가 80% 이상일 때는 병해 발생 위험이 높으므로 즉시 환기를 실시합니다. 야간에는 순환팬을 가동하고, 주간에는 측창/천창을 열어 습도를 60-70%로 낮춥니다. 과습 지속 시 잿빛곰팡이병, 노균병 등이 발생할 수 있습니다.",
        "metadata": {"주제": "환경-습도", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_nutrient_ph_low",
        "text": "양액 pH가 5.5 이하로 떨어졌을 때는 pH 상향제(수산화칼륨 KOH 또는 수산화나트륨 NaOH)를 소량 투입하여 5.8-6.2 범위로 조절합니다. pH가 낮으면 철, 망간 등 미량원소 과잉 흡수로 독성 증상이 나타날 수 있으므로 주의합니다.",
        "metadata": {"주제": "양액-pH", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_low_temp_damage",
        "text": "야간 온도가 10℃ 이하로 떨어지면 대부분 작물의 생육이 정지되고 저온 장해가 발생합니다. 토마토, 파프리카는 15℃ 이하에서 화분 발아율이 저하되고, 딸기는 8℃ 이하에서 꽃눈 분화가 지연됩니다. 저온 장해 예방을 위해 보온 커튼, 난방기 가동이 필요합니다.",
        "metadata": {"주제": "환경-저온장해", "출처": "농촌진흥청"}
    },
    {
        "id": "kb_tomato_leaf_disease",
        "text": "토마토 잎마름병 예방을 위한 환경 관리 기준은 습도 60-70%, 주간 온도 22-25℃, 야간 온도 16-18℃입니다. 과습과 온도 편차가 크면 병 발생이 증가하므로 환기와 난방을 통해 안정적인 환경을 유지합니다. 발병 시 이병엽을 제거하고 구리제 또는 친환경 미생물제를 살포합니다.",
        "metadata": {"작물": "토마토", "주제": "병해충-잎마름병", "출처": "농촌진흥청"}
    },
]

def ingest_document(host: str, doc: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """단일 문서 인게스트"""
    try:
        resp = requests.post(
            f"{host}/ingest",
            json=doc,
            timeout=timeout_s,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "added": data.get("added", 0)}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest built-in SmartFarm KB into /ingest.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="RAG API host (default: ERA_RAG_HOST or 127.0.0.1:41177)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout seconds")
    args = parser.parse_args()

    print("=" * 80)
    print("스마트팜 기본 지식 베이스 인게스트")
    print("=" * 80)
    print(f"총 문서: {len(KNOWLEDGE_BASE)}개")
    print(f"HOST: {args.host}")
    print("=" * 80)
    
    total_chunks = 0
    
    for doc in KNOWLEDGE_BASE:
        result = ingest_document(args.host, doc, timeout_s=args.timeout)
        if result["success"]:
            chunks = result["added"]
            total_chunks += chunks
            print(f"✓ {doc['id']:25s} | {chunks:2d} chunks | {doc['metadata'].get('주제', 'N/A')}")
        else:
            print(f"✗ {doc['id']:25s} | ERROR: {result['error']}")
    
    print("=" * 80)
    print(f"총 {total_chunks}개 청크가 인덱스에 추가되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()
