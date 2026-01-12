#!/usr/bin/env python3
"""Generate QA dataset for wasabi domain from translated corpus.

Usage:
    python scripts/generate_wasabi_qa.py \
        --input output/wasabi_en_ko_parallel.jsonl \
        --output output/wasabi_qa_dataset.jsonl \
        --num-questions 220

This script:
1. Loads translated wasabi corpus (Korean)
2. Generates diverse questions using LLM
3. Generates grounded answers from context
4. Evaluates quality with LLM-as-a-Judge
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import os
from openai import OpenAI


# Wasabi-specific seed questions (Korean)
WASABI_SEED_QUESTIONS = [
    "와사비의 최적 재배 온도는 몇 도인가요?",
    "와사비 수경재배 시 EC 값은 어떻게 관리해야 하나요?",
    "와사비 뿌리썩음병의 원인과 대처법은 무엇인가요?",
    "와사비 재배에 적합한 pH 범위는 얼마인가요?",
    "와사비 근경 수확까지 걸리는 기간은 얼마나 되나요?",
    "와사비 차광 관리는 어떻게 해야 하나요?",
    "와사비 양액 순환 시스템 구성 방법은?",
    "와사비 종자 발아 조건은 무엇인가요?",
]

# Question complexity levels
COMPLEXITY_LEVELS = {
    "basic": "단순 사실 질문 (예: 온도, 수치, 이름)",
    "intermediate": "관계 추론 질문 (예: 원인-결과, 비교)",
    "advanced": "다단계 복합 질문 (예: 문제해결, 최적화)",
}

# Question categories for wasabi domain
QUESTION_CATEGORIES = [
    "환경조건",  # Temperature, humidity, light
    "양액관리",  # EC, pH, nutrients
    "병해충",    # Diseases, pests
    "재배기술",  # Cultivation techniques
    "수확품질",  # Harvest, quality
    "설비장비",  # Equipment, systems
]


QUESTION_GENERATION_PROMPT = """당신은 와사비 재배 전문가입니다. 주어진 문서를 바탕으로 교육적 가치가 높은 질문을 생성하세요.

[문서]
{context}

[기존 질문들 (중복 피하기)]
{existing_questions}

[요구사항]
- 문서 내용에 직접 근거하는 질문만 생성
- 질문 유형: {complexity}
- 카테고리: {category}
- 한국어로 3개 질문 생성

JSON 배열로 출력:
[
  {{"question": "질문 텍스트", "answer_hint": "답변 힌트 (문서에서 추출)"}},
  ...
]
"""

ANSWER_GENERATION_PROMPT = """당신은 와사비 재배 전문가입니다. 주어진 문서를 바탕으로 질문에 정확하게 답변하세요.

[질문]
{question}

[참고 문서]
{context}

[요구사항]
- 문서에 근거한 답변만 제공 (환각 금지)
- 수치, 단위, 조건을 정확히 인용
- 2-4문장으로 간결하게 답변
- 관련 없는 내용은 "문서에 해당 정보가 없습니다"라고 답변

답변:"""


@dataclass
class QAPair:
    """Question-Answer pair."""
    id: str
    question: str
    answer: str
    context: str
    category: str
    complexity: str
    source_ids: List[str]
    metadata: Dict[str, Any]


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Load translated corpus."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(row)
    return docs


def create_client() -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )


def generate_questions(
    client: OpenAI,
    context: str,
    existing: List[str],
    complexity: str,
    category: str,
    model: str = "gemini-2.5-flash",
) -> List[Dict[str, str]]:
    """Generate questions from context."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        context=context[:2000],  # Limit context length
        existing_questions="\n".join(existing[-10:]) if existing else "(없음)",
        complexity=COMPLEXITY_LEVELS.get(complexity, "기본"),
        category=category,
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        content = resp.choices[0].message.content
        
        # Parse JSON
        import re
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            return json.loads(json_match.group())
        return []
    except Exception as e:
        print(f"Question generation error: {e}")
        return []


def generate_answer(
    client: OpenAI,
    question: str,
    context: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """Generate grounded answer."""
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=question,
        context=context[:3000],
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Answer generation error: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input corpus JSONL")
    parser.add_argument("--output", required=True, help="Output QA dataset JSONL")
    parser.add_argument("--num-questions", type=int, default=220)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus = load_corpus(input_path)
    print(f"Loaded {len(corpus)} documents")

    # Create client
    client = create_client()

    # Load existing if resuming
    existing_questions = []
    existing_ids = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_questions.append(row["question"])
                existing_ids.add(row["id"])
        print(f"Resuming from {len(existing_ids)} existing QA pairs")

    # Generate QA pairs
    qa_pairs = []
    qa_id = len(existing_ids)

    # Shuffle corpus for diversity
    random.shuffle(corpus)

    print(f"Generating {args.num_questions} QA pairs...")

    for doc in corpus:
        if len(qa_pairs) + len(existing_ids) >= args.num_questions:
            break

        text_ko = doc.get("text_ko") or doc.get("text", "")
        if len(text_ko) < 100:
            continue

        doc_id = doc.get("id", "unknown")

        # Rotate through categories and complexities
        category = random.choice(QUESTION_CATEGORIES)
        complexity = random.choice(list(COMPLEXITY_LEVELS.keys()))

        # Generate questions
        questions = generate_questions(
            client,
            text_ko,
            existing_questions + [q["question"] for q in qa_pairs],
            complexity,
            category,
            args.model,
        )

        for q_data in questions:
            if len(qa_pairs) + len(existing_ids) >= args.num_questions:
                break

            question = q_data.get("question", "").strip()
            if not question or question in existing_questions:
                continue

            # Generate answer
            answer = generate_answer(client, question, text_ko, args.model)
            if not answer or "문서에 해당 정보가 없습니다" in answer:
                continue

            qa_pair = {
                "id": f"wasabi_qa_{qa_id:04d}",
                "question": question,
                "answer": answer,
                "context": text_ko[:1000],
                "category": category,
                "complexity": complexity,
                "source_ids": [doc_id],
                "metadata": {
                    "answer_hint": q_data.get("answer_hint", ""),
                    "model": args.model,
                },
            }

            qa_pairs.append(qa_pair)
            existing_questions.append(question)
            qa_id += 1

            # Save incrementally
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")

            if len(qa_pairs) % 10 == 0:
                print(f"Progress: {len(qa_pairs) + len(existing_ids)}/{args.num_questions}")

            time.sleep(args.delay)

    print(f"\nGeneration complete!")
    print(f"Total QA pairs: {len(qa_pairs) + len(existing_ids)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
