# RAG 기반 LLM-as-a-Judge 데이터셋 구축 시스템

스마트팜 도메인을 위한 고품질 QA 데이터셋 구축 파이프라인.  
**Self-Instruct**, **Evol-Instruct**, **RAFT**, **LLM-as-a-Judge** 등 검증된 연구 방법론을 계승.

## 참고 연구

| 연구 | 출처 | 핵심 기법 | 본 시스템 적용 |
|------|------|----------|---------------|
| **Self-Instruct** | Wang et al., 2023, ACL | Seed 기반 instruction 자동 생성 | 질문 생성 프롬프트 구조 |
| **Evol-Instruct** | Xu et al., 2023, ICLR | 난이도 점진적 진화 | 질문 복잡도 단계적 확장 |
| **RAFT** | Zhang et al., 2024, COLM | RAG + Fine-tuning 데이터셋 | 문서 컨텍스트 기반 QA 생성 |
| **LLM-as-a-Judge** | Zheng et al., 2024, NeurIPS | SOTA 모델 평가 체계 | Judge 프롬프트 및 루브릭 |
| **Prometheus** | Kim et al., 2024, NeurIPS | 세분화된 평가 기준 | 다차원 점수 체계 |

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Question Generation (Self-Instruct + RAFT)             │
│  - Seed questions에서 다양한 질문 자동 생성                      │
│  - RAG 컨텍스트 기반으로 도메인 특화 질문 확장                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Answer Generation (RAFT)                               │
│  - 검색된 문서 컨텍스트 + 질문 → 초기 답변 생성                  │
│  - Chain-of-Thought 추론 과정 포함                              │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Judge Evaluation (LLM-as-a-Judge + Prometheus)         │
│  - Groundedness, Accuracy, Completeness 다차원 평가             │
│  - 구체적 피드백 및 개선점 제시                                  │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Iterative Refinement (Evol-Instruct)                   │
│  - 피드백 기반 답변 개선                                         │
│  - 임계값 도달까지 반복 (최대 N회)                               │
└─────────────────────────────────────────────────────────────────┘
```

## 설치

```bash
cd dataset
pip install -r requirements.txt
```

## 설정

`config/settings.yaml`에서 API 설정:

```yaml
llm:
  generator:  # 질문/답변 생성용 (저비용 모델 권장)
    base_url: "https://api.openai.com/v1"
    model: "gemini-2.5-flash"
    api_key: "${API_KEY}"
  
  judge:  # SOTA 평가용 (고성능 모델 권장)
    base_url: "https://api.openai.com/v1"
    model: "gemini-2.5-flash"
    api_key: "${API_KEY}"
```

환경변수 설정:
```bash
# 필수
export API_KEY="YOUR_API_KEY"

# (선택) OpenAI-compatible 프록시/사설 엔드포인트 사용 시
export OPENAI_BASE_URL="https://api.openai.com/v1"

# (선택) Hugging Face 토큰 (gated dataset 접근 시)
export HF_TOKEN="YOUR_HF_TOKEN"
```

참고: `dataset/pipeline/llm_connector.py` 및 `dataset/pipeline/corpus_cli.py`는 기본적으로
레포 루트의 `.env` 또는 `dataset/.env`를 자동 로드합니다. 템플릿은 `.env.example`을 참고하세요.

## 사용법

### 1. 환경 설정
```bash
cd dataset
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# 필수
export API_KEY="YOUR_API_KEY"

# (선택) OpenAI-compatible 프록시/사설 엔드포인트 사용 시
export OPENAI_BASE_URL="https://api.openai.com/v1"

# (선택) Hugging Face 토큰 (gated dataset 접근 시)
export HF_TOKEN="YOUR_HF_TOKEN"
```

### 3. 데이터셋 구축 실행
```bash
# 기본 실행 (스마트팜 문서에서 QA 데이터셋 생성)
python pipeline/main.py --documents ../data/smartfarm_docs/ --output smartfarm_qa_dataset.jsonl

# 커스텀 설정으로 실행
python pipeline/main.py \
  --config config/settings.yaml \
  --documents ../docs/smartfarm_kb.pdf \
  --output custom_dataset.jsonl
```

### 4. 기존 데이터셋 평가
```bash
# 생성된 데이터셋 품질 평가
python pipeline/main.py --evaluate smartfarm_qa_dataset.jsonl
```

### 5. 테스트 실행
```bash
# 파이프라인 스모크 테스트 (로컬 테스트 문서 사용)
python test_pipeline.py

# RAG grounding 스모크 테스트 (RAG가 실제로 답변에 반영되는지 확인)
# ※ LLM 호출이 발생하므로 API_KEY/OPENAI_BASE_URL 설정이 필요합니다.
python test_rag_grounding.py
```

## (추가) CGIAR -> EN/KO 말뭉치 구축 도구 (corpus_cli)

이 레포의 `dataset/` 구현을 재사용하여, CGIAR 공식 Hugging Face 데이터셋에서 문서를 추출/청킹하고
EN->KO 번역 + MQM-style LLM 평가를 수행하는 보조 CLI입니다. 기존 QA 데이터셋 파이프라인과는 별도로 동작합니다.

### 1) EN 말뭉치 export
```bash
python -m dataset.pipeline.corpus_cli export-cgiar \
  --config dataset/config/settings.yaml \
  --output dataset/output/wasabi_en_corpus.jsonl \
  --limit-per-dataset 200
```

### 2) EN -> KO 번역 (LLM)
```bash
python -m dataset.pipeline.corpus_cli translate \
  --config dataset/config/settings.yaml \
  --input dataset/output/wasabi_en_corpus.jsonl \
  --output dataset/output/wasabi_en_ko_parallel.jsonl
```

### 3) MQM-style 점수 산출 (LLM-as-a-judge)
```bash
python -m dataset.pipeline.corpus_cli mqm-score \
  --config dataset/config/settings.yaml \
  --input dataset/output/wasabi_en_ko_parallel.jsonl \
  --output dataset/output/wasabi_en_ko_parallel_scored.jsonl
```

설정은 `dataset/config/settings.yaml`의 `mqm.judges`(선택) 및 `llm.generator`/`llm.judge`를 참고하세요.

## 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--documents` | 입력 문서 경로 (파일/디렉토리) | 필수 |
| `--output` | 출력 데이터셋 파일 경로 | `smartfarm_qa_dataset.jsonl` |
| `--config` | 설정 파일 경로 | `config/settings.yaml` |
| `--evaluate` | 평가 모드 (기존 데이터셋 평가) | - |

## 출력 형식

최종 데이터셋은 JSONL 형식으로 저장:

```json
{
  "id": "qa_001",
  "question": "와사비 재배 시 최적 수온은?",
  "context": "와사비는 13-17°C의 수온에서 최적 생육...",
  "answer": "와사비의 최적 수온은 13-17°C입니다...",
  "reasoning": "문서에 따르면 와사비는 냉수성 작물로...",
  "metadata": {
    "source": "wasabi_manual.pdf",
    "chunk_id": 42,
    "iteration": 2,
    "final_score": 4.7,
    "scores": {
      "groundedness": 5.0,
      "accuracy": 4.5,
      "completeness": 4.5
    }
  }
}
```

## 라이선스

MIT License
