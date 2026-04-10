# 반도체 R&D 기술 전략 분석 에이전트

경쟁사 반도체 R&D 동향을 자동 수집·분석하여 전략 보고서를 생성하는 AI 에이전트 시스템.

## 아키텍처 (그래프 흐름)

```
User Query
    ↓
Supervisor Agent          ← 전체 흐름 제어 (LangGraph 조건부 라우팅)
    ↓
Query Planning Agent      ← 기업/기술/이슈 중심 쿼리 혼합 + web/rag 타겟
    ↓
Parallel Retrieve         ← Web Search + RAG (Thread 병렬 수집 → retrieved_documents 병합)
    ↓
Balanced Retrieval Node   ← 중복 제거 · scope/신뢰도 필터 · 점수 정렬 · MMR 다양화
    ↓
TRL Preparation Node      ← 논문/특허/뉴스/채용 Signal 집계
    ↓
Competitive Analyst Agent ← Task4 구조화 추출 + TRL·위협 분석
    ↓
Draft Generation Agent
    ↓
Validation Node
    ↓
Formatting Node
    ↓
Supervisor (승인 로그) → END
    ↓
Final Report (Markdown)
```

## 파일 구조

```
├── state.py              # SupervisorState, RetrievedDocument, DocumentExtraction 등
├── config.py             # LLM / ChromaDB / 기본값
├── retrieval_utils.py    # 기업명 정규화, scope 날짜, MMR
├── document_metadata.py  # 적재 시 파일명 파싱
├── agents.py             # 에이전트·노드
├── graph.py              # LangGraph
├── main.py               # 실행 진입점
├── langsmith_setup.py    # LangSmith 환경변수 (graph import 전 적용)
├── ingest.py             # ChromaDB 적재 (txt/md/pdf)
├── eval/retrieval_eval.py  # Hit@K / MRR 소규모 벤치
├── docs/corpus.md        # 내부 PDF 파일명 매핑 정본
├── requirements.txt
└── .env                  # (로컬) API 키 — gitignore
```

## State 요약 (`state.py`)

- **입력:** `user_query`, `target_technologies`, `target_companies`, `scope` (`date_from`, `date_to` 등)
- **검색:** `search_queries` (`target`: `web` | `rag`)
- **수집:** `retrieved_documents` → `balanced_documents`
- **분석:** `aggregated_signals`, `document_extractions` (Task4 구조화), `analysis_result`
- **출력:** `current_draft`, `validation`, `final_report`
- **제어:** `next_action` (`plan` … `format` | `end`)

## 빠른 시작

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env 에 OPENAI_API_KEY, (선택) HF_TOKEN
cp .env.example .env  # 저장소에 예시가 있으면 사용

# 내부 PDF 적재 — 파일명 규칙은 docs/corpus.md
python ingest.py --dir docs/pdfs

python main.py
```

## 검색 품질 소평가 (Hit@K / MRR)

Chroma에 코퍼스가 적재된 뒤:

```bash
python -m eval.retrieval_eval
```

골든 쿼리는 `eval/retrieval_eval.py`의 `GOLDEN` 리스트에서 조정합니다.

## 환경 변수 (선택)

| 변수 | 의미 |
|------|------|
| `MIN_CREDIBILITY_FOR_BALANCE` | 균형 단계 신뢰도 하한 (기본 `0.38`) |
| `MMR_LAMBDA` | MMR 관련성 가중 (기본 `0.65`) |
| `MMR_POOL_SIZE` | MMR 후보 풀 크기 (기본 `60`) |
| `EMBEDDING_MODEL` | Chroma 임베딩 모델 id |
| `CHROMA_PERSIST_DIR` | Chroma 저장 경로 |
| `HF_TOKEN` | Hugging Face Hub 다운로드 가속 |

## LangSmith (추적)

[LangSmith](https://smith.langchain.com)에서 LLM·LangGraph 실행(노드·프롬프트·지연)을 확인하려면 API 키를 넣습니다.

1. [API Keys](https://smith.langchain.com/settings)에서 키 발급.
2. `.env` 예시:

```bash
LANGCHAIN_API_KEY=lsv2_pt_...
# 또는 LANGSMITH_API_KEY= (동일하게 인식)
LANGCHAIN_PROJECT=semiconductor-rnd
```

`LANGCHAIN_API_KEY`(또는 `LANGSMITH_API_KEY`)만 설정해 두면 `main.py`의 `configure_langsmith()`가 추적을 켭니다. 끄려면 `LANGSMITH_TRACING=false` 또는 `LANGCHAIN_TRACING_V2=false`를 명시하세요.

`python main.py` 실행 시 콘솔에 대시보드 URL이 출력됩니다.

## 커스텀 실행

```python
from main import run_analysis

run_analysis(
    user_query="HBM4 관련 삼성·마이크론 R&D 동향 분석",
    target_technologies=["HBM4", "PIM", "CXL"],
    target_companies=["Samsung", "Micron", "SK Hynix"],
    output_path="reports/my_report.md",
)
```

## 내부 문서 / 코퍼스

- **정본 매핑:** [docs/corpus.md](docs/corpus.md)
- 적재: `python ingest.py --dir ./docs/pdfs`

## 임베딩 모델

- 기본 **BAAI/bge-m3** (다국어)
- 대안: `intfloat/multilingual-e5-large`
- 오프라인 평가: `eval/retrieval_eval.py` (Hit@K, MRR)

## TRL 분석 한계

TRL 4~6 구간은 기업 비공개 영역으로, 특허·논문·채용 등 **간접 지표 기반 추정**입니다.
