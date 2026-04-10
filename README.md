# 반도체 R&D 기술 전략 분석 에이전트

경쟁사 반도체 R&D 동향을 자동 수집·분석하여 전략 보고서를 생성하는 AI 에이전트 시스템.

## 아키텍처

```
User Query
    ↓
Supervisor Agent          ← 전체 흐름 제어 (LangGraph 조건부 라우팅)
    ↓
Query Planning Agent      ← 기업/기술/이슈 중심 쿼리 혼합 생성 + 약어 확장
    ↓
Web Search Agent ──┐
                   ├─→ Balanced Retrieval Node  ← 중복제거·재정렬·편향제거
RAG Agent ─────────┘
    ↓
TRL Preparation Node      ← 논문/특허/뉴스/채용 Signal 집계
    ↓
Competitive Analyst Agent ← TRL 추정 + 위협 수준 평가
    ↓
Draft Generation Agent    ← 전략 보고서 초안 생성
    ↓
Validation Node           ← 품질 검증 (최대 2회 재시도)
    ↓
Formatting Node           ← 최종 형식 통일 + 면책 고지 추가
    ↓
Final Report (Markdown)
```

## 파일 구조

```
semiconductor_agent/
├── state.py        # SupervisorState 및 하위 TypedDict 정의
├── config.py       # LLM / ChromaDB / 기본값 설정
├── agents.py       # 각 에이전트·노드 구현
├── graph.py        # LangGraph 그래프 정의 및 라우팅
├── main.py         # 실행 진입점
├── ingest.py       # ChromaDB 문서 적재 스크립트
├── requirements.txt
└── .env.example
```

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. (선택) 내부 문서 적재
python ingest.py --dir ./docs

# 4. 분석 실행
python main.py
```

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

## 내부 문서 적재 규칙 (ChromaDB)

파일명을 `{기업}_{기술}_{YYYYMM}.txt` 형식으로 저장 후 ingest.py 실행.
예: `Samsung_HBM4_202501.txt`

## 임베딩 모델

다국어 지원 + 반도체 도메인 성능 기준으로 선택:
- **BAAI/bge-m3** (기본): 한·영 혼용 문서에 강함
- `intfloat/multilingual-e5-large`: 대안 옵션

성능 평가 기준: **Hit@K**, **MRR**

## TRL 분석 한계

TRL 4~6 구간은 기업 비공개 영역으로, 특허 출원 수·논문 발표 빈도·
채용 공고 키워드 등 **간접 지표 기반 추정**입니다.
