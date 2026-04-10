"""
state.py — SupervisorState 및 하위 상태 타입 정의
보고서 설계의 State/Graph 흐름을 그대로 반영
"""
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional


# ─────────────────────────────────────────
# Query Planning
# ─────────────────────────────────────────
class SearchQuery(TypedDict, total=False):
    query: str
    target: Literal["web", "rag"]
    purpose: str
    query_type: Literal["company", "technology", "issue"]


# ─────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────
class RetrievedDocument(TypedDict, total=False):
    source: Literal["web", "rag"]
    title: str
    content: str
    url: str
    company: str
    technology: str
    published_at: str
    source_type: str
    publisher: str
    is_official_source: str  # "true" | "false" (Chroma 메타데이터와 동일)
    doc_id: str
    chunk_index: int
    trl_aggregation_key: str
    relevance_score: float
    credibility_score: float  # 신뢰도 (balance 판단)
    freshness_score: float    # 최신성
    diversity_score: float    # 출처 다양성
    final_score: float        # 종합 점수


# ─────────────────────────────────────────
# Signal (TRL 근거 데이터)
# ─────────────────────────────────────────
class Signal(TypedDict, total=False):
    company: str
    technology: str
    signal_type: Literal["paper", "patent", "news", "job"]
    count: int
    trend: Literal["increasing", "stable", "decreasing"]


# ─────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────
class CompetitorAnalysis(TypedDict, total=False):
    company: str
    technology: str
    trend_summary: str
    trl_level: int
    trl_evidence: List[str]
    threat_level: str          # "HIGH" | "MEDIUM" | "LOW"
    key_findings: List[str]
    source_urls: List[str]


# ─────────────────────────────────────────
# Task 4 — 문서별 구조화 추출 (Analyst 입력용)
# ─────────────────────────────────────────
class DocumentExtraction(TypedDict, total=False):
    title: str
    url: str
    company: str
    technology: str
    research_topic: str
    development_purpose: str
    evidence_notes: str
    citation_line: str


# ─────────────────────────────────────────
# Validation
# ─────────────────────────────────────────
class ValidationState(TypedDict, total=False):
    passed: bool
    feedback: str
    validation_retry_count: int


# ─────────────────────────────────────────
# SupervisorState (전체 그래프 공유 상태)
# ─────────────────────────────────────────
class SupervisorState(TypedDict, total=False):
    # ── 입력
    user_query: str
    target_technologies: List[str]   # 예: ["HBM4", "PIM", "CXL"]
    target_companies: List[str]      # 예: ["Samsung", "Micron", "Broadcom"]
    scope: Dict[str, Any]            # 분석 기간, 지역 등 부가 조건

    # ── Planning
    search_queries: List[SearchQuery]

    # ── Retrieval
    retrieved_documents: List[RetrievedDocument]
    balanced_documents: List[RetrievedDocument]

    # ── TRL 준비 (Signal 집계)
    aggregated_signals: List[Signal]

    # ── Analysis
    analysis_result: List[CompetitorAnalysis]
    document_extractions: List[DocumentExtraction]

    # ── Draft
    draft_history: List[str]
    current_draft: str

    # ── Validation
    validation: ValidationState

    # ── Output
    final_report: str

    # ── Flow Control
    next_action: Literal[
        "plan",
        "retrieve",
        "balance",
        "prepare_trl",
        "analyze",
        "draft",
        "validate",
        "format",
        "end",
    ]
    flow_retry_count: int
    error: Optional[str]
    error_history: List[str]
