"""
agents.py — 각 에이전트/노드 구현
Supervisor → QueryPlanning → ParallelRetrieve(Web+RAG) → Balanced →
TRLPrep → CompetitiveAnalyst(Task4추출) → Draft → Validation → Formatting → Supervisor
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults

from config import (
    get_llm, get_fast_llm,
    get_chroma_client, get_collection,
    DEFAULT_TECHNOLOGIES, DEFAULT_COMPANIES,
    MAX_VALIDATION_RETRY, MAX_FLOW_RETRY,
)
from state import (
    SupervisorState,
    SearchQuery,
    RetrievedDocument,
    Signal,
    CompetitorAnalysis,
    ValidationState,
    DocumentExtraction,
)
from retrieval_utils import (
    canonical_company,
    parse_scope_dates,
    doc_within_scope,
    mmr_select,
    MIN_CREDIBILITY_FOR_BALANCE,
    MMR_POOL_SIZE,
)

llm = get_llm()
fast_llm = get_fast_llm()
search_tool = DuckDuckGoSearchResults(output_format="list", num_results=5)


def _meta_truthy_flag(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


def _credibility_from_rag_meta(meta: dict) -> float:
    if _meta_truthy_flag(meta.get("is_official_source")):
        return 0.92
    st = (meta.get("source_type") or "").lower()
    if st in ("standard", "patent"):
        return 0.88
    if st in ("paper", "whitepaper", "ir_report"):
        return 0.82
    return 0.74


def _freshness_from_published_at(pub: str) -> float:
    if not pub or len(pub) < 7:
        return 0.55
    try:
        y, m = int(pub[:4]), int(pub[5:7])
        doc_dt = datetime(y, m, 1)
        months = (datetime.now() - doc_dt).days / 30.44
        return max(0.25, min(1.0, 1.0 - min(months, 48) * 0.018))
    except Exception:
        return 0.55


def _chunk_index_from_meta(meta: dict) -> int:
    v = meta.get("chunk_index", 0)
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _retrieval_dedup_key(doc: RetrievedDocument) -> tuple:
    if doc.get("source") == "rag" and doc.get("doc_id"):
        return ("rag", doc["doc_id"], str(doc.get("chunk_index", 0)))
    u = (doc.get("url") or "").strip()
    if u:
        return ("web", u)
    return ("web", "__nourl__", id(doc))


# ═══════════════════════════════════════════════════════
# 1. Supervisor Agent
# ═══════════════════════════════════════════════════════
def supervisor_node(state: SupervisorState) -> SupervisorState:
    """
    전체 흐름 제어.
    - 첫 진입: next_action = "plan"
    - 각 단계 완료 후 다음 액션 결정
    - 검증 실패 시 재시도 또는 종료
    """
    current = state.get("next_action", "plan")
    retry = state.get("flow_retry_count", 0)

    # 에러 누적이 임계치 초과 → 강제 종료
    if retry >= MAX_FLOW_RETRY:
        print("[Supervisor] 최대 재시도 초과 → 강제 종료")
        return {**state, "next_action": "end"}

    # 흐름 순서 결정
    # "plan": 검색 쿼리가 아직 없으면 라우터가 query_planning으로 가도록 next_action을 "plan"으로 유지
    if current == "plan" and not state.get("search_queries"):
        next_action = "plan"
    else:
        flow_map = {
            "plan": "retrieve",
            "retrieve": "balance",
            "balance": "prepare_trl",
            "prepare_trl": "analyze",
            "analyze": "draft",
            "draft": "validate",
            "validate": _decide_after_validate(state),
            "format": "end",
        }
        if current in flow_map:
            next_action = flow_map[current]
        else:
            next_action = "end"

    if current == "format" and next_action == "end":
        print("[Supervisor] 최종 보고서 정리 완료 → 종료")
    print(f"[Supervisor] {current} → {next_action}")
    return {**state, "next_action": next_action}


def _decide_after_validate(state: SupervisorState) -> str:
    validation = state.get("validation", {})
    if validation.get("passed", False):
        return "format"
    retry_count = validation.get("validation_retry_count", 0)
    if retry_count >= MAX_VALIDATION_RETRY:
        print("[Supervisor] 검증 재시도 초과 → format 강제 진행")
        return "format"
    return "draft"  # 재초안 생성


# ═══════════════════════════════════════════════════════
# 2. Query Planning Agent
# ═══════════════════════════════════════════════════════
QUERY_PLANNING_PROMPT = """
당신은 반도체 R&D 기술 인텔리전스 전문가입니다.
사용자 요청을 분석하여 다양한 관점의 검색 쿼리를 생성하세요.

규칙:
1. 기업 중심 / 기술 중심 / 문제·이슈 중심 쿼리를 혼합하여 확증 편향 방지
2. 반도체 약어는 풀네임도 포함 (HBM4 → High Bandwidth Memory 4, PIM → Processing-In-Memory)
3. 문서 유형을 활용한 우회 쿼리 포함 (논문·특허·채용공고·발표자료)
4. web 타겟: 최신 뉴스·특허·발표 / rag 타겟: 내부 기술 기준·히스토리
5. **반드시 `target`이 \"rag\"인 쿼리를 최소 4개 포함** (ChromaDB 내부 PDF·표준 문서 검색용)

JSON 배열로만 응답하세요. 각 항목:
{{"query": "...", "target": "web"|"rag", "purpose": "...", "query_type": "company"|"technology"|"issue"}}
"""

def query_planning_node(state: SupervisorState) -> SupervisorState:
    user_query = state.get("user_query", "")
    techs = state.get("target_technologies", DEFAULT_TECHNOLOGIES)
    companies = state.get("target_companies", DEFAULT_COMPANIES)

    prompt = f"""
사용자 요청: {user_query}
분석 기술: {techs}
분석 기업: {companies}

위 조건에 맞는 검색 쿼리를 최소 10개 이상 생성하세요.
"""
    messages = [
        SystemMessage(content=QUERY_PLANNING_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = fast_llm.invoke(messages)
    raw = response.content.strip()

    # JSON 파싱
    try:
        json_str = re.search(r"\[.*\]", raw, re.DOTALL).group()
        queries: List[SearchQuery] = json.loads(json_str)
    except Exception as e:
        print(f"[QueryPlanning] 파싱 실패: {e} → 기본 쿼리 사용")
        queries = _default_queries(techs, companies)

    queries = _ensure_rag_queries(queries, techs, companies)

    print(f"[QueryPlanning] {len(queries)}개 쿼리 생성")
    return {**state, "search_queries": queries, "next_action": "plan"}


def _default_rag_queries(techs: List[str], companies: List[str]) -> List[SearchQuery]:
    """LLM이 rag 쿼리를 안 주거나 파싱 실패 시 Chroma 검색용 보강"""
    out: List[SearchQuery] = []
    for tech in techs:
        out.append(
            {
                "query": f"{tech} HBM CXL PIM memory standard JEDEC whitepaper specification",
                "target": "rag",
                "purpose": f"{tech} 관련 내부·표준·백서 청크 검색",
                "query_type": "technology",
            }
        )
    for company in companies:
        compact = company.replace(" ", "")
        out.append(
            {
                "query": f"{company} {compact} IR report whitepaper CXL HBM PIM semiconductor",
                "target": "rag",
                "purpose": f"{company} 내부 성격 문서 검색",
                "query_type": "company",
            }
        )
    return out


def _ensure_rag_queries(
    queries: List[SearchQuery], techs: List[str], companies: List[str]
) -> List[SearchQuery]:
    if any(q.get("target") == "rag" for q in queries):
        return queries
    print("[QueryPlanning] rag 타겟 쿼리 없음 → 기본 RAG 쿼리 자동 추가")
    return queries + _default_rag_queries(techs, companies)


def _default_queries(techs: List[str], companies: List[str]) -> List[SearchQuery]:
    queries = []
    for tech in techs:
        for company in companies[:3]:
            queries.append({
                "query": f"{company} {tech} R&D 2024 2025",
                "target": "web",
                "purpose": f"{company}의 {tech} 개발 동향 파악",
                "query_type": "company",
            })
        queries.append({
            "query": f"{tech} semiconductor research paper patent 2024",
            "target": "web",
            "purpose": f"{tech} 기술 논문·특허 현황",
            "query_type": "technology",
        })
    queries.extend(_default_rag_queries(techs, companies))
    return queries


# ═══════════════════════════════════════════════════════
# 3. Web Search / RAG 수집 (병렬 노드에서 공용)
# ═══════════════════════════════════════════════════════
def collect_web_documents(state: SupervisorState) -> List[RetrievedDocument]:
    queries = [q for q in state.get("search_queries", []) if q.get("target") == "web"]
    targets = state.get("target_companies", DEFAULT_COMPANIES)
    new_docs: List[RetrievedDocument] = []

    for q in queries:
        try:
            results = search_tool.invoke(q["query"])
            if isinstance(results, str):
                results = json.loads(results)
            for r in results:
                co = _extract_company(r.get("title", "") + r.get("snippet", ""), state)
                doc: RetrievedDocument = {
                    "source": "web",
                    "title": r.get("title", ""),
                    "content": r.get("snippet", r.get("body", "")),
                    "url": r.get("link", r.get("href", "")),
                    "company": canonical_company(co, targets) if co else "",
                    "technology": _extract_technology(
                        r.get("title", "") + r.get("snippet", ""), state
                    ),
                    "published_at": r.get("date", datetime.now().strftime("%Y-%m")),
                    "relevance_score": 0.7,
                    "credibility_score": _calc_credibility(r.get("link", "")),
                    "freshness_score": 0.8,
                    "diversity_score": 0.5,
                    "final_score": 0.0,
                }
                new_docs.append(doc)
        except Exception as e:
            print(f"[WebSearch] 쿼리 실패 '{q['query']}': {e}")

    return new_docs


def web_search_node(state: SupervisorState) -> SupervisorState:
    existing = state.get("retrieved_documents", [])
    new_docs = collect_web_documents(state)
    print(f"[WebSearch] {len(new_docs)}개 문서 수집")
    return {**state, "retrieved_documents": existing + new_docs, "next_action": "retrieve"}


def _calc_credibility(url: str) -> float:
    """신뢰도 높은 출처 판별"""
    high_cred = ["ieee.org", "acm.org", "arxiv.org", "patent", "sec.gov",
                 "samsung.com", "micron.com", "skhynix.com"]
    return 0.9 if any(h in url for h in high_cred) else 0.6


# ═══════════════════════════════════════════════════════
# 4. RAG Agent
# ═══════════════════════════════════════════════════════
def collect_rag_documents(state: SupervisorState) -> List[RetrievedDocument]:
    queries = [q for q in state.get("search_queries", []) if q.get("target") == "rag"]
    targets = state.get("target_companies", DEFAULT_COMPANIES)
    new_docs: List[RetrievedDocument] = []

    try:
        client = get_chroma_client()
        collection = get_collection(client)

        for q in queries:
            results = collection.query(
                query_texts=[q["query"]],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.5
                pub = meta.get("published_at") or ""
                co_raw = meta.get("company", "") or ""
                doc: RetrievedDocument = {
                    "source": "rag",
                    "title": meta.get("title", "내부 문서"),
                    "content": doc_text,
                    "url": meta.get("url", "internal"),
                    "company": canonical_company(str(co_raw), targets)
                    if co_raw
                    else "",
                    "technology": meta.get("technology", ""),
                    "published_at": pub,
                    "source_type": meta.get("source_type", ""),
                    "publisher": meta.get("publisher", ""),
                    "is_official_source": str(meta.get("is_official_source", "false")),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_index": _chunk_index_from_meta(meta),
                    "trl_aggregation_key": meta.get("trl_aggregation_key", ""),
                    "relevance_score": max(0.0, 1.0 - distance),
                    "credibility_score": _credibility_from_rag_meta(meta),
                    "freshness_score": _freshness_from_published_at(pub),
                    "diversity_score": 0.4,
                    "final_score": 0.0,
                }
                new_docs.append(doc)
    except Exception as e:
        print(f"[RAG] ChromaDB 조회 실패: {e}")

    return new_docs


def rag_node(state: SupervisorState) -> SupervisorState:
    existing = state.get("retrieved_documents", [])
    new_docs = collect_rag_documents(state)
    print(f"[RAG] {len(new_docs)}개 문서 검색")
    return {**state, "retrieved_documents": existing + new_docs, "next_action": "retrieve"}


def parallel_retrieve_node(state: SupervisorState) -> SupervisorState:
    """Web + RAG 동시 수집(Thread) 후 retrieved_documents 병합."""
    from concurrent.futures import ThreadPoolExecutor

    existing = list(state.get("retrieved_documents") or [])
    with ThreadPoolExecutor(max_workers=2) as pool:
        fw = pool.submit(collect_web_documents, state)
        fr = pool.submit(collect_rag_documents, state)
        w_docs = fw.result()
        r_docs = fr.result()

    merged = existing + w_docs + r_docs
    print(
        f"[Retrieve/parallel] web={len(w_docs)} rag={len(r_docs)} "
        f"→ 합계 {len(merged)}개"
    )
    return {**state, "retrieved_documents": merged, "next_action": "retrieve"}


# ═══════════════════════════════════════════════════════
# 5. Balanced Retrieval Node
# ═══════════════════════════════════════════════════════
def balanced_retrieval_node(state: SupervisorState) -> SupervisorState:
    """
    중복 제거 + scope·신뢰도 필터 + 점수 정렬 + MMR로 다양성 확보
    """
    docs = state.get("retrieved_documents", [])
    targets = state.get("target_companies", DEFAULT_COMPANIES)
    scope = state.get("scope") or {}
    date_from, date_to = parse_scope_dates(scope)

    seen: set[tuple] = set()
    unique_docs: List[RetrievedDocument] = []
    for doc in docs:
        key = _retrieval_dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    for doc in unique_docs:
        if doc.get("company"):
            doc["company"] = canonical_company(str(doc["company"]), targets)

    scoped = [d for d in unique_docs if doc_within_scope(d, date_from, date_to)]
    if len(scoped) < max(3, len(unique_docs) // 4):
        pool_docs = unique_docs
        print("[BalancedRetrieval] scope 적용 시 문서 과소 → 전체 풀 사용")
    else:
        pool_docs = scoped

    cred_floor = MIN_CREDIBILITY_FOR_BALANCE
    official_ok = [
        d
        for d in pool_docs
        if float(d.get("credibility_score", 0)) >= cred_floor
        or _meta_truthy_flag(d.get("is_official_source"))
    ]
    if not official_ok:
        filtered = pool_docs
        print("[BalancedRetrieval] 신뢰도 하한 적용 시 비어 있음 → 필터 완화")
    else:
        filtered = official_ok

    for doc in filtered:
        div_hint = 0.35
        if doc.get("source") == "rag" and _meta_truthy_flag(doc.get("is_official_source")):
            div_hint = 0.75
        elif doc.get("source") == "web" and doc.get("url"):
            div_hint = 0.55
        doc["diversity_score"] = div_hint
        doc["final_score"] = (
            doc.get("relevance_score", 0.5) * 0.4
            + doc.get("credibility_score", 0.5) * 0.3
            + doc.get("freshness_score", 0.5) * 0.2
            + doc.get("diversity_score", 0.5) * 0.1
        )

    ranked = sorted(filtered, key=lambda d: d["final_score"], reverse=True)
    pool = ranked[: max(MMR_POOL_SIZE, 30)]
    balanced = mmr_select(pool, k=30)

    print(
        f"[BalancedRetrieval] {len(docs)} → 고유 {len(unique_docs)} "
        f"→ 필터 후 {len(filtered)} → MMR 상위 {len(balanced)}개"
    )
    return {**state, "balanced_documents": balanced, "next_action": "balance"}


# ═══════════════════════════════════════════════════════
# 6. TRL Preparation Node
# ═══════════════════════════════════════════════════════
def trl_preparation_node(state: SupervisorState) -> SupervisorState:
    """
    문서에서 Signal(논문/특허/뉴스/채용) 집계 → TRL 추정 데이터 구조화
    """
    docs = state.get("balanced_documents", [])
    techs = state.get("target_technologies", DEFAULT_TECHNOLOGIES)
    companies = state.get("target_companies", DEFAULT_COMPANIES)

    signal_map: dict = {}
    for tech in techs:
        for company in companies:
            key = f"{company}_{tech}"
            signal_map[key] = {
                "company": company, "technology": tech,
                "paper": 0, "patent": 0, "news": 0, "job": 0,
            }

    for doc in docs:
        company = canonical_company(str(doc.get("company", "")), companies)
        tech = doc.get("technology", "")
        content = (doc.get("content", "") + " " + doc.get("title", "")).lower()
        url = doc.get("url", "").lower()
        stype = (doc.get("source_type") or "").lower()

        if not company or not tech:
            continue
        key = f"{company}_{tech}"
        if key not in signal_map:
            signal_map[key] = {
                "company": company, "technology": tech,
                "paper": 0, "patent": 0, "news": 0, "job": 0,
            }

        if doc.get("source") == "rag" and stype:
            if stype in ("paper", "whitepaper"):
                signal_map[key]["paper"] += 1
                continue
            if stype == "patent":
                signal_map[key]["patent"] += 1
                continue
            if stype in ("ir_report", "standard"):
                signal_map[key]["news"] += 1
                continue

        if any(k in url for k in ["arxiv", "ieee", "acm", "scholar"]):
            signal_map[key]["paper"] += 1
        elif any(k in url for k in ["patent", "uspto", "kipris"]):
            signal_map[key]["patent"] += 1
        elif any(k in content for k in ["채용", "job", "hiring", "engineer wanted"]):
            signal_map[key]["job"] += 1
        else:
            signal_map[key]["news"] += 1

    signals: List[Signal] = []
    for key, s in signal_map.items():
        total = s["paper"] + s["patent"] + s["news"] + s["job"]
        trend = "increasing" if total >= 3 else ("stable" if total >= 1 else "decreasing")
        for stype in ["paper", "patent", "news", "job"]:
            if s[stype] > 0:
                signals.append({
                    "company": s["company"],
                    "technology": s["technology"],
                    "signal_type": stype,
                    "count": s[stype],
                    "trend": trend,
                })

    print(f"[TRLPrep] {len(signals)}개 Signal 생성")
    return {**state, "aggregated_signals": signals, "next_action": "prepare_trl"}


# ═══════════════════════════════════════════════════════
# 7. Competitive Analyst Agent
# ═══════════════════════════════════════════════════════
ANALYST_PROMPT = """
당신은 반도체 기술 전략 분석 전문가입니다.
수집된 문서와 Signal을 바탕으로 아래 항목을 분석하세요.

분석 항목:
1. trend_summary: 해당 기업-기술 조합의 최신 개발 방향 요약 (3~5문장)
2. trl_level: TRL 1~9 추정 (간접 지표 기반, 4~6은 추정임을 인지)
3. trl_evidence: TRL 추정 근거 목록 (최소 2개)
4. threat_level: "HIGH" | "MEDIUM" | "LOW"
5. key_findings: 핵심 발견사항 목록 (최소 3개)
6. source_urls: 참고 URL 목록

JSON 배열로만 응답. 각 항목:
{{"company":"...", "technology":"...", "trend_summary":"...",
  "trl_level":5, "trl_evidence":["..."], "threat_level":"HIGH",
  "key_findings":["..."], "source_urls":["..."]}}

주의: TRL 4~6 구간은 기업 비공개 영역이므로 간접 지표 기반 추정임을 trl_evidence에 명시.
"""

EXTRACT_PROMPT = """
각 문서 블록을 읽고 Task 4용 필드만 채운 JSON 배열을 출력하세요.
필드: title, url, company, technology, research_topic, development_purpose,
evidence_notes(기술수준·TRL 근거가 될 만한 문장), citation_line(한 줄 출처 표기)
모르면 빈 문자열. JSON 배열만 출력.
"""


def _extract_structured_documents(
    docs: List[RetrievedDocument],
) -> List[DocumentExtraction]:
    if not docs:
        return []
    blocks = []
    for i, d in enumerate(docs):
        body = (d.get("content") or "")[:1400]
        blocks.append(
            f"--- doc{i} ---\n"
            f"title:{d.get('title','')}\nurl:{d.get('url','')}\n"
            f"company:{d.get('company','')}\ntechnology:{d.get('technology','')}\n"
            f"body:{body}"
        )
    payload = "\n".join(blocks)
    messages = [
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=payload),
    ]
    try:
        response = fast_llm.invoke(messages)
        raw = response.content.strip()
        json_str = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_str:
            return []
        arr = json.loads(json_str.group())
        out: List[DocumentExtraction] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "company": str(item.get("company", "")),
                    "technology": str(item.get("technology", "")),
                    "research_topic": str(item.get("research_topic", "")),
                    "development_purpose": str(item.get("development_purpose", "")),
                    "evidence_notes": str(item.get("evidence_notes", "")),
                    "citation_line": str(item.get("citation_line", "")),
                }
            )
        return out
    except Exception as e:
        print(f"[Analyst/Extract] 구조화 추출 실패: {e}")
        return []


def competitive_analyst_node(state: SupervisorState) -> SupervisorState:
    docs = state.get("balanced_documents", [])
    signals = state.get("aggregated_signals", [])
    techs = state.get("target_technologies", DEFAULT_TECHNOLOGIES)
    companies = state.get("target_companies", DEFAULT_COMPANIES)

    extract_slice = docs[:12]
    extractions = _extract_structured_documents(extract_slice)
    extraction_json = json.dumps(extractions[:20], ensure_ascii=False, indent=2)

    doc_summary = "\n".join([
        f"[{d.get('company','?')}/{d.get('technology','?')}] {d.get('title','')} — "
        f"{(d.get('content') or '')[:1200]}"
        for d in docs[:14]
    ])
    signal_summary = json.dumps(signals[:30], ensure_ascii=False, indent=2)

    prompt = f"""
분석 대상 기업: {companies}
분석 기술: {techs}

[문서 구조화 추출 Task4]
{extraction_json}

[수집 문서 발췌]
{doc_summary}

[Signal 집계]
{signal_summary}

위 데이터를 기반으로 기업별·기술별 분석 결과를 생성하세요.
"""
    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        json_str = re.search(r"\[.*\]", raw, re.DOTALL).group()
        results: List[CompetitorAnalysis] = json.loads(json_str)
    except Exception as e:
        print(f"[Analyst] 파싱 실패: {e}")
        results = []

    print(f"[Analyst] {len(results)}개 분석 결과 생성")
    return {
        **state,
        "analysis_result": results,
        "document_extractions": extractions,
        "next_action": "analyze",
    }


# ═══════════════════════════════════════════════════════
# 8. Draft Generation Agent
# ═══════════════════════════════════════════════════════
DRAFT_PROMPT = """
당신은 기술 전략 보고서 작성 전문가입니다.
아래 분석 결과를 바탕으로 전략 보고서 초안을 작성하세요.

보고서 구조:
- SUMMARY: 핵심 분석 결과 요약 / 주요 경쟁사 전략 정리
- 1장. 분석 배경: HBM 시장 현황 / 차세대 기술 전략적 중요성
- 2장. 기술 현황: HBM4·PIM·CXL 개요 / 한계 및 발전 방향
- 3장. 경쟁사 동향 분석: 기업별 상세 분석 및 근거 제시
- 4장. 기술 성숙도 및 경쟁 비교: TRL 분석 / 포지션 비교 매트릭스
- 5장. 전략적 시사점: 위협·기회 요소 / 대응 전략 제안
- REFERENCE: 출처 목록 (하이퍼링크 포함)
- 분석 한계 고지 (TRL 4~6 간접 지표 기반 추정 명시)

마크다운 형식으로 작성.
언어는 한국어로 작성.
"""

def draft_generation_node(state: SupervisorState) -> SupervisorState:
    analysis = state.get("analysis_result", [])
    validation = state.get("validation", {})
    history = state.get("draft_history", [])
    retry_count = validation.get("validation_retry_count", 0)

    feedback_section = ""
    if retry_count > 0:
        feedback_section = f"\n[이전 검증 피드백]\n{validation.get('feedback', '')}\n위 피드백을 반영하여 개선하세요."

    prompt = f"""
분석 결과:
{json.dumps(analysis, ensure_ascii=False, indent=2)[:4000]}
{feedback_section}

위 내용을 바탕으로 전략 보고서 초안을 작성하세요.
"""
    messages = [
        SystemMessage(content=DRAFT_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    draft = response.content

    print(f"[DraftGen] 초안 생성 완료 ({len(draft)}자) / 재시도: {retry_count}")
    return {
        **state,
        "current_draft": draft,
        "draft_history": history + [draft],
        "next_action": "draft",
    }


# ═══════════════════════════════════════════════════════
# 9. Validation (검증 노드)
# ═══════════════════════════════════════════════════════
VALIDATION_PROMPT = """
당신은 기술 보고서 품질 검토자입니다. 아래 기준으로 보고서를 평가하세요.

검증 기준:
1. 모든 필수 섹션(SUMMARY, 1~5장, REFERENCE) 포함 여부
2. 기업별 분석에 근거 자료(출처) 포함 여부
3. TRL 추정 시 한계 고지 여부
4. 전략적 시사점의 구체성 (단순 나열이 아닌 실행 가능한 제안)
5. 비교 가능한 형태(매트릭스 또는 표)로 기업 비교 포함 여부

JSON으로 응답:
{{"passed": true|false, "feedback": "개선 사항 구체적으로..."}}
"""

def validation_node(state: SupervisorState) -> SupervisorState:
    draft = state.get("current_draft", "")
    validation = state.get("validation", {})
    retry_count = validation.get("validation_retry_count", 0)

    messages = [
        SystemMessage(content=VALIDATION_PROMPT),
        HumanMessage(content=f"보고서 초안:\n{draft[:5000]}"),
    ]
    response = fast_llm.invoke(messages)
    raw = response.content.strip()

    try:
        json_str = re.search(r"\{.*\}", raw, re.DOTALL).group()
        result = json.loads(json_str)
    except Exception:
        result = {"passed": True, "feedback": "파싱 실패 — 기본 통과"}

    new_validation: ValidationState = {
        "passed": result.get("passed", True),
        "feedback": result.get("feedback", ""),
        "validation_retry_count": retry_count + (0 if result.get("passed") else 1),
    }
    status = "통과" if new_validation["passed"] else f"실패 (재시도 {new_validation['validation_retry_count']})"
    print(f"[Validation] {status}")
    return {**state, "validation": new_validation, "next_action": "validate"}


# ═══════════════════════════════════════════════════════
# 10. Formatting Node
# ═══════════════════════════════════════════════════════
def formatting_node(state: SupervisorState) -> SupervisorState:
    """
    최종 보고서 형식 통일 + 헤더/날짜/분석 한계 고지 추가
    """
    draft = state.get("current_draft", "")
    companies = state.get("target_companies", [])
    techs = state.get("target_technologies", [])
    today = datetime.now().strftime("%Y년 %m월 %d일")

    header = f"""# 반도체 R&D 기술 전략 분석 보고서

> **분석 대상 기업:** {', '.join(companies)}
> **분석 기술:** {', '.join(techs)}
> **작성일:** {today}
> **분석 도구:** AI 에이전트 시스템 (Web Search + RAG)

---

"""
    disclaimer = """

---

> **⚠️ 분석 한계 고지**
> TRL 4~6 구간은 기업 비공개 영역으로, 본 보고서의 기술 성숙도 평가는
> 특허 출원 수·논문 발표 빈도·채용 공고 키워드 등 **간접 지표 기반 추정**입니다.
> 직접적인 기술 수준 확인이 아님을 명시합니다.
"""
    final_report = header + draft + disclaimer

    print(f"[Formatting] 최종 보고서 완성 ({len(final_report)}자)")
    return {**state, "final_report": final_report, "next_action": "format"}


# ═══════════════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════════════
def _extract_company(text: str, state: SupervisorState) -> str:
    companies = state.get("target_companies", DEFAULT_COMPANIES)
    text_lower = text.lower()
    for c in companies:
        if c.lower() in text_lower:
            return c
    return ""

def _extract_technology(text: str, state: SupervisorState) -> str:
    techs = state.get("target_technologies", DEFAULT_TECHNOLOGIES)
    text_lower = text.lower()
    for t in techs:
        if t.lower() in text_lower:
            return t
    return ""
