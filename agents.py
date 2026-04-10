"""
agents.py — 각 에이전트/노드 구현
Supervisor → QueryPlanning → ParallelRetrieve(Web+RAG) → Balanced →
TRLPrep → CompetitiveAnalyst(Task4추출) → Draft → Validation → Formatting → Supervisor
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, List
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

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

WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "8"))
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "8"))
BALANCED_K = int(os.getenv("BALANCED_K", "80"))
ANALYST_DOCS_MAX = int(os.getenv("ANALYST_DOCS_MAX", "35"))
EXTRACTIONS_MAX = int(os.getenv("EXTRACTIONS_MAX", "60"))

llm = get_llm()
fast_llm = get_fast_llm()

_tavily_search: TavilySearch | None = None


def _get_tavily_search() -> TavilySearch:
    """TAVILY_API_KEY 검증이 있어 모듈 import 시점이 아니라 첫 웹 검색 시 생성."""
    global _tavily_search
    if _tavily_search is None:
        _tavily_search = TavilySearch(max_results=WEB_MAX_RESULTS, search_depth="basic")
    return _tavily_search


def _normalize_web_published(raw: Any) -> str:
    """Tavily published_date 등 → YYYY-MM-DD 또는 YYYY-MM. 없으면 빈 문자열."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = s.replace("Z", "").replace("T", " ")
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if len(s) >= 7 and s[4] == "-":
        return s[:7]
    return s[:32]


def _publisher_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _tavily_fetch_results(query: str, state: SupervisorState) -> list[dict[str, Any]]:
    """
    tool.invoke 대신 api_wrapper.raw_results 로 원본 results 를 받아
    published_date 등 메타를 유지 (langchain-tavily / Bearer 인증).
    """
    tool = _get_tavily_search()
    w = tool.api_wrapper
    scope = state.get("scope") or {}
    date_from, date_to = parse_scope_dates(scope)
    start_s = date_from.strftime("%Y-%m-%d") if date_from else None
    end_s = date_to.strftime("%Y-%m-%d") if date_to else None

    topic: Any = tool.topic if tool.topic is not None else "general"
    env_topic = os.getenv("TAVILY_TOPIC", "").strip().lower()
    if env_topic in ("general", "news", "finance"):
        topic = env_topic

    raw = w.raw_results(
        query=query,
        max_results=tool.max_results if tool.max_results is not None else 5,
        search_depth=tool.search_depth if tool.search_depth is not None else "basic",
        include_domains=tool.include_domains,
        exclude_domains=tool.exclude_domains,
        include_answer=tool.include_answer if tool.include_answer is not None else False,
        include_raw_content=tool.include_raw_content
        if tool.include_raw_content is not None
        else False,
        include_images=tool.include_images if tool.include_images is not None else False,
        include_image_descriptions=tool.include_image_descriptions,
        include_favicon=tool.include_favicon,
        topic=topic,
        time_range=tool.time_range,
        country=tool.country,
        auto_parameters=tool.auto_parameters,
        start_date=start_s,
        end_date=end_s,
        include_usage=tool.include_usage,
    )
    return list(raw.get("results") or [])


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
            results = _tavily_fetch_results(q["query"], state)
            print(f"[Tavily] query_results={len(results)} | {q.get('query','')}")
            for r in results:
                body = (r.get("content") or r.get("snippet") or r.get("body") or "")
                url = (r.get("url") or r.get("link") or r.get("href") or "").strip()
                pub = _normalize_web_published(
                    r.get("published_date")
                    or r.get("publishedDate")
                    or r.get("date")
                )
                score = r.get("score")
                rel = float(score) if isinstance(score, (int, float)) else 0.7
                co = _extract_company(r.get("title", "") + body, state)
                doc: RetrievedDocument = {
                    "source": "web",
                    "title": (r.get("title") or "")[:500],
                    "content": body,
                    "url": url,
                    "company": canonical_company(co, targets) if co else "",
                    "technology": _extract_technology(
                        r.get("title", "") + body, state
                    ),
                    "published_at": pub,
                    "source_type": "web",
                    "publisher": _publisher_from_url(url),
                    "relevance_score": rel,
                    "credibility_score": _calc_credibility(url),
                    "freshness_score": _freshness_from_published_at(pub)
                    if pub
                    else 0.55,
                    "diversity_score": 0.5,
                    "final_score": 0.0,
                }
                new_docs.append(doc)
        except Exception as e:
            print(f"[Tavily] 쿼리 실패 '{q['query']}': {e}")

    return new_docs


def web_search_node(state: SupervisorState) -> SupervisorState:
    existing = state.get("retrieved_documents", [])
    new_docs = collect_web_documents(state)
    print(f"[Tavily] {len(new_docs)}개 문서 수집")
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
                n_results=RAG_N_RESULTS,
                include=["documents", "metadatas", "distances"],
            )
            n = len(results.get("documents", [[]])[0]) if results.get("documents") else 0
            print(f"[RAG] query_results={n} | {q.get('query','')}")
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
    balanced = mmr_select(pool, k=BALANCED_K)

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
        # tech = doc.get("technology", "")
        if not tech:
            tech = _extract_technology(content, state)
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
당신은 반도체 기술 전략 분석 전문가입니다. 아래의 [TRL 판정 기준]과 [정보 공개 패턴]을 참조하여 수집된 데이터를 분석하고 JSON 배열로 응답하세요.

[TRL 9단계 판정 기준]
1. TRL 1~2: 기초 원리 관찰, 아이디어/이론 수준
2. TRL 3: 개념 검증, 실험실 수준 실증
3. TRL 4: 부품 검증, 실험실 환경 통합
4. TRL 5: 부품 검증(실환경), 유사 환경 통합 테스트
5. TRL 6: 시스템 시연, 실제 환경 유사 조건 시연
6. TRL 7: 시스템 시제품, 실제 운용 환경 시연 (샘플 공급)
7. TRL 8: 시스템 완성, 양산 적합성 검증 완료
8. TRL 9: 실제 운용, 상용 양산 및 납품

[정보 공개 패턴 분석 지침]
- TRL 1~3: 논문, 학회, 특허 출원 등 공개 활동 활발 (인재 채용, 투자 유치 목적)
- TRL 4~6 (GAP 구간): 수율, 공정 파라미터 등 영업 비밀로 인해 정보가 극도로 제한됨. 과거 활동 대비 침묵이 길어지면 이 구간으로 추정.
- TRL 7~9: 고객사 샘플 공급, 양산 발표, IR 실적 공시 등 비즈니스 목적으로 다시 정보 공개 시작.

[분석 항목 및 출력 형식]
반드시 아래 키워드를 가진 JSON 배열로만 응답하세요:
{
  "company": "기업명",
  "technology": "기술명",
  "trend_summary": "최신 개발 방향 요약 (3~5문장)",
  "trl_level": 추정 숫자 (1~9),
  "trl_evidence": ["단계 판정 근거 1", "단계 판정 근거 2"],
  "information_gap_analysis": "4~6단계를 포함한 정보 노출 패턴 분석 결과",
  "threat_level": "HIGH" | "MEDIUM" | "LOW",
  "key_findings": ["핵심 발견사항 3개 이상"],
  "source_urls": ["참고 URL"]
}

주의: TRL 4~6 구간은 기업 비공개 영역이므로 간접 지표 기반 추정임을 trl_evidence에 명시.
- source_urls는 반드시 입력 데이터(documents)에 존재하는 URL만 사용
"""

EXTRACT_PROMPT = """
당신은 반도체 문서 데이터 추출기입니다. 각 문서 블록을 읽고 아래 필드를 채운 JSON 배열을 출력하세요.

[필드 정의]
- title: 문서 제목
- url: 문서 주소
- company: 대상 기업
- technology: 대상 기술
- research_topic: 구체적 연구 주제
- development_purpose: 기술 개발 목적 (PPA 개선 등)
- evidence_notes: TRL 판정 근거 (예: 수율 언급, 샘플 공급, 학회 발표 여부 등)
- citation_line: 한 줄 출처 (예: [작성자/매체, 연도])

[핵심 원칙: 원문 보존]
- 과도한 요약 금지. 입력 body의 핵심 문장을 가능한 한 원문 어휘로 유지.
- research_topic / development_purpose / evidence_notes는 "짧은 키워드"가 아니라 문장 단위로 작성.
- 가능한 경우 문서의 표현을 그대로 인용하거나 준인용하여 정보 손실을 최소화.
- 정보가 불명확하면 빈칸 대신 "문서에 명시적 근거 부족"이라고 명시.

[길이 가이드]
- research_topic: 2~4문장 (최소 120자)
- development_purpose: 2~4문장 (최소 120자)
- evidence_notes: 3~6문장 (최소 220자), TRL 간접근거/직접근거를 구분해서 작성
- citation_line: [매체/학회, 연도] 형식 유지
"""


def _extract_structured_documents(
    docs: List[RetrievedDocument],
) -> List[DocumentExtraction]:
    if not docs:
        return []
    blocks = []
    for i, d in enumerate(docs):
        body = (d.get("content") or "")[:5000]
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
            research_topic = str(item.get("research_topic", "")).strip()
            development_purpose = str(item.get("development_purpose", "")).strip()
            evidence_notes = str(item.get("evidence_notes", "")).strip()

            # 과도한 축약 방지: 너무 짧은 필드는 원문 기반 보정 문구를 붙여 최소 정보량 확보
            if len(research_topic) < 80:
                research_topic = (
                    research_topic + " "
                    + "문서 본문의 핵심 기술 설명을 원문 표현 중심으로 추가 보완 필요."
                ).strip()
            if len(development_purpose) < 80:
                development_purpose = (
                    development_purpose + " "
                    + "문서에 나타난 개발 목적/적용 맥락을 원문 기반으로 확장 기술."
                ).strip()
            if len(evidence_notes) < 120:
                evidence_notes = (
                    evidence_notes + " "
                    + "TRL 근거(논문/특허/시제품/양산/고객사 언급 여부)를 문서 표현 기준으로 보강."
                ).strip()

            out.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "company": str(item.get("company", "")),
                    "technology": str(item.get("technology", "")),
                    "research_topic": research_topic,
                    "development_purpose": development_purpose,
                    "evidence_notes": evidence_notes,
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

    extract_slice = docs[:ANALYST_DOCS_MAX]
    extractions = _extract_structured_documents(extract_slice)
    extraction_json = json.dumps(extractions[:EXTRACTIONS_MAX], ensure_ascii=False, indent=2)

    doc_summary = "\n".join([
        f"[{d.get('company','?')}/{d.get('technology','?')}] {d.get('title','')} — "
        f"{(d.get('content') or '')[:3200]}"
        for d in docs[:ANALYST_DOCS_MAX]
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
당신은 SK hynix의 R&D 전략 담당자에게 보고하는 기술 전략 보고서 작성 전문가입니다.
반드시 "SK hynix 기준에서 경쟁사를 평가하고 대응 전략을 도출"하는 관점으로 작성하세요.

다음 요구사항을 반드시 만족해야 한다:
- 분량 규칙 (엄격 적용):
  - REFERENCE를 제외한 본문(SUMMARY ~ 5장)만 기준으로 최소 8000자 이상 작성
  - REFERENCE, URL, 표의 단순 나열은 글자 수에 포함되지 않음
  - 각 장(1~5장)은 최소 1000자 이상 작성
  - SUMMARY는 최소 500자 이상 작성
  - 분량이 부족할 경우 반드시 추가 분석을 생성하여 보완

[핵심 작성 원칙]
- SK hynix를 기준으로 경쟁사 기술 수준과 위협도를 비교 분석
- 분량을 충분히 확보할 것: 전체 본문(REFERENCE 제외) **최소 8000자 이상** 작성할 것. 
- **반드시 REFFERENCE는 최대 20개 이내로 제한**
- 단순 설명이 아닌 "정량적 지표 + 근거" 중심으로 작성
- 수치가 없을 경우: "데이터 부족" 또는 "추정"이라고 명시
- 경쟁사 분석은 반드시 기업별로 분리
- TRL은 반드시 근거 기반으로 추정 (추정 여부 명시)
- REFERENCE에는 실제 URL 포함
- 전체 한국어 작성
- 반드시 아래 목차 구조와 동일한 마크다운 헤더 사용 (변형 금지)
- 각 장은 최소 4개 이상의 문단 또는 4개 이상의 불릿으로 작성
- 단순 요약이 아니라 분석 중심으로 서술할 것
- 가능한 한 구체적인 사례와 근거를 포함할 것
- 3장/5장은 특히 상세히 작성: 기업별 근거와 대응 전략을 각각 최소 2문단 이상 제시
- 본문이 8000자 미만일 경우 보고서를 완성하지 말고 내용을 추가하여 기준을 충족시킬 것

---

## SUMMARY

- 전체 분석 결과를 1000자 이상 요약
- 주요 경쟁사(삼성전자, 마이크론 등)와의 격차를 정량적으로 비교
- 핵심 위협 요소 및 기회 요소를 수치 기반으로 요약

---

## 1장. 분석 배경

- HBM 시장 규모, CAGR, 주요 기업 점유율 (SK hynix 포함) 제시
- 왜 지금 HBM4/PIM/CXL 분석이 중요한지:
  → AI 시장 성장률, GPU 수요 증가 등 데이터 기반 설명

---

## 2장. 기술 현황

- HBM4, PIM, CXL 각각에 대해:
  - 성능 지표 (대역폭, 전력 효율 등)
  - 기존 HBM3E 대비 개선 수치
- 기술 한계 (발열, 수율, 비용 등)도 가능하면 정량적으로 표현

---

## 3장. 경쟁사 동향 분석

대상 기업:
- TRL 레벨이 높은 기업들을 대상으로 분석

각 기업별 필수 포함 요소:
- SK hynix 대비 기술 격차 (성능, 제품 출시 여부 등)
- R&D 투자 규모 / CAPEX / 비율
- 특허 수 / 논문 수 / 최근 발표 건수
- 제품 양산 여부 또는 샘플 공급 단계
- 채용 공고 수 또는 기술 키워드 등장 빈도
- 기술 방향 (예: PIM 집중, CXL 연계 등)

→ 반드시 "SK hynix 대비 위협 수준"을 명시 (높음/중간/낮음 + 근거)

---

## 4장. 기술 성숙도 및 경쟁 비교

- 기업별 TRL 수준 명시 (숫자 필수)
- TRL 판단 근거:
  (논문, 특허, 제품 출시, 고객사 공급 여부 등과 연결)

- SK hynix를 기준으로 경쟁사 위치를 비교:
  → "기술 성숙도 vs 시장 적용 수준" 관점

- 반드시 비교 가능한 형태 포함:
  - Markdown 표 또는 매트릭스

예시 항목:
| 기업 | TRL | 양산 여부 | 주요 기술 | 위협 수준 |

---

## 5장. 전략적 시사점

- 반드시 SK hynix의 R&D 의사결정을 지원하는 형태로 작성

1) 위협 요소
- 경쟁사의 TRL, 투자 규모, 고객사 확보 여부 기반 분석

2) 기회 요소
- SK hynix가 선점 가능한 기술 영역
- 경쟁사 대비 공백 영역

3) 대응 전략 (핵심)
- 실행 가능한 수준으로 작성
  (예: HBM4 특정 기술 선행 투자, PIM 통합 가속 등)
- 우선순위 또는 영향도 포함 (High / Medium / Low)

→ "즉시 실행 가능한 전략" 최소 2개 이상 포함

---

## REFERENCE

- 논문, 특허, 기사, 발표자료 등
- 많은 문서를 참조하더라도 문서 url은 최대 20개 이내로 제한
- 반드시 URL 포함

---

"""

def draft_generation_node(state: SupervisorState) -> SupervisorState:
    analysis = state.get("analysis_result", [])
    balanced_docs = state.get("balanced_documents", [])
    signals = state.get("aggregated_signals", [])
    validation = state.get("validation", {})
    history = state.get("draft_history", [])
    retry_count = validation.get("validation_retry_count", 0)

    feedback_section = ""
    if retry_count > 0:
        feedback_section = f"\n[이전 검증 피드백]\n{validation.get('feedback', '')}\n위 피드백을 반영하여 개선하세요."

    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    signal_json = json.dumps(signals[:40], ensure_ascii=False, indent=2)

    doc_digest = "\n".join(
        [
            f"- [{d.get('source','?')}] {d.get('company','?')}/{d.get('technology','?')} | "
            f"{d.get('title','')[:180]} | "
            f"url={d.get('url','')[:160]} | "
            f"excerpt={(d.get('content','') or '')[:800]}"
            for d in balanced_docs[:15]
        ]
    )
    fallback_note = ""
    if len(analysis) < 8:
        fallback_note = (
            "\n[중요]\n"
            f"- 현재 analysis_result 개수={len(analysis)}로 충분히 압축되어 있습니다.\n"
            "- 보고서를 짧게 요약하지 말고, 아래 문서 구조화 추출/원문 발췌/시그널을 직접 근거로 확장 작성하세요.\n"
            "- 특히 3장(경쟁사 동향)과 5장(전략적 시사점)은 문서 근거를 인용해 상세히 작성하세요.\n"
        )
    prompt = f"""
분석 결과:
{analysis_json[:12000]}

[문서 구조화 추출 (원문 보존형)]
[수집 문서 원문 발췌]
{doc_digest[:9000]}

[Signal 집계]
{signal_json[:5000]}
{fallback_note}
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
당신은 기술 전략 보고서를 검토하는 시니어 컨설턴트입니다.
아래 기준에 따라 보고서를 엄격하게 평가하세요.

[평가 원칙]
- 단순 포함 여부가 아니라 "품질 수준"까지 평가
- 기준을 만족하지 못하면 반드시 failed 처리
- 모호한 표현, 근거 없는 주장, 정량 데이터 부족은 모두 감점 요인

---

[검증 기준]

1. 구조 완전성
- SUMMARY, 1~5장, REFERENCE, 분석 한계 고지가 모두 존재해야 함
- 마크다운 헤더 구조가 유지되어야 함

2. 정량 데이터 포함 여부 (중요)
- 시장 규모, 점유율, CAGR, 특허 수, 논문 수, 성능 수치 등 정량 지표를 포함해야 함
- 최소 2개 이상의 정량 지표 포함 필수
- 서로 다른 유형의 지표 포함 필요 (예: 시장 규모 + 기술 성능, 특허 수 + 점유율 등)
- 단순 설명 위주이거나 정량 데이터가 부족하면 실패 처리

3. 근거 기반 분석
- 경쟁사 분석에 출처 또는 근거 명시 여부
- "추정", "가능성" 등의 표현만 있고 근거가 없으면 실패

4. TRL 분석 타당성
- TRL 숫자가 명시되어야 함
- TRL 판단 근거 (논문, 특허, 제품, 뉴스 등)와 연결되어야 함
- 분석 한계 고지 포함 여부

5. 경쟁사 비교 구조
- 기업 간 비교가 가능한 형태로 정리되어야 함
  (매트릭스, 표, 또는 명확한 비교 서술)
- 단순 나열일 경우 실패

6. 전략적 시사점의 실행 가능성
- "해야 한다" 수준이 아닌 구체적 실행 전략 포함
- 최소 1개 이상의 실행 가능한 액션 아이템 포함

---

[평가 결과 생성 규칙]

- 모든 기준을 만족하면: "passed": true
- 하나라도 부족하면: "passed": false

- feedback에는 반드시:
  1) 부족한 항목
  2) 왜 문제인지
  3) 어떻게 수정해야 하는지
를 구체적으로 작성

---

JSON 형식으로만 응답:
{
  "passed": true | false,
  "feedback": "구체적인 개선 사항"
}
"""

def validation_node(state: SupervisorState) -> SupervisorState:
    draft = state.get("current_draft", "")
    validation = state.get("validation", {})
    retry_count = validation.get("validation_retry_count", 0)

    messages = [
        SystemMessage(content=VALIDATION_PROMPT),
        HumanMessage(content=f"보고서 초안:\n{draft[:]}"),
    ]
    response = fast_llm.invoke(messages)
    raw = response.content.strip()

    try:
        json_str = re.search(r"\{.*\}", raw, re.DOTALL).group()
        result = json.loads(json_str)
    except Exception:
        result = {"passed": False, "feedback": "JSON 파싱 실패 — 형식 오류로 재생성 필요"}

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
    # REFERENCE는 LLM 생성 결과가 빈약할 수 있어, state의 문서 메타(title/url)로 보강한다.
    balanced_docs = list(state.get("balanced_documents") or [])
    extractions = list(state.get("document_extractions") or [])
    companies = state.get("target_companies", [])
    techs = state.get("target_technologies", [])
    today = datetime.now().strftime("%Y년 %m월 %d일")

    header = f"""# 반도체 R&D 기술 전략 분석 보고서

> **분석 대상 기업:** {', '.join(companies)}
> **분석 기술:** {', '.join(techs)}
> **작성일:** {today}
> **분석 도구:** AI 에이전트 시스템 (Tavily 웹 검색 + RAG)

---

"""
    disclaimer = """

---

> **분석 한계 고지**
> TRL 4~6 구간은 기업 비공개 영역으로, 본 보고서의 기술 성숙도 평가는
> 특허 출원 수·논문 발표 빈도·채용 공고 키워드 등 **간접 지표 기반 추정**입니다.
> 직접적인 기술 수준 확인이 아님을 명시합니다.
"""

    def _build_reference_block() -> str:
        items: list[tuple[str, str]] = []

        # 1) 구조화 추출(문서별) 우선
        for ex in extractions:
            if not isinstance(ex, dict):
                continue
            title = (ex.get("title") or "").strip()
            url = (ex.get("url") or "").strip()
            if title:
                items.append((title, url))

        # 2) balanced_documents에서 제목/URL 보강 (RAG는 url이 internal일 수 있음)
        for d in balanced_docs:
            if not isinstance(d, dict):
                continue
            title = (d.get("title") or "").strip()
            url = (d.get("url") or "").strip()
            if title:
                items.append((title, url))

        # dedupe
        seen: set[str] = set()
        lines: list[str] = []
        for title, url in items:
            key = (url or "") + "||" + title
            if key in seen:
                continue
            seen.add(key)
            if url and url.lower() not in ("internal", "null", "none"):
                lines.append(f"- {title} — {url}")
            else:
                lines.append(f"- {title} (internal)")

        if not lines:
            return ""
        return "\n".join(lines)

    ref_block = _build_reference_block()
    if ref_block:
        if "## REFERENCE" in draft:
            # 이미 REFERENCE 섹션이 있으면 뒤에 자동 수집 목록을 추가
            draft = draft.replace(
                "## REFERENCE",
                "## REFERENCE\n\n### (자동 수집: 메타데이터 기반)\n" + ref_block + "\n\n",
                1,
            )
        else:
            # REFERENCE 섹션 자체가 없으면 생성
            draft = draft.rstrip() + "\n\n---\n\n## REFERENCE\n\n" + ref_block + "\n"

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
