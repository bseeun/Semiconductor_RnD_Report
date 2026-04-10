"""
검색·균형 단계 공통: 기업명 정규화, 분석 기간(scope), MMR용 유사도
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, List, Optional

from state import RetrievedDocument

# config에서 가져오면 순환 참조 가능 → 환경변수 또는 기본값
MIN_CREDIBILITY_FOR_BALANCE = float(os.getenv("MIN_CREDIBILITY_FOR_BALANCE", "0.38"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.65"))
MMR_POOL_SIZE = int(os.getenv("MMR_POOL_SIZE", "60"))


def compact_name(s: str) -> str:
    return re.sub(r"[\s\-_]", "", (s or "").lower())


# 별칭(compact) → target_companies에 쓰는 표기
_COMPANY_ALIAS_COMPACT: dict[str, str] = {
    "skhynix": "SK Hynix",
    "sk하이닉스": "SK Hynix",
    "samsung": "Samsung",
    "삼성": "Samsung",
    "micron": "Micron",
    "마이크론": "Micron",
    "broadcom": "Broadcom",
    "브로드컴": "Broadcom",
    "tsmc": "TSMC",
    "jedec": "JEDEC",
    "ieee": "IEEE",
}


def canonical_company(name: str, targets: List[str]) -> str:
    """문서 메타/본문의 기업명을 target_companies 표기로 맞춤."""
    if not name or not targets:
        return name or ""
    raw = name.strip()
    cpt = compact_name(raw)
    target_compact = {compact_name(t): t for t in targets}

    if cpt in target_compact:
        return target_compact[cpt]
    if cpt in _COMPANY_ALIAS_COMPACT:
        t = _COMPANY_ALIAS_COMPACT[cpt]
        if t in targets:
            return t
    for tc, orig in target_compact.items():
        if tc and (tc in cpt or cpt in tc):
            return orig
    return raw


def parse_scope_dates(scope: dict[str, Any] | None) -> tuple[Optional[datetime], Optional[datetime]]:
    """scope.date_from / date_to → datetime (일 단위, 시작 00:00 / 끝 포함)."""
    if not scope:
        return None, None
    df = scope.get("date_from")
    dt_to = scope.get("date_to")

    def _parse(d: Any) -> Optional[datetime]:
        if not d or not isinstance(d, str):
            return None
        d = d.strip()[:10]
        try:
            return datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(d + "-01", "%Y-%m-%d")
            except ValueError:
                return None

    return _parse(df), _parse(dt_to)


def parse_doc_published_datetime(pub: str | None) -> Optional[datetime]:
    """published_at: YYYY-MM 또는 YYYY-MM-DD."""
    if not pub or not isinstance(pub, str):
        return None
    p = pub.strip()[:10]
    if len(p) >= 7 and p[4:5] == "-":
        if len(p) == 7:
            p = p + "-01"
        try:
            return datetime.strptime(p[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return None


def doc_within_scope(
    doc: RetrievedDocument,
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> bool:
    """날짜 없으면 기간 필터 통과(보수적으로 유지)."""
    if date_from is None and date_to is None:
        return True
    d = parse_doc_published_datetime(doc.get("published_at"))
    if d is None:
        return True
    if date_from is not None and d < date_from:
        return False
    if date_to is not None and d > date_to:
        return False
    return True


def word_jaccard_similarity(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-zA-Z0-9가-힣]{2,}", (a or "").lower()))
    sb = set(re.findall(r"[a-zA-Z0-9가-힣]{2,}", (b or "").lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mmr_select(
    docs: List[RetrievedDocument],
    k: int,
    lambda_param: float = MMR_LAMBDA,
) -> List[RetrievedDocument]:
    """이미 relevance proxy(final_score 사전 계산 가정) 기준 정렬된 풀에서 MMR로 k개 선택."""
    if not docs or k <= 0:
        return []
    pool = list(docs)
    selected: List[RetrievedDocument] = []
    candidates = pool[:]

    def rel_score(d: RetrievedDocument) -> float:
        return float(d.get("final_score", 0.5))

    while len(selected) < k and candidates:
        best_i = -1
        best_mmr = -1.0
        for i, c in enumerate(candidates):
            rel = rel_score(c)
            if not selected:
                mmr = rel
            else:
                texts = [c.get("content", "") + c.get("title", "")]
                max_sim = max(
                    word_jaccard_similarity(
                        texts[0],
                        s.get("content", "") + s.get("title", ""),
                    )
                    for s in selected
                )
                mmr = lambda_param * rel - (1.0 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_i = i
        chosen = candidates.pop(best_i)
        selected.append(chosen)
    return selected
