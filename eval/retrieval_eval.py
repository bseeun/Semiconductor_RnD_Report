#!/usr/bin/env python3
"""
소규모 내부 코퍼스 검색 평가 (Hit@K, MRR).
골든 쿼리는 문서 제목/메타 키워드 매칭으로 관련 청크가 상위 K 안에 오는지 본다.

사용:
  cd 프로젝트 루트 && python -m eval.retrieval_eval
"""
from __future__ import annotations

import os
import sys

# 프로젝트 루트를 path에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_chroma_client, get_collection  # noqa: E402


# (query, 반드시 메타데이터나 본문에 포함되면 hit로 보는 부분문자열)
GOLDEN = [
    ("JEDEC HBM4 memory standard", "JEDEC"),
    ("Samsung CXL memory expansion whitepaper", "Samsung"),
    ("Micron CXL platform", "Micron"),
    ("SK Hynix PIM processing in memory", "SKHynix"),
]


def _doc_matches(doc: str, meta: dict, needle: str) -> bool:
    n = needle.lower()
    blob = (doc or "").lower()
    for v in meta.values():
        if v and n in str(v).lower():
            return True
    return n in blob


def hit_at_k_and_mrr(k: int = 5) -> None:
    client = get_chroma_client()
    collection = get_collection(client)
    hits = []
    rranks: list[float] = []

    for query, needle in GOLDEN:
        res = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        rank = None
        for i, d in enumerate(docs):
            m = metas[i] if i < len(metas) else {}
            if _doc_matches(d, m if isinstance(m, dict) else {}, needle):
                rank = i + 1
                break
        hits.append(1 if rank is not None else 0)
        if rank is not None:
            rranks.append(1.0 / rank)
        else:
            rranks.append(0.0)

    n = len(GOLDEN)
    hitk = sum(hits) / n if n else 0.0
    mrr = sum(rranks) / n if n else 0.0
    print(f"Golden 쿼리 {n}개 | Hit@{k} = {hitk:.3f} | MRR = {mrr:.3f}")


if __name__ == "__main__":
    hit_at_k_and_mrr(5)
