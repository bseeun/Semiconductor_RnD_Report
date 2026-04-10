#!/usr/bin/env python3
"""
FAISS 기반 내부 코퍼스 검색 평가 (Hit@K, MRR).

기본 동작:
1) ChromaDB에서 documents + metadatas를 읽는다.
2) SentenceTransformer 임베딩으로 FAISS 인덱스를 만든다.
3) 골든 쿼리(약라벨 또는 사용자 qrels)로 Hit@K/MRR 계산.

사용:
  python -m eval.faiss_retrieval_eval --k 5

선택:
  --qrels ./eval/qrels.jsonl
  qrels 포맷(JSONL):
    {"query":"JEDEC HBM4 memory standard", "needles":["JEDEC","HBM4"]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

# 프로젝트 루트를 path에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import EMBEDDING_MODEL_NAME, get_chroma_client, get_collection  # noqa: E402


# qrels가 없을 때 쓰는 기본 약라벨
DEFAULT_GOLDEN: list[dict[str, Any]] = [
    {"query": "JEDEC HBM4 memory standard", "needles": ["JEDEC", "HBM4"]},
    {"query": "Samsung CXL memory expansion whitepaper", "needles": ["Samsung", "CXL"]},
    {"query": "Micron CXL platform", "needles": ["Micron", "CXL"]},
    {"query": "SK Hynix PIM processing in memory", "needles": ["SKHynix", "PIM"]},
]


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _load_qrels(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_GOLDEN
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if isinstance(row, dict) and row.get("query") and row.get("needles"):
                out.append(row)
    return out or DEFAULT_GOLDEN


def _doc_matches(doc: str, meta: dict[str, Any], needles: list[str]) -> bool:
    blob = (doc or "").lower()
    meta_blob = " ".join(str(v) for v in (meta or {}).values()).lower()
    for n in needles:
        t = str(n).lower()
        if t in blob or t in meta_blob:
            return True
    return False


def run_eval(k: int, qrels_path: str | None) -> None:
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise RuntimeError("faiss-cpu 미설치: pip install faiss-cpu") from e

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers 미설치: pip install sentence-transformers"
        ) from e

    client = get_chroma_client()
    collection = get_collection(client)
    data = collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    if not docs:
        raise RuntimeError("ChromaDB 문서가 비어 있습니다. 먼저 ingest를 수행하세요.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    doc_vec = model.encode(docs, convert_to_numpy=True, show_progress_bar=True).astype(
        np.float32
    )
    doc_vec = _normalize(doc_vec)

    dim = int(doc_vec.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(doc_vec)

    qrels = _load_qrels(qrels_path)
    hits: list[int] = []
    rranks: list[float] = []

    for item in qrels:
        query = str(item["query"])
        needles = [str(x) for x in item.get("needles", [])]
        qvec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        qvec = _normalize(qvec)
        _scores, idx = index.search(qvec, k)
        rank = None
        for i, didx in enumerate(idx[0]):
            if didx < 0 or didx >= len(docs):
                continue
            if _doc_matches(docs[didx], metas[didx] if didx < len(metas) else {}, needles):
                rank = i + 1
                break

        hit = 1 if rank is not None else 0
        rr = (1.0 / rank) if rank is not None else 0.0
        hits.append(hit)
        rranks.append(rr)
        print(
            f"[Eval] q='{query[:60]}' needles={needles} "
            f"-> hit@{k}={hit} rank={rank if rank is not None else '-'} rr={rr:.3f}"
        )

    n = len(qrels)
    hitk = (sum(hits) / n) if n else 0.0
    mrr = (sum(rranks) / n) if n else 0.0
    print(f"\nFAISS Eval | queries={n} | Hit@{k}={hitk:.3f} | MRR={mrr:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--qrels", type=str, default=None)
    args = ap.parse_args()
    run_eval(k=args.k, qrels_path=args.qrels)


if __name__ == "__main__":
    main()

