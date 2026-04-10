#!/usr/bin/env python3
"""
qrels 작성을 위한 doc_id 추출 도우미.

빠르게 결과를 보기 위해 FAISS 전체 재인덱싱 대신 Chroma query를 직접 사용한다.

사용:
  python -m eval.export_doc_ids --query "JEDEC HBM4 memory standard" --topk 10
  python -m eval.export_doc_ids --queries-file eval/qrels.sample.jsonl --topk 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# 프로젝트 루트를 path에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_chroma_client, get_collection  # noqa: E402


def _load_queries(path: str | None, single_query: str | None) -> list[str]:
    if single_query:
        return [single_query]
    if not path:
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if isinstance(row, dict) and row.get("query"):
                out.append(str(row["query"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None, help="단일 검색 쿼리")
    ap.add_argument("--queries-file", type=str, default=None, help="JSONL 파일(query 키 사용)")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    queries = _load_queries(args.queries_file, args.query)
    if not queries:
        raise SystemExit("query 또는 queries-file 중 하나를 지정하세요.")

    client = get_chroma_client()
    collection = get_collection(client)

    for q in queries:
        res = collection.query(
            query_texts=[q],
            n_results=args.topk,
            include=["metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0] if res.get("ids") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dists = res.get("distances", [[]])[0] if res.get("distances") else []
        print(f"\n=== query: {q} ===")
        for rank, doc_id in enumerate(ids, start=1):
            meta = metas[rank - 1] if (rank - 1) < len(metas) and isinstance(metas[rank - 1], dict) else {}
            title = str(meta.get("title", ""))[:90]
            url = str(meta.get("url", ""))
            dist = float(dists[rank - 1]) if (rank - 1) < len(dists) else float("nan")
            print(
                f"{rank:>2}. distance={dist:.4f} "
                f"doc_id={doc_id} title={title} url={url}"
            )


if __name__ == "__main__":
    main()

