"""
ingest.py — ChromaDB에 내부 문서 적재 스크립트
사용법: python ingest.py --dir ./docs
"""
import os
import argparse
import uuid
from pathlib import Path

from config import get_chroma_client, get_collection


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """단순 고정 크기 청킹 (실제 운영 시 semantic chunking 권장)"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_file(collection, filepath: str, metadata: dict):
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"  ✓ {filepath} → {len(chunks)}개 청크 적재")


def ingest_directory(doc_dir: str):
    """
    디렉터리 내 .txt / .md 파일을 일괄 적재
    파일명 규칙: {company}_{technology}_{YYYYMM}.txt
    예: Samsung_HBM4_202501.txt
    """
    client = get_chroma_client()
    collection = get_collection(client)

    doc_path = Path(doc_dir)
    files = list(doc_path.glob("**/*.txt")) + list(doc_path.glob("**/*.md"))

    print(f"\nChromaDB 문서 적재 시작: {len(files)}개 파일")
    for fp in files:
        parts = fp.stem.split("_")
        meta = {
            "company": parts[0] if len(parts) > 0 else "unknown",
            "technology": parts[1] if len(parts) > 1 else "unknown",
            "published_at": parts[2] if len(parts) > 2 else "unknown",
            "source": "internal",
            "title": fp.name,
            "url": str(fp),
        }
        ingest_file(collection, str(fp), meta)

    print(f"\n적재 완료. 총 문서 수: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB 문서 적재")
    parser.add_argument("--dir", default="./docs", help="문서 디렉터리 경로")
    args = parser.parse_args()
    ingest_directory(args.dir)
