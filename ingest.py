"""
ingest.py — ChromaDB에 내부 문서 적재 (txt / md / pdf)
사용법:
  python ingest.py --dir ./docs
  python ingest.py --dir ./pdfs --chunk-size 1200 --overlap 150
"""
from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from config import get_chroma_client, get_collection
from document_metadata import build_chroma_document_metadata


def normalize_text(text: str) -> str:
    """PDF 추출물 등 연속 공백·빈 줄 정리"""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """문자 기준 고정 윈도 청킹 (overlap은 이전 청크와 겹치는 글자 수)"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    step = chunk_size - overlap
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step
    return chunks


def read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """페이지별 (1-based page_no, text). 추출 실패 시 빈 문자열."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "PDF 적재에는 pypdf가 필요합니다. pip install pypdf"
        ) from e

    reader = PdfReader(str(path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        out.append((i, normalize_text(raw)))
    return out


def ingest_text_document(
    collection,
    filepath: Path,
    *,
    chunk_size: int,
    overlap: int,
    source_label: str,
    use_llm_fallback: bool,
) -> int:
    """단일 텍스트 파일 → 청크 후 add. 반환: 청크 개수"""
    text = read_plain_text(filepath)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        print(f"  (건너뜀) 내용 없음: {filepath}")
        return 0

    base, _doc_id = build_chroma_document_metadata(
        filepath, use_llm_fallback=use_llm_fallback
    )
    meta = {
        **base,
        "source": "internal",
        "file_type": source_label,
    }
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {**meta, "chunk_index": i, "page": "", "page_chunk_index": ""}
        for i in range(len(chunks))
    ]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"  ✓ {filepath} → {len(chunks)}개 청크 적재")
    return len(chunks)


def ingest_pdf_document(
    collection,
    filepath: Path,
    *,
    chunk_size: int,
    overlap: int,
    use_llm_fallback: bool,
) -> int:
    """PDF 페이지별 텍스트 추출 → 페이지 단위로 청킹 후 add. 반환: 청크 개수"""
    pages = read_pdf_pages(filepath)
    base_meta, _doc_id = build_chroma_document_metadata(
        filepath, use_llm_fallback=use_llm_fallback
    )
    base = {
        **base_meta,
        "source": "internal",
        "file_type": "pdf",
    }

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    global_idx = 0

    for page_no, page_text in pages:
        if not page_text:
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for local_i, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_ids.append(str(uuid.uuid4()))
            all_meta.append(
                {
                    **base,
                    "page": str(page_no),
                    "chunk_index": global_idx,
                    "page_chunk_index": local_i,
                }
            )
            global_idx += 1

    if not all_docs:
        print(f"  (건너뜀) 추출 텍스트 없음: {filepath}")
        return 0

    collection.add(documents=all_docs, ids=all_ids, metadatas=all_meta)
    print(f"  ✓ {filepath} → {len(all_docs)}개 청크 적재 (PDF)")
    return len(all_docs)


def ingest_path(
    collection,
    filepath: Path,
    *,
    chunk_size: int,
    overlap: int,
    use_llm_fallback: bool,
) -> int:
    suf = filepath.suffix.lower()
    if suf in (".txt", ".md"):
        return ingest_text_document(
            collection,
            filepath,
            chunk_size=chunk_size,
            overlap=overlap,
            source_label=suf.lstrip("."),
            use_llm_fallback=use_llm_fallback,
        )
    if suf == ".pdf":
        return ingest_pdf_document(
            collection,
            filepath,
            chunk_size=chunk_size,
            overlap=overlap,
            use_llm_fallback=use_llm_fallback,
        )
    return 0


def collect_files(doc_dir: Path) -> list[Path]:
    exts = ("*.txt", "*.md", "*.pdf")
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in exts:
        for p in doc_dir.glob(f"**/{pattern}"):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def ingest_directory(
    doc_dir: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    use_llm_fallback: bool = True,
) -> None:
    """
    디렉터리 내 .txt / .md / .pdf 일괄 적재.
    파일명 규칙: {company}_{technology}_{source_type}_{YYYYMM}.pdf
    """
    doc_path = Path(doc_dir)
    if not doc_path.is_dir():
        raise FileNotFoundError(f"디렉터리가 없습니다: {doc_path.resolve()}")

    client = get_chroma_client()
    collection = get_collection(client)
    files = collect_files(doc_path)

    print(f"\nChromaDB 문서 적재 시작: {doc_path.resolve()}")
    print(f"  파일 {len(files)}개 | chunk_size={chunk_size} overlap={overlap}")
    total_chunks = 0
    for fp in files:
        total_chunks += ingest_path(
            collection,
            fp,
            chunk_size=chunk_size,
            overlap=overlap,
            use_llm_fallback=use_llm_fallback,
        )

    print(f"\n적재 완료. 이번 실행 청크: {total_chunks}개 | 컬렉션 총 문서 수: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChromaDB 문서 적재 (txt / md / pdf, 청킹 후 임베딩)"
    )
    parser.add_argument(
        "--dir",
        default="./docs",
        help="문서가 있는 폴더 (하위 폴더 포함 재귀 탐색)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="청크 최대 문자 수 (기본 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="청크 간 겹침 문자 수 (기본 120)",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="파일명 파싱 실패 시 LLM 보완 호출 안 함 (비용·속도 절약)",
    )
    args = parser.parse_args()
    ingest_directory(
        args.dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_llm_fallback=not args.no_llm_fallback,
    )
