"""
ingest.py — ChromaDB에 내부 문서 적재 (txt / md / pdf)
사용법:
  python ingest.py --dir ./docs
  python ingest.py --dir ./pdfs --chunk-size 1200 --overlap 150
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
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

## PDF 텍스트 추출 보조 클래스 및 함수 (pdfminer.six 활용-> 텍스트 위치 기반으로 2단 레이아웃 감지 및 라인 병합)

@dataclass
class PdfTextSpan:
    x: float
    y: float
    text: str


def _merge_spans_into_lines(spans: list[PdfTextSpan], y_tolerance: float = 3.0) -> list[str]:
    """좌표가 비슷한 span을 한 줄로 합친다."""
    if not spans:
        return []

    ordered = sorted(spans, key=lambda item: (-item.y, item.x))
    lines: list[tuple[float, list[PdfTextSpan]]] = []

    for span in ordered:
        if not lines or abs(lines[-1][0] - span.y) > y_tolerance:
            lines.append((span.y, [span]))
            continue
        lines[-1][1].append(span)

    merged: list[str] = []
    for _y, line_spans in lines:
        line_spans.sort(key=lambda item: item.x)
        parts: list[str] = []
        for span in line_spans:
            piece = span.text.strip()
            if not piece:
                continue
            if not parts:
                parts.append(piece)
                continue
            prev = parts[-1]
            if prev.endswith("-"):
                parts[-1] = prev[:-1] + piece
            elif re.match(r"^[,.;:)\]%]$", piece):
                parts[-1] = prev + piece
            else:
                parts.append(piece)
        line = " ".join(parts).strip()
        if line:
            merged.append(line)
    return merged


def _page_is_probably_two_column(spans: list[PdfTextSpan], page_width: float) -> bool:
    """본문 span 분포를 보고 2단 레이아웃 여부를 추정한다."""
    if len(spans) < 20 or page_width <= 0:
        return False

    center = page_width / 2.0
    gutter = page_width * 0.08
    left = [span for span in spans if span.x < center - gutter]
    right = [span for span in spans if span.x > center + gutter]
    middle = [span for span in spans if center - gutter <= span.x <= center + gutter]

    if len(left) < 8 or len(right) < 8:
        return False

    return len(middle) <= max(6, len(spans) // 10)


def _read_pdf_pages_with_pdfminer(path: Path) -> list[tuple[int, str]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages: list[tuple[int, str]] = []
    for i, page_layout in enumerate(extract_pages(str(path)), start=1):
        spans: list[PdfTextSpan] = []
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            text = normalize_text(element.get_text() or "")
            if not text:
                continue
            spans.append(
                PdfTextSpan(
                    x=float(element.x0),
                    y=float(element.y0),
                    text=text,
                )
            )

        if not spans:
            pages.append((i, ""))
            continue

        page_width = max(1.0, float(page_layout.x1) - float(page_layout.x0))
        if not _page_is_probably_two_column(spans, page_width):
            ordered = sorted(spans, key=lambda item: (-item.y, item.x))
            raw = "\n".join(span.text for span in ordered if span.text.strip())
            pages.append((i, normalize_text(raw)))
            continue

        center = page_width / 2.0
        gutter = page_width * 0.08
        left_spans = [span for span in spans if span.x < center - gutter]
        right_spans = [span for span in spans if span.x > center + gutter]
        middle_spans = [span for span in spans if center - gutter <= span.x <= center + gutter]

        blocks: list[str] = []
        if middle_spans:
            blocks.append("\n".join(_merge_spans_into_lines(middle_spans)))
        blocks.append("\n".join(_merge_spans_into_lines(left_spans)))
        blocks.append("\n".join(_merge_spans_into_lines(right_spans)))
        pages.append((i, normalize_text("\n\n".join(block for block in blocks if block.strip()))))
    return pages


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """pdfminer.six로 페이지 텍스트를 읽는다."""
    return _read_pdf_pages_with_pdfminer(path)


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

    base, doc_id = build_chroma_document_metadata(
        filepath, use_llm_fallback=use_llm_fallback
    )
    meta = {
        **base,
        "source": "internal",
        "file_type": source_label,
    }
    # 재적재해도 동일 청크는 동일 ID를 갖도록 안정적 ID 사용
    ids = [f"{doc_id}::c{i}" for i in range(len(chunks))]
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
    base_meta, doc_id = build_chroma_document_metadata(
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
            all_ids.append(f"{doc_id}::p{page_no}::c{local_i}")
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