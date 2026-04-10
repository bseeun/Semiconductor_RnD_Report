"""
파일명 기반 문서 메타데이터 파싱 + (선택) LLM 보완
규칙: {company}_{technology}_{source_type}_{YYYYMM}.pdf|txt|md
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

VALID_COMPANIES = [
    "Samsung",
    "Micron",
    "SKHynix",
    "JEDEC",
    "IEEE",
    "Broadcom",
    "TSMC",
]
VALID_TECHS = ["HBM4", "HBM", "PIM", "CXL", "ALL"]
VALID_SOURCE_TYPES = [
    "standard",
    "whitepaper",
    "ir_report",
    "patent",
    "paper",
]

_CORP_COMPANIES = frozenset({"Samsung", "Micron", "SKHynix", "Broadcom", "TSMC"})


def stable_doc_id(path: Path) -> str:
    """동일 경로 → 동일 ID (재적재·중복 제거용)"""
    key = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def parse_filename(filepath: str, *, use_llm_fallback: bool = True) -> dict[str, Any]:
    stem = Path(filepath).stem
    parts = stem.split("_")

    meta: dict[str, Any] = {
        "company": None,
        "technology": None,
        "source_type": None,
        "published_at": None,
        "source_file": Path(filepath).name,
    }

    if len(parts) > 0 and parts[0] in VALID_COMPANIES:
        meta["company"] = parts[0]

    if len(parts) > 1 and parts[1] in VALID_TECHS:
        meta["technology"] = None if parts[1] == "ALL" else parts[1]

    if len(parts) > 2 and parts[2] in VALID_SOURCE_TYPES:
        meta["source_type"] = parts[2]

    if len(parts) > 3 and re.fullmatch(r"\d{6}", parts[3]):
        raw = parts[3]
        meta["published_at"] = f"{raw[:4]}-{raw[4:6]}"

    missing: list[str] = []
    for k, v in meta.items():
        if k == "source_file":
            continue
        if v is not None:
            continue
        if k == "technology" and len(parts) > 1 and parts[1] == "ALL":
            continue
        missing.append(k)

    if missing and use_llm_fallback:
        meta = _llm_fallback(filepath, meta, missing)

    return meta


def _pdf_snippet_pypdf(path: str, max_chars: int = 800) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(path)
        buf: list[str] = []
        n = 0
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            buf.append(t)
            n += len(t)
            if n >= max_chars:
                break
        return "".join(buf)[:max_chars]
    except Exception:
        return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        chunk = fence.group(1).strip()
        if chunk.startswith("{"):
            try:
                out = json.loads(chunk)
                return out if isinstance(out, dict) else None
            except json.JSONDecodeError:
                pass
    i = text.find("{")
    if i < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_fallback(filepath: str, meta: dict[str, Any], missing_fields: list[str]) -> dict[str, Any]:
    path = Path(filepath)
    snippet = ""
    if path.suffix.lower() == ".pdf":
        snippet = _pdf_snippet_pypdf(filepath, max_chars=800)
    elif path.is_file():
        try:
            snippet = path.read_text(encoding="utf-8", errors="ignore")[:800]
        except OSError:
            snippet = ""

    try:
        from config import get_fast_llm
        from langchain_core.messages import HumanMessage
    except Exception:
        return meta

    prompt = f"""아래 텍스트를 보고 누락된 메타데이터만 JSON으로 채우세요.
이미 알려진 값(그대로 두어도 됨): {json.dumps({k: meta[k] for k in meta if k != "source_file"}, ensure_ascii=False)}
반드시 채울 필드: {missing_fields}

허용 값:
- company: {VALID_COMPANIES}
- technology: {[t for t in VALID_TECHS if t != "ALL"]} (기술 전체이면 null)
- source_type: {VALID_SOURCE_TYPES}
- published_at: YYYY-MM 문자열 (모르면 null)

텍스트 일부:
{snippet[:800]}

응답은 JSON 객체 하나만. 모르는 키는 null."""

    try:
        llm = get_fast_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "") or ""
        result = _parse_json_object(content)
        if not result:
            return meta
        for field in missing_fields:
            val = result.get(field)
            if val is None or val == "":
                continue
            if field == "company" and val not in VALID_COMPANIES:
                continue
            if field == "technology" and val not in VALID_TECHS:
                continue
            if field == "source_type" and val not in VALID_SOURCE_TYPES:
                continue
            if field == "published_at":
                if not re.fullmatch(r"\d{4}-\d{2}", str(val)):
                    continue
            meta[field] = val
    except Exception:
        pass

    return meta


def infer_publisher(meta: dict[str, Any]) -> str:
    c = meta.get("company")
    if not c:
        return ""
    if c in ("IEEE", "JEDEC"):
        return c
    if c in VALID_COMPANIES:
        return c
    return ""


def infer_is_official_source(meta: dict[str, Any]) -> bool:
    st = meta.get("source_type") or ""
    co = meta.get("company") or ""
    if st == "standard":
        return True
    if co in ("JEDEC", "IEEE"):
        return True
    if st == "ir_report" and co in _CORP_COMPANIES:
        return True
    if st == "patent":
        return True
    if st == "paper" and co == "IEEE":
        return True
    return False


def trl_aggregation_key(meta: dict[str, Any]) -> str:
    """TRL Prep 등에서 출처·기술·유형별 버킷 키"""
    c = meta.get("company") or "*"
    t = meta.get("technology") or "*"
    s = meta.get("source_type") or "*"
    p = meta.get("published_at") or "*"
    return f"{c}|{t}|{s}|{p}"


def none_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_chroma_document_metadata(
    path: Path,
    *,
    use_llm_fallback: bool = True,
) -> tuple[dict[str, str], str]:
    """
    ChromaDB용 메타데이터(값은 모두 문자열)와 doc_id 반환.
    """
    raw_path = str(path.resolve())
    parsed = parse_filename(raw_path, use_llm_fallback=use_llm_fallback)
    doc_id = stable_doc_id(path)

    publisher = infer_publisher(parsed)
    official = infer_is_official_source(parsed)
    trl_key = trl_aggregation_key(parsed)

    out: dict[str, str] = {
        "company": none_to_str(parsed.get("company")),
        "technology": none_to_str(parsed.get("technology")),
        "source_type": none_to_str(parsed.get("source_type")),
        "published_at": none_to_str(parsed.get("published_at")),
        "source_file": none_to_str(parsed.get("source_file")) or path.name,
        "title": path.name,
        "url": raw_path,
        "publisher": publisher,
        "is_official_source": "true" if official else "false",
        "trl_aggregation_key": trl_key,
        "doc_id": doc_id,
    }
    return out, doc_id
