"""
LangSmith 연동 — LangChain / LangGraph 추적용 환경변수 정리.
langchain·langgraph import 전에 호출할 것 (main.py).
"""
from __future__ import annotations

import os


def configure_langsmith() -> None:
    """
    - LANGSMITH_API_KEY → LANGCHAIN_API_KEY 별칭 동기화
    - LANGCHAIN_ENDPOINT 기본값 (미설정 시 공식 엔드포인트)
    - LANGCHAIN_TRACING_V2 미설정 + 키 존재 시 자동 true (LANGSMITH_TRACING=false 로 끌 수 있음)
    """
    key = (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
    if key:
        os.environ.setdefault("LANGCHAIN_API_KEY", key)

    if not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ.setdefault(
            "LANGCHAIN_ENDPOINT",
            "https://api.smith.langchain.com",
        )

    tracing_explicit = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    if tracing_explicit in ("true", "1", "yes"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    elif tracing_explicit in ("false", "0", "no"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        off = os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("false", "0", "no")
        if key and not off:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        elif off:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

    os.environ.setdefault("LANGCHAIN_PROJECT", "semiconductor-rnd")
