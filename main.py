"""
main.py — 에이전트 실행 진입점
"""
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv

# LangSmith 등 .env는 LangChain/LangGraph import 전에 로드해야 추적이 안정적으로 켜짐
load_dotenv()

from langsmith_setup import configure_langsmith

configure_langsmith()

from graph import app
from state import SupervisorState
from config import DEFAULT_TECHNOLOGIES, DEFAULT_COMPANIES


def run_analysis(
    user_query: str,
    target_technologies: list[str] | None = None,
    target_companies: list[str] | None = None,
    output_path: str | None = None,
) -> str:
    """
    반도체 R&D 경쟁사 분석 실행

    Args:
        user_query: 사용자 분석 요청 (예: "HBM4 관련 삼성·마이크론 최신 R&D 동향 분석")
        target_technologies: 분석 기술 목록 (기본: HBM4, PIM, CXL)
        target_companies: 분석 기업 목록 (기본: Samsung, Micron 등)
        output_path: 최종 보고서 저장 경로 (None이면 저장 안 함)

    Returns:
        최종 보고서 문자열
    """
    techs = target_technologies or DEFAULT_TECHNOLOGIES
    companies = target_companies or DEFAULT_COMPANIES

    # 초기 상태 구성
    initial_state: SupervisorState = {
        "user_query": user_query,
        "target_technologies": techs,
        "target_companies": companies,
        "scope": {
            "date_from": "2024-01-01",
            "date_to": datetime.now().strftime("%Y-%m-%d"),
            "language": ["ko", "en"],
        },
        "search_queries": [],
        "retrieved_documents": [],
        "balanced_documents": [],
        "aggregated_signals": [],
        "analysis_result": [],
        "document_extractions": [],
        "draft_history": [],
        "current_draft": "",
        "validation": {
            "passed": False,
            "feedback": "",
            "validation_retry_count": 0,
        },
        "final_report": "",
        "next_action": "plan",
        "flow_retry_count": 0,
        "error": None,
        "error_history": [],
    }

    # 스레드 ID (체크포인터용)
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        # LangSmith: https://smith.langchain.com 에서 실행 단위로 묶여 보임
        "run_name": f"semiconductor_rnd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "tags": ["semiconductor_rnd", "langgraph"],
        "metadata": {"thread_id": thread_id},
    }

    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes"):
        proj = os.getenv("LANGCHAIN_PROJECT", "semiconductor-rnd")
        print(
            f"[LangSmith] 추적 활성화 → 프로젝트: {proj}\n"
            f"            대시보드: https://smith.langchain.com"
        )

    print(f"\n{'='*60}")
    print(f"반도체 R&D 기술 전략 분석 에이전트 시작")
    print(f"분석 기술: {techs}")
    print(f"분석 기업: {companies}")
    print(f"{'='*60}\n")

    # 그래프 실행
    final_state = app.invoke(initial_state, config=config)

    report = final_state.get("final_report", "보고서 생성 실패")

    # 파일 저장 (옵션)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[완료] 보고서 저장: {output_path}")
    else:
        print("\n" + "="*60)
        print(report[:2000])  # 콘솔에는 앞부분만 출력
        print("="*60)

    return report


if __name__ == "__main__":
    # 예시 실행
    run_analysis(
        user_query="HBM4, PIM, CXL 기술에 대한 삼성전자·마이크론·SK하이닉스의 최신 R&D 동향을 분석하고 전략 보고서를 작성해줘",
        output_path=f"reports/semiconductor_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
    )
