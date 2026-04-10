"""
graph.py — LangGraph 그래프 정의
Supervisor 기반 중앙 제어 구조
Parallel Retrieve (Web+RAG) → Balanced → TRL → Analyze → Draft → Validate → Format → Supervisor → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import SupervisorState
from agents import (
    supervisor_node,
    query_planning_node,
    parallel_retrieve_node,
    balanced_retrieval_node,
    trl_preparation_node,
    competitive_analyst_node,
    draft_generation_node,
    validation_node,
    formatting_node,
)


def _route_supervisor(state: SupervisorState) -> str:
    """Supervisor가 결정한 next_action을 노드 이름으로 라우팅"""
    action = state.get("next_action", "end")
    routing = {
        "plan": "query_planning",
        "retrieve": "parallel_retrieve",
        "balance": "balanced_retrieval",
        "prepare_trl": "trl_preparation",
        "analyze": "competitive_analyst",
        "draft": "draft_generation",
        "validate": "validation",
        "format": "formatting",
        "end": END,
    }
    return routing.get(action, END)


def _route_after_validation(state: SupervisorState) -> str:
    """검증 결과에 따라 재초안 생성 또는 포맷팅으로 분기"""
    validation = state.get("validation", {})
    if validation.get("passed", False):
        return "supervisor"  # supervisor → format
    retry = validation.get("validation_retry_count", 0)
    from config import MAX_VALIDATION_RETRY
    if retry >= MAX_VALIDATION_RETRY:
        return "supervisor"  # 강제 통과
    return "draft_generation"  # 재초안


def build_graph(checkpointer=None):
    """
    그래프 구성:

    [START] → supervisor → (route) → query_planning
                                    → parallel_retrieve (Web+RAG 스레드 병렬)
                                    → balanced_retrieval
                                    → trl_preparation
                                    → competitive_analyst
                                    → draft_generation
                                    → validation → (조건부) draft_generation | supervisor
                                    → formatting → supervisor → [END]
    """
    graph = StateGraph(SupervisorState)

    # ── 노드 등록
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("query_planning", query_planning_node)
    graph.add_node("parallel_retrieve", parallel_retrieve_node)
    graph.add_node("balanced_retrieval", balanced_retrieval_node)
    graph.add_node("trl_preparation", trl_preparation_node)
    graph.add_node("competitive_analyst", competitive_analyst_node)
    graph.add_node("draft_generation", draft_generation_node)
    graph.add_node("validation", validation_node)
    graph.add_node("formatting", formatting_node)

    # ── 엣지: START → supervisor
    graph.set_entry_point("supervisor")

    # ── Supervisor → 각 노드 (조건부 라우팅)
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "query_planning": "query_planning",
            "parallel_retrieve": "parallel_retrieve",
            "balanced_retrieval": "balanced_retrieval",
            "trl_preparation": "trl_preparation",
            "competitive_analyst": "competitive_analyst",
            "draft_generation": "draft_generation",
            "validation": "validation",
            "formatting": "formatting",
            END: END,
        },
    )

    # ── 각 노드 완료 후 supervisor로 복귀 (순차 흐름)
    graph.add_edge("query_planning", "supervisor")

    graph.add_edge("parallel_retrieve", "supervisor")

    graph.add_edge("balanced_retrieval", "supervisor")
    graph.add_edge("trl_preparation", "supervisor")
    graph.add_edge("competitive_analyst", "supervisor")

    # draft_generation → validation → 조건부 분기
    graph.add_edge("draft_generation", "validation")
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {
            "supervisor": "supervisor",
            "draft_generation": "draft_generation",
        },
    )

    graph.add_edge("formatting", "supervisor")

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# 그래프 싱글톤
app = build_graph()


if __name__ == "__main__":
    # 그래프 구조 출력 (디버깅용)
    print("노드 목록:", list(app.nodes.keys()) if hasattr(app, 'nodes') else "컴파일됨")
    print("그래프 빌드 성공 ✓")
