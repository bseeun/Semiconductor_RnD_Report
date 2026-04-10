"""
main.py — 에이전트 실행 진입점
"""
import os
import textwrap
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


def _render_markdown_pdf(report_markdown: str, output_path: str) -> bool:
    """
    Markdown -> HTML -> PDF 렌더링.
    성공하면 True, 실패하면 False를 반환한다.
    """
    try:
        import markdown as md
        from weasyprint import HTML
    except Exception:
        return False

    html_body = md.markdown(
        report_markdown,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    html = f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <style>
    @page {{
      size: A4;
      margin: 18mm 14mm;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
      line-height: 1.6;
      color: #1f2937;
      font-size: 11pt;
      word-break: break-word;
    }}
    h1, h2, h3 {{
      color: #111827;
      margin-top: 18px;
      margin-bottom: 8px;
      line-height: 1.35;
    }}
    h1 {{ font-size: 20pt; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
    h2 {{ font-size: 15pt; border-left: 4px solid #2563eb; padding-left: 8px; }}
    h3 {{ font-size: 12pt; }}
    p {{ margin: 7px 0; }}
    ul, ol {{ margin: 6px 0 8px 20px; }}
    li {{ margin: 2px 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #f3f4f6;
      border-radius: 4px;
      padding: 1px 4px;
      font-size: 9.8pt;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 6px 8px;
      vertical-align: top;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 700;
    }}
    blockquote {{
      margin: 10px 0;
      padding: 6px 10px;
      border-left: 4px solid #93c5fd;
      background: #eff6ff;
      color: #1e3a8a;
    }}
    hr {{
      border: 0;
      border-top: 1px solid #e5e7eb;
      margin: 14px 0;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    try:
        HTML(string=html).write_pdf(output_path)
        return True
    except Exception:
        return False


def _save_report_pdf(report: str, output_path: str) -> None:
    """텍스트 보고서를 PDF로 저장."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    left, top, bottom = 40, height - 40, 40
    line_height = 14

    # 한글 표시용 CID 폰트 시도 (환경에 따라 실패할 수 있어 fallback 포함)
    font_name = "Helvetica"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        font_name = "HYSMyeongJo-Medium"
    except Exception:
        font_name = "Helvetica"

    y = top
    c.setFont(font_name, 10)
    max_chars = 95 if font_name == "Helvetica" else 70

    for raw_line in report.splitlines():
        wrapped = textwrap.wrap(raw_line, width=max_chars) or [""]
        for line in wrapped:
            if y <= bottom:
                c.showPage()
                c.setFont(font_name, 10)
                y = top
            c.drawString(left, y, line)
            y -= line_height

    c.save()


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

    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes"):
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()
        except Exception:
            pass

    report = final_state.get("final_report", "보고서 생성 실패")

    # 파일 저장 (옵션)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if output_path.lower().endswith(".pdf"):
            # 1) 예쁜 렌더링: Markdown -> HTML -> PDF
            # 2) 실패 시 텍스트 PDF fallback
            ok = _render_markdown_pdf(report, output_path)
            if not ok:
                _save_report_pdf(report, output_path)
        else:
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
        output_path=f"reports/semiconductor_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    )
