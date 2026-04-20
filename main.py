"""Financial Analyst Multi-Agent System.

Orchestrates three specialized agents (Researcher, Quantitative, Analyst)
plus a Gemini-based evaluator using LangGraph to produce comprehensive
financial analysis reports with quality assurance.

Usage:
    python main.py --company "Apple"
    python main.py --company "Tesla" --max-revisions 2
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from graph.workflow import build_graph
from memory.store import load_memory, save_memory
from tools.mcp_tools import create_mcp_client, get_firecrawl_tools, get_fmp_tools
from tools.tavily_tools import get_tavily_search_tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_analysis(company: str, max_revisions: int = 3) -> dict:
    """Run the full financial analysis pipeline for a company.

    Returns the full final state dict (report, evaluation, errors, etc.).
    """
    logger.info("Starting analysis for: %s", company)

    # Load persistent memory from prior runs
    memory_context = load_memory(company)
    if memory_context:
        logger.info("Loaded prior analysis context for %s", company)

    # Initialize MCP client and load tools
    client = create_mcp_client()
    tavily_tool = get_tavily_search_tool()

    try:
        firecrawl_tools = await get_firecrawl_tools(client)
        logger.info("Loaded %d Firecrawl tools", len(firecrawl_tools))
    except Exception as e:
        logger.warning("Failed to load Firecrawl tools: %s. Continuing without.", e)
        firecrawl_tools = []

    try:
        fmp_tools = await get_fmp_tools(client)
        logger.info("Loaded %d FMP tools", len(fmp_tools))
    except Exception as e:
        logger.warning("Failed to load FMP tools: %s. Continuing without.", e)
        fmp_tools = []

    researcher_tools = [tavily_tool] + firecrawl_tools
    quantitative_tools = fmp_tools

    if not quantitative_tools:
        logger.warning(
            "No quantitative tools available. "
            "The quantitative agent will have limited functionality."
        )

    # Build and run the graph
    app = build_graph(researcher_tools, quantitative_tools)

    initial_state = {
        "company_name": company,
        "messages": [],
        "financial_data": [],
        "news_articles": [],
        "analysis_report": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "needs_revision": False,
        "revision_feedback": "",
        "memory_context": memory_context,
        "evaluation": {},
        "errors": [],
    }

    logger.info("Executing analysis graph...")
    result = await app.ainvoke(initial_state)

    # Save to persistent memory for future runs
    save_memory(
        company=company,
        report=result.get("analysis_report", ""),
        financial_data=result.get("financial_data", []),
        evaluation=result.get("evaluation"),
    )

    if result.get("errors"):
        logger.warning("Analysis completed with %d errors", len(result["errors"]))

    return result


def _build_report_text(result: dict) -> str:
    """Build the full report as a markdown string."""
    lines = []
    lines.append("# Financial Analysis Report\n")
    lines.append(result.get("analysis_report", "No report generated."))

    evaluation = result.get("evaluation", {})
    if evaluation and "overall_score" in evaluation:
        lines.append("\n---\n")
        lines.append("## Quality Evaluation (Gemini LLM-as-Judge)\n")
        lines.append(f"**Overall Score: {evaluation['overall_score']}/5 — Verdict: {evaluation.get('verdict', 'N/A')}**\n")
        lines.append(f"| Dimension | Score | Reasoning |")
        lines.append(f"|-----------|-------|-----------|")
        lines.append(f"| Grounding | {evaluation.get('grounding_score', 'N/A')}/5 | {evaluation.get('grounding_reasoning', '')} |")
        lines.append(f"| Completeness | {evaluation.get('completeness_score', 'N/A')}/5 | {evaluation.get('completeness_reasoning', '')} |")
        lines.append(f"| Consistency | {evaluation.get('consistency_score', 'N/A')}/5 | {evaluation.get('consistency_reasoning', '')} |")
        lines.append(f"| Actionability | {evaluation.get('actionability_score', 'N/A')}/5 | {evaluation.get('actionability_reasoning', '')} |")
        issues = evaluation.get("issues", [])
        if issues:
            lines.append(f"\n**Issues ({len(issues)}):**")
            for issue in issues:
                lines.append(f"- {issue}")

    return "\n".join(lines)


def _save_report(result: dict, company: str) -> str:
    """Save the report to reports/ directory. Returns the file path."""
    from datetime import datetime, timezone
    from pathlib import Path

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = company.strip().lower().replace(" ", "_")
    filepath = reports_dir / f"{slug}_{date_str}.md"

    report_text = _build_report_text(result)
    filepath.write_text(report_text)

    return str(filepath)


def _print_report(result: dict) -> None:
    """Pretty-print the analysis results."""
    print("\n" + "=" * 80)
    print(_build_report_text(result))
    print("=" * 80)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multi-Agent Financial Analyst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--company",
        required=True,
        help="Company name or ticker to analyze (e.g., 'Apple', 'TSLA')",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum number of revision cycles (default: 3)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_analysis(args.company, args.max_revisions))
    _print_report(result)
    filepath = _save_report(result, args.company)
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
