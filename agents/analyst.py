import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.state import FinancialAnalystState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior investment analyst tasked with producing a professional
financial analysis memo. You will receive:
1. News articles and qualitative research about a company
2. Quantitative financial data (statements, ratios, market data)
3. Any errors that occurred during data collection
4. Historical context from prior analyses (if available)

Your job:
- Synthesize all available data into a coherent, professional analysis memo
- Evaluate the company's financial health, growth prospects, and risks
- Check for data consistency (e.g., do the news align with the financial trends?)
- If historical context is available, note any significant changes or trends
- Account for any data collection errors — if tools failed, acknowledge the gaps

Write the memo in this format:
## Executive Summary
(2-3 sentences)

## Financial Overview
(Key metrics and trends)

## News & Market Sentiment
(Recent developments and their implications)

## Risk Assessment
(Key risks identified)

## Conclusion
(Investment thesis or recommendation direction)
"""


class AnalysisOutput(BaseModel):
    """Structured output from the analyst agent."""

    report: str = Field(description="The full analysis memo in markdown format")


async def analyst_node(state: FinancialAnalystState) -> dict:
    """Analyst agent: synthesizes data into a professional report."""
    company = state["company_name"]
    news = state.get("news_articles", [])
    financials = state.get("financial_data", [])
    errors = state.get("errors", [])
    memory_context = state.get("memory_context", "")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=8192,
    )

    data_summary = _format_data_for_analyst(
        company, news, financials, errors, memory_context, revision_count, max_revisions
    )

    try:
        # Try structured output first; fall back to plain invoke if it fails
        try:
            structured_llm = llm.with_structured_output(AnalysisOutput)
            result: AnalysisOutput = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=data_summary),
            ])
            report = result.report
        except Exception:
            # Fallback: invoke without structured output and use raw text
            logger.info("Structured output failed, falling back to plain invoke")
            response = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=data_summary),
            ])
            report = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "analysis_report": report,
            "revision_count": revision_count + 1,
            "messages": [HumanMessage(
                content=f"[Analyst] Report produced (revision {revision_count + 1}/{max_revisions})."
            )],
        }

    except Exception as e:
        logger.error("Analyst agent failed: %s", e)
        return {
            "analysis_report": f"Analysis failed: {e}",
            "revision_count": revision_count + 1,
            "messages": [HumanMessage(content=f"[Analyst] Error: {e}")],
            "errors": [{
                "agent": "analyst",
                "tool": "llm",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "recoverable": False,
            }],
        }


def _format_data_for_analyst(
    company: str,
    news: list[dict],
    financials: list[dict],
    errors: list[dict],
    memory_context: str,
    revision_count: int,
    max_revisions: int,
) -> str:
    """Format all collected data into a prompt for the analyst."""
    parts = [f"# Analysis Request: {company}\n"]
    parts.append(f"Revision: {revision_count + 1}/{max_revisions}\n")

    if memory_context:
        parts.append(memory_context)
        parts.append("")

    parts.append("## Collected News & Research")
    if news:
        parts.append(json.dumps(news, indent=2, default=str))
    else:
        parts.append("NO NEWS DATA AVAILABLE — this is a critical gap.")

    parts.append("\n## Collected Financial Data")
    if financials:
        parts.append(json.dumps(financials, indent=2, default=str))
    else:
        parts.append("NO FINANCIAL DATA AVAILABLE — this is a critical gap.")

    if errors:
        parts.append("\n## Data Collection Errors")
        parts.append("The following errors occurred during data gathering. "
                      "Account for these gaps in your analysis:")
        for err in errors:
            parts.append(f"- [{err.get('agent')}] {err.get('tool')}: {err.get('message')}")

    if revision_count >= max_revisions - 1:
        parts.append(
            "\n⚠️ This is the final revision. Produce the best analysis possible "
            "with the data available, noting any gaps."
        )

    return "\n".join(parts)
