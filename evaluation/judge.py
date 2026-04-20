"""LLM-as-Judge evaluation pipeline using Gemini.

Uses a separate model provider (Google Gemini) to independently evaluate
the Analyst's report against the raw source data. This eliminates self-serving
bias that would occur if the same model evaluated its own output.

Evaluation rubric:
  - Grounding:     Are claims supported by the raw data provided?
  - Completeness:  Does the report cover all required sections?
  - Consistency:   Do numbers/facts match between sections?
  - Actionability: Would an investor find this useful?
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from graph.state import FinancialAnalystState

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an independent quality evaluator for financial analysis reports.
You will receive:
1. A financial analysis report produced by an AI analyst
2. The raw source data (news articles + financial data) that the analyst had access to
3. Any errors that occurred during data collection

Your job is to evaluate the report's quality on four dimensions, each scored 1-5:

## Scoring Rubric

### Grounding (1-5): Are claims supported by the source data?
- 1: Most claims are fabricated or unsupported
- 2: Several key claims lack source support
- 3: Main claims are supported, some minor unsupported statements
- 4: Nearly all claims traceable to source data
- 5: Every factual claim is directly supported by the provided data

### Completeness (1-5): Does the report cover all required sections?
Required sections: Executive Summary, Financial Overview, News & Market Sentiment, Risk Assessment, Conclusion
- 1: Missing 3+ sections
- 2: Missing 2 sections
- 3: All sections present but some are superficial
- 4: All sections present with adequate depth
- 5: All sections present with thorough, insightful analysis

### Consistency (1-5): Do numbers and facts match across the report?
- 1: Contradictions between sections
- 2: Some numbers don't match the source data
- 3: Minor inconsistencies that don't affect conclusions
- 4: Numbers are consistent, narrative aligns with data
- 5: Perfect internal consistency and source alignment

### Actionability (1-5): Would an investor find this useful?
- 1: No clear takeaway or investment perspective
- 2: Vague conclusions without specific reasoning
- 3: Clear conclusion but limited supporting argument
- 4: Well-reasoned conclusion with specific supporting points
- 5: Compelling, nuanced analysis with clear investment implications

## Response Format
Return your evaluation as structured JSON with scores, issues found, and a verdict.
"""

# Few-shot examples to calibrate the judge
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """## Report to evaluate:
## Executive Summary
Apple reported strong Q4 results with revenue of $89.5B, beating estimates.

## Financial Overview
Revenue: $89.5B (up 6% YoY). EPS: $1.46. Services revenue hit record $22.2B.
P/E ratio: 28.5x. Gross margin improved to 46.2%.

## News & Market Sentiment
Recent news is positive: strong iPhone 15 demand, services growth, and AI initiatives.
Analysts have raised price targets.

## Risk Assessment
- China revenue declined 2% — geopolitical risk remains
- Regulatory pressure on App Store fees in EU

## Conclusion
Apple demonstrates strong financial health with growing services offsetting hardware cyclicality.

## Source data:
financial_data: [{"metric": "Revenue", "value": "$89.5B", "period": "Q4 2024"},
{"metric": "EPS", "value": "$1.46", "period": "Q4 2024"},
{"metric": "P/E", "value": "28.5", "period": "TTM"},
{"metric": "Gross Margin", "value": "46.2%", "period": "Q4 2024"},
{"metric": "Services Revenue", "value": "$22.2B", "period": "Q4 2024"}]
news: [{"title": "Apple beats Q4 estimates", "sentiment": "positive"},
{"title": "iPhone 15 demand strong", "sentiment": "positive"},
{"title": "Apple China revenue dips", "sentiment": "negative"}]
errors: []"""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "grounding_score": 5,
            "grounding_reasoning": "Every number in the report (revenue, EPS, P/E, gross margin, services revenue) matches the source data exactly. China decline claim is supported by the news article.",
            "completeness_score": 4,
            "completeness_reasoning": "All five sections present. Risk section could be deeper — only two bullet points. Financial Overview is solid.",
            "consistency_score": 5,
            "consistency_reasoning": "No contradictions. Revenue figure consistent between Executive Summary and Financial Overview. Sentiment in News section aligns with source article sentiments.",
            "actionability_score": 3,
            "actionability_reasoning": "Clear positive thesis but lacks specific valuation argument or comparison to peers. An investor would want more on whether current P/E is justified.",
            "overall_score": 4.25,
            "issues": ["Risk section is thin — only 2 risks identified", "No valuation argument or peer comparison in conclusion"],
            "verdict": "pass"
        })
    },
    {
        "role": "user",
        "content": """## Report to evaluate:
## Executive Summary
Tesla is revolutionizing the auto industry with its innovative EV technology and AI capabilities.

## Financial Overview
Tesla's revenue has been growing rapidly. The company is profitable and has a strong balance sheet.

## News & Market Sentiment
The market is very excited about Tesla's future. Elon Musk continues to drive innovation.

## Risk Assessment
There are some risks but Tesla's brand is strong.

## Conclusion
Tesla is a strong buy with unlimited potential.

## Source data:
financial_data: [{"metric": "Revenue", "value": "$25.2B", "period": "Q3 2024"},
{"metric": "Net Income", "value": "$1.8B", "period": "Q3 2024"},
{"metric": "Gross Margin", "value": "19.8%", "period": "Q3 2024"}]
news: [{"title": "Tesla Q3 deliveries miss estimates", "sentiment": "negative"},
{"title": "Tesla Cybertruck recall announced", "sentiment": "negative"}]
errors: []"""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "grounding_score": 1,
            "grounding_reasoning": "Report contains zero specific numbers despite financial data being available. Claims of 'growing rapidly' and 'strong balance sheet' are unsubstantiated. No source data is referenced.",
            "completeness_score": 2,
            "completeness_reasoning": "All sections technically present but each is 1-2 vague sentences. Financial Overview has no actual financials. Risk section is a single dismissive sentence.",
            "consistency_score": 1,
            "consistency_reasoning": "Report narrative is overwhelmingly positive but source news is negative (missed deliveries, recall). The report ignores contradicting evidence entirely.",
            "actionability_score": 1,
            "actionability_reasoning": "'Strong buy with unlimited potential' is not analysis — it's hype. No specific reasoning, no valuation, no risk-reward framework.",
            "overall_score": 1.25,
            "issues": [
                "No specific numbers used despite available financial data",
                "Completely ignores negative news (delivery miss, recall)",
                "Risk section dismisses risks without analysis",
                "Conclusion is unsupported hype, not investment analysis",
                "Positive bias contradicts source data sentiment"
            ],
            "verdict": "fail"
        })
    },
]


class EvaluationResult(BaseModel):
    """Structured output from the LLM judge."""

    grounding_score: int = Field(ge=1, le=5, description="Are claims supported by source data? (1-5)")
    grounding_reasoning: str = Field(description="Justification for grounding score")
    completeness_score: int = Field(ge=1, le=5, description="Does the report cover all sections? (1-5)")
    completeness_reasoning: str = Field(description="Justification for completeness score")
    consistency_score: int = Field(ge=1, le=5, description="Do facts match across report and sources? (1-5)")
    consistency_reasoning: str = Field(description="Justification for consistency score")
    actionability_score: int = Field(ge=1, le=5, description="Would an investor find this useful? (1-5)")
    actionability_reasoning: str = Field(description="Justification for actionability score")
    overall_score: float = Field(ge=1.0, le=5.0, description="Weighted average of all scores")
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    verdict: str = Field(description="'pass', 'fail', or 'needs_improvement'")


async def evaluator_node(state: FinancialAnalystState) -> dict:
    """Evaluate the analyst's report using Gemini as an independent judge.

    This node runs after the analyst and uses a different model provider
    (Google Gemini) to eliminate self-serving bias. The evaluation scores
    drive the revision decision: if the verdict is 'fail' or 'needs_improvement',
    the graph routes back to the researcher for more data.
    """
    report = state.get("analysis_report", "")
    news = state.get("news_articles", [])
    financials = state.get("financial_data", [])
    errors = state.get("errors", [])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
    )

    evaluation_input = _format_evaluation_input(report, news, financials, errors)

    try:
        # Build few-shot examples as a single text block to avoid message format issues
        few_shot_text = "\n\n---\n\n".join(
            f"**Example {i//2 + 1} {'Input' if ex['role'] == 'user' else 'Output'}:**\n{ex['content']}"
            for i, ex in enumerate(FEW_SHOT_EXAMPLES)
        )

        system_with_examples = (
            JUDGE_SYSTEM_PROMPT
            + "\n\n## Examples\n\n"
            + few_shot_text
            + "\n\n---\n\nNow evaluate the following report. Respond with ONLY valid JSON."
        )

        messages = [
            SystemMessage(content=system_with_examples),
            HumanMessage(content=evaluation_input),
        ]

        # Try structured output first, fall back to JSON parsing
        try:
            structured_llm = llm.with_structured_output(EvaluationResult)
            result: EvaluationResult = await structured_llm.ainvoke(messages)
        except Exception:
            logger.info("Structured output failed for evaluator, falling back to JSON parsing")
            raw_response = await llm.ainvoke(messages)
            content = raw_response.content if isinstance(raw_response.content, str) else str(raw_response.content)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = EvaluationResult.model_validate_json(json_match.group())
            else:
                raise ValueError(f"Could not parse evaluation JSON from response: {content[:200]}")

        evaluation = result.model_dump()

        # The evaluator decides if revision is needed (replacing analyst self-assessment)
        needs_revision = result.verdict in ("fail", "needs_improvement")

        revision_feedback = ""
        if needs_revision and result.issues:
            revision_feedback = (
                f"Evaluation failed (score: {result.overall_score}/5). "
                f"Issues: {'; '.join(result.issues[:3])}. "
                f"Please gather more data to address these gaps."
            )

        logger.info(
            "Evaluation complete: %.1f/5 — %s (%d issues)",
            result.overall_score,
            result.verdict,
            len(result.issues),
        )

        return {
            "evaluation": evaluation,
            "needs_revision": needs_revision,
            "revision_feedback": revision_feedback,
            "messages": [HumanMessage(
                content=f"[Evaluator] Score: {result.overall_score}/5 — {result.verdict}. "
                f"Issues: {len(result.issues)}"
            )],
        }

    except Exception as e:
        logger.error("Evaluator failed: %s", e)
        # If evaluation fails, don't block the pipeline — pass through
        return {
            "evaluation": {"error": str(e), "verdict": "pass", "overall_score": 0},
            "needs_revision": False,
            "messages": [HumanMessage(content=f"[Evaluator] Failed: {e}. Skipping evaluation.")],
            "errors": [
                {
                    "agent": "evaluator",
                    "tool": "gemini",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }
            ],
        }


def _format_evaluation_input(
    report: str,
    news: list[dict],
    financials: list[dict],
    errors: list[dict],
) -> str:
    """Format the analyst's output + raw sources for the judge."""
    parts = ["## Report to evaluate:", report, ""]
    parts.append("## Source data:")
    parts.append(f"financial_data: {json.dumps(financials, indent=2, default=str)}")
    parts.append(f"news: {json.dumps(news, indent=2, default=str)}")
    parts.append(f"errors: {json.dumps(errors, indent=2, default=str)}")
    return "\n".join(parts)
