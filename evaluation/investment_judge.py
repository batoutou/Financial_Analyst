import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import InvestmentState

logger = logging.getLogger(__name__)

_MAX_SINGLE = {"aggressive": 30.0, "balanced": 20.0, "defensive": 10.0}

JUDGE_SYSTEM_PROMPT = """You are an independent portfolio risk manager reviewing an investment
portfolio allocation. Evaluate the portfolio holistically and produce a quality score.

Score these dimensions (1-5 each):
- Risk/Reward quality: Are R/R ratios >= 2.0 on all positions?
- Diversification: Are there >= 3 asset types? Max 2 stocks in same sector?
- Data quality: Are broker targets recent and specific?
- Allocation logic: Does sizing reflect conviction? Does it match the risk regime?

Return ONLY this JSON object:
{
    "verdict": "<pass | fail>",
    "overall_score": <float 1-5>,
    "rr_score": <float 1-5>,
    "diversification_score": <float 1-5>,
    "data_quality_score": <float 1-5>,
    "allocation_score": <float 1-5>,
    "issues": ["<specific issue 1>", "<specific issue 2>"],
    "feedback": "<actionable guidance for improvement if verdict is fail>"
}"""


def _validate_portfolio(portfolio: list[dict], market_regime: dict) -> list[str]:
    """Pure validation — returns list of issues. Empty = no issues."""
    issues = []
    risk_level = market_regime.get("risk_level", "balanced")
    max_single = _MAX_SINGLE.get(risk_level, 20.0)
    non_cash = [p for p in portfolio if p.get("asset_type") != "cash"]

    for item in non_cash:
        rr = item.get("risk_reward")
        if rr is not None and rr < 2.0:
            issues.append(f"{item['ticker']}: risk/reward {rr:.2f} is below minimum 2.0")

    for item in non_cash:
        pct = item.get("allocation_pct", 0)
        if pct > max_single:
            issues.append(
                f"{item['ticker']}: allocation {pct}% exceeds {risk_level} regime max {max_single}%"
            )

    asset_types = {p.get("asset_type") for p in non_cash if p.get("asset_type")}
    if len(asset_types) < 3:
        issues.append(
            f"Portfolio has only {len(asset_types)} asset type(s); need at least 3 for diversification"
        )

    # Guard against stock overconcentration (proxy for sector concentration — max 3 stocks)
    stock_count = sum(1 for p in non_cash if p.get("asset_type") == "stock")
    if stock_count > 3:
        issues.append(
            f"Portfolio has {stock_count} stock positions; limit to 3 to avoid sector concentration"
        )

    total = sum(p.get("allocation_pct", 0) for p in portfolio)
    if abs(total - 100.0) > 1.0:
        issues.append(f"Allocations sum to {total:.1f}%, expected 100%")

    return issues


def _parse_evaluation(content: str | list) -> dict:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()
    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"verdict": "pass", "overall_score": 3.0, "issues": [], "feedback": ""}


def create_investment_judge_node():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    async def investment_judge_node(state: InvestmentState) -> dict:
        portfolio = state.get("portfolio", [])
        market_regime = state.get("market_regime", {})
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 2)

        # Pure validation first
        pure_issues = _validate_portfolio(portfolio, market_regime)

        judge_input = (
            f"Market regime: {market_regime.get('risk_level', 'balanced')} "
            f"(VIX {market_regime.get('vix', 'N/A')})\n\n"
            f"Portfolio:\n{json.dumps(portfolio, indent=2, default=str)}\n\n"
            f"Pre-validation issues found: {pure_issues if pure_issues else 'None'}"
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=judge_input),
            ])
            evaluation = _parse_evaluation(response.content)

            # Merge pure validation issues
            all_issues = pure_issues + [
                i for i in evaluation.get("issues", []) if i not in pure_issues
            ]
            evaluation["issues"] = all_issues

            # Override verdict if pure validation found critical issues
            if pure_issues:
                evaluation["verdict"] = "fail"

            needs_revision = (
                evaluation.get("verdict") == "fail"
                and revision_count < max_revisions
            )
            revision_feedback = ""
            if needs_revision and all_issues:
                revision_feedback = (
                    f"Portfolio rejected (score {evaluation.get('overall_score', 0):.1f}/5). "
                    f"Issues: {'; '.join(all_issues[:3])}. "
                    f"{evaluation.get('feedback', 'Find better candidates.')}"
                )

            logger.info(
                "InvestmentJudge: %.1f/5 — %s (%d issues)",
                evaluation.get("overall_score", 0),
                evaluation.get("verdict", "unknown"),
                len(all_issues),
            )

            return {
                "evaluation": evaluation,
                "needs_revision": needs_revision,
                "revision_count": revision_count + 1,
                "revision_feedback": revision_feedback,
                "messages": [HumanMessage(
                    content=f"[InvestmentJudge] Score: {evaluation.get('overall_score', 0):.1f}/5 — "
                            f"{evaluation.get('verdict', 'unknown')}. Issues: {len(all_issues)}"
                )],
            }

        except Exception as e:
            logger.error("InvestmentJudge failed: %s", e)
            return {
                "evaluation": {"verdict": "pass", "overall_score": 0, "issues": pure_issues},
                "needs_revision": False,
                "revision_count": revision_count + 1,
                "revision_feedback": "",
                "messages": [HumanMessage(content=f"[InvestmentJudge] Failed: {e}. Skipping evaluation.")],
                "errors": [{
                    "agent": "investment_judge", "tool": "gemini",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return investment_judge_node
