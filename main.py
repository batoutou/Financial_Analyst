"""Investment Opportunity Scanner — Multi-Agent System.

Autonomously scans public markets (US stocks, EU stocks, ETFs, crypto, bonds)
and produces a portfolio allocation with entry/target/stop prices.

Usage:
    python main.py --budget 1000
    python main.py --budget 5000 --max-revisions 2
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from graph.workflow import build_graph
from tools.mcp_tools import (
    create_mcp_client,
    get_alphavantage_tools,
    get_firecrawl_tools,
    get_fmp_tools,
)
from tools.tavily_tools import get_tavily_search_tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_analysis(budget: float, max_revisions: int = 2) -> dict:
    """Run the full investment opportunity pipeline.

    Returns the full final state dict (portfolio, evaluation, errors, etc.).
    """
    logger.info("Starting investment scan with budget: %.0f", budget)

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

    try:
        alphavantage_tools = await get_alphavantage_tools(client)
        if alphavantage_tools:
            logger.info("Loaded %d AlphaVantage tools", len(alphavantage_tools))
    except Exception as e:
        logger.warning("Failed to load AlphaVantage tools: %s. Continuing without.", e)
        alphavantage_tools = []

    researcher_tools = [tavily_tool] + firecrawl_tools
    # AlphaVantage supplements FMP for RSI, quotes, and earnings data
    quantitative_tools = fmp_tools + alphavantage_tools

    if not quantitative_tools:
        logger.warning("No quantitative tools available. Analysis will have limited financial data.")

    app = build_graph(researcher_tools, quantitative_tools)

    initial_state = {
        "budget": budget,
        "market_regime": {},
        "candidates": [],
        "opportunities": [],
        "portfolio": [],
        "evaluation": {},
        "revision_count": 0,
        "max_revisions": max_revisions,
        "needs_revision": False,
        "revision_feedback": "",
        "messages": [],
        "errors": [],
    }

    logger.info("Executing investment scan graph...")
    result = await app.ainvoke(initial_state)

    if result.get("errors"):
        logger.warning("Scan completed with %d errors", len(result["errors"]))

    return result


def _build_report_text(result: dict) -> str:
    from datetime import datetime, timezone

    lines = []
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    budget = result.get("budget", 0)
    regime = result.get("market_regime", {})
    risk_level = regime.get("risk_level", "N/A")
    vix = regime.get("vix", "N/A")

    lines.append(f"# Investment Portfolio — {date_str}")
    lines.append(f"**Budget:** {budget:,.0f} | **Risk regime:** {risk_level.title()} (VIX {vix})\n")

    lines.append("## Market Regime")
    lines.append(f"- VIX: {vix}")
    lines.append(f"- Yield curve spread: {regime.get('yield_curve_spread', 'N/A')}")
    lines.append(f"- S&P 500 vs 200MA: {'Above' if regime.get('spx_above_200ma') else 'Below'}")
    lines.append(f"- Summary: {regime.get('summary', 'N/A')}\n")

    portfolio = result.get("portfolio", [])
    if portfolio:
        lines.append("## Portfolio Allocation\n")
        lines.append("| # | Ticker | Name | Type | Horizon | Buy at | Target | Stop | Return | R/R | Alloc % | Amount |")
        lines.append("|---|--------|------|------|---------|--------|--------|------|--------|-----|---------|--------|")
        i = 1
        for item in portfolio:
            if item["ticker"] == "CASH":
                lines.append(
                    f"| — | **CASH** | Cash Reserve | — | — | — | — | — | — | — | "
                    f"**{item['allocation_pct']}%** | {item['allocation_amount']:,.0f} |"
                )
            else:
                lines.append(
                    f"| {i} | **{item['ticker']}** | {item['name']} | {item['asset_type']} | "
                    f"{item.get('horizon', 'N/A')} | {item.get('entry_price', 'N/A')} | "
                    f"{item.get('target_price', 'N/A')} | {item.get('stop_loss', 'N/A')} | "
                    f"+{item.get('implied_return_pct', 'N/A')}% | {item.get('risk_reward', 'N/A')} | "
                    f"**{item['allocation_pct']}%** | {item['allocation_amount']:,.0f} |"
                )
                i += 1
        lines.append("")

        lines.append("## Asset Detail\n")
        for item in portfolio:
            if item["ticker"] == "CASH":
                continue
            lines.append(f"### {item['ticker']} — {item['name']}")
            lines.append(
                f"**Type:** {item['asset_type'].upper()} | **Horizon:** {item.get('horizon', 'N/A')} | "
                f"**Conviction:** {item.get('conviction', 'N/A')}/10\n"
            )

            broker = item.get("broker_consensus")
            if broker and broker.get("brokers"):
                buy = broker.get("buy_count", 0)
                hold = broker.get("hold_count", 0)
                sell = broker.get("sell_count", 0)
                lines.append(
                    f"**Broker consensus:** {buy}B/{hold}H/{sell}S | "
                    f"Mean target: {broker.get('mean_target', 'N/A')} | "
                    f"Implied return: +{item.get('implied_return_pct', 'N/A')}%\n"
                )
                lines.append("| Broker | Analyst | Rating | Target | Date | Implied Return |")
                lines.append("|--------|---------|--------|--------|------|----------------|")
                for b in broker["brokers"]:
                    lines.append(
                        f"| {b.get('broker', '')} | {b.get('analyst', '')} | "
                        f"{b.get('rating', '')} | {b.get('target_price', '')} | "
                        f"{b.get('rating_date', '')} | +{b.get('implied_return_pct', '')}% |"
                    )
                lines.append("")

            signals = item.get("signals") or {}
            if signals:
                sig_parts = [f"{k}: {v}" for k, v in signals.items() if v is not None]
                if sig_parts:
                    lines.append(f"**Signals:** {', '.join(sig_parts)}\n")

            lines.append(f"**Rationale:** {item.get('rationale', 'N/A')}\n")
            lines.append(
                f"**Entry:** {item.get('entry_price')} | "
                f"**Target:** {item.get('target_price')} | "
                f"**Stop:** {item.get('stop_loss')} | "
                f"**R/R:** {item.get('risk_reward')}\n"
            )

    evaluation = result.get("evaluation", {})
    if evaluation and "overall_score" in evaluation:
        lines.append("---\n## Quality Evaluation\n")
        lines.append(f"**Score: {evaluation['overall_score']}/5 — {evaluation.get('verdict', 'N/A')}**\n")
        for issue in evaluation.get("issues", []):
            lines.append(f"- {issue}")

    errors = result.get("errors", [])
    if errors:
        lines.append(f"\n---\n**{len(errors)} error(s) during scan** (see logs)")

    return "\n".join(lines)


def _save_report(result: dict) -> str:
    from datetime import datetime, timezone
    from pathlib import Path

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = reports_dir / f"{date_str}_portfolio.md"

    filepath.write_text(_build_report_text(result))
    return str(filepath)


def _print_report(result: dict) -> None:
    print("\n" + "=" * 80)
    print(_build_report_text(result))
    print("=" * 80)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Investment Opportunity Scanner — Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budget",
        required=True,
        type=float,
        help="Total budget to invest (e.g. 1000)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum revision cycles if judge rejects portfolio (default: 2)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_analysis(args.budget, args.max_revisions))
    _print_report(result)
    filepath = _save_report(result)
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
