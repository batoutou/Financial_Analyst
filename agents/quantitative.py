import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import InvestmentState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative financial analyst. Extract hard financial data for a given asset.

Instructions vary by asset type:

For stocks and ETFs:
1. Get analyst price targets from multiple brokers: use the price-target endpoint for the ticker.
   For each analyst entry, extract: broker name, analyst name, rating, target price, date, price on date.
2. Get key financial ratios: P/E, EV/EBITDA, P/B, ROE, debt-to-equity, revenue growth, gross margin.
3. Get current stock quote (latest price).
4. Get RSI (14-period daily) to assess momentum.

For crypto:
1. Get current price and 30-day price change.
2. Get RSI if available.

For bonds:
1. Get current yield and duration from the ETF data.
2. Get recent price and 52-week range.

Return your findings as a structured JSON array where each item has:
{
    "metric": "<name>",
    "category": "<broker_consensus | income_statement | balance_sheet | ratio | market_data | technical>",
    "value": <numeric value or nested object>,
    "period": "<FY2024 | TTM | current | etc.>",
    "unit": "<USD | percentage | ratio | count>"
}

For broker_consensus, use value as a nested object:
{
    "metric": "broker_consensus",
    "category": "broker_consensus",
    "value": {
        "mean_target": <float>,
        "buy_count": <int>,
        "hold_count": <int>,
        "sell_count": <int>,
        "brokers": [
            {"broker": str, "analyst": str, "rating": str, "target_price": float,
             "rating_date": str, "price_on_rating_date": float, "implied_return_pct": float}
        ]
    },
    "period": "current",
    "unit": "USD"
}

Return ONLY the JSON array, no other text."""


def create_quantitative_node(tools: list[BaseTool]):
    """Create the quantitative agent node with the given tools."""

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def quantitative_node(state: InvestmentState) -> dict:
        candidate = state.get("candidate", {})
        ticker = candidate.get("ticker", state.get("company_name", ""))
        name = candidate.get("name", ticker)
        asset_type = candidate.get("asset_type", "stock")
        collected_errors: list[dict] = []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract financial data for {asset_type}: {name} (ticker: {ticker})"),
        ]

        try:
            for _ in range(10):
                response: AIMessage = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool = tools_by_name.get(tool_call["name"])
                    if tool is None:
                        result = f"Tool '{tool_call['name']}' not found."
                    else:
                        result, error = await retry_tool_call(tool, tool_call["args"], "quantitative")
                        if error:
                            collected_errors.append(error)

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

            financial_data = _parse_financial_data(response.content)

            return {
                "financial_data": financial_data,
                "messages": [HumanMessage(content=f"[Quantitative] Extracted {len(financial_data)} data points for {name}.")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("Quantitative agent failed: %s", e)
            collected_errors.append({
                "agent": "quantitative",
                "tool": "agent_loop",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "recoverable": False,
            })
            return {
                "financial_data": [],
                "messages": [HumanMessage(content=f"[Quantitative] Error: {e}")],
                "errors": collected_errors,
            }

    return quantitative_node


def _parse_financial_data(content: str | list) -> list[dict]:
    """Extract structured financial data from the LLM response."""
    text = content if isinstance(content, str) else content[0].get("text", "") if content else ""
    text = text.strip()

    if "[" in text:
        json_str = text[text.index("["):text.rindex("]") + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return [{"metric": "Raw Data", "category": "other", "value": text, "period": "N/A", "unit": "N/A"}]
