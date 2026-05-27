import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import InvestmentState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a macro financial analyst. Assess the current market regime.

Search for exactly these three data points:
1. Current VIX index level (fear gauge)
2. US 10-year minus 2-year Treasury yield spread (positive = normal, negative = inverted)
3. Whether the S&P 500 is trading above or below its 200-day moving average

Return ONLY this JSON object (no other text):
{
    "vix": <float>,
    "yield_curve_spread": <float>,
    "spx_above_200ma": <bool>,
    "risk_level": <"aggressive" | "balanced" | "defensive">,
    "summary": "<1-2 sentence market description>"
}

Risk level rules:
- "aggressive": VIX < 15 AND yield_curve_spread > 0 AND spx_above_200ma = true
- "defensive": VIX > 25 OR spx_above_200ma = false
- "balanced": everything else"""

_FALLBACK_REGIME = {
    "vix": 20.0,
    "yield_curve_spread": 0.0,
    "spx_above_200ma": True,
    "risk_level": "balanced",
    "summary": "Could not determine market regime. Defaulting to balanced.",
}


def _parse_regime(content: str | list) -> dict:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            required = ["vix", "yield_curve_spread", "spx_above_200ma", "risk_level", "summary"]
            if all(k in data for k in required):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return dict(_FALLBACK_REGIME)


def create_macro_scanner_node(tools: list[BaseTool]):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=2048)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def macro_scanner_node(state: InvestmentState) -> dict:
        collected_errors: list[dict] = []
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                "Search for: current VIX level, US 10y-2y yield spread, "
                "and whether S&P 500 is above its 200-day moving average."
            )),
        ]

        try:
            for _ in range(6):
                response: AIMessage = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool = tools_by_name.get(tool_call["name"])
                    if tool is None:
                        result = f"Tool '{tool_call['name']}' not found."
                    else:
                        result, error = await retry_tool_call(tool, tool_call["args"], "macro_scanner")
                        if error:
                            collected_errors.append(error)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            regime = _parse_regime(response.content)
            logger.info("Market regime: %s (VIX %.1f)", regime["risk_level"], regime.get("vix", 0))

            return {
                "market_regime": regime,
                "messages": [HumanMessage(content=f"[MacroScanner] Regime: {regime['risk_level']} — {regime['summary']}")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("MacroScanner failed: %s", e)
            return {
                "market_regime": dict(_FALLBACK_REGIME),
                "messages": [HumanMessage(content=f"[MacroScanner] Error: {e}. Using balanced defaults.")],
                "errors": collected_errors + [{
                    "agent": "macro_scanner", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": True,
                }],
            }

    return macro_scanner_node
