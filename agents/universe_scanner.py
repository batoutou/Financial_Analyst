import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import InvestmentState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a market analyst scanning global public markets for investment opportunities.

Based on the provided market regime, identify approximately 15 investment candidates across:
- ~4 US stocks (NYSE/NASDAQ) showing momentum, oversold value, or strong earnings catalysts
- ~3 European stocks (Euronext, LSE, Xetra) with analyst upgrades or sector tailwinds
- ~3 global ETFs (MSCI World, sector ETFs, thematic) with strong trend or attractive entry
- ~3 crypto assets (BTC, ETH, or major tokens) with technical setups or on-chain catalysts
- ~2 bond ETFs (TLT, BND, BUND equivalent) attractive given the current rate environment

Use search tools to find current market opportunities, analyst recommendations, and trending assets.

Return ONLY a JSON array where each item has:
{
    "ticker": "<exchange ticker symbol>",
    "name": "<full asset name>",
    "asset_type": "<stock | etf | bond | crypto>",
    "exchange": "<NASDAQ | NYSE | Euronext | LSE | Xetra | Crypto>",
    "rationale": "<one sentence: why this asset is interesting right now>"
}"""


def _parse_candidates(content: str | list) -> list[dict]:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "[" in text:
        try:
            json_str = text[text.index("["):text.rindex("]") + 1]
            data = json.loads(json_str)
            return [
                item for item in data
                if all(k in item for k in ["ticker", "name", "asset_type"])
            ]
        except (json.JSONDecodeError, ValueError):
            pass

    return []


def create_universe_scanner_node(tools: list[BaseTool]):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def universe_scanner_node(state: InvestmentState) -> dict:
        regime = state.get("market_regime", {})
        revision_count = state.get("revision_count", 0)
        revision_feedback = state.get("revision_feedback", "")
        collected_errors: list[dict] = []

        user_msg = (
            f"Market regime: {regime.get('risk_level', 'balanced')} "
            f"(VIX {regime.get('vix', 'N/A')}, {regime.get('summary', '')})\n\n"
            "Find the best investment opportunities across US stocks, EU stocks, "
            "global ETFs, crypto, and bond ETFs right now."
        )
        if revision_feedback:
            user_msg += (
                f"\n\nPrevious portfolio was rejected. Feedback: {revision_feedback}\n"
                "Find DIFFERENT assets that address these issues."
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
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
                        result, error = await retry_tool_call(tool, tool_call["args"], "universe_scanner")
                        if error:
                            collected_errors.append(error)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            raw_candidates = _parse_candidates(response.content)
            # Tag each candidate with the current revision round for deduplication downstream
            candidates = [{**c, "revision": revision_count} for c in raw_candidates]
            logger.info("UniverseScanner found %d candidates (revision %d)", len(candidates), revision_count)

            return {
                "candidates": candidates,
                "messages": [HumanMessage(content=f"[UniverseScanner] Found {len(candidates)} candidates.")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("UniverseScanner failed: %s", e)
            return {
                "candidates": [],
                "messages": [HumanMessage(content=f"[UniverseScanner] Error: {e}")],
                "errors": collected_errors + [{
                    "agent": "universe_scanner", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return universe_scanner_node
