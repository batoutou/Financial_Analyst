import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

_RESEARCH_PROMPT = """You are a financial research analyst. Research the following asset for a
potential investment opportunity. Find recent news, analyst coverage, earnings, and market sentiment.
Focus on what is material to an investment decision right now.

Return findings as a JSON array: [{"title": str, "source": str, "date": str, "summary": str, "sentiment": str}]
Return ONLY the JSON array."""

_QUANT_PROMPT = """You are a quantitative analyst. For the given asset, fetch:
1. Analyst price targets from all brokers (use price-target endpoint)
2. Key financial ratios (P/E, EV/EBITDA, ROE, revenue growth, gross margin, debt/equity)
3. Current stock quote / price
4. RSI (14-day)

Return findings as a JSON array of metric objects. For broker targets, use category "broker_consensus"
with value as a nested object containing mean_target, buy_count, hold_count, sell_count, and brokers array.
Return ONLY the JSON array."""

_SCORING_PROMPT = """You are a senior investment analyst. Given the research and quantitative data
for an asset, produce a structured investment opportunity assessment.

Rules:
- entry_price: use the current market price (or slightly below if RSI < 35)
- target_price: anchor to broker consensus mean target; adjust to nearest technical resistance
- stop_loss: place below the nearest key support level
- risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
- implied_return_pct = round((target_price - entry_price) / entry_price * 100, 1)
- conviction: 1-10 score based on: broker consensus (40%), valuation vs sector (30%), momentum/RSI (20%), news sentiment (10%)
- horizon: "short-term" (days-weeks, RSI reversal), "medium-term" (1-6 months, earnings cycle), "long-term" (6m+, DCF/dividend)

Return ONLY this JSON object:
{
    "entry_price": <float>,
    "target_price": <float>,
    "stop_loss": <float>,
    "conviction": <int 1-10>,
    "horizon": <"short-term" | "medium-term" | "long-term">,
    "implied_return_pct": <float>,
    "risk_reward": <float>,
    "broker_consensus": {
        "mean_target": <float>,
        "buy_count": <int>,
        "hold_count": <int>,
        "sell_count": <int>,
        "brokers": [{"broker": str, "analyst": str, "rating": str, "target_price": float, "rating_date": str, "price_on_rating_date": float, "implied_return_pct": float}]
    },
    "signals": {"rsi": <float or null>, "pe_ratio": <float or null>, "revenue_growth": <float or null>},
    "rationale": "<2-3 sentences explaining the investment thesis>"
}

If risk_reward < 2.0, still return the JSON but set conviction to 0."""


def _parse_opportunity(content: str | list, candidate: dict) -> dict | None:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            required = [
                "entry_price", "target_price", "stop_loss", "conviction",
                "horizon", "implied_return_pct", "risk_reward",
                "broker_consensus", "signals", "rationale",
            ]
            if not all(k in data for k in required):
                return None
            if data.get("risk_reward", 0) < 2.0:
                return None
            return {**candidate, **data}
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_json_list(content: str | list) -> list:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()
    if "[" in text:
        try:
            return json.loads(text[text.index("["):text.rindex("]") + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def create_asset_analyst_node(researcher_tools: list[BaseTool], quantitative_tools: list[BaseTool]):
    researcher_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    quant_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    scoring_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=2048)

    researcher_llm_with_tools = researcher_llm.bind_tools(researcher_tools) if researcher_tools else researcher_llm
    quant_llm_with_tools = quant_llm.bind_tools(quantitative_tools) if quantitative_tools else quant_llm

    researcher_tools_by_name = {t.name: t for t in researcher_tools}
    quant_tools_by_name = {t.name: t for t in quantitative_tools}

    async def _run_tool_loop(llm_with_tools, tools_by_name, messages, agent_name, max_iter=10):
        collected_errors = []
        for _ in range(max_iter):
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:
                    result = f"Tool '{tool_call['name']}' not found."
                else:
                    result, error = await retry_tool_call(tool, tool_call["args"], agent_name)
                    if error:
                        collected_errors.append(error)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        return response, collected_errors

    async def asset_analyst_node(state: dict) -> dict:
        candidate = state.get("candidate", {})
        market_regime = state.get("market_regime", {})
        ticker = candidate.get("ticker", "UNKNOWN")
        name = candidate.get("name", ticker)
        asset_type = candidate.get("asset_type", "stock")
        collected_errors: list[dict] = []

        try:
            # Phase 1: Research
            research_messages = [
                SystemMessage(content=_RESEARCH_PROMPT),
                HumanMessage(content=f"Research {asset_type}: {name} (ticker: {ticker})"),
            ]
            research_response, r_errors = await _run_tool_loop(
                researcher_llm_with_tools, researcher_tools_by_name, research_messages, "asset_analyst_research"
            )
            collected_errors.extend(r_errors)
            news_data = _parse_json_list(research_response.content)

            # Phase 2: Quantitative
            quant_messages = [
                SystemMessage(content=_QUANT_PROMPT),
                HumanMessage(content=f"Get quantitative data for {asset_type}: {name} (ticker: {ticker})"),
            ]
            quant_response, q_errors = await _run_tool_loop(
                quant_llm_with_tools, quant_tools_by_name, quant_messages, "asset_analyst_quant"
            )
            collected_errors.extend(q_errors)
            quant_data = _parse_json_list(quant_response.content)

            # Phase 3: Score the opportunity
            scoring_input = (
                f"Asset: {name} ({ticker}, {asset_type})\n"
                f"Market regime: {market_regime.get('risk_level', 'balanced')} "
                f"(VIX {market_regime.get('vix', 'N/A')})\n\n"
                f"Research findings:\n{json.dumps(news_data, indent=2, default=str)}\n\n"
                f"Quantitative data:\n{json.dumps(quant_data, indent=2, default=str)}"
            )
            scoring_response = await scoring_llm.ainvoke([
                SystemMessage(content=_SCORING_PROMPT),
                HumanMessage(content=scoring_input),
            ])

            opportunity = _parse_opportunity(scoring_response.content, candidate)

            if opportunity:
                logger.info(
                    "AssetAnalyst scored %s: conviction=%d, R/R=%.2f, target=%s",
                    ticker, opportunity["conviction"], opportunity["risk_reward"], opportunity["target_price"]
                )
                return {
                    "opportunities": [opportunity],
                    "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: conviction={opportunity['conviction']}/10, R/R={opportunity['risk_reward']:.2f}")],
                    "errors": collected_errors,
                }
            else:
                logger.info("AssetAnalyst dropped %s: insufficient R/R or parse failure", ticker)
                return {
                    "opportunities": [],
                    "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: dropped (R/R < 2.0 or insufficient data)")],
                    "errors": collected_errors,
                }

        except Exception as e:
            logger.error("AssetAnalyst failed for %s: %s", ticker, e)
            return {
                "opportunities": [],
                "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: error — {e}")],
                "errors": collected_errors + [{
                    "agent": "asset_analyst", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return asset_analyst_node
