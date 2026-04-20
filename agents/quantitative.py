import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import FinancialAnalystState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative financial analyst. Your job is to extract
hard financial data about a given company using the financial data tools available.

Instructions:
1. Retrieve the company's latest financial statements: income statement, balance sheet,
   and cash flow statement.
2. Extract key financial ratios and metrics: P/E ratio, EV/EBITDA, debt-to-equity,
   current ratio, ROE, revenue growth, profit margins.
3. Get recent stock price data if available.

Return your findings as a structured JSON array where each item has:
- "metric": name of the metric or statement
- "category": one of "income_statement", "balance_sheet", "cash_flow", "ratio", "market_data"
- "value": the numeric value or data
- "period": the time period (e.g., "FY2024", "Q4 2024", "TTM")
- "unit": "USD", "percentage", "ratio", etc.

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

    async def quantitative_node(state: FinancialAnalystState) -> dict:
        company = state["company_name"]
        collected_errors: list[dict] = []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract financial data for: {company}"),
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
                "messages": [HumanMessage(content=f"[Quantitative] Extracted {len(financial_data)} data points for {company}.")],
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
