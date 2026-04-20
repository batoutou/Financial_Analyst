import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import FinancialAnalystState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial research analyst. Your job is to gather
comprehensive news and qualitative information about a given company.

Instructions:
1. Search for recent news, earnings announcements, analyst opinions, and market sentiment
   about the company using the search tools available.
2. If a web scraping tool is available, use it to extract detailed content from relevant
   financial report URLs (SEC filings, earnings transcripts, analyst reports).
3. Return your findings as a structured JSON array where each item has:
   - "title": article/report title
   - "source": source name or URL
   - "date": publication date if available
   - "summary": 2-3 sentence summary of key points
   - "sentiment": "positive", "negative", or "neutral"

Focus on information that would be material to an investment decision.
Return ONLY the JSON array, no other text."""


def create_researcher_node(tools: list[BaseTool]):
    """Create the researcher agent node with the given tools."""

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def researcher_node(state: FinancialAnalystState) -> dict:
        company = state["company_name"]
        revision_feedback = state.get("revision_feedback", "")
        memory_context = state.get("memory_context", "")
        collected_errors: list[dict] = []

        user_msg = f"Research the company: {company}"
        if revision_feedback:
            user_msg += f"\n\nThe analyst requested additional research:\n{revision_feedback}"
        if memory_context:
            user_msg += f"\n\n{memory_context}"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            # Agentic tool-use loop (max 10 iterations to prevent runaway)
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
                        result, error = await retry_tool_call(tool, tool_call["args"], "researcher")
                        if error:
                            collected_errors.append(error)

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

            # Parse the final response into structured articles
            articles = _parse_articles(response.content)

            return {
                "news_articles": articles,
                "messages": [HumanMessage(content=f"[Researcher] Found {len(articles)} articles about {company}.")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("Researcher agent failed: %s", e)
            collected_errors.append({
                "agent": "researcher",
                "tool": "agent_loop",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "recoverable": False,
            })
            return {
                "news_articles": [],
                "messages": [HumanMessage(content=f"[Researcher] Error: {e}")],
                "errors": collected_errors,
            }

    return researcher_node


def _parse_articles(content: str | list) -> list[dict]:
    """Extract structured articles from the LLM response."""
    text = content if isinstance(content, str) else content[0].get("text", "") if content else ""
    text = text.strip()

    # Try to extract JSON array from the response
    if "[" in text:
        json_str = text[text.index("["):text.rindex("]") + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: return the raw text as a single article
    return [{"title": "Research Summary", "source": "LLM", "summary": text, "sentiment": "neutral"}]
