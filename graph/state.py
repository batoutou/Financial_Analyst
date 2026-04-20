import operator
from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class FinancialAnalystState(TypedDict):
    """Shared state for the financial analyst multi-agent workflow."""

    company_name: str

    # Agent communication via LangGraph message reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Accumulated data from specialized agents (append-only)
    financial_data: Annotated[list[dict], operator.add]
    news_articles: Annotated[list[dict], operator.add]

    # Analyst output
    analysis_report: str

    # Revision loop control
    revision_count: int
    max_revisions: int
    needs_revision: bool
    revision_feedback: str

    # Persistent memory: context from prior runs on the same company
    memory_context: str

    # LLM-as-judge evaluation results
    evaluation: dict

    # Structured error log (append-only across all agents)
    errors: Annotated[list[dict], operator.add]
