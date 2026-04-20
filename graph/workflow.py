import logging
from typing import Literal

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst import analyst_node
from agents.quantitative import create_quantitative_node
from agents.researcher import create_researcher_node
from evaluation.judge import evaluator_node
from graph.state import FinancialAnalystState

logger = logging.getLogger(__name__)


def route_after_evaluation(state: FinancialAnalystState) -> Literal["researcher", "__end__"]:
    """Route after the evaluator: revise or finish.

    The evaluator (Gemini) sets needs_revision based on its verdict.
    We also enforce the revision cap to prevent infinite loops.
    """
    if state.get("needs_revision") and state.get("revision_count", 0) < state.get("max_revisions", 3):
        logger.info(
            "Evaluator requested revision %d/%d",
            state["revision_count"],
            state["max_revisions"],
        )
        return "researcher"
    return END


def build_graph(
    researcher_tools: list[BaseTool],
    quantitative_tools: list[BaseTool],
) -> CompiledStateGraph:
    """Build the financial analyst multi-agent graph.

    Architecture:
        START ──┬──> researcher ──┐
                └──> quantitative ─┤
                                   └──> analyst ──> evaluator ──[condition]──> END
                                                       │
                                                       └─── revise ──> researcher
    """
    researcher = create_researcher_node(researcher_tools)
    quantitative = create_quantitative_node(quantitative_tools)

    graph = StateGraph(FinancialAnalystState)

    # Add nodes
    graph.add_node("researcher", researcher)
    graph.add_node("quantitative", quantitative)
    graph.add_node("analyst", analyst_node)
    graph.add_node("evaluator", evaluator_node)

    # Parallel fan-out: START -> researcher + quantitative
    graph.add_edge(START, "researcher")
    graph.add_edge(START, "quantitative")

    # Fan-in: both feed into analyst (LangGraph waits for both automatically)
    graph.add_edge("researcher", "analyst")
    graph.add_edge("quantitative", "analyst")

    # Analyst produces report, then evaluator judges it
    graph.add_edge("analyst", "evaluator")

    # Evaluator decides: pass -> END, fail/needs_improvement -> researcher
    graph.add_conditional_edges("evaluator", route_after_evaluation)

    return graph.compile()
