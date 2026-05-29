import logging

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.asset_analyst import create_asset_analyst_node
from agents.macro_scanner import create_macro_scanner_node
from agents.portfolio_constructor import portfolio_constructor_node
from agents.universe_scanner import create_universe_scanner_node
from evaluation.investment_judge import create_investment_judge_node
from graph.state import InvestmentState

logger = logging.getLogger(__name__)


def build_graph(researcher_tools: list[BaseTool], quantitative_tools: list[BaseTool]):
    graph = StateGraph(InvestmentState)

    graph.add_node("macro_scanner", create_macro_scanner_node(researcher_tools))
    graph.add_node("universe_scanner", create_universe_scanner_node(researcher_tools))
    graph.add_node("analyze_candidate", create_asset_analyst_node(researcher_tools, quantitative_tools))
    graph.add_node("portfolio_constructor", portfolio_constructor_node)
    graph.add_node("investment_judge", create_investment_judge_node())

    graph.add_edge(START, "macro_scanner")
    graph.add_edge("macro_scanner", "universe_scanner")

    def dispatch_candidates(state: InvestmentState):
        candidates = state.get("candidates", [])
        revision_count = state.get("revision_count", 0)
        # Use only candidates from the current revision round
        current = [c for c in candidates if c.get("revision", 0) == revision_count]
        if not current:
            # No candidates this round — skip directly to portfolio constructor
            return [Send("portfolio_constructor", state)]
        return [
            Send("analyze_candidate", {
                "candidate": candidate,
                "market_regime": state.get("market_regime", {}),
                "budget": state["budget"],
                "revision_count": revision_count,
                "opportunities": [],
                "errors": [],
                "messages": [],
            })
            for candidate in current
        ]

    graph.add_conditional_edges("universe_scanner", dispatch_candidates, ["analyze_candidate", "portfolio_constructor"])
    graph.add_edge("analyze_candidate", "portfolio_constructor")
    graph.add_edge("portfolio_constructor", "investment_judge")

    def route_after_judge(state: InvestmentState) -> str:
        if state.get("needs_revision") and state.get("revision_count", 0) < state.get("max_revisions", 2):
            return "universe_scanner"
        return END

    graph.add_conditional_edges("investment_judge", route_after_judge, ["universe_scanner", END])

    return graph.compile()
