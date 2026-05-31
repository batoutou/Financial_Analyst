from unittest.mock import MagicMock, patch

from langchain_core.tools import BaseTool


def _make_mock_tool():
    tool = MagicMock(spec=BaseTool)
    tool.name = "mock_tool"
    return tool


def _mock_llm():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    return llm


def _patched_build_graph(mock_tools):
    mock_llm = _mock_llm()
    with patch("agents.macro_scanner.ChatAnthropic", return_value=mock_llm), \
         patch("agents.universe_scanner.ChatAnthropic", return_value=mock_llm), \
         patch("agents.asset_analyst.ChatAnthropic", return_value=mock_llm), \
         patch("evaluation.investment_judge.ChatGoogleGenerativeAI", return_value=mock_llm):
        from graph.workflow import build_graph
        return build_graph(mock_tools, mock_tools)


def test_build_graph_compiles_without_error():
    mock_tools = [_make_mock_tool()]
    graph = _patched_build_graph(mock_tools)
    assert graph is not None


def test_build_graph_has_expected_nodes():
    mock_tools = [_make_mock_tool()]
    graph = _patched_build_graph(mock_tools)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"macro_scanner", "universe_scanner", "analyze_candidate",
                "portfolio_constructor", "investment_judge"}
    for node in expected:
        assert node in node_names, f"Missing node: {node}"
