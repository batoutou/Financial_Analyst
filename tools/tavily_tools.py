from langchain_tavily import TavilySearch


def get_tavily_search_tool() -> TavilySearch:
    """Return a Tavily search tool configured for financial news."""
    return TavilySearch(
        max_results=5,
        topic="news",
        search_depth="advanced",
    )
