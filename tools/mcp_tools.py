import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _build_connections() -> dict:
    """Build MCP server connection configs from environment variables."""
    connections: dict = {
        "firecrawl": {
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "transport": "stdio",
            "env": {"FIRECRAWL_API_KEY": os.environ["FIRECRAWL_API_KEY"]},
        },
        "fmp": {
            "command": "npx",
            "args": ["-y", "aigroup-fmp-mcp"],
            "transport": "stdio",
            "env": {"FMP_API_KEY": os.environ["FMP_API_KEY"]},
        },
    }
    # AlphaVantage is optional — only added when the key is present
    if os.environ.get("ALPHA_VANTAGE_API_KEY"):
        connections["alphavantage"] = {
            "command": "npx",
            "args": ["-y", "@mcpdotdirect/mcp-server-alpha-vantage"],
            "transport": "stdio",
            "env": {"ALPHA_VANTAGE_API_KEY": os.environ["ALPHA_VANTAGE_API_KEY"]},
        }
    return connections


def create_mcp_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient with Firecrawl, FMP, and (optionally) AlphaVantage."""
    return MultiServerMCPClient(_build_connections(), tool_name_prefix=True)


async def get_firecrawl_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Get Firecrawl MCP tools for web scraping."""
    return await client.get_tools(server_name="firecrawl")


async def get_fmp_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Get Financial Modeling Prep MCP tools for financial data."""
    return await client.get_tools(server_name="fmp")


async def get_alphavantage_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Get AlphaVantage MCP tools for market data (RSI, quotes, earnings, forex).

    Returns empty list if ALPHA_VANTAGE_API_KEY is not set.
    """
    if not os.environ.get("ALPHA_VANTAGE_API_KEY"):
        return []
    return await client.get_tools(server_name="alphavantage")
