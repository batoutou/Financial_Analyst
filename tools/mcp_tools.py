import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _build_connections() -> dict:
    """Build MCP server connection configs from environment variables."""
    return {
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


def create_mcp_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient with Firecrawl and FMP servers.

    Usage:
        client = create_mcp_client()
        firecrawl_tools = await client.get_tools(server_name="firecrawl")
        fmp_tools = await client.get_tools(server_name="fmp")
    """
    return MultiServerMCPClient(_build_connections(), tool_name_prefix=True)


async def get_firecrawl_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Get Firecrawl MCP tools for web scraping."""
    return await client.get_tools(server_name="firecrawl")


async def get_fmp_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Get Financial Modeling Prep MCP tools for financial data."""
    return await client.get_tools(server_name="fmp")
