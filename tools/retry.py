import logging
from datetime import datetime, timezone

from langchain_core.tools import BaseTool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Transient exceptions worth retrying
_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _invoke_with_retry(tool: BaseTool, args: dict) -> str:
    """Invoke a tool with retries on transient failures."""
    return await tool.ainvoke(args)


async def retry_tool_call(
    tool: BaseTool,
    args: dict,
    agent_name: str,
) -> tuple[str, dict | None]:
    """Call a tool with retry logic. Returns (result, error_record_or_None).

    On success: returns (result_string, None)
    On permanent failure: returns (error_message, structured_error_dict)
    """
    try:
        result = await _invoke_with_retry(tool, args)
        return str(result), None
    except _TRANSIENT_EXCEPTIONS as e:
        error_record = _build_error(agent_name, tool.name, "transient", str(e), recoverable=True)
        logger.warning("Tool %s failed after retries: %s", tool.name, e)
        return f"Tool error after retries: {e}", error_record
    except Exception as e:
        error_record = _build_error(agent_name, tool.name, type(e).__name__, str(e), recoverable=False)
        logger.error("Tool %s failed permanently: %s", tool.name, e)
        return f"Tool error: {e}", error_record


def _build_error(
    agent: str,
    tool: str,
    error_type: str,
    message: str,
    recoverable: bool,
) -> dict:
    return {
        "agent": agent,
        "tool": tool,
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recoverable": recoverable,
    }
