# Multi-Agent Financial Analyst

An autonomous multi-agent system that produces investment-grade financial analysis reports. The system coordinates specialized AI agents that research, extract data, synthesize findings, and independently evaluate output quality — all orchestrated as a parallel, self-correcting workflow.

## Why This Project

This project demonstrates production-level agentic AI architecture:

- **Multi-agent orchestration** with parallel execution and typed shared state
- **Cross-provider LLM evaluation** (Gemini judges Claude's output — no self-serving bias)
- **Model Context Protocol (MCP)** for standardized external tool integration
- **Self-correcting feedback loops** with bounded revision cycles
- **Persistent cross-run memory** for longitudinal analysis
- **Resilient design** with retries, structured error tracking, and graceful degradation

---

## Architecture

```
                    ┌─────────────────────┐
                    │       START         │
                    └────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │          PARALLEL EXECUTION
                ▼                         ▼
   ┌────────────────────┐   ┌────────────────────────┐
   │  Researcher Agent  │   │  Quantitative Agent    │
   │  (Claude Sonnet 4) │   │  (Claude Sonnet 4)     │
   │                    │   │                        │
   │  Tools:            │   │  Tools:                │
   │  • Tavily Search   │   │  • FMP MCP Server      │
   │  • Firecrawl MCP   │   │    (balance sheets,    │
   │    (web scraping)  │   │     ratios, prices)    │
   └─────────┬──────────┘   └───────────┬────────────┘
             │                           │
             └─────────────┬─────────────┘
                           │             FAN-IN (waits for both)
                           ▼
              ┌────────────────────────┐
              │     Analyst Agent      │
              │     (Claude Sonnet 4)  │
              │                        │
              │  Synthesizes all data   │
              │  into investment memo   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Evaluator (Judge)    │
              │   (Gemini 2.5 Flash)   │
              │                        │
              │  Scores: grounding,    │
              │  completeness,         │
              │  consistency,          │
              │  actionability         │
              └────────────┬───────────┘
                           │
                    ┌──────┴──────┐
                    │  VERDICT?   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         verdict: pass             verdict: fail
              │                         │
              ▼                         ▼
           ┌─────┐          ┌───────────────────┐
           │ END │          │ Back to Researcher │
           └─────┘          │ (with feedback)    │
                            └───────────────────┘
                                max 3 cycles
```

---

## How It Works

### 1. Parallel Agent Execution

The Researcher and Quantitative agents run **simultaneously** via LangGraph's fan-out mechanism. When two edges leave the same node, LangGraph schedules both as concurrent async tasks:

```python
graph.add_edge(START, "researcher")      # both edges from START
graph.add_edge(START, "quantitative")    # → parallel execution

graph.add_edge("researcher", "analyst")  # fan-in: analyst waits
graph.add_edge("quantitative", "analyst")# for BOTH to complete
```

This cuts total latency nearly in half compared to sequential execution.

### 2. Agent Instantiation (Closure Pattern)

Each agent is created by a factory function that captures its tools in a closure:

```python
def create_researcher_node(tools: list[BaseTool]):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def researcher_node(state: FinancialAnalystState) -> dict:
        # ReAct loop: call LLM → execute tools → feed results back → repeat
        ...
        return {"news_articles": articles, "errors": collected_errors}

    return researcher_node
```

Tools are bound once at graph-build time. The node signature stays clean (`state → partial_state`) as LangGraph requires.

### 3. Shared State with Typed Reducers

Agents communicate exclusively through a shared `TypedDict` state:

```python
class FinancialAnalystState(TypedDict):
    financial_data: Annotated[list[dict], operator.add]   # append-only
    news_articles: Annotated[list[dict], operator.add]    # append-only
    errors: Annotated[list[dict], operator.add]           # append-only
    analysis_report: str                                   # last-write-wins
    revision_count: int
    memory_context: str                                    # from prior runs
    evaluation: dict                                       # judge scores
```

The `Annotated[list, operator.add]` reducer means each agent **appends** to the list rather than overwriting. When both parallel agents complete, their results merge automatically.

### 4. LLM-as-Judge Evaluation (Cross-Provider)

After the Analyst produces a report, an **independent model from a different provider** evaluates it against the raw source data:

```python
# Evaluator uses Gemini (Google) to judge Claude's (Anthropic) output
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
```

**Why a different provider?** If Claude evaluated its own output, it could rate itself favorably. Using Gemini eliminates self-serving bias — it has no knowledge of the analyst's reasoning process and can only judge the report against the evidence.

The evaluation uses a **structured rubric with few-shot calibration**:

| Dimension | What it measures |
|-----------|-----------------|
| **Grounding** (1-5) | Are claims supported by the source data provided? |
| **Completeness** (1-5) | Does the report cover all required sections? |
| **Consistency** (1-5) | Do numbers and facts match across sections and sources? |
| **Actionability** (1-5) | Would an investor find this useful for a decision? |

Two few-shot examples calibrate the judge: a strong report (4.25/5, pass) and a poor report (1.25/5, fail) — anchoring what "good" and "bad" look like.

### 5. Self-Correcting Revision Loop

If the evaluator returns `verdict: "fail"` or `"needs_improvement"`, the graph routes back to the Researcher with specific feedback:

```
Evaluation failed (score: 2.3/5). Issues: No financial data referenced;
Risk section ignores available evidence. Please gather more data.
```

The loop is **bounded** (`max_revisions`, default 3) to prevent infinite cycles.

### 6. Persistent Cross-Run Memory

The system remembers prior analyses. Run Apple today and again next month — the second run references the first:

```
## Prior Analysis History for Apple
### Analysis from 2025-01-15
**Summary**: Record Q4 revenue of $89.5B...
**Key metrics**: Revenue: $89.5B, P/E: 28.5
**Flags/Risks**: China revenue declined 2%
**Quality score**: 4.25/5
```

This enables trend comparisons ("Revenue increased from $89.5B to $94.9B since our last analysis") and tracks which risks persist across time.

Storage: `~/.financial-analyst/memory.json` — human-readable, inspectable.

### 7. MCP Tool Integration

External data sources are connected via **Model Context Protocol** — a standardized interface for LLM tool servers:

```python
client = MultiServerMCPClient({
    "firecrawl": {
        "command": "npx",
        "args": ["-y", "firecrawl-mcp"],
        "transport": "stdio",
    },
    "fmp": {
        "command": "npx",
        "args": ["-y", "aigroup-fmp-mcp"],
        "transport": "stdio",
    },
})
```

MCP servers run as **child processes** communicating over stdin/stdout (JSON-RPC). The `MultiServerMCPClient` converts their tools into standard LangChain `BaseTool` objects. The agents don't know they're talking to MCP — they just see tools with names and schemas.

**Why MCP?** It decouples tool implementation from agent code. Swapping Firecrawl for a different scraper requires zero agent changes — just a config update.

### 8. Retry with Exponential Backoff

Every external tool call is wrapped in a resilient retry layer:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),   # 1s, 2s, 4s
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
)
async def _invoke_with_retry(tool, args):
    return await tool.ainvoke(args)
```

Transient failures are retried silently. Permanent failures produce a structured error record that propagates through the state — the Analyst acknowledges gaps, and the Evaluator adjusts scoring expectations accordingly.

### 9. Structured Error Tracking

Errors are not lost strings — they're an append-only log:

```python
{
    "agent": "researcher",
    "tool": "firecrawl_scrape",
    "error_type": "ConnectionError",
    "message": "Connection refused after 3 retries",
    "timestamp": "2025-01-15T10:30:00+00:00",
    "recoverable": True
}
```

Downstream agents read the error log to account for missing data. The final output includes a categorized error summary.

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js (for MCP servers via `npx`)

### Installation

```bash
cd financial-analyst-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

| Variable | Purpose | Provider |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Agent LLM (Claude Sonnet 4) | [console.anthropic.com](https://console.anthropic.com) |
| `GOOGLE_API_KEY` | Evaluator LLM (Gemini 2.5 Flash Lite) | [aistudio.google.com](https://aistudio.google.com) |
| `TAVILY_API_KEY` | Web search | [app.tavily.com](https://app.tavily.com) |
| `FIRECRAWL_API_KEY` | Web scraping via MCP | [firecrawl.dev](https://firecrawl.dev) |
| `FMP_API_KEY` | Financial data via MCP | [financialmodelingprep.com](https://financialmodelingprep.com) |
| `LANGSMITH_API_KEY` | Tracing & observability | [smith.langchain.com](https://smith.langchain.com) |

### Run

```bash
python main.py --company "Apple"
python main.py --company "Tesla" --max-revisions 2
```

Reports are saved to `reports/<company>_<date>.md`.

---

## Project Structure

```
financial-analyst-agents/
├── main.py                    # Entry point, orchestration, report output
├── agents/
│   ├── researcher.py          # News + web scraping (Tavily, Firecrawl MCP)
│   ├── quantitative.py        # Financial data extraction (FMP MCP)
│   └── analyst.py             # Report synthesis with memory awareness
├── evaluation/
│   └── judge.py               # Gemini LLM-as-Judge (rubric + few-shot)
├── graph/
│   ├── state.py               # TypedDict state with Annotated reducers
│   └── workflow.py            # StateGraph wiring + conditional routing
├── memory/
│   └── store.py               # Persistent JSON memory (cross-run)
├── tools/
│   ├── tavily_tools.py        # Tavily search config
│   ├── mcp_tools.py           # MCP client (Firecrawl + FMP)
│   └── retry.py               # Tenacity exponential backoff
├── reports/                   # Generated analysis reports (gitignored)
├── mcp_config.json
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Technology | Why |
|------------|-----|
| **LangGraph** | DAG-based agent orchestration with native parallelism, typed state, and conditional routing — more control than simple chains |
| **Claude Sonnet 4** | High-quality reasoning for financial analysis with strong tool-use capabilities |
| **Gemini 2.5 Flash Lite** | Fast, cheap evaluation from a different provider — eliminates self-serving bias in quality assessment |
| **MCP** | Standardized tool protocol — decouple tool servers from agent logic, swap implementations without code changes |
| **LangSmith** | Zero-code observability: every node, tool call, and token traced automatically via env vars |
| **Tavily** | Purpose-built search API for AI agents with relevance scoring |
| **Firecrawl** | Reliable web scraping that handles JS-rendered pages (SEC filings, earnings reports) |
| **Financial Modeling Prep** | Structured financial data API (statements, ratios, market data) |
| **Pydantic** | Typed structured output — forces LLM responses into validated schemas, no fragile parsing |
| **Tenacity** | Battle-tested retry library with exponential backoff for transient API failures |
