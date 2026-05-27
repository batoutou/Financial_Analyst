# Investment Opportunities Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the financial analyst agent into an autonomous market scanner that identifies investment opportunities across stocks, ETFs, bonds, and crypto, producing a portfolio allocation with entry/target/stop prices and % allocation per asset.

**Architecture:** 5-phase LangGraph pipeline — MacroScanner (regime detection) → UniverseScanner (~15 candidates) → parallel AssetAnalyst per candidate via Send API (research + quant + scoring) → PortfolioConstructor (conviction-weighted allocation) → InvestmentJudge (Gemini validation with revision loop). Graph files are rebuilt from scratch (they were deleted).

**Tech Stack:** LangGraph, LangChain Anthropic/Google, FMP MCP (broker consensus + ratios), Tavily (search), pytest + pytest-asyncio

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `graph/__init__.py` | Rebuild | Exports `build_graph` |
| `graph/state.py` | Rebuild | `InvestmentState` TypedDict with reducers |
| `graph/workflow.py` | Rebuild | LangGraph wiring with Send fan-out |
| `agents/macro_scanner.py` | New | Market regime detection (VIX, yield curve, trend) |
| `agents/universe_scanner.py` | New | Autonomous candidate discovery (~15 assets) |
| `agents/asset_analyst.py` | New | Per-candidate: research + quant + conviction scoring |
| `agents/portfolio_constructor.py` | New | Conviction-weighted allocation engine |
| `evaluation/investment_judge.py` | New | Portfolio validation (Gemini LLM-as-judge) |
| `agents/researcher.py` | Modify | Add asset-type context to prompts |
| `agents/quantitative.py` | Modify | Add broker consensus data fetching |
| `main.py` | Modify | `--budget` CLI, portfolio report format |
| `requirements.txt` | Modify | Add pytest, pytest-asyncio |
| `tests/__init__.py` | New | Test package |
| `tests/test_state.py` | New | InvestmentState reducer tests |
| `tests/test_macro_scanner.py` | New | Regime parsing tests |
| `tests/test_universe_scanner.py` | New | Candidate parsing tests |
| `tests/test_asset_analyst.py` | New | Opportunity parsing + R/R enforcement |
| `tests/test_portfolio_constructor.py` | New | Allocation logic tests |
| `tests/test_investment_judge.py` | New | Portfolio validation logic tests |
| `tests/test_workflow.py` | New | Graph builds + routes correctly |

---

### Task 1: Test infrastructure + graph/state.py

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `graph/__init__.py`
- Create: `graph/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Add test dependencies to requirements.txt**

Append these two lines to `requirements.txt`:
```
pytest>=8.0
pytest-asyncio>=0.24
```

- [ ] **Step 2: Install dependencies**

Run: `.venv/bin/pip install pytest pytest-asyncio`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` as an empty file.

Create `tests/test_state.py`:
```python
import operator

from graph.state import InvestmentState


def test_opportunities_reducer_accumulates():
    existing = [{"ticker": "AAPL"}]
    update = [{"ticker": "MSFT"}]
    result = operator.add(existing, update)
    assert len(result) == 2
    assert result[0]["ticker"] == "AAPL"
    assert result[1]["ticker"] == "MSFT"


def test_investment_state_has_required_fields():
    fields = InvestmentState.__annotations__
    required = [
        "budget", "market_regime", "candidates", "opportunities",
        "portfolio", "evaluation", "revision_count", "max_revisions",
        "needs_revision", "revision_feedback", "messages", "errors",
    ]
    for field in required:
        assert field in fields, f"Missing field: {field}"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 5: Create graph/__init__.py (stub — will be updated in Task 9)**

Create `graph/__init__.py`:
```python
```
(empty file)

- [ ] **Step 6: Create graph/state.py**

```python
import operator
from typing import Annotated

from typing_extensions import TypedDict


class InvestmentState(TypedDict):
    # Input
    budget: float

    # Phase 1: MacroScanner output
    market_regime: dict  # {vix, yield_curve_spread, spx_above_200ma, risk_level, summary}

    # Phase 2: UniverseScanner output — accumulates across revision rounds, tagged with revision number
    candidates: Annotated[list[dict], operator.add]

    # Phase 3: AssetAnalyst fan-out results — accumulates across revision rounds
    opportunities: Annotated[list[dict], operator.add]

    # Phase 4: PortfolioConstructor output
    portfolio: list[dict]

    # Eval / revision
    evaluation: dict
    revision_count: int
    max_revisions: int
    needs_revision: bool
    revision_feedback: str

    # Infra
    messages: Annotated[list, operator.add]
    errors: Annotated[list[dict], operator.add]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_state.py -v`
Expected: 2 PASSED

- [ ] **Step 8: Commit**

```bash
git add requirements.txt tests/__init__.py graph/__init__.py graph/state.py tests/test_state.py
git commit -m "feat: add InvestmentState and test infrastructure"
```

---

### Task 2: agents/macro_scanner.py

**Files:**
- Create: `agents/macro_scanner.py`
- Create: `tests/test_macro_scanner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_macro_scanner.py`:
```python
import json

from agents.macro_scanner import _parse_regime


def test_parse_regime_aggressive():
    raw = json.dumps({
        "vix": 12.5,
        "yield_curve_spread": 0.8,
        "spx_above_200ma": True,
        "risk_level": "aggressive",
        "summary": "Low volatility, bull market.",
    })
    regime = _parse_regime(raw)
    assert regime["risk_level"] == "aggressive"
    assert regime["vix"] == 12.5
    assert regime["spx_above_200ma"] is True


def test_parse_regime_defensive():
    raw = json.dumps({
        "vix": 28.0,
        "yield_curve_spread": -0.5,
        "spx_above_200ma": False,
        "risk_level": "defensive",
        "summary": "High volatility, bear market.",
    })
    regime = _parse_regime(raw)
    assert regime["risk_level"] == "defensive"


def test_parse_regime_falls_back_on_invalid_json():
    regime = _parse_regime("not valid json at all")
    assert regime["risk_level"] == "balanced"
    assert "vix" in regime


def test_parse_regime_extracts_json_from_text():
    raw = 'Here is the regime: {"vix": 18.0, "yield_curve_spread": 0.1, "spx_above_200ma": true, "risk_level": "balanced", "summary": "Moderate market."}'
    regime = _parse_regime(raw)
    assert regime["risk_level"] == "balanced"
    assert regime["vix"] == 18.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_macro_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.macro_scanner'`

- [ ] **Step 3: Create agents/macro_scanner.py**

```python
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import InvestmentState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a macro financial analyst. Assess the current market regime.

Search for exactly these three data points:
1. Current VIX index level (fear gauge)
2. US 10-year minus 2-year Treasury yield spread (positive = normal, negative = inverted)
3. Whether the S&P 500 is trading above or below its 200-day moving average

Return ONLY this JSON object (no other text):
{
    "vix": <float>,
    "yield_curve_spread": <float>,
    "spx_above_200ma": <bool>,
    "risk_level": <"aggressive" | "balanced" | "defensive">,
    "summary": "<1-2 sentence market description>"
}

Risk level rules:
- "aggressive": VIX < 15 AND yield_curve_spread > 0 AND spx_above_200ma = true
- "defensive": VIX > 25 OR spx_above_200ma = false
- "balanced": everything else"""

_FALLBACK_REGIME = {
    "vix": 20.0,
    "yield_curve_spread": 0.0,
    "spx_above_200ma": True,
    "risk_level": "balanced",
    "summary": "Could not determine market regime. Defaulting to balanced.",
}


def _parse_regime(content: str | list) -> dict:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            required = ["vix", "yield_curve_spread", "spx_above_200ma", "risk_level", "summary"]
            if all(k in data for k in required):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return dict(_FALLBACK_REGIME)


def create_macro_scanner_node(tools: list[BaseTool]):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=2048)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def macro_scanner_node(state: InvestmentState) -> dict:
        collected_errors: list[dict] = []
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                "Search for: current VIX level, US 10y-2y yield spread, "
                "and whether S&P 500 is above its 200-day moving average."
            )),
        ]

        try:
            for _ in range(6):
                response: AIMessage = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    tool = tools_by_name.get(tool_call["name"])
                    if tool is None:
                        result = f"Tool '{tool_call['name']}' not found."
                    else:
                        result, error = await retry_tool_call(tool, tool_call["args"], "macro_scanner")
                        if error:
                            collected_errors.append(error)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            regime = _parse_regime(response.content)
            logger.info("Market regime: %s (VIX %.1f)", regime["risk_level"], regime.get("vix", 0))

            return {
                "market_regime": regime,
                "messages": [HumanMessage(content=f"[MacroScanner] Regime: {regime['risk_level']} — {regime['summary']}")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("MacroScanner failed: %s", e)
            return {
                "market_regime": dict(_FALLBACK_REGIME),
                "messages": [HumanMessage(content=f"[MacroScanner] Error: {e}. Using balanced defaults.")],
                "errors": collected_errors + [{
                    "agent": "macro_scanner", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": True,
                }],
            }

    return macro_scanner_node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_macro_scanner.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/macro_scanner.py tests/test_macro_scanner.py
git commit -m "feat: add MacroScanner agent for market regime detection"
```

---

### Task 3: agents/universe_scanner.py

**Files:**
- Create: `agents/universe_scanner.py`
- Create: `tests/test_universe_scanner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_scanner.py`:
```python
import json

from agents.universe_scanner import _parse_candidates


def test_parse_candidates_valid():
    raw = json.dumps([
        {"ticker": "NVDA", "name": "NVIDIA", "asset_type": "stock", "exchange": "NASDAQ", "rationale": "AI demand"},
        {"ticker": "IWDA", "name": "iShares MSCI World", "asset_type": "etf", "exchange": "LSE", "rationale": "Diversification"},
        {"ticker": "BTC-USD", "name": "Bitcoin", "asset_type": "crypto", "exchange": "Crypto", "rationale": "Digital gold"},
    ])
    candidates = _parse_candidates(raw)
    assert len(candidates) == 3
    assert candidates[0]["ticker"] == "NVDA"


def test_parse_candidates_filters_incomplete_items():
    raw = json.dumps([
        {"ticker": "NVDA", "name": "NVIDIA", "asset_type": "stock"},  # valid
        {"name": "No ticker"},  # invalid — missing ticker
        {"ticker": "AAPL"},  # invalid — missing name and asset_type
    ])
    candidates = _parse_candidates(raw)
    assert len(candidates) == 1
    assert candidates[0]["ticker"] == "NVDA"


def test_parse_candidates_returns_empty_on_invalid_json():
    candidates = _parse_candidates("not json")
    assert candidates == []


def test_parse_candidates_extracts_json_array_from_text():
    raw = 'Here are the candidates: [{"ticker": "AAPL", "name": "Apple", "asset_type": "stock", "exchange": "NASDAQ", "rationale": "Quality"}]'
    candidates = _parse_candidates(raw)
    assert len(candidates) == 1
    assert candidates[0]["ticker"] == "AAPL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_universe_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.universe_scanner'`

- [ ] **Step 3: Create agents/universe_scanner.py**

```python
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from graph.state import InvestmentState
from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a market analyst scanning global public markets for investment opportunities.

Based on the provided market regime, identify approximately 15 investment candidates across:
- ~4 US stocks (NYSE/NASDAQ) showing momentum, oversold value, or strong earnings catalysts
- ~3 European stocks (Euronext, LSE, Xetra) with analyst upgrades or sector tailwinds
- ~3 global ETFs (MSCI World, sector ETFs, thematic) with strong trend or attractive entry
- ~3 crypto assets (BTC, ETH, or major tokens) with technical setups or on-chain catalysts
- ~2 bond ETFs (TLT, BND, BUND equivalent) attractive given the current rate environment

Use search tools to find current market opportunities, analyst recommendations, and trending assets.

Return ONLY a JSON array where each item has:
{
    "ticker": "<exchange ticker symbol>",
    "name": "<full asset name>",
    "asset_type": "<stock | etf | bond | crypto>",
    "exchange": "<NASDAQ | NYSE | Euronext | LSE | Xetra | Crypto>",
    "rationale": "<one sentence: why this asset is interesting right now>"
}"""


def _parse_candidates(content: str | list) -> list[dict]:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "[" in text:
        try:
            json_str = text[text.index("["):text.rindex("]") + 1]
            data = json.loads(json_str)
            return [
                item for item in data
                if all(k in item for k in ["ticker", "name", "asset_type"])
            ]
        except (json.JSONDecodeError, ValueError):
            pass

    return []


def create_universe_scanner_node(tools: list[BaseTool]):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def universe_scanner_node(state: InvestmentState) -> dict:
        regime = state.get("market_regime", {})
        revision_count = state.get("revision_count", 0)
        revision_feedback = state.get("revision_feedback", "")
        collected_errors: list[dict] = []

        user_msg = (
            f"Market regime: {regime.get('risk_level', 'balanced')} "
            f"(VIX {regime.get('vix', 'N/A')}, {regime.get('summary', '')})\n\n"
            "Find the best investment opportunities across US stocks, EU stocks, "
            "global ETFs, crypto, and bond ETFs right now."
        )
        if revision_feedback:
            user_msg += (
                f"\n\nPrevious portfolio was rejected. Feedback: {revision_feedback}\n"
                "Find DIFFERENT assets that address these issues."
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
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
                        result, error = await retry_tool_call(tool, tool_call["args"], "universe_scanner")
                        if error:
                            collected_errors.append(error)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            raw_candidates = _parse_candidates(response.content)
            # Tag each candidate with the current revision round for deduplication downstream
            candidates = [{**c, "revision": revision_count} for c in raw_candidates]
            logger.info("UniverseScanner found %d candidates (revision %d)", len(candidates), revision_count)

            return {
                "candidates": candidates,
                "messages": [HumanMessage(content=f"[UniverseScanner] Found {len(candidates)} candidates.")],
                "errors": collected_errors,
            }

        except Exception as e:
            logger.error("UniverseScanner failed: %s", e)
            return {
                "candidates": [],
                "messages": [HumanMessage(content=f"[UniverseScanner] Error: {e}")],
                "errors": collected_errors + [{
                    "agent": "universe_scanner", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return universe_scanner_node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_universe_scanner.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/universe_scanner.py tests/test_universe_scanner.py
git commit -m "feat: add UniverseScanner agent for autonomous candidate discovery"
```

---

### Task 4: Modify agents/researcher.py

Make the researcher asset-type aware so it fetches relevant data per asset class.

**Files:**
- Modify: `agents/researcher.py`

- [ ] **Step 1: Update the SYSTEM_PROMPT in agents/researcher.py**

Replace the existing `SYSTEM_PROMPT` constant (lines 13–29) with:

```python
SYSTEM_PROMPT = """You are a senior financial research analyst. Gather comprehensive news and
qualitative information about a given asset.

Instructions:
1. Search for recent news, earnings announcements, analyst opinions, and market sentiment.
2. Adapt your focus to the asset type:
   - stock: earnings results, revenue growth, analyst upgrades/downgrades, insider activity, catalysts
   - etf: underlying theme performance, fund flows, sector rotation signals
   - bond: interest rate environment, Fed policy signals, yield curve positioning
   - crypto: exchange flows (buy/sell pressure), on-chain metrics, regulatory news, whale activity
3. If a web scraping tool is available, extract detailed content from relevant URLs.
4. Return your findings as a structured JSON array where each item has:
   - "title": article/report title
   - "source": source name or URL
   - "date": publication date if available
   - "summary": 2-3 sentence summary of key points
   - "sentiment": "positive", "negative", or "neutral"

Return ONLY the JSON array, no other text."""
```

- [ ] **Step 2: Update the user message in researcher_node to include asset_type**

In the `researcher_node` function, replace:
```python
user_msg = f"Research the company: {company}"
```
With:
```python
candidate = state.get("candidate", {})
asset_type = candidate.get("asset_type", "stock")
ticker = candidate.get("ticker", state.get("company_name", ""))
name = candidate.get("name", ticker)
user_msg = f"Research {asset_type}: {name} (ticker: {ticker})"
```

Also update the references to `company` further down in the function:
```python
# Replace:
user_msg += f"\n\nThe analyst requested additional research:\n{revision_feedback}"
# With (same — no change needed here):
user_msg += f"\n\nThe analyst requested additional research:\n{revision_feedback}"

# Replace the return statement's message:
# From:
"messages": [HumanMessage(content=f"[Researcher] Found {len(articles)} articles about {company}.")],
# To:
"messages": [HumanMessage(content=f"[Researcher] Found {len(articles)} articles about {name}.")],
```

- [ ] **Step 3: Verify existing tests still pass (no tests for researcher yet — smoke check)**

Run: `.venv/bin/pytest tests/ -v`
Expected: All prior tests still PASSED

- [ ] **Step 4: Commit**

```bash
git add agents/researcher.py
git commit -m "feat: make researcher agent asset-type aware"
```

---

### Task 5: Modify agents/quantitative.py

Add broker consensus data fetching (analyst price targets per ticker).

**Files:**
- Modify: `agents/quantitative.py`

- [ ] **Step 1: Replace the SYSTEM_PROMPT in agents/quantitative.py**

Replace the existing `SYSTEM_PROMPT` constant with:

```python
SYSTEM_PROMPT = """You are a quantitative financial analyst. Extract hard financial data for a given asset.

Instructions vary by asset type:

For stocks and ETFs:
1. Get analyst price targets from multiple brokers: use the price-target endpoint for the ticker.
   For each analyst entry, extract: broker name, analyst name, rating, target price, date, price on date.
2. Get key financial ratios: P/E, EV/EBITDA, P/B, ROE, debt-to-equity, revenue growth, gross margin.
3. Get current stock quote (latest price).
4. Get RSI (14-period daily) to assess momentum.

For crypto:
1. Get current price and 30-day price change.
2. Get RSI if available.

For bonds:
1. Get current yield and duration from the ETF data.
2. Get recent price and 52-week range.

Return your findings as a structured JSON array where each item has:
{
    "metric": "<name>",
    "category": "<broker_consensus | income_statement | balance_sheet | ratio | market_data | technical>",
    "value": <numeric value or nested object>,
    "period": "<FY2024 | TTM | current | etc.>",
    "unit": "<USD | percentage | ratio | count>"
}

For broker_consensus, use value as a nested object:
{
    "metric": "broker_consensus",
    "category": "broker_consensus",
    "value": {
        "mean_target": <float>,
        "buy_count": <int>,
        "hold_count": <int>,
        "sell_count": <int>,
        "brokers": [
            {"broker": str, "analyst": str, "rating": str, "target_price": float,
             "rating_date": str, "price_on_rating_date": float, "implied_return_pct": float}
        ]
    },
    "period": "current",
    "unit": "USD"
}

Return ONLY the JSON array, no other text."""
```

- [ ] **Step 2: Update the user message in quantitative_node to use candidate context**

In the `quantitative_node` function, replace:
```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"Extract financial data for: {company}"),
]
```
With:
```python
candidate = state.get("candidate", {})
ticker = candidate.get("ticker", state.get("company_name", ""))
name = candidate.get("name", ticker)
asset_type = candidate.get("asset_type", "stock")
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"Extract financial data for {asset_type}: {name} (ticker: {ticker})"),
]
```

And update the return message:
```python
# From:
"messages": [HumanMessage(content=f"[Quantitative] Extracted {len(financial_data)} data points for {company}.")],
# To:
"messages": [HumanMessage(content=f"[Quantitative] Extracted {len(financial_data)} data points for {name}.")],
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `.venv/bin/pytest tests/ -v`
Expected: All prior tests PASSED

- [ ] **Step 4: Commit**

```bash
git add agents/quantitative.py
git commit -m "feat: add broker consensus fetching to quantitative agent"
```

---

### Task 6: agents/asset_analyst.py

The per-candidate combined analysis node. Receives one candidate via the Send API, runs research + quantitative phases internally, then scores the opportunity.

**Files:**
- Create: `agents/asset_analyst.py`
- Create: `tests/test_asset_analyst.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_asset_analyst.py`:
```python
import json

from agents.asset_analyst import _parse_opportunity


_CANDIDATE = {
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "asset_type": "stock",
    "exchange": "NASDAQ",
    "rationale": "AI demand",
    "revision": 0,
}


def test_parse_opportunity_valid():
    raw = json.dumps({
        "entry_price": 118.50,
        "target_price": 145.00,
        "stop_loss": 108.00,
        "conviction": 8,
        "horizon": "medium-term",
        "implied_return_pct": 22.4,
        "risk_reward": 2.75,
        "broker_consensus": {
            "mean_target": 145.00,
            "buy_count": 4,
            "hold_count": 1,
            "sell_count": 0,
            "brokers": [],
        },
        "signals": {"rsi": 38, "pe_ratio": 28.5},
        "rationale": "Strong AI tailwinds with oversold RSI.",
    })
    opp = _parse_opportunity(raw, _CANDIDATE)
    assert opp is not None
    assert opp["ticker"] == "NVDA"
    assert opp["entry_price"] == 118.50
    assert opp["conviction"] == 8


def test_parse_opportunity_rejects_low_rr():
    raw = json.dumps({
        "entry_price": 100.0,
        "target_price": 108.0,
        "stop_loss": 95.0,
        "conviction": 7,
        "horizon": "short-term",
        "implied_return_pct": 8.0,
        "risk_reward": 1.6,  # below 2.0 — should be rejected
        "broker_consensus": {"mean_target": 108.0, "buy_count": 2, "hold_count": 1, "sell_count": 0, "brokers": []},
        "signals": {},
        "rationale": "Weak setup.",
    })
    opp = _parse_opportunity(raw, _CANDIDATE)
    assert opp is None


def test_parse_opportunity_returns_none_on_missing_fields():
    raw = json.dumps({"entry_price": 100.0})  # missing required fields
    opp = _parse_opportunity(raw, _CANDIDATE)
    assert opp is None


def test_parse_opportunity_returns_none_on_invalid_json():
    opp = _parse_opportunity("not json", _CANDIDATE)
    assert opp is None


def test_parse_opportunity_merges_candidate_fields():
    raw = json.dumps({
        "entry_price": 118.50,
        "target_price": 145.00,
        "stop_loss": 108.00,
        "conviction": 8,
        "horizon": "medium-term",
        "implied_return_pct": 22.4,
        "risk_reward": 2.75,
        "broker_consensus": {"mean_target": 145.0, "buy_count": 3, "hold_count": 0, "sell_count": 0, "brokers": []},
        "signals": {},
        "rationale": "Good setup.",
    })
    opp = _parse_opportunity(raw, _CANDIDATE)
    assert opp["asset_type"] == "stock"
    assert opp["exchange"] == "NASDAQ"
    assert opp["revision"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_asset_analyst.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.asset_analyst'`

- [ ] **Step 3: Create agents/asset_analyst.py**

```python
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from tools.retry import retry_tool_call

logger = logging.getLogger(__name__)

_RESEARCH_PROMPT = """You are a financial research analyst. Research the following asset for a
potential investment opportunity. Find recent news, analyst coverage, earnings, and market sentiment.
Focus on what is material to an investment decision right now.

Return findings as a JSON array: [{"title": str, "source": str, "date": str, "summary": str, "sentiment": str}]
Return ONLY the JSON array."""

_QUANT_PROMPT = """You are a quantitative analyst. For the given asset, fetch:
1. Analyst price targets from all brokers (use price-target endpoint)
2. Key financial ratios (P/E, EV/EBITDA, ROE, revenue growth, gross margin, debt/equity)
3. Current stock quote / price
4. RSI (14-day)

Return findings as a JSON array of metric objects. For broker targets, use category "broker_consensus"
with value as a nested object containing mean_target, buy_count, hold_count, sell_count, and brokers array.
Return ONLY the JSON array."""

_SCORING_PROMPT = """You are a senior investment analyst. Given the research and quantitative data
for an asset, produce a structured investment opportunity assessment.

Rules:
- entry_price: use the current market price (or slightly below if RSI < 35)
- target_price: anchor to broker consensus mean target; adjust to nearest technical resistance
- stop_loss: place below the nearest key support level
- risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
- implied_return_pct = round((target_price - entry_price) / entry_price * 100, 1)
- conviction: 1-10 score based on: broker consensus (40%), valuation vs sector (30%), momentum/RSI (20%), news sentiment (10%)
- horizon: "short-term" (days-weeks, RSI reversal), "medium-term" (1-6 months, earnings cycle), "long-term" (6m+, DCF/dividend)

Return ONLY this JSON object:
{
    "entry_price": <float>,
    "target_price": <float>,
    "stop_loss": <float>,
    "conviction": <int 1-10>,
    "horizon": <"short-term" | "medium-term" | "long-term">,
    "implied_return_pct": <float>,
    "risk_reward": <float>,
    "broker_consensus": {
        "mean_target": <float>,
        "buy_count": <int>,
        "hold_count": <int>,
        "sell_count": <int>,
        "brokers": [{"broker": str, "analyst": str, "rating": str, "target_price": float, "rating_date": str, "price_on_rating_date": float, "implied_return_pct": float}]
    },
    "signals": {"rsi": <float or null>, "pe_ratio": <float or null>, "revenue_growth": <float or null>},
    "rationale": "<2-3 sentences explaining the investment thesis>"
}

If risk_reward < 2.0, still return the JSON but set conviction to 0."""


def _parse_opportunity(content: str | list, candidate: dict) -> dict | None:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()

    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            required = [
                "entry_price", "target_price", "stop_loss", "conviction",
                "horizon", "implied_return_pct", "risk_reward",
                "broker_consensus", "signals", "rationale",
            ]
            if not all(k in data for k in required):
                return None
            if data.get("risk_reward", 0) < 2.0:
                return None
            return {**candidate, **data}
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_json_list(content: str | list) -> list:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()
    if "[" in text:
        try:
            return json.loads(text[text.index("["):text.rindex("]") + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def create_asset_analyst_node(researcher_tools: list[BaseTool], quantitative_tools: list[BaseTool]):
    researcher_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    quant_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=4096)
    scoring_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=2048)

    researcher_llm_with_tools = researcher_llm.bind_tools(researcher_tools) if researcher_tools else researcher_llm
    quant_llm_with_tools = quant_llm.bind_tools(quantitative_tools) if quantitative_tools else quant_llm

    researcher_tools_by_name = {t.name: t for t in researcher_tools}
    quant_tools_by_name = {t.name: t for t in quantitative_tools}

    async def _run_tool_loop(llm_with_tools, tools_by_name, messages, agent_name, max_iter=10):
        collected_errors = []
        for _ in range(max_iter):
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:
                    result = f"Tool '{tool_call['name']}' not found."
                else:
                    result, error = await retry_tool_call(tool, tool_call["args"], agent_name)
                    if error:
                        collected_errors.append(error)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        return response, collected_errors

    async def asset_analyst_node(state: dict) -> dict:
        candidate = state.get("candidate", {})
        market_regime = state.get("market_regime", {})
        ticker = candidate.get("ticker", "UNKNOWN")
        name = candidate.get("name", ticker)
        asset_type = candidate.get("asset_type", "stock")
        collected_errors: list[dict] = []

        try:
            # Phase 1: Research
            research_messages = [
                SystemMessage(content=_RESEARCH_PROMPT),
                HumanMessage(content=f"Research {asset_type}: {name} (ticker: {ticker})"),
            ]
            research_response, r_errors = await _run_tool_loop(
                researcher_llm_with_tools, researcher_tools_by_name, research_messages, "asset_analyst_research"
            )
            collected_errors.extend(r_errors)
            news_data = _parse_json_list(research_response.content)

            # Phase 2: Quantitative
            quant_messages = [
                SystemMessage(content=_QUANT_PROMPT),
                HumanMessage(content=f"Get quantitative data for {asset_type}: {name} (ticker: {ticker})"),
            ]
            quant_response, q_errors = await _run_tool_loop(
                quant_llm_with_tools, quant_tools_by_name, quant_messages, "asset_analyst_quant"
            )
            collected_errors.extend(q_errors)
            quant_data = _parse_json_list(quant_response.content)

            # Phase 3: Score the opportunity
            scoring_input = (
                f"Asset: {name} ({ticker}, {asset_type})\n"
                f"Market regime: {market_regime.get('risk_level', 'balanced')} "
                f"(VIX {market_regime.get('vix', 'N/A')})\n\n"
                f"Research findings:\n{json.dumps(news_data, indent=2, default=str)}\n\n"
                f"Quantitative data:\n{json.dumps(quant_data, indent=2, default=str)}"
            )
            scoring_response = await scoring_llm.ainvoke([
                SystemMessage(content=_SCORING_PROMPT),
                HumanMessage(content=scoring_input),
            ])

            opportunity = _parse_opportunity(scoring_response.content, candidate)

            if opportunity:
                logger.info(
                    "AssetAnalyst scored %s: conviction=%d, R/R=%.2f, target=%s",
                    ticker, opportunity["conviction"], opportunity["risk_reward"], opportunity["target_price"]
                )
                return {
                    "opportunities": [opportunity],
                    "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: conviction={opportunity['conviction']}/10, R/R={opportunity['risk_reward']:.2f}")],
                    "errors": collected_errors,
                }
            else:
                logger.info("AssetAnalyst dropped %s: insufficient R/R or parse failure", ticker)
                return {
                    "opportunities": [],
                    "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: dropped (R/R < 2.0 or insufficient data)")],
                    "errors": collected_errors,
                }

        except Exception as e:
            logger.error("AssetAnalyst failed for %s: %s", ticker, e)
            return {
                "opportunities": [],
                "messages": [HumanMessage(content=f"[AssetAnalyst] {ticker}: error — {e}")],
                "errors": collected_errors + [{
                    "agent": "asset_analyst", "tool": "agent_loop",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return asset_analyst_node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_asset_analyst.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/asset_analyst.py tests/test_asset_analyst.py
git commit -m "feat: add AssetAnalyst combined per-candidate analysis node"
```

---

### Task 7: agents/portfolio_constructor.py

Pure allocation logic: takes scored opportunities, applies conviction tiers + regime caps, outputs sized portfolio.

**Files:**
- Create: `agents/portfolio_constructor.py`
- Create: `tests/test_portfolio_constructor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_portfolio_constructor.py`:
```python
from agents.portfolio_constructor import _allocate


_OPP_STOCK = {
    "ticker": "NVDA", "name": "NVIDIA", "asset_type": "stock", "conviction": 9,
    "entry_price": 118.0, "target_price": 145.0, "stop_loss": 108.0,
    "horizon": "medium-term", "implied_return_pct": 22.4, "risk_reward": 2.75,
    "broker_consensus": None, "signals": {}, "rationale": "AI play.",
}
_OPP_ETF = {
    "ticker": "IWDA", "name": "iShares MSCI World", "asset_type": "etf", "conviction": 7,
    "entry_price": 88.0, "target_price": 102.0, "stop_loss": 82.0,
    "horizon": "long-term", "implied_return_pct": 15.9, "risk_reward": 2.3,
    "broker_consensus": None, "signals": {}, "rationale": "Global diversification.",
}
_OPP_BOND = {
    "ticker": "TLT", "name": "iShares 20+ Year Treasury", "asset_type": "bond", "conviction": 6,
    "entry_price": 92.0, "target_price": 105.0, "stop_loss": 88.0,
    "horizon": "medium-term", "implied_return_pct": 14.1, "risk_reward": 3.25,
    "broker_consensus": None, "signals": {}, "rationale": "Rate hedge.",
}


def test_allocate_balanced_three_assets():
    portfolio = _allocate([_OPP_STOCK, _OPP_ETF, _OPP_BOND], 1000.0, "balanced")
    cash = next(p for p in portfolio if p["ticker"] == "CASH")
    non_cash = [p for p in portfolio if p["ticker"] != "CASH"]

    # Cash reserve ~15%
    assert 14.0 <= cash["allocation_pct"] <= 16.0
    # No single asset > 20% (balanced cap)
    for item in non_cash:
        assert item["allocation_pct"] <= 20.1
    # Sum to 100%
    assert abs(sum(p["allocation_pct"] for p in portfolio) - 100.0) < 0.5
    # Budget amounts sum to budget
    assert abs(sum(p["allocation_amount"] for p in portfolio) - 1000.0) < 1.0


def test_allocate_returns_only_cash_when_empty():
    portfolio = _allocate([], 500.0, "balanced")
    assert len(portfolio) == 1
    assert portfolio[0]["ticker"] == "CASH"
    assert portfolio[0]["allocation_pct"] == 100.0
    assert portfolio[0]["allocation_amount"] == 500.0


def test_allocate_aggressive_allows_larger_positions():
    portfolio = _allocate([_OPP_STOCK], 1000.0, "aggressive")
    stock = next(p for p in portfolio if p["ticker"] == "NVDA")
    cash = next(p for p in portfolio if p["ticker"] == "CASH")
    assert stock["allocation_pct"] <= 30.1
    # Cash reserve ~10%
    assert 9.0 <= cash["allocation_pct"] <= 11.0


def test_allocate_defensive_caps_positions_at_10pct():
    opps = [_OPP_STOCK, _OPP_ETF, _OPP_BOND]
    portfolio = _allocate(opps, 1000.0, "defensive")
    non_cash = [p for p in portfolio if p["ticker"] != "CASH"]
    for item in non_cash:
        assert item["allocation_pct"] <= 10.1
    cash = next(p for p in portfolio if p["ticker"] == "CASH")
    assert cash["allocation_pct"] >= 24.0


def test_allocate_uses_only_latest_revision():
    old_opp = {**_OPP_STOCK, "ticker": "OLD", "revision": 0}
    new_opp = {**_OPP_ETF, "ticker": "NEW", "revision": 1}
    portfolio = _allocate([old_opp, new_opp], 1000.0, "balanced")
    tickers = {p["ticker"] for p in portfolio}
    assert "NEW" in tickers
    assert "OLD" not in tickers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_portfolio_constructor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.portfolio_constructor'`

- [ ] **Step 3: Create agents/portfolio_constructor.py**

```python
import logging

from langchain_core.messages import HumanMessage

from graph.state import InvestmentState

logger = logging.getLogger(__name__)

_CASH_RESERVE = {"aggressive": 10.0, "balanced": 15.0, "defensive": 25.0}
_MAX_SINGLE = {"aggressive": 30.0, "balanced": 20.0, "defensive": 10.0}


def _base_weight(conviction: int) -> float:
    if conviction >= 8:
        return 20.0
    if conviction >= 6:
        return 12.0
    return 6.5


def _allocate(opportunities: list[dict], budget: float, risk_level: str) -> list[dict]:
    cash_reserve = _CASH_RESERVE.get(risk_level, 15.0)
    max_single = _MAX_SINGLE.get(risk_level, 20.0)
    investable = 100.0 - cash_reserve

    _cash_item = {
        "ticker": "CASH", "name": "Cash Reserve", "asset_type": "cash",
        "allocation_pct": 100.0, "allocation_amount": round(budget, 2),
        "entry_price": None, "target_price": None, "stop_loss": None,
        "horizon": None, "conviction": None, "implied_return_pct": None,
        "risk_reward": None, "broker_consensus": None, "signals": None, "rationale": None,
    }

    if not opportunities:
        return [_cash_item]

    # Use only the latest revision round
    max_revision = max(o.get("revision", 0) for o in opportunities)
    opps = [o for o in opportunities if o.get("revision", 0) == max_revision]

    if not opps:
        return [_cash_item]

    raw = [_base_weight(o.get("conviction", 5)) for o in opps]
    total_raw = sum(raw)
    scaled = [w / total_raw * investable for w in raw]
    scaled = [min(w, max_single) for w in scaled]

    # Re-normalize after capping
    total_scaled = sum(scaled)
    final = [w / total_scaled * investable for w in scaled]

    portfolio = []
    for opp, pct in zip(opps, final):
        portfolio.append({
            "ticker": opp["ticker"],
            "name": opp["name"],
            "asset_type": opp["asset_type"],
            "allocation_pct": round(pct, 1),
            "allocation_amount": round(pct / 100.0 * budget, 2),
            "entry_price": opp.get("entry_price"),
            "target_price": opp.get("target_price"),
            "stop_loss": opp.get("stop_loss"),
            "horizon": opp.get("horizon"),
            "conviction": opp.get("conviction"),
            "implied_return_pct": opp.get("implied_return_pct"),
            "risk_reward": opp.get("risk_reward"),
            "broker_consensus": opp.get("broker_consensus"),
            "signals": opp.get("signals"),
            "rationale": opp.get("rationale"),
        })

    allocated_total = sum(item["allocation_pct"] for item in portfolio)
    cash_pct = round(100.0 - allocated_total, 1)
    portfolio.append({
        **_cash_item,
        "allocation_pct": cash_pct,
        "allocation_amount": round(cash_pct / 100.0 * budget, 2),
    })

    return portfolio


async def portfolio_constructor_node(state: InvestmentState) -> dict:
    opportunities = state.get("opportunities", [])
    budget = state["budget"]
    market_regime = state.get("market_regime", {})
    risk_level = market_regime.get("risk_level", "balanced")

    # Filter: only R/R >= 2.0, then sort by conviction desc
    valid = [o for o in opportunities if o.get("risk_reward", 0) >= 2.0]
    valid.sort(key=lambda x: x.get("conviction", 0), reverse=True)

    portfolio = _allocate(valid, budget, risk_level)
    non_cash_count = sum(1 for p in portfolio if p["ticker"] != "CASH")

    logger.info(
        "PortfolioConstructor: %d positions + cash (regime: %s, budget: %.0f)",
        non_cash_count, risk_level, budget,
    )

    return {
        "portfolio": portfolio,
        "messages": [HumanMessage(
            content=f"[PortfolioConstructor] Built portfolio: {non_cash_count} positions + cash ({risk_level} regime)."
        )],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_portfolio_constructor.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/portfolio_constructor.py tests/test_portfolio_constructor.py
git commit -m "feat: add PortfolioConstructor allocation engine"
```

---

### Task 8: evaluation/investment_judge.py

Gemini-powered portfolio validation. Validates R/R, concentration, diversification, and allocation sum.

**Files:**
- Create: `evaluation/investment_judge.py`
- Create: `tests/test_investment_judge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_investment_judge.py`:
```python
from evaluation.investment_judge import _validate_portfolio


_REGIME_BALANCED = {"risk_level": "balanced"}

_CLEAN_PORTFOLIO = [
    {"ticker": "NVDA", "asset_type": "stock", "allocation_pct": 18.0, "risk_reward": 2.75},
    {"ticker": "IWDA", "asset_type": "etf", "allocation_pct": 20.0, "risk_reward": 2.3},
    {"ticker": "TLT", "asset_type": "bond", "allocation_pct": 15.0, "risk_reward": 3.25},
    {"ticker": "BTC", "asset_type": "crypto", "allocation_pct": 12.0, "risk_reward": 2.5},
    {"ticker": "CASH", "asset_type": "cash", "allocation_pct": 35.0, "risk_reward": None},
]


def test_validate_portfolio_passes_clean_portfolio():
    issues = _validate_portfolio(_CLEAN_PORTFOLIO, _REGIME_BALANCED)
    assert issues == []


def test_validate_portfolio_catches_bad_rr():
    portfolio = [
        {"ticker": "BAD", "asset_type": "stock", "allocation_pct": 20.0, "risk_reward": 1.5},
        {"ticker": "IWDA", "asset_type": "etf", "allocation_pct": 20.0, "risk_reward": 2.5},
        {"ticker": "TLT", "asset_type": "bond", "allocation_pct": 20.0, "risk_reward": 2.0},
        {"ticker": "CASH", "asset_type": "cash", "allocation_pct": 40.0, "risk_reward": None},
    ]
    issues = _validate_portfolio(portfolio, _REGIME_BALANCED)
    assert any("BAD" in i and "risk/reward" in i for i in issues)


def test_validate_portfolio_catches_concentration():
    portfolio = [
        {"ticker": "NVDA", "asset_type": "stock", "allocation_pct": 25.0, "risk_reward": 2.5},
        {"ticker": "AAPL", "asset_type": "stock", "allocation_pct": 22.0, "risk_reward": 2.5},  # >20%
        {"ticker": "TLT", "asset_type": "bond", "allocation_pct": 38.0, "risk_reward": 2.0},
        {"ticker": "CASH", "asset_type": "cash", "allocation_pct": 15.0, "risk_reward": None},
    ]
    issues = _validate_portfolio(portfolio, _REGIME_BALANCED)
    assert any("exceeds" in i for i in issues)


def test_validate_portfolio_catches_low_diversification():
    portfolio = [
        {"ticker": "NVDA", "asset_type": "stock", "allocation_pct": 42.0, "risk_reward": 2.5},
        {"ticker": "AAPL", "asset_type": "stock", "allocation_pct": 43.0, "risk_reward": 2.5},
        {"ticker": "CASH", "asset_type": "cash", "allocation_pct": 15.0, "risk_reward": None},
    ]
    issues = _validate_portfolio(portfolio, _REGIME_BALANCED)
    assert any("asset type" in i for i in issues)


def test_validate_portfolio_catches_bad_sum():
    portfolio = [
        {"ticker": "NVDA", "asset_type": "stock", "allocation_pct": 50.0, "risk_reward": 2.5},
        {"ticker": "CASH", "asset_type": "cash", "allocation_pct": 40.0, "risk_reward": None},
        # Sums to 90%, not 100%
    ]
    issues = _validate_portfolio(portfolio, _REGIME_BALANCED)
    assert any("sum" in i.lower() or "100%" in i for i in issues)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_investment_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.investment_judge'`

- [ ] **Step 3: Create evaluation/investment_judge.py**

```python
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import InvestmentState

logger = logging.getLogger(__name__)

_MAX_SINGLE = {"aggressive": 30.0, "balanced": 20.0, "defensive": 10.0}

JUDGE_SYSTEM_PROMPT = """You are an independent portfolio risk manager reviewing an investment
portfolio allocation. Evaluate the portfolio holistically and produce a quality score.

Score these dimensions (1-5 each):
- Risk/Reward quality: Are R/R ratios >= 2.0 on all positions?
- Diversification: Are there >= 3 asset types? Max 2 stocks in same sector?
- Data quality: Are broker targets recent and specific?
- Allocation logic: Does sizing reflect conviction? Does it match the risk regime?

Return ONLY this JSON object:
{
    "verdict": "<pass | fail>",
    "overall_score": <float 1-5>,
    "rr_score": <float 1-5>,
    "diversification_score": <float 1-5>,
    "data_quality_score": <float 1-5>,
    "allocation_score": <float 1-5>,
    "issues": ["<specific issue 1>", "<specific issue 2>"],
    "feedback": "<actionable guidance for improvement if verdict is fail>"
}"""


def _validate_portfolio(portfolio: list[dict], market_regime: dict) -> list[str]:
    """Pure validation — returns list of issues. Empty = no issues."""
    issues = []
    risk_level = market_regime.get("risk_level", "balanced")
    max_single = _MAX_SINGLE.get(risk_level, 20.0)
    non_cash = [p for p in portfolio if p.get("asset_type") != "cash"]

    for item in non_cash:
        rr = item.get("risk_reward")
        if rr is not None and rr < 2.0:
            issues.append(f"{item['ticker']}: risk/reward {rr:.2f} is below minimum 2.0")

    for item in non_cash:
        pct = item.get("allocation_pct", 0)
        if pct > max_single:
            issues.append(
                f"{item['ticker']}: allocation {pct}% exceeds {risk_level} regime max {max_single}%"
            )

    asset_types = {p.get("asset_type") for p in non_cash if p.get("asset_type")}
    if len(asset_types) < 3:
        issues.append(
            f"Portfolio has only {len(asset_types)} asset type(s); need at least 3 for diversification"
        )

    total = sum(p.get("allocation_pct", 0) for p in portfolio)
    if abs(total - 100.0) > 1.0:
        issues.append(f"Allocations sum to {total:.1f}%, expected 100%")

    return issues


def _parse_evaluation(content: str | list) -> dict:
    text = content if isinstance(content, str) else (
        content[0].get("text", "") if content else ""
    )
    text = text.strip()
    if "{" in text:
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"verdict": "pass", "overall_score": 3.0, "issues": [], "feedback": ""}


def create_investment_judge_node():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    async def investment_judge_node(state: InvestmentState) -> dict:
        portfolio = state.get("portfolio", [])
        market_regime = state.get("market_regime", {})
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 2)

        # Pure validation first
        pure_issues = _validate_portfolio(portfolio, market_regime)

        judge_input = (
            f"Market regime: {market_regime.get('risk_level', 'balanced')} "
            f"(VIX {market_regime.get('vix', 'N/A')})\n\n"
            f"Portfolio:\n{json.dumps(portfolio, indent=2, default=str)}\n\n"
            f"Pre-validation issues found: {pure_issues if pure_issues else 'None'}"
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=judge_input),
            ])
            evaluation = _parse_evaluation(response.content)

            # Merge pure validation issues
            all_issues = pure_issues + [
                i for i in evaluation.get("issues", []) if i not in pure_issues
            ]
            evaluation["issues"] = all_issues

            # Override verdict if pure validation found critical issues
            if pure_issues:
                evaluation["verdict"] = "fail"

            needs_revision = (
                evaluation.get("verdict") == "fail"
                and revision_count < max_revisions
            )
            revision_feedback = ""
            if needs_revision and all_issues:
                revision_feedback = (
                    f"Portfolio rejected (score {evaluation.get('overall_score', 0):.1f}/5). "
                    f"Issues: {'; '.join(all_issues[:3])}. "
                    f"{evaluation.get('feedback', 'Find better candidates.')}"
                )

            logger.info(
                "InvestmentJudge: %.1f/5 — %s (%d issues)",
                evaluation.get("overall_score", 0),
                evaluation.get("verdict", "unknown"),
                len(all_issues),
            )

            return {
                "evaluation": evaluation,
                "needs_revision": needs_revision,
                "revision_count": revision_count + 1,
                "revision_feedback": revision_feedback,
                "messages": [HumanMessage(
                    content=f"[InvestmentJudge] Score: {evaluation.get('overall_score', 0):.1f}/5 — "
                            f"{evaluation.get('verdict', 'unknown')}. Issues: {len(all_issues)}"
                )],
            }

        except Exception as e:
            logger.error("InvestmentJudge failed: %s", e)
            return {
                "evaluation": {"verdict": "pass", "overall_score": 0, "issues": pure_issues},
                "needs_revision": False,
                "revision_count": revision_count + 1,
                "revision_feedback": "",
                "messages": [HumanMessage(content=f"[InvestmentJudge] Failed: {e}. Skipping evaluation.")],
                "errors": [{
                    "agent": "investment_judge", "tool": "gemini",
                    "error_type": type(e).__name__, "message": str(e),
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "recoverable": False,
                }],
            }

    return investment_judge_node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_investment_judge.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add evaluation/investment_judge.py tests/test_investment_judge.py
git commit -m "feat: add InvestmentJudge portfolio validation node"
```

---

### Task 9: graph/workflow.py + graph/__init__.py

Wire all nodes into a LangGraph StateGraph with Send-based fan-out for parallel per-candidate analysis.

**Files:**
- Create: `graph/workflow.py`
- Modify: `graph/__init__.py`
- Create: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow.py`:
```python
from unittest.mock import MagicMock

from graph.workflow import build_graph


def test_build_graph_compiles_without_error():
    mock_tools = [MagicMock()]
    mock_tools[0].name = "mock_tool"
    graph = build_graph(mock_tools, mock_tools)
    assert graph is not None


def test_build_graph_has_expected_nodes():
    mock_tools = [MagicMock()]
    mock_tools[0].name = "mock_tool"
    graph = build_graph(mock_tools, mock_tools)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"macro_scanner", "universe_scanner", "analyze_candidate",
                "portfolio_constructor", "investment_judge"}
    for node in expected:
        assert node in node_names, f"Missing node: {node}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graph.workflow'`

- [ ] **Step 3: Create graph/workflow.py**

```python
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
            # No candidates this round — skip to portfolio constructor
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
```

- [ ] **Step 4: Update graph/__init__.py**

```python
from graph.workflow import build_graph

__all__ = ["build_graph"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_workflow.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASSED

- [ ] **Step 7: Commit**

```bash
git add graph/workflow.py graph/__init__.py tests/test_workflow.py
git commit -m "feat: rebuild LangGraph workflow with Send-based parallel candidate analysis"
```

---

### Task 10: Update main.py

Replace `--company` CLI with `--budget`, update the initial state, and rewrite the report builder for the portfolio format.

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace run_analysis signature and initial state**

Replace the `run_analysis` function:

```python
async def run_analysis(budget: float, max_revisions: int = 2) -> dict:
    """Run the full investment opportunity pipeline.

    Returns the full final state dict (portfolio, evaluation, errors, etc.).
    """
    logger.info("Starting investment scan with budget: %.0f", budget)

    client = create_mcp_client()
    tavily_tool = get_tavily_search_tool()

    try:
        firecrawl_tools = await get_firecrawl_tools(client)
        logger.info("Loaded %d Firecrawl tools", len(firecrawl_tools))
    except Exception as e:
        logger.warning("Failed to load Firecrawl tools: %s. Continuing without.", e)
        firecrawl_tools = []

    try:
        fmp_tools = await get_fmp_tools(client)
        logger.info("Loaded %d FMP tools", len(fmp_tools))
    except Exception as e:
        logger.warning("Failed to load FMP tools: %s. Continuing without.", e)
        fmp_tools = []

    researcher_tools = [tavily_tool] + firecrawl_tools
    quantitative_tools = fmp_tools

    app = build_graph(researcher_tools, quantitative_tools)

    initial_state = {
        "budget": budget,
        "market_regime": {},
        "candidates": [],
        "opportunities": [],
        "portfolio": [],
        "evaluation": {},
        "revision_count": 0,
        "max_revisions": max_revisions,
        "needs_revision": False,
        "revision_feedback": "",
        "messages": [],
        "errors": [],
    }

    logger.info("Executing investment scan graph...")
    result = await app.ainvoke(initial_state)

    if result.get("errors"):
        logger.warning("Scan completed with %d errors", len(result["errors"]))

    return result
```

- [ ] **Step 2: Replace _build_report_text**

```python
def _build_report_text(result: dict) -> str:
    from datetime import datetime, timezone

    lines = []
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    budget = result.get("budget", 0)
    regime = result.get("market_regime", {})
    risk_level = regime.get("risk_level", "N/A")
    vix = regime.get("vix", "N/A")

    lines.append(f"# Investment Portfolio — {date_str}")
    lines.append(f"**Budget:** {budget:,.0f} | **Risk regime:** {risk_level.title()} (VIX {vix})\n")

    lines.append("## Market Regime")
    lines.append(f"- VIX: {vix}")
    lines.append(f"- Yield curve spread: {regime.get('yield_curve_spread', 'N/A')}")
    lines.append(f"- S&P 500 vs 200MA: {'Above' if regime.get('spx_above_200ma') else 'Below'}")
    lines.append(f"- Summary: {regime.get('summary', 'N/A')}\n")

    portfolio = result.get("portfolio", [])
    if portfolio:
        lines.append("## Portfolio Allocation\n")
        lines.append("| # | Ticker | Name | Type | Horizon | Buy at | Target | Stop | Return | R/R | Alloc % | Amount |")
        lines.append("|---|--------|------|------|---------|--------|--------|------|--------|-----|---------|--------|")
        for i, item in enumerate(portfolio, 1):
            if item["ticker"] == "CASH":
                lines.append(
                    f"| — | **CASH** | Cash Reserve | — | — | — | — | — | — | — | "
                    f"**{item['allocation_pct']}%** | {item['allocation_amount']:,.0f} |"
                )
            else:
                lines.append(
                    f"| {i} | **{item['ticker']}** | {item['name']} | {item['asset_type']} | "
                    f"{item.get('horizon', 'N/A')} | {item.get('entry_price', 'N/A')} | "
                    f"{item.get('target_price', 'N/A')} | {item.get('stop_loss', 'N/A')} | "
                    f"+{item.get('implied_return_pct', 'N/A')}% | {item.get('risk_reward', 'N/A')} | "
                    f"**{item['allocation_pct']}%** | {item['allocation_amount']:,.0f} |"
                )
        lines.append("")

        lines.append("## Asset Detail\n")
        for item in portfolio:
            if item["ticker"] == "CASH":
                continue
            lines.append(f"### {item['ticker']} — {item['name']}")
            lines.append(f"**Type:** {item['asset_type'].upper()} | **Horizon:** {item.get('horizon', 'N/A')} | "
                         f"**Conviction:** {item.get('conviction', 'N/A')}/10\n")

            broker = item.get("broker_consensus")
            if broker and broker.get("brokers"):
                buy = broker.get("buy_count", 0)
                hold = broker.get("hold_count", 0)
                sell = broker.get("sell_count", 0)
                lines.append(
                    f"**Broker consensus:** {buy}B/{hold}H/{sell}S | "
                    f"Mean target: {broker.get('mean_target', 'N/A')} | "
                    f"Implied return: +{item.get('implied_return_pct', 'N/A')}%\n"
                )
                lines.append("| Broker | Analyst | Rating | Target | Date | Implied Return |")
                lines.append("|--------|---------|--------|--------|------|----------------|")
                for b in broker["brokers"]:
                    lines.append(
                        f"| {b.get('broker', '')} | {b.get('analyst', '')} | "
                        f"{b.get('rating', '')} | {b.get('target_price', '')} | "
                        f"{b.get('rating_date', '')} | +{b.get('implied_return_pct', '')}% |"
                    )
                lines.append("")

            signals = item.get("signals") or {}
            if signals:
                sig_parts = [f"{k}: {v}" for k, v in signals.items() if v is not None]
                lines.append(f"**Signals:** {', '.join(sig_parts)}\n")

            lines.append(f"**Rationale:** {item.get('rationale', 'N/A')}\n")
            lines.append(f"**Entry:** {item.get('entry_price')} | **Target:** {item.get('target_price')} | "
                         f"**Stop:** {item.get('stop_loss')} | **R/R:** {item.get('risk_reward')}\n")

    evaluation = result.get("evaluation", {})
    if evaluation and "overall_score" in evaluation:
        lines.append("---\n## Quality Evaluation\n")
        lines.append(f"**Score: {evaluation['overall_score']}/5 — {evaluation.get('verdict', 'N/A')}**\n")
        for issue in evaluation.get("issues", []):
            lines.append(f"- {issue}")

    errors = result.get("errors", [])
    if errors:
        lines.append(f"\n---\n**{len(errors)} error(s) during scan** (see logs)")

    return "\n".join(lines)
```

- [ ] **Step 3: Replace _save_report and update the filename**

```python
def _save_report(result: dict) -> str:
    from datetime import datetime, timezone
    from pathlib import Path

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = reports_dir / f"{date_str}_portfolio.md"

    filepath.write_text(_build_report_text(result))
    return str(filepath)
```

- [ ] **Step 4: Update the CLI argument parser and main function**

Replace the argument parser and `main()` body:

```python
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Investment Opportunities Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budget",
        required=True,
        type=float,
        help="Total budget to invest (e.g. 1000)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum revision cycles if judge rejects portfolio (default: 2)",
    )
    args = parser.parse_args()

    result = asyncio.run(run_analysis(args.budget, args.max_revisions))
    _print_report(result)
    filepath = _save_report(result)
    print(f"\nReport saved to: {filepath}")
```

Also update `_print_report` — it stays the same (calls `_build_report_text`), but remove the `company` parameter since `_save_report` no longer takes it.

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASSED

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: update main.py for budget-based portfolio CLI and new report format"
```

---

## Self-Review

**Spec coverage check:**
- ✅ MacroScanner: VIX, yield curve, trend → Task 2
- ✅ UniverseScanner: ~15 candidates across US/EU/ETF/crypto/bond → Task 3
- ✅ Researcher asset-type aware → Task 4
- ✅ Broker consensus (per-broker table with rating, target, date, implied return) → Task 5
- ✅ AssetAnalyst: entry/target/stop/conviction/horizon/R/R → Task 6
- ✅ R/R ≥ 2.0 enforcement → Task 6 (`_parse_opportunity`) + Task 8 (`_validate_portfolio`)
- ✅ Portfolio allocation: conviction tiers, regime caps, cash reserve → Task 7
- ✅ Revision loop (max 2 cycles, tagged by revision round) → Tasks 7+9
- ✅ InvestmentJudge: diversification, R/R, concentration, allocation sum → Task 8
- ✅ LangGraph Send fan-out → Task 9
- ✅ `--budget` CLI, portfolio report with broker table → Task 10

**Placeholder scan:** None found. All steps contain real code.

**Type consistency check:**
- `InvestmentState.opportunities: Annotated[list[dict], operator.add]` — used correctly in Tasks 6, 7, 9
- `_allocate(opportunities, budget, risk_level)` defined in Task 7, called in `portfolio_constructor_node` in same file ✅
- `_validate_portfolio(portfolio, market_regime)` defined and tested in Task 8 ✅
- `dispatch_candidates` reads `state["candidates"]` filtered by `revision_count` — consistent with tagging in Task 3 ✅
- `revision_count` incremented by `investment_judge_node` (Task 8) — used as filter in `dispatch_candidates` (Task 9) ✅
