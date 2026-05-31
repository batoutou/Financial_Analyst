# Investment Opportunities Agent — Design Spec
**Date:** 2026-05-27
**Status:** Approved

---

## Problem

The current system takes a single company name and produces a narrative analysis memo. It does not scan markets autonomously, does not identify entry/exit prices, and does not produce portfolio allocations. The goal is to transform it into a system that autonomously discovers investment opportunities across public markets (stocks, ETFs, bonds, crypto) and produces actionable, sized portfolio recommendations.

---

## Goals

- Autonomously scan US, European, global ETF, and crypto markets for opportunities
- Incorporate real broker analyst recommendations (ratings, target prices, implied returns) as a primary signal
- Produce per-asset recommendations: buy price, target price, stop loss, horizon, conviction
- Produce a portfolio allocation table: % of budget per asset, in EUR amount
- Calibrate risk profile dynamically from live market regime data (VIX, yield curve, trend)

---

## Non-Goals

- No watchlist input mode (fully autonomous scan only)
- No intraday / HFT signals
- No trade execution
- No backtesting

---

## Architecture

5-phase pipeline orchestrated by LangGraph. Graph files (`graph/state.py`, `graph/workflow.py`, `graph/__init__.py`) are deleted on disk and rebuilt from scratch.

```
START
  │
  ▼
MacroScanner                 ← detects VIX, yield curve, trend → sets risk_level
  │
  ▼
UniverseScanner              ← finds ~15 candidates: ~4 US stocks, ~3 EU stocks, ~3 global ETFs, ~3 crypto, ~2 bond ETFs
  │
  │  [dynamic fan-out via LangGraph Send API — one branch per candidate, parallel]
  │
  ├── Candidate N: Researcher ──┐
  │              Quantitative  ┤→ AssetAnalyst → scored opportunity
  ├── ...all candidates
  │
  ▼  [fan-in]
PortfolioConstructor         ← allocates % weights: conviction × regime × diversification
  │
  ▼
InvestmentJudge (Gemini)     ← validates R/R, diversification, target recency
  │
  ┌──────────┤
  │          │
PASS       FAIL → back to UniverseScanner (feedback, max 2 cycles)
  │
  ▼
END → reports/YYYY-MM-DD_portfolio.md
```

---

## New & Modified Files

| File | Status | Purpose |
|------|--------|---------|
| `graph/state.py` | Rebuild | `InvestmentState` TypedDict |
| `graph/workflow.py` | Rebuild | LangGraph wiring with Send fan-out |
| `graph/__init__.py` | Rebuild | Module export |
| `agents/macro_scanner.py` | New | Detects market regime |
| `agents/universe_scanner.py` | New | Autonomous candidate discovery |
| `agents/asset_analyst.py` | New | Per-candidate scorer |
| `agents/portfolio_constructor.py` | New | Allocation engine |
| `evaluation/investment_judge.py` | New (extends judge.py) | Portfolio validation |
| `agents/researcher.py` | Modify | Asset-type aware prompts |
| `agents/quantitative.py` | Modify | Add broker consensus endpoints |
| `main.py` | Modify | `--budget` CLI, new report format |

---

## State Design

```python
class InvestmentState(TypedDict):
    # Input
    budget: float

    # Phase 1
    market_regime: dict                                  # VIX, yield_curve, trend, risk_level

    # Phase 2
    candidates: Annotated[list[dict], operator.add]      # {ticker, name, asset_type, rationale}

    # Phase 3 (fan-out results)
    opportunities: Annotated[list[dict], operator.add]   # scored assets

    # Phase 4
    portfolio: list[dict]                                # final allocations

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

### Opportunity object (AssetAnalyst output per candidate)

```python
{
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "asset_type": "stock",           # stock | etf | bond | crypto
    "horizon": "medium-term",        # short-term | medium-term | long-term
    "conviction": 8,                 # 1–10
    "entry_price": 118.50,
    "target_price": 145.00,          # anchored to broker mean consensus target
    "stop_loss": 108.00,
    "implied_return_pct": 22.4,
    "risk_reward": 2.75,             # must be ≥ 2.0 to be included
    "broker_consensus": {
        "mean_target": 145.00,
        "buy_count": 4,
        "hold_count": 1,
        "sell_count": 0,
        "brokers": [
            {
                "broker": "KBC Securities",
                "analyst": "Jacob Mekhael",
                "rating": "Buy",
                "target_price": 15.00,
                "rating_date": "2026-05-27",
                "price_on_rating_date": 2.81,
                "implied_return_pct": 434.8
            }
        ]
    },
    "signals": {
        "rsi": 38,
        "price_vs_50ma": -4.2,
        "pe_ratio": 28.5,
        "pe_vs_sector": -12.0,
        "revenue_growth_yoy": 0.22,
        "roe": 0.31
    },
    "rationale": "2–3 sentence explanation of the thesis."
}
```

### Portfolio allocation object (PortfolioConstructor output)

```python
{
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "asset_type": "stock",
    "allocation_pct": 20.0,
    "allocation_amount": 200.0,      # budget × pct
    "entry_price": 118.50,
    "target_price": 145.00,
    "stop_loss": 108.00,
    "horizon": "medium-term",
    "conviction": 8,
    "implied_return_pct": 22.4,
    "risk_reward": 2.75
}
```

---

## Investment Signals Framework

### Stocks
| Layer | Signals | Source |
|-------|---------|--------|
| Broker consensus | Per-broker target, rating, date, implied return, mean target | FMP `/price-target` |
| Valuation | P/E vs sector, PEG, EV/EBITDA, P/B | FMP ratios |
| Momentum | RSI, price vs 50/200 MA, MACD | FMP technical |
| Quality | ROE, revenue growth, margin trend, debt/equity | FMP statements |
| Sentiment | News tone, insider activity, short interest | Tavily + FMP |
| Catalyst | Upcoming earnings, product events, regulatory news | Tavily |

### ETFs
| Signal | Source |
|--------|--------|
| Price vs 50/200 MA (trend strength) | FMP |
| Underlying theme performance | Tavily |
| Expense ratio | FMP / Tavily |
| Broker consensus on ETF or underlying index | FMP |

### Bonds (via ETFs: TLT, BND, BUND ETF)
| Signal | Source |
|--------|--------|
| Current yield vs 5yr historical | FMP + Tavily |
| Duration risk | FMP |
| Yield curve position (Fed policy context) | Tavily (macro scan) |

### Crypto
| Signal | Source |
|--------|--------|
| RSI weekly + price structure | FMP or Tavily |
| Exchange flow sentiment (accumulation vs distribution) | Tavily |
| Market cap rank + dominance | Tavily |

---

## Conviction Scoring (1–10)

Weighted composite computed by AssetAnalyst:

| Component | Weight | Description |
|-----------|--------|-------------|
| Broker consensus strength | 40% | % Buy ratings + mean implied return magnitude |
| Valuation attractiveness | 30% | Discount to fair value / sector average |
| Momentum alignment | 20% | RSI, price vs moving averages |
| News sentiment | 10% | Recency and quality of catalyst |

---

## Entry / Target / Stop Logic

- **Entry price**: current price if at/near support; otherwise current last close
- **Target price**: anchored to broker consensus mean target; adjusted for technical resistance
- **Stop loss**: below nearest key technical support; R/R must be ≥ 2:1
- **Horizon**: inferred from catalyst timing + analyst target date + signal type
  - RSI reversal → short-term
  - Earnings cycle + fundamental re-rating → medium-term
  - DCF discount + dividend → long-term

---

## Market Regime Detection (MacroScanner)

MacroScanner runs first and sets `risk_level` which gates allocation sizes:

| VIX Level | Yield Curve | SPX vs 200MA | Risk Level | Max Single Asset |
|-----------|-------------|--------------|------------|-----------------|
| < 15 | Steep (>0.5%) | Above | Aggressive | 30% |
| 15–25 | Flat | Above | Balanced | 20% |
| > 25 | Inverted | Below | Defensive | 10% |

Cash reserve: 10% (aggressive) / 15% (balanced) / 25% (defensive).

---

## Portfolio Allocation Rules (PortfolioConstructor)

- Conviction 8–10 → 15–25% allocation
- Conviction 6–7 → 8–15% allocation
- Conviction 4–5 → 5–8% allocation
- Max 2 assets per sector
- At least 3 different asset types in final portfolio
- Always reserve cash (see regime table above)
- Final allocations sum to 100% (including cash reserve)

---

## InvestmentJudge Validation (Gemini)

Fails the portfolio (triggers revision) if any of:
- Any position has R/R < 2:1
- Any single asset > 30% of portfolio
- Fewer than 3 asset types represented
- Any broker target older than 90 days used as primary signal without flagging
- Allocation percentages do not sum to 100%

---

## CLI

```bash
python main.py --budget 1000
python main.py --budget 5000 --max-revisions 2
```

---

## Output Report Format

```
reports/YYYY-MM-DD_portfolio.md
```

```markdown
# Investment Portfolio — 2026-05-27
**Budget:** €1,000 | **Risk regime:** Balanced (VIX 18.2)

## Market Regime
...

## Portfolio Allocation Summary
| # | Ticker | Name | Type | Horizon | Buy at | Target | Stop | Return | R/R | Alloc % | Amount |
...
| CASH | Reserve | — | — | — | — | — | — | — | 15% | €150 |

## Asset Detail

### 1. NVDA — NVIDIA Corporation
**Broker consensus:** Overweight (4 Buy, 1 Hold) | Mean target: $145 | Implied: +22%
| Broker | Analyst | Rating | Target | Date | Implied Return |
...
**Signals:** RSI 38, P/E below sector avg, ...
**Rationale:** ...
```

---

## Error Handling

Same resilience pattern as the existing system:
- All tool calls wrapped in `retry_tool_call` with exponential backoff
- Structured error log appended to `errors` state field
- If a candidate's deep-dive fails, it is dropped from the universe (not included in portfolio)
- If MacroScanner fails, default to `risk_level: "balanced"`
- If < 3 candidates survive deep-dive, skip to PortfolioConstructor with available data and flag in report

---

## Out of Scope

- Watchlist / manual ticker input
- Backtesting or historical portfolio performance
- Trade execution or brokerage integration
- Real-time streaming prices
