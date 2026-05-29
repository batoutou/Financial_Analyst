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

    # Cash is at least the reserve (15%); may be higher when positions hit max_single cap
    assert cash["allocation_pct"] >= 14.0
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
    # Cash is at least the reserve (10%); may be higher when single asset hits 30% cap
    assert cash["allocation_pct"] >= 9.0


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
