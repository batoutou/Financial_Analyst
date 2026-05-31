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
