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
