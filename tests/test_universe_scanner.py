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
