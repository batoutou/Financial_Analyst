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
