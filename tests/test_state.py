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
