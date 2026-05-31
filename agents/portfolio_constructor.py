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
    n = len(opps)
    weights = [w / total_raw for w in raw]

    # Iterative water-filling: cap each weight at max_single/investable,
    # redistribute excess proportionally to uncapped slots until stable.
    cap_ratio = max_single / investable  # cap as fraction of investable
    for _ in range(n + 1):
        capped_mask = [w >= cap_ratio - 1e-9 for w in weights]
        if not any(capped_mask):
            break
        # Lock capped positions at cap_ratio, distribute remaining to free ones
        n_capped = sum(capped_mask)
        locked = n_capped * cap_ratio
        remaining = 1.0 - locked
        free_indices = [i for i, c in enumerate(capped_mask) if not c]
        if not free_indices or remaining <= 0:
            # All positions hit the cap: distribute evenly up to cap
            weights = [min(cap_ratio, 1.0 / n)] * n
            break
        free_raw_sum = sum(weights[i] for i in free_indices)
        new_weights = list(weights)
        for i, c in enumerate(capped_mask):
            if c:
                new_weights[i] = cap_ratio
            else:
                new_weights[i] = (weights[i] / free_raw_sum) * remaining
        weights = new_weights

    final = [w * investable for w in weights]

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
