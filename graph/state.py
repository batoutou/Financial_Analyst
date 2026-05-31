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
