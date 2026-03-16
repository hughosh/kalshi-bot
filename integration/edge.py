"""Edge computation and quarter-Kelly position sizing.

Edge formula:
  edge_yes = model_prob - (best_ask / 100)
  edge_no  = (1 - model_prob) - ((100 - best_bid) / 100)

Trade signal emitted when:
  1. abs(edge) > MIN_EDGE_PCT
  2. model_confidence >= MIN_CONFIDENCE
  3. points_observed >= MIN_POINTS_OBSERVED
  4. net_dollar_edge (after fees) > NET_EDGE_MIN_DOLLARS

Position sizing: quarter-Kelly
  kelly_fraction = edge / odds_against
  contracts = floor(kelly_fraction / 4 * bankroll / cost_per_contract)
  capped at MAX_CONTRACTS_PER_SIGNAL
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    side: str           # "yes" or "no"
    edge: float         # raw edge (positive)
    contracts: int      # recommended quantity
    price_cents: int    # limit price (the ask we'd take)
    model_prob: float
    confidence: float   # 1 - CI_width


def compute_edge(
    model_prob: float,
    best_ask_cents: Optional[int],
    best_bid_cents: Optional[int],
) -> tuple[Optional[str], float, int]:
    """Return (side, edge, price_cents) or (None, 0, 0) if no edge.

    side: "yes" if we think the market underprices YES, "no" if it underprices NO.
    edge: positive float representing our advantage.
    price_cents: the price we'd pay (ask for YES, 100-bid for NO).
    """
    if best_ask_cents is not None:
        market_yes = best_ask_cents / 100.0
        edge_yes = model_prob - market_yes
    else:
        edge_yes = -999.0

    if best_bid_cents is not None:
        market_no = (100 - best_bid_cents) / 100.0
        edge_no = (1.0 - model_prob) - market_no
    else:
        edge_no = -999.0

    if edge_yes > edge_no and edge_yes > 0:
        return "yes", edge_yes, best_ask_cents
    elif edge_no > 0:
        return "no", edge_no, 100 - best_bid_cents
    return None, 0.0, 0


def quarter_kelly_size(
    edge: float,
    price_cents: int,
    bankroll: float,
    max_contracts: int,
    fee_per_contract: float = 0.07,
) -> int:
    """Quarter-Kelly position size in contracts.

    kelly_fraction = edge / (payout_ratio)
    payout_ratio = (100 - price_cents) / price_cents  (net win per dollar risked)
    """
    if price_cents <= 0 or price_cents >= 100 or edge <= 0:
        return 0
    cost = price_cents / 100.0
    payout_ratio = (100 - price_cents) / price_cents
    if payout_ratio <= 0:
        return 0
    kelly = edge / payout_ratio
    quarter = kelly / 4.0
    cost_per = cost + fee_per_contract
    if cost_per <= 0:
        return 0
    contracts = int(math.floor(quarter * bankroll / cost_per))
    return max(0, min(contracts, max_contracts))


def evaluate_signal(
    ticker: str,
    model_prob: float,
    confidence: float,
    points_observed: int,
    best_ask_cents: Optional[int],
    best_bid_cents: Optional[int],
    bankroll: float,
    min_edge: float = 0.05,
    min_confidence: float = 0.60,
    min_points: int = 10,
    net_edge_min: float = 0.10,
    max_contracts: int = 20,
    fee_per_contract: float = 0.07,
) -> Optional[TradeSignal]:
    """Full signal evaluation pipeline. Returns TradeSignal or None."""
    # Gate checks
    if confidence < min_confidence:
        log.debug("Signal rejected: confidence %.3f < %.3f", confidence, min_confidence)
        return None
    if points_observed < min_points:
        log.debug("Signal rejected: %d points < %d minimum", points_observed, min_points)
        return None

    side, edge, price_cents = compute_edge(model_prob, best_ask_cents, best_bid_cents)
    if side is None or edge < min_edge:
        log.debug("Signal rejected: edge %.4f < %.4f (side=%s)", edge, min_edge, side)
        return None

    # Net dollar edge after fees
    net_dollar = edge * 100 - fee_per_contract  # per contract
    if net_dollar < net_edge_min:
        log.debug("Signal rejected: net dollar edge $%.3f < $%.3f", net_dollar, net_edge_min)
        return None

    contracts = quarter_kelly_size(edge, price_cents, bankroll, max_contracts, fee_per_contract)
    if contracts <= 0:
        log.debug("Signal rejected: kelly size = 0")
        return None

    signal = TradeSignal(
        ticker=ticker,
        side=side,
        edge=edge,
        contracts=contracts,
        price_cents=price_cents,
        model_prob=model_prob,
        confidence=confidence,
    )
    log.info("Trade signal: %s %s %d @ %dc (edge=%.3f, prob=%.3f)",
             side.upper(), ticker, contracts, price_cents, edge, model_prob)
    return signal
