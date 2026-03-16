"""Risk management: per-match and portfolio-level exposure limits.

Limits enforced:
  1. Per-match contract cap (MAX_CONTRACTS_PER_MATCH = 50)
  2. Total dollar exposure cap (MAX_TOTAL_EXPOSURE = $500)
  3. Reduces signal size to stay within limits (never exceeds)
"""
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class MatchExposure:
    """Tracks contracts held for a single match."""
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0

    @property
    def total_contracts(self) -> int:
        return self.yes_contracts + self.no_contracts


class RiskManager:
    """Enforces position limits across the portfolio.

    Args:
        max_contracts_per_match: Hard cap on contracts per match ticker.
        max_total_exposure: Dollar cap across all active positions.
        fee_per_contract: Kalshi fee per contract.
    """

    def __init__(
        self,
        max_contracts_per_match: int = 50,
        max_total_exposure: float = 500.0,
        fee_per_contract: float = 0.07,
    ) -> None:
        self._max_per_match = max_contracts_per_match
        self._max_exposure = max_total_exposure
        self._fee = fee_per_contract
        self._matches: dict[str, MatchExposure] = {}  # ticker → exposure
        self._total_cost: float = 0.0  # cumulative dollar cost of open positions

    def check_and_reduce(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
    ) -> int:
        """Return the number of contracts allowed (may be reduced or zero).

        Updates internal state if contracts > 0 are returned.
        Caller must actually place the order and call confirm_fill or rollback.
        """
        if contracts <= 0:
            return 0

        exposure = self._matches.get(ticker, MatchExposure(ticker=ticker))

        # Per-match limit
        room_match = self._max_per_match - exposure.total_contracts
        if room_match <= 0:
            log.info("Risk: match %s at cap (%d contracts)", ticker, exposure.total_contracts)
            return 0
        allowed = min(contracts, room_match)

        # Total exposure limit
        cost_per = price_cents / 100.0 + self._fee
        room_dollars = self._max_exposure - self._total_cost
        if room_dollars <= 0:
            log.info("Risk: total exposure at cap ($%.2f)", self._total_cost)
            return 0
        max_by_dollars = int(room_dollars / cost_per) if cost_per > 0 else 0
        allowed = min(allowed, max_by_dollars)

        if allowed <= 0:
            return 0

        if allowed < contracts:
            log.info("Risk: reduced %s %d→%d contracts (match=%d, exposure=$%.2f)",
                     ticker, contracts, allowed, exposure.total_contracts, self._total_cost)

        return allowed

    def confirm_fill(self, ticker: str, side: str, contracts: int, price_cents: int) -> None:
        """Record a filled order."""
        if contracts <= 0:
            return
        if ticker not in self._matches:
            self._matches[ticker] = MatchExposure(ticker=ticker)
        exp = self._matches[ticker]
        if side == "yes":
            exp.yes_contracts += contracts
        else:
            exp.no_contracts += contracts
        cost = contracts * (price_cents / 100.0 + self._fee)
        self._total_cost += cost
        log.info("Risk: filled %s %s %d @ %dc — match=%d, total=$%.2f",
                 side, ticker, contracts, price_cents, exp.total_contracts, self._total_cost)

    def release_match(self, ticker: str) -> None:
        """Release exposure when a match settles."""
        if ticker in self._matches:
            exp = self._matches.pop(ticker)
            # Approximate cost release — in practice settlement PnL would adjust this
            released = exp.total_contracts * self._fee  # conservative: only release fee portion
            self._total_cost = max(0.0, self._total_cost - released)
            log.info("Risk: released match %s (%d contracts)", ticker, exp.total_contracts)

    @property
    def total_exposure(self) -> float:
        return self._total_cost

    @property
    def active_matches(self) -> int:
        return len(self._matches)

    def summary(self) -> dict:
        return {
            "total_exposure": round(self._total_cost, 2),
            "active_matches": self.active_matches,
            "matches": {
                t: {"yes": e.yes_contracts, "no": e.no_contracts, "total": e.total_contracts}
                for t, e in self._matches.items()
            },
        }
