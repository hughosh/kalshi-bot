"""TradingGuard — stop-loss, profit-target, pause/resume, and force-exit logic.

Inserted between signal evaluation and order execution in bot._on_point.
No changes to integration/ modules required.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import aiosqlite
    from integration.kalshi_client import KalshiClient
    from web.app import BroadcastHub
    from web.state import BotState

log = logging.getLogger(__name__)


@dataclass
class MatchGuardState:
    match_id: str
    ticker: str
    player1: str
    player2: str
    # YES position (positive = holding YES contracts)
    yes_contracts: int = 0
    no_contracts: int = 0
    cost_basis_cents: int = 0       # total cash spent (including fees)
    unrealized_pnl_cents: int = 0
    stop_loss_triggered: bool = False
    profit_target_triggered: bool = False
    force_exit_requested: bool = False

    @property
    def contracts_held(self) -> int:
        return self.yes_contracts + self.no_contracts


class TradingGuard:
    """Risk gate and PnL tracker sitting in front of order execution.

    Usage in bot._on_point:
        guard.update_mark_to_market(match_id, ticker, ask, bid)
        allowed, reason = guard.should_trade(match_id, ticker, signal)
        if not allowed:
            ...record guard_blocked signal...
            return
        # proceed to place order
        guard.record_fill(match_id, ticker, side, contracts, price_cents)
    """

    def __init__(
        self,
        state: "BotState",
        db_conn: "aiosqlite.Connection",
        hub: "BroadcastHub",
    ) -> None:
        self._state = state
        self._db = db_conn
        self._hub = hub
        self._matches: dict[str, MatchGuardState] = {}
        self._portfolio_halted: bool = False
        # kalshi client injected after construction (set by bot.py)
        self.kalshi: Optional["KalshiClient"] = None

    # ------------------------------------------------------------------
    # Public interface called from bot._on_point
    # ------------------------------------------------------------------

    def should_trade(
        self,
        match_id: str,
        ticker: str,
        signal=None,  # TradeSignal | None
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Called before every potential order."""
        if not self._state.trading_enabled:
            return False, "trading_disabled"

        if self._portfolio_halted:
            return False, "portfolio_stop_loss"

        gs = self._matches.get(match_id)
        if gs is None:
            return True, ""

        if gs.stop_loss_triggered:
            return False, "match_stop_loss"

        if gs.profit_target_triggered:
            return False, "profit_target_reached"

        if gs.force_exit_requested:
            # Caller must handle the actual exit order asynchronously;
            # we just block further entries here.
            return False, "force_exit"

        return True, ""

    def record_fill(
        self,
        match_id: str,
        ticker: str,
        player1: str,
        player2: str,
        side: str,
        contracts: int,
        price_cents: int,
        fee_per_contract: float = 0.07,
    ) -> None:
        """Update guard state after a confirmed order fill."""
        gs = self._get_or_create(match_id, ticker, player1, player2)
        cost = contracts * (price_cents / 100 + fee_per_contract)
        gs.cost_basis_cents += int(round(cost * 100))
        if side == "yes":
            gs.yes_contracts += contracts
        else:
            gs.no_contracts += contracts
        log.debug(
            "Guard fill: %s %s %d@%d¢ cost_basis=%d¢",
            match_id, side, contracts, price_cents, gs.cost_basis_cents,
        )

    def update_mark_to_market(
        self,
        match_id: str,
        ticker: str,
        player1: str,
        player2: str,
        ask_cents: Optional[int],
        bid_cents: Optional[int],
    ) -> None:
        """Recompute unrealized PnL and check thresholds."""
        gs = self._matches.get(match_id)
        if gs is None or gs.contracts_held == 0:
            return

        mid = None
        if ask_cents and bid_cents:
            mid = (ask_cents + bid_cents) / 2
        elif ask_cents:
            mid = ask_cents
        elif bid_cents:
            mid = bid_cents

        if mid is None:
            return

        # Mark-to-market: value of position at current mid price
        position_value_cents = 0
        if gs.yes_contracts > 0:
            position_value_cents += int(gs.yes_contracts * mid)
        if gs.no_contracts > 0:
            position_value_cents += int(gs.no_contracts * (100 - mid))

        gs.unrealized_pnl_cents = position_value_cents - gs.cost_basis_cents
        self._check_match_thresholds(match_id)
        self._check_portfolio_stop_loss()

    def force_exit(self, match_id: str) -> None:
        """Request a force-exit for a match; picked up on next point cycle."""
        gs = self._matches.get(match_id)
        if gs is None:
            log.warning("Guard force_exit: unknown match_id %s", match_id)
            return
        gs.force_exit_requested = True
        log.info("Guard: force exit requested for %s", match_id)

    def release_match(self, match_id: str) -> None:
        """Called when a match finishes; removes guard state."""
        self._matches.pop(match_id, None)

    @property
    def portfolio_pnl_summary(self) -> dict:
        total_unrealized = sum(
            gs.unrealized_pnl_cents for gs in self._matches.values()
        )
        return {
            "total_unrealized_pnl_cents": total_unrealized,
            "portfolio_halted": self._portfolio_halted,
            "matches": {
                mid: {
                    "ticker":                  gs.ticker,
                    "player1":                 gs.player1,
                    "player2":                 gs.player2,
                    "yes_contracts":           gs.yes_contracts,
                    "no_contracts":            gs.no_contracts,
                    "cost_basis_cents":        gs.cost_basis_cents,
                    "unrealized_pnl_cents":    gs.unrealized_pnl_cents,
                    "stop_loss_triggered":     gs.stop_loss_triggered,
                    "profit_target_triggered": gs.profit_target_triggered,
                    "force_exit_requested":    gs.force_exit_requested,
                }
                for mid, gs in self._matches.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(
        self, match_id: str, ticker: str, player1: str, player2: str
    ) -> MatchGuardState:
        if match_id not in self._matches:
            self._matches[match_id] = MatchGuardState(
                match_id=match_id,
                ticker=ticker,
                player1=player1,
                player2=player2,
            )
        return self._matches[match_id]

    def _check_match_thresholds(self, match_id: str) -> None:
        gs = self._matches.get(match_id)
        if gs is None:
            return

        pnl_dollars = gs.unrealized_pnl_cents / 100

        if (
            not gs.stop_loss_triggered
            and pnl_dollars < -self._state.match_stop_loss_dollars
        ):
            gs.stop_loss_triggered = True
            msg = (
                f"Stop loss triggered for {gs.player1} vs {gs.player2}: "
                f"PnL ${pnl_dollars:.2f} < -${self._state.match_stop_loss_dollars:.2f}"
            )
            log.warning(msg)
            self._broadcast_alert("warning", msg)

        if (
            not gs.profit_target_triggered
            and pnl_dollars > self._state.profit_target_dollars
        ):
            gs.profit_target_triggered = True
            self.force_exit(match_id)
            msg = (
                f"Profit target reached for {gs.player1} vs {gs.player2}: "
                f"PnL ${pnl_dollars:.2f} > ${self._state.profit_target_dollars:.2f}"
            )
            log.info(msg)
            self._broadcast_alert("info", msg)

    def _check_portfolio_stop_loss(self) -> None:
        if self._portfolio_halted:
            return
        total_pnl_dollars = (
            sum(gs.unrealized_pnl_cents for gs in self._matches.values()) / 100
        )
        if total_pnl_dollars < -self._state.portfolio_stop_loss_dollars:
            self._portfolio_halted = True
            msg = (
                f"Portfolio stop loss triggered: "
                f"total PnL ${total_pnl_dollars:.2f} < "
                f"-${self._state.portfolio_stop_loss_dollars:.2f}. "
                f"All trading halted."
            )
            log.warning(msg)
            self._broadcast_alert("error", msg)

    def _broadcast_alert(self, level: str, message: str) -> None:
        try:
            self._hub.broadcast_sync({"type": "alert", "data": {"level": level, "message": message}})
        except Exception:
            pass
