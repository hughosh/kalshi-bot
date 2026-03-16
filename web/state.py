"""BotState — shared live configuration between the bot hot path and the web server.

All reads/writes happen within a single asyncio event loop, so no threading
locks are required. The web server's POST /api/config mutates this object;
bot._on_point reads it on every point event.
"""
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

import config as _cfg
from web.db import get_config, set_config

log = logging.getLogger(__name__)

# Keys that map to numeric (float/int) fields on BotState
_FLOAT_KEYS = {
    "min_edge_pct",
    "min_confidence",
    "max_total_exposure",
    "match_stop_loss_dollars",
    "portfolio_stop_loss_dollars",
    "profit_target_dollars",
}
_INT_KEYS = {
    "min_points_observed",
    "max_contracts_per_match",
}
_BOOL_KEYS = {
    "trading_enabled",
    "paper_mode",
}


class BotState:
    """Runtime-adjustable bot configuration, persisted to bot_config table."""

    def __init__(self, paper: bool, db_conn: "aiosqlite.Connection") -> None:
        self._db = db_conn
        self.paper_mode: bool = paper

        # Trading thresholds — defaults from config.py
        self.trading_enabled: bool = True
        self.min_edge_pct: float = _cfg.MIN_EDGE_PCT
        self.min_confidence: float = _cfg.MIN_CONFIDENCE
        self.min_points_observed: int = _cfg.MIN_POINTS_OBSERVED
        self.max_contracts_per_match: int = _cfg.MAX_CONTRACTS_PER_MATCH
        self.max_total_exposure: float = _cfg.MAX_TOTAL_EXPOSURE

        # Stop-loss / profit controls
        self.match_stop_loss_dollars: float = _cfg.MATCH_STOP_LOSS_DOLLARS
        self.portfolio_stop_loss_dollars: float = _cfg.PORTFOLIO_STOP_LOSS_DOLLARS
        self.profit_target_dollars: float = _cfg.PROFIT_TARGET_DOLLARS

    async def load_from_db(self) -> None:
        """Apply any persisted overrides from the bot_config table."""
        stored = await get_config(self._db)
        for key, raw in stored.items():
            self._apply(key, raw, persist=False)
        if stored:
            log.info("BotState: loaded %d config overrides from DB", len(stored))

    async def apply_update(self, key: str, value: str) -> None:
        """Update in-memory and persist to bot_config table."""
        changed = self._apply(key, value, persist=False)
        if changed:
            await set_config(self._db, key, value)

    def _apply(self, key: str, raw: str, *, persist: bool = True) -> bool:
        """Parse and apply a single config key. Returns True if recognised."""
        try:
            if key in _BOOL_KEYS:
                val = raw.lower() in ("1", "true", "yes")
                setattr(self, key, val)
            elif key in _FLOAT_KEYS:
                setattr(self, key, float(raw))
            elif key in _INT_KEYS:
                setattr(self, key, int(raw))
            else:
                return False
        except (ValueError, TypeError) as e:
            log.warning("BotState: bad value for %s=%r: %s", key, raw, e)
            return False
        log.debug("BotState: %s = %s", key, raw)
        return True

    def to_dict(self) -> dict:
        return {
            "trading_enabled":             self.trading_enabled,
            "paper_mode":                  self.paper_mode,
            "min_edge_pct":                self.min_edge_pct,
            "min_confidence":              self.min_confidence,
            "min_points_observed":         self.min_points_observed,
            "max_contracts_per_match":     self.max_contracts_per_match,
            "max_total_exposure":          self.max_total_exposure,
            "match_stop_loss_dollars":     self.match_stop_loss_dollars,
            "portfolio_stop_loss_dollars": self.portfolio_stop_loss_dollars,
            "profit_target_dollars":       self.profit_target_dollars,
        }
