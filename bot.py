"""Live tennis trading bot entry point.

Usage:
  python bot.py                   # paper trading (default)
  python bot.py --live            # real execution (Kalshi demo env)
  python bot.py --live --prod     # real execution (Kalshi production)

Web portal:
  Starts automatically at http://127.0.0.1:8765 (configure WEB_PORT in .env)

Environment variables:
  API_TENNIS_KEY, KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH (see config.py)
"""
import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import uvicorn

import config
from integration.edge import TradeSignal, evaluate_signal
from integration.kalshi_client import KalshiClient, HybridKalshiClient, KalshiError
from integration.market_finder import MarketFinder, MarketMatch
from integration.match_tracker import MatchTracker
from integration.player_resolver import PlayerResolver
from integration.risk import RiskManager
from integration.tennis_feed import PointEvent, TennisFeed
from tennis.engine import PredictionEngine
from web.app import BroadcastHub, create_app
from web.db import (
    init_db, insert_trade, insert_signal,
    upsert_match_state, migrate_json_log,
)
from web.guard import TradingGuard
from web.state import BotState

log = logging.getLogger("bot")


class TradingBot:
    """Orchestrates the live trading loop.

    Connects the TennisFeed → MatchTracker → Edge/Risk/Guard → KalshiClient
    pipeline, with a co-resident FastAPI web portal for monitoring and control.
    """

    def __init__(
        self,
        paper: bool = True,
        demo: bool = True,
    ) -> None:
        self.paper = paper
        self.demo = demo
        self._kalshi: KalshiClient | None = None
        self._market_finder: MarketFinder | None = None
        self._match_tracker: MatchTracker | None = None
        self._risk: RiskManager | None = None
        self._feed: TennisFeed | None = None
        # Web portal components
        self._db_conn = None
        self._bot_state: BotState | None = None
        self._hub: BroadcastHub | None = None
        self._guard: TradingGuard | None = None

    async def start(self) -> None:
        """Load artifacts, wire components, and run the main loop."""
        mode = "PAPER" if self.paper else ("DEMO" if self.demo else "PRODUCTION")
        log.info("=== Tennis Trading Bot starting [%s mode] ===", mode)

        # ── Persistence layer ──────────────────────────────────────────
        self._db_conn = await init_db(config.DB_PATH)
        await migrate_json_log(self._db_conn, "data/trade_log.json")

        # ── Shared bot state (thresholds, trading_enabled, etc.) ───────
        self._bot_state = BotState(paper=self.paper, db_conn=self._db_conn)
        await self._bot_state.load_from_db()

        # ── WebSocket broadcast hub ────────────────────────────────────
        self._hub = BroadcastHub()

        # ── Load model artifacts ───────────────────────────────────────
        log.info("Loading model artifacts from %s ...", config.ARTIFACTS_DIR)
        engine = PredictionEngine.load(config.ARTIFACTS_DIR)

        model        = engine._model
        calibrator   = engine._calibrator
        elo_tracker  = engine._elo
        player_stats = engine._player_stats
        h2h          = engine._h2h

        # ── Player resolver ────────────────────────────────────────────
        resolver = PlayerResolver(config.ALIASES_PATH, config.ATP_RAW_DIR)

        # ── Match tracker ──────────────────────────────────────────────
        self._match_tracker = MatchTracker(
            model=model,
            calibrator=calibrator,
            elo_tracker=elo_tracker,
            player_stats_df=player_stats,
            h2h_df=h2h,
            resolver=resolver,
            n_mc_samples=200,
        )

        # ── Risk manager ───────────────────────────────────────────────
        self._risk = RiskManager(
            max_contracts_per_match=config.MAX_CONTRACTS_PER_MATCH,
            max_total_exposure=config.MAX_TOTAL_EXPOSURE,
            fee_per_contract=config.KALSHI_FEE_PER_CONTRACT,
        )

        # ── Kalshi client ──────────────────────────────────────────────
        if config.HYBRID_PRICING and config.PROD_KALSHI_API_KEY:
            log.info("Using hybrid client: production prices + demo trading")
            self._kalshi = HybridKalshiClient(
                prod_api_key=config.PROD_KALSHI_API_KEY,
                prod_key_path=config.PROD_KALSHI_PRIVATE_KEY_PATH,
                demo_api_key=config.DEMO_KALSHI_API_KEY,
                demo_key_path=config.DEMO_KALSHI_PRIVATE_KEY_PATH,
            )
        else:
            if self.demo and config.DEMO_KALSHI_API_KEY:
                kalshi_key = config.DEMO_KALSHI_API_KEY
                kalshi_pem = config.DEMO_KALSHI_PRIVATE_KEY_PATH
            else:
                kalshi_key = config.KALSHI_API_KEY
                kalshi_pem = config.KALSHI_PRIVATE_KEY_PATH

            self._kalshi = KalshiClient(
                api_key=kalshi_key,
                private_key_path=kalshi_pem,
                demo=self.demo,
            )

        # ── Market finder ──────────────────────────────────────────────
        self._market_finder = MarketFinder(
            kalshi=self._kalshi,
            refresh_secs=config.MARKET_REFRESH_SECS,
        )

        # ── Trading guard (stop-loss, profit target, pause) ───────────
        self._guard = TradingGuard(
            state=self._bot_state,
            db_conn=self._db_conn,
            hub=self._hub,
        )
        self._guard.kalshi = self._kalshi

        # ── FastAPI web portal in the same event loop ──────────────────
        fastapi_app = create_app(
            state=self._bot_state,
            db_conn=self._db_conn,
            hub=self._hub,
            kalshi=self._kalshi,
            guard=self._guard,
            risk=self._risk,
        )
        uvi_config = uvicorn.Config(
            fastapi_app,
            host="127.0.0.1",
            port=config.WEB_PORT,
            loop="none",          # attach to the already-running asyncio loop
            log_level="warning",
        )
        uvi_server = uvicorn.Server(uvi_config)
        loop = asyncio.get_event_loop()
        loop.create_task(uvi_server.serve(), name="uvicorn")
        loop.create_task(self._hub._dispatch_loop(), name="ws-dispatch")
        log.info("Web portal: http://127.0.0.1:%d", config.WEB_PORT)

        # ── Tennis feed ────────────────────────────────────────────────
        self._feed = TennisFeed(
            api_key=config.API_TENNIS_KEY,
            on_point=self._on_point,
            rest_poll_secs=config.REST_POLL_SECS,
        )

        # ── Run ────────────────────────────────────────────────────────
        async with self._kalshi:
            log.info("Bot running. Press Ctrl+C to stop.")
            try:
                await self._feed.run()
            except asyncio.CancelledError:
                log.info("Bot cancelled.")
            finally:
                await self._shutdown()

    async def _on_point(self, event: PointEvent) -> None:
        """Callback for each live point event from the feed."""
        # 1. Update match tracker → get prediction
        result = self._match_tracker.process_point(event)
        if result is None:
            return

        session = self._match_tracker.get_session(event.match_id)
        if session is None:
            return

        log.debug(
            "Match %s: %s vs %s — P1 win=%.3f conf=%.3f (pts=%d)",
            event.match_id, event.player1_name, event.player2_name,
            result.win_prob_p1, result.confidence, result.points_observed,
        )

        # 2. Find Kalshi market for this match
        market = await self._market_finder.find(event.player1_name, event.player2_name)

        # 3. Prices from the market listing
        prices = market.prices if market else None

        # 4. Mark-to-market for guard (even before signal evaluation)
        if market and self._guard:
            self._guard.update_mark_to_market(
                event.match_id, market.ticker,
                event.player1_name, event.player2_name,
                prices.best_ask_cents if prices else None,
                prices.best_bid_cents if prices else None,
            )

        # 5. Persist match state snapshot to DB and broadcast to portal
        match_state_row = {
            "match_id":      event.match_id,
            "player1":       event.player1_name,
            "player2":       event.player2_name,
            "ticker":        market.ticker if market else None,
            "sets_p1":       event.sets_p1,
            "sets_p2":       event.sets_p2,
            "games_p1":      event.games_p1,
            "games_p2":      event.games_p2,
            "points_p1":     event.points_p1,
            "points_p2":     event.points_p2,
            "server":        event.server,
            "is_tiebreak":   event.is_tiebreak,
            "win_prob_p1":   result.win_prob_p1,
            "confidence":    result.confidence,
            "ci_lower":      result.ci_lower,
            "ci_upper":      result.ci_upper,
            "best_ask_cents": prices.best_ask_cents if prices else None,
            "best_bid_cents": prices.best_bid_cents if prices else None,
            "is_active":     True,
            "updated_at":    time.time(),
        }
        await upsert_match_state(self._db_conn, match_state_row)
        self._hub.broadcast_sync({"type": "match_update", "data": match_state_row})

        if market is None:
            return

        # 6. Read thresholds from BotState (runtime-adjustable)
        bs = self._bot_state
        model_prob = result.win_prob_p1

        # 7. Evaluate trade signal
        signal = evaluate_signal(
            ticker=market.ticker,
            model_prob=model_prob,
            confidence=result.confidence,
            points_observed=result.points_observed,
            best_ask_cents=prices.best_ask_cents,
            best_bid_cents=prices.best_bid_cents,
            bankroll=bs.max_total_exposure - self._risk.total_exposure,
            min_edge=bs.min_edge_pct,
            min_confidence=bs.min_confidence,
            min_points=bs.min_points_observed,
            net_edge_min=config.NET_EDGE_MIN_DOLLARS,
            max_contracts=bs.max_contracts_per_match,
            fee_per_contract=config.KALSHI_FEE_PER_CONTRACT,
        )

        # Build base signal record for the DB (completed below)
        sig_base = {
            "timestamp":       time.time(),
            "match_id":        event.match_id,
            "ticker":          market.ticker,
            "model_prob":      round(result.win_prob_p1, 4),
            "confidence":      round(result.confidence, 4),
            "points_observed": result.points_observed,
            "best_ask_cents":  prices.best_ask_cents,
            "best_bid_cents":  prices.best_bid_cents,
            "edge":            round(signal.edge, 4) if signal else None,
            "side":            signal.side if signal else None,
            "contracts":       signal.contracts if signal else None,
        }

        if signal is None:
            await insert_signal(self._db_conn, {**sig_base, "decision": "skipped", "skip_reason": "no_edge"})
            return

        # 8. TradingGuard check (stop-loss, pause, force-exit)
        allowed_by_guard, guard_reason = self._guard.should_trade(
            event.match_id, signal.ticker, signal
        )
        if not allowed_by_guard:
            await insert_signal(
                self._db_conn,
                {**sig_base, "decision": "guard_blocked", "skip_reason": guard_reason},
            )
            # Update match card with last signal info
            match_state_row["last_signal"] = {
                "decision": "guard_blocked", "skip_reason": guard_reason,
                "timestamp": sig_base["timestamp"],
            }
            self._hub.broadcast_sync({"type": "match_update", "data": match_state_row})
            return

        # 9. Risk check
        allowed = self._risk.check_and_reduce(
            signal.ticker, signal.side, signal.contracts, signal.price_cents
        )
        if allowed <= 0:
            await insert_signal(
                self._db_conn,
                {**sig_base, "decision": "risk_blocked", "skip_reason": "exposure_limit"},
            )
            return

        # 10. Execute or log
        await self._execute_signal(
            signal, allowed, market, result, event, sig_base
        )

    async def _execute_signal(
        self,
        signal: TradeSignal,
        contracts: int,
        market: MarketMatch,
        result,
        event: PointEvent,
        sig_base: dict,
    ) -> None:
        """Execute a trade (paper or live) and persist to DB."""
        trade_id = str(uuid.uuid4())
        trade_record: dict = {
            "trade_id":    trade_id,
            "timestamp":   time.time(),
            "match_id":    event.match_id,
            "player1":     event.player1_name,
            "player2":     event.player2_name,
            "ticker":      signal.ticker,
            "side":        signal.side,
            "contracts":   contracts,
            "price_cents": signal.price_cents,
            "edge":        round(signal.edge, 4),
            "model_prob":  round(signal.model_prob, 4),
            "confidence":  round(signal.confidence, 4),
            "score":       f"{event.sets_p1}-{event.sets_p2} {event.games_p1}-{event.games_p2}",
            "mode":        "paper" if self.paper else "live",
            "order_id":    None,
            "error":       None,
            "pnl_cents":   None,
            "settled_at":  None,
        }

        if self.paper:
            log.info(
                "PAPER TRADE: %s %s %d @ %dc | edge=%.3f | %s vs %s | %s",
                signal.side.upper(), signal.ticker, contracts, signal.price_cents,
                signal.edge, event.player1_name, event.player2_name,
                trade_record["score"],
            )
            self._risk.confirm_fill(signal.ticker, signal.side, contracts, signal.price_cents)
        else:
            try:
                order_id = str(uuid.uuid4())
                resp = await self._kalshi.place_order(
                    ticker=signal.ticker,
                    side=signal.side,
                    contracts=contracts,
                    price_cents=signal.price_cents,
                    client_order_id=order_id,
                )
                trade_record["order_id"] = order_id
                self._risk.confirm_fill(signal.ticker, signal.side, contracts, signal.price_cents)
                log.info(
                    "LIVE ORDER: %s %s %d @ %dc | order_id=%s",
                    signal.side.upper(), signal.ticker, contracts, signal.price_cents, order_id,
                )
            except KalshiError as e:
                log.error("Order failed: %s", e)
                trade_record["error"] = str(e)

        # Persist trade immediately (not just on shutdown)
        await insert_trade(self._db_conn, trade_record)

        # Update guard position tracking
        if not trade_record.get("error"):
            self._guard.record_fill(
                event.match_id, signal.ticker,
                event.player1_name, event.player2_name,
                signal.side, contracts, signal.price_cents,
                config.KALSHI_FEE_PER_CONTRACT,
            )

        # Record signal as traded
        await insert_signal(
            self._db_conn,
            {**sig_base, "decision": "traded", "trade_id": trade_id},
        )

        # Broadcast trade to portal
        self._hub.broadcast_sync({"type": "trade", "data": trade_record})

        # Broadcast updated risk summary
        risk_data = {
            **self._risk.summary(),
            **self._guard.portfolio_pnl_summary,
            "trading_enabled": self._bot_state.trading_enabled,
            "paper_mode":      self._bot_state.paper_mode,
        }
        self._hub.broadcast_sync({"type": "risk_update", "data": risk_data})

    async def _shutdown(self) -> None:
        """Clean up on exit."""
        if self._feed:
            await self._feed.stop()

        if self._db_conn:
            await self._db_conn.close()
            log.info("DB connection closed.")

        if self._risk:
            log.info("Final risk summary: %s", self._risk.summary())

        log.info("Bot shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Tennis trading bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--prod", action="store_true", help="Use production Kalshi (default: demo)")
    parser.add_argument("--log-level", default=config.LOG_LEVEL, help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    paper = not args.live
    demo = not args.prod

    if not paper:
        if demo:
            api_key = config.DEMO_KALSHI_API_KEY or config.KALSHI_API_KEY
            key_path = config.DEMO_KALSHI_PRIVATE_KEY_PATH or config.KALSHI_PRIVATE_KEY_PATH
        else:
            api_key = config.KALSHI_API_KEY
            key_path = config.KALSHI_PRIVATE_KEY_PATH
        if not api_key:
            log.error("KALSHI_API_KEY (or DEMO_KALSHI_API_KEY) required for live trading")
            sys.exit(1)
        if not key_path:
            log.error("KALSHI_PRIVATE_KEY_PATH (or DEMO_KALSHI_PRIVATE_KEY_PATH) required for live trading")
            sys.exit(1)

    if not config.API_TENNIS_KEY:
        log.error("API_TENNIS_KEY required. Set it in your environment.")
        sys.exit(1)

    bot = TradingBot(paper=paper, demo=demo)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")


if __name__ == "__main__":
    main()
