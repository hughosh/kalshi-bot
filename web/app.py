"""FastAPI web application for the trading portal.

Serves the static dashboard and exposes REST + WebSocket endpoints.
Runs in the same asyncio event loop as bot.py via uvicorn with loop="none".
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    import aiosqlite
    from integration.kalshi_client import KalshiClient
    from web.guard import TradingGuard
    from web.state import BotState

import web.db as db

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Broadcast hub
# ---------------------------------------------------------------------------

class BroadcastHub:
    """Fan-out WebSocket hub.

    The bot hot path calls broadcast_sync() (non-blocking) or broadcast()
    (async); both push to an asyncio.Queue which the _dispatch_loop drains
    and fans out to all connected browser clients.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=512)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        log.debug("WS client connected (%d total)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.debug("WS client disconnected (%d total)", len(self._clients))

    def broadcast_sync(self, message: dict) -> None:
        """Non-blocking put from bot hot path. Drops if queue full."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    async def broadcast(self, message: dict) -> None:
        """Async put; awaits if queue is full (use from async contexts only)."""
        await self._queue.put(message)

    async def _dispatch_loop(self) -> None:
        """Background task: drain queue and fan-out to all WS clients."""
        while True:
            try:
                msg = await self._queue.get()
                if not self._clients:
                    continue
                text = json.dumps(msg)
                dead: set[WebSocket] = set()
                for ws in list(self._clients):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        dead.add(ws)
                self._clients -= dead
            except Exception as e:
                log.debug("dispatch_loop error: %s", e)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    key: str
    value: str


class TradingModeUpdate(BaseModel):
    mode: str  # "paper" | "live"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    state: "BotState",
    db_conn: "aiosqlite.Connection",
    hub: BroadcastHub,
    kalshi: Optional["KalshiClient"],
    guard: Optional["TradingGuard"] = None,
    risk=None,  # RiskManager — passed in so /api/risk can query it
) -> FastAPI:

    app = FastAPI(title="Tennis Bot Portal", docs_url=None, redoc_url=None)

    # Mount static files
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    # ------------------------------------------------------------------
    # REST — read endpoints
    # ------------------------------------------------------------------

    @app.get("/api/trades")
    async def get_trades(
        limit: int = 100,
        offset: int = 0,
        match_id: Optional[str] = None,
    ):
        trades = await db.get_trades(db_conn, limit=limit, offset=offset, match_id=match_id)
        total = await db.count_trades(db_conn, match_id=match_id)
        return {"trades": trades, "total": total}

    @app.get("/api/signals")
    async def get_signals(
        match_id: Optional[str] = None,
        limit: int = 100,
    ):
        signals = await db.get_signals(db_conn, match_id=match_id, limit=limit)
        return {"signals": signals}

    @app.get("/api/matches")
    async def get_matches():
        matches = await db.get_active_matches(db_conn)
        return {"matches": matches}

    @app.get("/api/risk")
    async def get_risk():
        risk_summary = risk.summary() if risk else {}
        guard_summary = guard.portfolio_pnl_summary if guard else {}
        return {
            **risk_summary,
            **guard_summary,
            "trading_enabled": state.trading_enabled,
            "paper_mode": state.paper_mode,
        }

    @app.get("/api/config")
    async def get_config():
        return state.to_dict()

    # ------------------------------------------------------------------
    # REST — write endpoints
    # ------------------------------------------------------------------

    @app.post("/api/config")
    async def post_config(update: ConfigUpdate):
        await state.apply_update(update.key, update.value)
        cfg = state.to_dict()
        hub.broadcast_sync({"type": "config_update", "data": cfg})
        return cfg

    @app.post("/api/trading/toggle")
    async def toggle_trading():
        state.trading_enabled = not state.trading_enabled
        await db.set_config(db_conn, "trading_enabled", str(state.trading_enabled))
        cfg = state.to_dict()
        hub.broadcast_sync({"type": "config_update", "data": cfg})
        status = "enabled" if state.trading_enabled else "disabled"
        log.info("Trading %s via web portal", status)
        return {"trading_enabled": state.trading_enabled}

    @app.post("/api/trading/mode")
    async def set_trading_mode(update: TradingModeUpdate):
        if update.mode not in ("paper", "live"):
            raise HTTPException(400, "mode must be 'paper' or 'live'")
        state.paper_mode = update.mode == "paper"
        await db.set_config(db_conn, "paper_mode", str(state.paper_mode))
        hub.broadcast_sync({"type": "config_update", "data": state.to_dict()})
        return {"paper_mode": state.paper_mode}

    @app.post("/api/matches/{match_id}/exit")
    async def exit_match(match_id: str):
        if guard is None:
            raise HTTPException(503, "Guard not initialised")
        guard.force_exit(match_id)
        hub.broadcast_sync({
            "type": "alert",
            "data": {"level": "info", "message": f"Force exit requested for {match_id}"},
        })
        return {"match_id": match_id, "status": "exit_requested"}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await hub.connect(ws)
        # Send full current state on connect so the page populates immediately
        try:
            matches = await db.get_active_matches(db_conn)
            await ws.send_text(json.dumps({"type": "init", "data": {
                "config": state.to_dict(),
                "matches": matches,
            }}))
        except Exception:
            pass
        try:
            while True:
                # Keep connection alive; ignore inbound (controls use REST)
                await ws.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(ws)
        except Exception:
            hub.disconnect(ws)

    return app
