"""SQLite persistence layer for the trading portal.

All functions accept an open aiosqlite.Connection and are safe to call
from within the asyncio event loop — aiosqlite runs SQLite in a thread
pool internally, so the event loop is never blocked.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    TEXT UNIQUE NOT NULL,
    timestamp   REAL NOT NULL,
    match_id    TEXT NOT NULL,
    player1     TEXT NOT NULL,
    player2     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,
    contracts   INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    edge        REAL NOT NULL,
    model_prob  REAL NOT NULL,
    confidence  REAL NOT NULL,
    score       TEXT NOT NULL DEFAULT '',
    mode        TEXT NOT NULL,
    order_id    TEXT,
    error       TEXT,
    pnl_cents   INTEGER,
    settled_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_match  ON trades(match_id);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades(timestamp DESC);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    match_id        TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    model_prob      REAL NOT NULL,
    confidence      REAL NOT NULL,
    points_observed INTEGER NOT NULL,
    best_ask_cents  INTEGER,
    best_bid_cents  INTEGER,
    edge            REAL,
    side            TEXT,
    contracts       INTEGER,
    decision        TEXT NOT NULL,
    skip_reason     TEXT,
    trade_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_match ON signals(match_id);
CREATE INDEX IF NOT EXISTS idx_signals_ts    ON signals(timestamp DESC);

CREATE TABLE IF NOT EXISTS match_state (
    match_id        TEXT PRIMARY KEY,
    player1         TEXT NOT NULL,
    player2         TEXT NOT NULL,
    ticker          TEXT,
    sets_p1         INTEGER DEFAULT 0,
    sets_p2         INTEGER DEFAULT 0,
    games_p1        INTEGER DEFAULT 0,
    games_p2        INTEGER DEFAULT 0,
    points_p1       INTEGER DEFAULT 0,
    points_p2       INTEGER DEFAULT 0,
    server          INTEGER DEFAULT 0,
    is_tiebreak     INTEGER DEFAULT 0,
    win_prob_p1     REAL,
    confidence      REAL,
    ci_lower        REAL,
    ci_upper        REAL,
    best_ask_cents  INTEGER,
    best_bid_cents  INTEGER,
    edge            REAL,
    edge_side       TEXT,
    last_signal     TEXT,
    unrealized_pnl  REAL DEFAULT 0.0,
    contracts_held  INTEGER DEFAULT 0,
    is_active       INTEGER DEFAULT 1,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


async def init_db(db_path: str) -> aiosqlite.Connection:
    """Create tables if not exist; return open connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_SCHEMA)
    await conn.commit()
    log.info("DB initialised at %s", db_path)
    return conn


async def insert_trade(conn: aiosqlite.Connection, record: dict) -> None:
    await conn.execute(
        """INSERT OR IGNORE INTO trades
           (trade_id, timestamp, match_id, player1, player2, ticker,
            side, contracts, price_cents, edge, model_prob, confidence,
            score, mode, order_id, error, pnl_cents, settled_at)
           VALUES (:trade_id,:timestamp,:match_id,:player1,:player2,:ticker,
                   :side,:contracts,:price_cents,:edge,:model_prob,:confidence,
                   :score,:mode,:order_id,:error,:pnl_cents,:settled_at)""",
        {
            "trade_id":    record.get("trade_id", ""),
            "timestamp":   record.get("timestamp", time.time()),
            "match_id":    record.get("match_id", ""),
            "player1":     record.get("player1", ""),
            "player2":     record.get("player2", ""),
            "ticker":      record.get("ticker", ""),
            "side":        record.get("side", ""),
            "contracts":   record.get("contracts", 0),
            "price_cents": record.get("price_cents", 0),
            "edge":        record.get("edge", 0.0),
            "model_prob":  record.get("model_prob", 0.0),
            "confidence":  record.get("confidence", 0.0),
            "score":       record.get("score", ""),
            "mode":        record.get("mode", "paper"),
            "order_id":    record.get("order_id"),
            "error":       record.get("error"),
            "pnl_cents":   record.get("pnl_cents"),
            "settled_at":  record.get("settled_at"),
        },
    )
    await conn.commit()


async def insert_signal(conn: aiosqlite.Connection, record: dict) -> None:
    await conn.execute(
        """INSERT INTO signals
           (timestamp, match_id, ticker, model_prob, confidence,
            points_observed, best_ask_cents, best_bid_cents,
            edge, side, contracts, decision, skip_reason, trade_id)
           VALUES (:timestamp,:match_id,:ticker,:model_prob,:confidence,
                   :points_observed,:best_ask_cents,:best_bid_cents,
                   :edge,:side,:contracts,:decision,:skip_reason,:trade_id)""",
        {
            "timestamp":       record.get("timestamp", time.time()),
            "match_id":        record.get("match_id", ""),
            "ticker":          record.get("ticker", ""),
            "model_prob":      record.get("model_prob", 0.0),
            "confidence":      record.get("confidence", 0.0),
            "points_observed": record.get("points_observed", 0),
            "best_ask_cents":  record.get("best_ask_cents"),
            "best_bid_cents":  record.get("best_bid_cents"),
            "edge":            record.get("edge"),
            "side":            record.get("side"),
            "contracts":       record.get("contracts"),
            "decision":        record.get("decision", "skipped"),
            "skip_reason":     record.get("skip_reason"),
            "trade_id":        record.get("trade_id"),
        },
    )
    await conn.commit()


async def upsert_match_state(conn: aiosqlite.Connection, state: dict) -> None:
    await conn.execute(
        """INSERT OR REPLACE INTO match_state
           (match_id, player1, player2, ticker,
            sets_p1, sets_p2, games_p1, games_p2, points_p1, points_p2,
            server, is_tiebreak, win_prob_p1, confidence, ci_lower, ci_upper,
            best_ask_cents, best_bid_cents, edge, edge_side,
            last_signal, unrealized_pnl, contracts_held, is_active, updated_at)
           VALUES
           (:match_id,:player1,:player2,:ticker,
            :sets_p1,:sets_p2,:games_p1,:games_p2,:points_p1,:points_p2,
            :server,:is_tiebreak,:win_prob_p1,:confidence,:ci_lower,:ci_upper,
            :best_ask_cents,:best_bid_cents,:edge,:edge_side,
            :last_signal,:unrealized_pnl,:contracts_held,:is_active,:updated_at)""",
        {
            "match_id":      state.get("match_id", ""),
            "player1":       state.get("player1", ""),
            "player2":       state.get("player2", ""),
            "ticker":        state.get("ticker"),
            "sets_p1":       state.get("sets_p1", 0),
            "sets_p2":       state.get("sets_p2", 0),
            "games_p1":      state.get("games_p1", 0),
            "games_p2":      state.get("games_p2", 0),
            "points_p1":     state.get("points_p1", 0),
            "points_p2":     state.get("points_p2", 0),
            "server":        state.get("server", 0),
            "is_tiebreak":   int(state.get("is_tiebreak", False)),
            "win_prob_p1":   state.get("win_prob_p1"),
            "confidence":    state.get("confidence"),
            "ci_lower":      state.get("ci_lower"),
            "ci_upper":      state.get("ci_upper"),
            "best_ask_cents":state.get("best_ask_cents"),
            "best_bid_cents":state.get("best_bid_cents"),
            "edge":          state.get("edge"),
            "edge_side":     state.get("edge_side"),
            "last_signal":   json.dumps(state["last_signal"]) if state.get("last_signal") else None,
            "unrealized_pnl":state.get("unrealized_pnl", 0.0),
            "contracts_held":state.get("contracts_held", 0),
            "is_active":     int(state.get("is_active", True)),
            "updated_at":    state.get("updated_at", time.time()),
        },
    )
    await conn.commit()


async def deactivate_match(conn: aiosqlite.Connection, match_id: str) -> None:
    await conn.execute(
        "UPDATE match_state SET is_active=0, updated_at=? WHERE match_id=?",
        (time.time(), match_id),
    )
    await conn.commit()


async def get_trades(
    conn: aiosqlite.Connection,
    limit: int = 100,
    offset: int = 0,
    match_id: Optional[str] = None,
) -> list[dict]:
    if match_id:
        cur = await conn.execute(
            "SELECT * FROM trades WHERE match_id=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (match_id, limit, offset),
        )
    else:
        cur = await conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def count_trades(
    conn: aiosqlite.Connection, match_id: Optional[str] = None
) -> int:
    if match_id:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM trades WHERE match_id=?", (match_id,)
        )
    else:
        cur = await conn.execute("SELECT COUNT(*) FROM trades")
    row = await cur.fetchone()
    return row[0] if row else 0


async def get_signals(
    conn: aiosqlite.Connection,
    match_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    if match_id:
        cur = await conn.execute(
            "SELECT * FROM signals WHERE match_id=? ORDER BY timestamp DESC LIMIT ?",
            (match_id, limit),
        )
    else:
        cur = await conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_active_matches(conn: aiosqlite.Connection) -> list[dict]:
    cur = await conn.execute(
        "SELECT * FROM match_state WHERE is_active=1 ORDER BY updated_at DESC"
    )
    rows = await cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("last_signal"):
            try:
                d["last_signal"] = json.loads(d["last_signal"])
            except Exception:
                pass
        result.append(d)
    return result


async def get_match_state(
    conn: aiosqlite.Connection, match_id: str
) -> Optional[dict]:
    cur = await conn.execute(
        "SELECT * FROM match_state WHERE match_id=?", (match_id,)
    )
    row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("last_signal"):
        try:
            d["last_signal"] = json.loads(d["last_signal"])
        except Exception:
            pass
    return d


async def settle_trade(
    conn: aiosqlite.Connection,
    trade_id: str,
    pnl_cents: int,
    settled_at: float,
) -> None:
    await conn.execute(
        "UPDATE trades SET pnl_cents=?, settled_at=? WHERE trade_id=?",
        (pnl_cents, settled_at, trade_id),
    )
    await conn.commit()


async def get_config(conn: aiosqlite.Connection) -> dict[str, str]:
    cur = await conn.execute("SELECT key, value FROM bot_config")
    rows = await cur.fetchall()
    return {r["key"]: r["value"] for r in rows}


async def set_config(
    conn: aiosqlite.Connection, key: str, value: str
) -> None:
    await conn.execute(
        "INSERT OR REPLACE INTO bot_config (key, value, updated_at) VALUES (?,?,?)",
        (key, value, time.time()),
    )
    await conn.commit()


async def migrate_json_log(
    conn: aiosqlite.Connection, json_path: str
) -> int:
    """Import legacy data/trade_log.json into SQLite (idempotent)."""
    path = Path(json_path)
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            old_trades = json.load(f)
    except Exception as e:
        log.warning("migrate_json_log: could not read %s: %s", json_path, e)
        return 0

    inserted = 0
    for record in old_trades:
        # Synthesise a stable trade_id if missing
        if not record.get("trade_id"):
            raw = f"{record.get('timestamp',0)}_{record.get('ticker','')}_{record.get('side','')}"
            record["trade_id"] = hashlib.md5(raw.encode()).hexdigest()
        record.setdefault("order_id", None)
        record.setdefault("error", None)
        record.setdefault("pnl_cents", None)
        record.setdefault("settled_at", None)
        record.setdefault("score", "")
        try:
            await insert_trade(conn, record)
            inserted += 1
        except Exception:
            pass

    if inserted > 0:
        migrated = path.with_suffix(".json.migrated")
        path.rename(migrated)
        log.info("Migrated %d trades from %s → SQLite", inserted, json_path)
    return inserted
