# Kalshi Tennis Trading Bot

A live tennis trading bot that predicts match win probabilities in real time using a point-by-point Bayesian model and trades (or paper trades) on Kalshi prediction markets.

---

## How it works

### Prediction pipeline

Each live point from the API-Tennis feed triggers the following chain:

```
API-Tennis WebSocket
    → TennisFeed          parse point event
    → MatchTracker        GBDT + BayesianUpdater → win probability
    → MarketFinder        fetch real Kalshi bid/ask prices
    → evaluate_signal     edge + quarter-Kelly sizing
    → RiskManager         exposure cap
    → TradingGuard        stop-loss / pause
    → PaperKalshiClient   simulate fill, log to DB
    → BroadcastHub        push update to web portal
```

### Model

The model is a three-layer stack:

| Layer | What it does | File |
|-------|-------------|------|
| **GBDT (XGBoost)** | Predicts P(server wins point) from pre-match + in-play features | `tennis/model.py` |
| **BayesianUpdater** | Beta-Binomial conjugate update of p_serve after each observed point | `tennis/updater.py` |
| **Markov chain** | Propagates p_serve analytically through game → set → match (Newton & Keller 2005) | `tennis/markov.py` |
| **Calibrator** | Isotonic regression to correct match-level probability bias | `tennis/calibration.py` |

The GBDT prior initialises the Bayesian updater at match start. As points are observed the posterior mean shifts toward the player's actual in-match serve rate, with an optional exponential decay (`decay_rate`) that gives more weight to recent points.

### Features

Pre-match features (computed once per match):
- Surface-specific ELO ratings
- Rolling serve/return win % (last 52 weeks)
- Head-to-head record
- Tournament level, surface, ranking difference

In-play features (updated each point):
- Running serve points won %
- Return points won %
- Momentum over last 5 and 10 points
- Set/game score, tiebreak flag, final-set flag

### Signal evaluation

`integration/edge.py` — four sequential gates before a trade is placed:

1. **Confidence** — Bayesian CI width narrow enough (`MIN_CONFIDENCE`, default 0.60)
2. **Points observed** — at least `MIN_POINTS_OBSERVED` (default 10) played
3. **Edge** — `model_prob − market_ask > MIN_EDGE_PCT` (default 0.05) on YES, or equivalent on NO
4. **Net dollar edge** — edge after $0.07/contract fee > `NET_EDGE_MIN_DOLLARS` (default $0.10)

Position sizing: **quarter-Kelly**
```
kelly = edge / payout_ratio
contracts = floor(kelly / 4 × bankroll / cost_per_contract)
```
Capped at `MAX_CONTRACTS_PER_SIGNAL` (default 20).

---

## Paper trading

By default (`python bot.py`) the bot runs in pure paper mode using `PaperKalshiClient`:

- **Prices** are fetched from the **production Kalshi API** with no authentication — `/markets` is publicly readable, giving real live bid/ask spreads.
- **Orders** are never sent anywhere. `place_order()` deducts cost from an in-memory balance and increments a position counter.
- **All fills are persisted** to `data/trading.db` with `mode="paper"` and are visible in the web portal.
- If the bot restarts, in-memory balance/positions reset but the DB history is preserved.

```python
# integration/kalshi_client.py
class PaperKalshiClient:
    async def place_order(self, ticker, side, contracts, price_cents, ...):
        self._balance -= contracts * price_cents / 100.0
        self._positions[ticker][side] += contracts
        # logs fill, returns {"status": "paper_filled"}
        # no HTTP call made
```

---

## Setup

### Requirements

- Python 3.12+
- Dependencies: `pip install -r requirements.txt`

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
# Required for live score feed
API_TENNIS_KEY=your_api_tennis_key

# Required only for live trading (--live --prod)
# Not used in paper mode
KALSHI_API_KEY=your_kalshi_uuid
KALSHI_PRIVATE_KEY_PATH=/path/to/rsa_private_key.pem

# Optional overrides (defaults shown)
MIN_EDGE_PCT=0.05
MIN_CONFIDENCE=0.60
MIN_POINTS_OBSERVED=10
MAX_CONTRACTS_PER_MATCH=50
MAX_TOTAL_EXPOSURE=500
MAX_CONTRACTS_PER_SIGNAL=20
WEB_PORT=8765
```

### Data

Download the raw ATP/slam point-by-point data before training:

```bash
python scripts/download_data.py
```

### Training

```bash
python scripts/build_features.py   # build parquet feature files
python scripts/train.py            # train GBDT + calibrator
python scripts/audit_model.py      # check for leakage / overfitting
python scripts/backtest.py         # evaluate on held-out test set
python scripts/tune_updater.py     # grid search BayesianUpdater params
```

Artifacts are saved to `data/artifacts/`: `gbdt_model.pkl`, `calibrator.pkl`, `player_elo.parquet`, `player_stats.parquet`, `h2h.parquet`.

---

## Running the bot

```bash
# Paper trading (default) — real prices, simulated fills
python bot.py

# Live trading on Kalshi production (real money)
python bot.py --live --prod
```

Web portal starts automatically at **http://127.0.0.1:8765**.

---

## Project structure

```
bot.py                      Main entry point
config.py                   All env-var configuration

tennis/
    model.py                XGBoost point-win GBDT
    updater.py              Beta-Binomial Bayesian updater
    markov.py               Point→game→set→match Markov chain
    calibration.py          Isotonic regression calibrator
    engine.py               PredictionEngine (ties it all together)
    features.py             Feature engineering
    elo.py                  Surface-specific ELO tracker
    types.py                MatchState, PredictionResult dataclasses

integration/
    kalshi_client.py        PaperKalshiClient, HybridKalshiClient, KalshiClient
    market_finder.py        Resolves live matches to Kalshi market tickers
    tennis_feed.py          API-Tennis WebSocket + REST fallback
    match_tracker.py        Per-match session management
    edge.py                 Edge computation + quarter-Kelly sizing
    risk.py                 Exposure tracking and caps
    player_resolver.py      Player name → ATP ID resolution

web/
    app.py                  FastAPI portal + WebSocket hub
    db.py                   SQLite persistence (trades, signals, match state)
    guard.py                Stop-loss, profit target, force-exit logic
    state.py                Runtime-adjustable thresholds

scripts/
    build_features.py       Feature dataset construction
    train.py                Model training
    backtest.py             Held-out test evaluation
    audit_model.py          Leakage and overfitting audit
    tune_updater.py         BayesianUpdater hyperparameter search
```

---

## Key design decisions

**No demo API.** The Kalshi demo environment returns stale/zero prices, making it useless for paper trading edge evaluation. `PaperKalshiClient` reads production prices (no auth required) and simulates fills locally.

**No GBDT/updater blending.** An earlier design blended the GBDT estimate and the Bayesian posterior at 20 points. This was removed — the updater already handles the prior-to-data transition cleanly through its Beta-Binomial parameterisation.

**Leakage-free calibration.** The isotonic calibrator is fitted on actual match winners (`match_winner` from ATP/slam metadata), not on `server_won.mean() > 0.5` which is derivable from training features. The validation set is split temporally: H1 for GBDT early stopping, H2 for calibrator fitting.

**Server identity.** Main-tour PBP data provides only a raw S/R sequence with no server labels. `_assign_server_to_points()` in `build_features.py` reconstructs the correct server for each point by tracking game boundaries, service alternation, and tiebreak rules.

---

## Academic references

| Paper | Relevance |
|-------|-----------|
| Newton & Keller (2005) — *Probability of winning at tennis I* | Foundation of the Markov chain in `markov.py` |
| Klaassen & Magnus (2001) — *On the Independence and Identical Distribution of Points in Tennis* | Points are not iid; server advantage varies with score pressure |
| Ingram (2019) — *A point-based Bayesian hierarchical model to predict the outcome of tennis matches* | Validates the Beta-Binomial conjugate approach in `updater.py` |
| Sipko & Knottenbelt (2015) — *Machine Learning for Prediction of Professional Tennis Matches* | GBDT approach and feature selection |
