"""Configuration loaded from environment variables.

Required env vars:
  API_TENNIS_KEY          — API-Tennis authentication key
  KALSHI_API_KEY          — Kalshi access key (UUID)
  KALSHI_PRIVATE_KEY_PATH — Path to RSA private key PEM file

Optional env vars (with defaults):
  MIN_EDGE_PCT            — Minimum edge threshold (default: 0.05)
  MIN_CONFIDENCE          — Minimum model confidence (default: 0.60)
  MIN_POINTS_OBSERVED     — Minimum points before trading (default: 10)
  MAX_CONTRACTS_PER_MATCH — Hard cap per match (default: 50)
  MAX_TOTAL_EXPOSURE      — Dollar cap across all matches (default: 500)
  MAX_CONTRACTS_PER_SIGNAL— Per-signal contract cap (default: 20)
  ARTIFACTS_DIR           — Path to model artifacts (default: data/artifacts)
  ALIASES_PATH            — Path to player aliases JSON (default: data/aliases.json)
  ATP_RAW_DIR             — Path to raw ATP CSVs (default: data/raw/tennis_atp)
  KALSHI_SERIES_TICKER    — Kalshi tennis series (default: KXATPMATCH)
  MARKET_REFRESH_SECS     — Market cache refresh interval seconds (default: 60)
  REST_POLL_SECS          — REST fallback polling interval seconds (default: 3)
  LOG_LEVEL               — Logging level (default: INFO)
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from cwd

# --- Required ---
API_TENNIS_KEY: str = os.environ.get("API_TENNIS_KEY", "")
KALSHI_API_KEY: str = os.environ.get("KALSHI_API_KEY", "")
KALSHI_PRIVATE_KEY_PATH: str = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

# --- Demo Kalshi credentials (used when --demo, which is the default) ---
DEMO_KALSHI_API_KEY: str = os.environ.get("DEMO_KALSHI_API_KEY", "")
DEMO_KALSHI_PRIVATE_KEY_PATH: str = os.environ.get("DEMO_KALSHI_PRIVATE_KEY_PATH", "")

# --- Hybrid pricing: read real prices from prod, trade on demo ---
HYBRID_PRICING: bool = os.environ.get("HYBRID_PRICING", "").lower() in ("1", "true", "yes")
PROD_KALSHI_API_KEY: str = os.environ.get("PROD_KALSHI_API_KEY", "")
PROD_KALSHI_PRIVATE_KEY_PATH: str = os.environ.get("PROD_KALSHI_PRIVATE_KEY_PATH", "")

# --- Trading parameters ---
MIN_EDGE_PCT: float = float(os.environ.get("MIN_EDGE_PCT", "0.05"))
MIN_CONFIDENCE: float = float(os.environ.get("MIN_CONFIDENCE", "0.60"))
MIN_POINTS_OBSERVED: int = int(os.environ.get("MIN_POINTS_OBSERVED", "10"))
MAX_CONTRACTS_PER_MATCH: int = int(os.environ.get("MAX_CONTRACTS_PER_MATCH", "50"))
MAX_TOTAL_EXPOSURE: float = float(os.environ.get("MAX_TOTAL_EXPOSURE", "500.0"))
MAX_CONTRACTS_PER_SIGNAL: int = int(os.environ.get("MAX_CONTRACTS_PER_SIGNAL", "20"))
NET_EDGE_MIN_DOLLARS: float = float(os.environ.get("NET_EDGE_MIN_DOLLARS", "0.10"))
KALSHI_FEE_PER_CONTRACT: float = 0.07  # fixed fee, not configurable

# --- Infrastructure ---
ARTIFACTS_DIR: str = os.environ.get("ARTIFACTS_DIR", "data/artifacts")
ALIASES_PATH: str = os.environ.get("ALIASES_PATH", "data/aliases.json")
ATP_RAW_DIR: str = os.environ.get("ATP_RAW_DIR", "data/raw/tennis_atp")
KALSHI_SERIES_TICKER: str = os.environ.get("KALSHI_SERIES_TICKER", "KXATPMATCH")
MARKET_REFRESH_SECS: int = int(os.environ.get("MARKET_REFRESH_SECS", "60"))
REST_POLL_SECS: float = float(os.environ.get("REST_POLL_SECS", "3.0"))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# --- Kalshi endpoints ---
KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_WS_PROD = "wss://trading-api.kalshi.com/trade-api/ws/v2"
KALSHI_WS_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"

# --- API-Tennis endpoints ---
API_TENNIS_WS = "wss://wss.api-tennis.com/live"
API_TENNIS_REST = "https://api.api-tennis.com/tennis/"

# --- Web portal ---
WEB_PORT: int = int(os.environ.get("WEB_PORT", "8765"))
DB_PATH: str = os.environ.get("DB_PATH", "data/trading.db")
MATCH_STOP_LOSS_DOLLARS: float = float(os.environ.get("MATCH_STOP_LOSS_DOLLARS", "10.0"))
PORTFOLIO_STOP_LOSS_DOLLARS: float = float(os.environ.get("PORTFOLIO_STOP_LOSS_DOLLARS", "50.0"))
PROFIT_TARGET_DOLLARS: float = float(os.environ.get("PROFIT_TARGET_DOLLARS", "20.0"))
