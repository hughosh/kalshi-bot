"""Kalshi REST client with RSA-PSS auth and token-bucket rate limiting.

Auth: RSA-PSS SHA-256, MGF1, salt_length=DIGEST_LENGTH
  Sign: f"{timestamp_ms}{METHOD}{/trade-api/v2/endpoint}"
  Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE

Production URL: https://api.elections.kalshi.com/trade-api/v2
Demo URL: https://demo-api.kalshi.co/trade-api/v2

Rate limits: 20 read/s, 10 write/s (basic tier)
"""
import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

log = logging.getLogger(__name__)

_READ_METHODS = {"GET", "HEAD"}


class KalshiError(Exception):
    """Non-retryable Kalshi API error (4xx)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


@dataclass
class MarketPrices:
    """Prices from the /markets listing (no separate orderbook needed)."""
    ticker: str
    yes_ask: Optional[float]   # dollars (0.00-1.00) or None
    yes_bid: Optional[float]
    no_ask: Optional[float]
    no_bid: Optional[float]
    last_price: Optional[float]
    timestamp: float

    @property
    def best_ask_cents(self) -> Optional[int]:
        """YES ask in cents for edge computation."""
        if self.yes_ask is not None and self.yes_ask > 0:
            return int(round(self.yes_ask * 100))
        return None

    @property
    def best_bid_cents(self) -> Optional[int]:
        """YES bid in cents for edge computation."""
        if self.yes_bid is not None and self.yes_bid > 0:
            return int(round(self.yes_bid * 100))
        return None


class _TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


class KalshiClient:
    """Async Kalshi REST client.

    Args:
        api_key: Kalshi access key (UUID string).
        private_key_path: Path to RSA private key PEM file.
        demo: If True, use the demo environment.
    """

    def __init__(self, api_key: str, private_key_path: str, demo: bool = True) -> None:
        self._api_key = api_key
        self._private_key = self._load_key(private_key_path)
        self._base = (
            "https://demo-api.kalshi.co/trade-api/v2"
            if demo
            else "https://api.elections.kalshi.com/trade-api/v2"
        )
        self._path_prefix = "/trade-api/v2"
        self._read_bucket = _TokenBucket(rate=20.0)
        self._write_bucket = _TokenBucket(rate=10.0)
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _load_key(path: str):
        p = Path(path)
        if not p.exists():
            log.warning("Private key not found at %s — auth will fail on live calls", path)
            return None
        with open(p, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=10.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _sign(self, method: str, path: str) -> tuple[str, str]:
        """Return (timestamp_ms_str, signature_b64).

        path: full API path e.g. /trade-api/v2/portfolio/balance (no query string).
        """
        ts_ms = str(int(time.time() * 1000))
        parsed = urlparse(path)
        path_only = parsed.path
        msg = f"{ts_ms}{method.upper()}{path_only}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return ts_ms, base64.b64encode(sig).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._private_key is None:
            return {}
        ts, sig = self._sign(method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        auth: bool = False,
        _retry: int = 0,
    ) -> Any:
        bucket = self._read_bucket if method in _READ_METHODS else self._write_bucket
        await bucket.acquire()

        url = self._base + endpoint
        headers = {"Content-Type": "application/json"}
        if auth:
            full_path = self._path_prefix + endpoint
            headers.update(self._auth_headers(method, full_path))

        try:
            resp = await self._client.request(
                method, url, params=params, json=json, headers=headers
            )
        except httpx.RequestError as e:
            log.warning("Kalshi request error: %s", e)
            raise

        if resp.status_code == 429:
            if _retry < 2:
                wait = 2 ** _retry
                log.warning("Kalshi 429 — backing off %.1fs", wait)
                await asyncio.sleep(wait)
                return await self._request(method, endpoint, params, json, auth, _retry + 1)
            raise KalshiError(429, "Rate limited after retries")

        if resp.status_code >= 500:
            if _retry < 3:
                wait = 2 ** _retry
                log.warning("Kalshi %d — backing off %.1fs", resp.status_code, wait)
                await asyncio.sleep(wait)
                return await self._request(method, endpoint, params, json, auth, _retry + 1)
            raise KalshiError(resp.status_code, resp.text)

        if resp.status_code >= 400:
            raise KalshiError(resp.status_code, resp.text)

        return resp.json()

    # --- Public API ---

    async def get_markets(
        self,
        series_ticker: str,
        limit: int = 200,
        status: str = "open",
    ) -> list[dict]:
        """Return list of market dicts from GET /markets.

        Uses status='open' to get only active markets server-side,
        avoiding pagination through thousands of finalized markets.
        Pass status='' to disable the filter.
        """
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": limit}
        if status:
            params["status"] = status
        data = await self._request("GET", "/markets", params=params)
        return data.get("markets", [])

    def extract_prices(self, market: dict) -> MarketPrices:
        """Extract prices from a market dict (from get_markets response)."""
        def _to_float(val) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                return f if f > 0 else None
            except (ValueError, TypeError):
                return None

        return MarketPrices(
            ticker=market.get("ticker", ""),
            yes_ask=_to_float(market.get("yes_ask_dollars")),
            yes_bid=_to_float(market.get("yes_bid_dollars")),
            no_ask=_to_float(market.get("no_ask_dollars")),
            no_bid=_to_float(market.get("no_bid_dollars")),
            last_price=_to_float(market.get("last_price_dollars")),
            timestamp=time.time(),
        )

    async def place_order(
        self,
        ticker: str,
        side: str,         # "yes" or "no"
        contracts: int,
        price_cents: int,  # limit price 1-99
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Place a limit order. Requires auth."""
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": contracts,
            "yes_price": price_cents if side == "yes" else (100 - price_cents),
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return await self._request("POST", "/portfolio/orders", json=body, auth=True)

    async def get_positions(self) -> list[dict]:
        """Return current portfolio positions. Requires auth."""
        data = await self._request("GET", "/portfolio/positions", auth=True)
        return data.get("market_positions", [])

    async def get_balance(self) -> float:
        """Return available balance in dollars."""
        data = await self._request("GET", "/portfolio/balance", auth=True)
        return data.get("balance", 0) / 100.0

    async def get_series(self, limit: int = 200) -> list[dict]:
        """Return list of series dicts from GET /series."""
        data = await self._request("GET", "/series", params={"limit": limit})
        return data.get("series", [])


class HybridKalshiClient:
    """Composes two KalshiClient instances: production for prices, demo for trades.

    Reads market data (prices, series) from the production API where prices are
    accurate, while routing all order execution to the demo environment for
    paper trading.

    Duck-typed to match KalshiClient's public interface so MarketFinder,
    TradingGuard, and bot.py can use it transparently.
    """

    def __init__(
        self,
        prod_api_key: str,
        prod_key_path: str,
        demo_api_key: str = "",
        demo_key_path: str = "",
    ) -> None:
        self._price_client = KalshiClient(
            api_key=prod_api_key,
            private_key_path=prod_key_path,
            demo=False,
        )
        # Trade client is optional — if no demo credentials, paper-only mode
        self._trade_client: KalshiClient | None = None
        if demo_api_key and demo_key_path:
            self._trade_client = KalshiClient(
                api_key=demo_api_key,
                private_key_path=demo_key_path,
                demo=True,
            )

    async def __aenter__(self):
        await self._price_client.__aenter__()
        if self._trade_client:
            await self._trade_client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._price_client.__aexit__(*args)
        if self._trade_client:
            await self._trade_client.__aexit__(*args)

    # --- Price reads (production) ---

    async def get_markets(
        self,
        series_ticker: str,
        limit: int = 200,
        status: str = "open",
    ) -> list[dict]:
        return await self._price_client.get_markets(series_ticker, limit, status)

    def extract_prices(self, market: dict) -> MarketPrices:
        return self._price_client.extract_prices(market)

    async def get_series(self, limit: int = 200) -> list[dict]:
        return await self._price_client.get_series(limit)

    # --- Trade execution (demo) ---

    async def place_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        client_order_id: Optional[str] = None,
    ) -> dict:
        if self._trade_client is None:
            log.warning("HybridClient: no trade client configured, order dropped")
            return {"status": "paper_only"}
        return await self._trade_client.place_order(
            ticker, side, contracts, price_cents, client_order_id
        )

    async def get_positions(self) -> list[dict]:
        if self._trade_client is None:
            return []
        return await self._trade_client.get_positions()

    async def get_balance(self) -> float:
        if self._trade_client is None:
            return 0.0
        return await self._trade_client.get_balance()
