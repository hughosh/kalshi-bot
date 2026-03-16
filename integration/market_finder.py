"""Match → Kalshi market resolution.

Kalshi tennis markets live under many series tickers:
  KXATPMATCH (ATP main tour), KXWTAMATCH (WTA main tour),
  KXATPCHALLENGERMATCH (ATP Challengers), KXWTACHALLENGERMATCH (WTA Challengers),
  KXITFMATCH, KXUNITEDCUPMATCH, KXDAVISCUPMATCH, etc.

Rather than hardcoding series, we discover them dynamically at startup by
scanning the /series endpoint for Sports-category series whose ticker or
title contains tennis-related keywords.

Each match has two markets (one per player), grouped by event_ticker.
We always return the market where YES = player1 winning.

Market query uses status='open' to get only active (non-finalized) markets
directly from the API, avoiding pagination through thousands of old results.
"""
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from integration.kalshi_client import KalshiClient, MarketPrices

log = logging.getLogger(__name__)

# Known tennis match series as fallback if discovery fails.
_FALLBACK_SERIES = [
    "KXATPMATCH",
    "KXWTAMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTACHALLENGERMATCH",
    "KXCHALLENGERMATCH",
    "KXITFMATCH",
    "KXUNITEDCUPMATCH",
    "KXDAVISCUPMATCH",
    "KXTENNISEXHIBITION",
    "KXEXHIBITIONMEN",
    "KXEXHIBITIONWOMEN",
    "KXSIXKINGSMATCH",
    "KXSIXKINGSSLAMMATCH",
    "KXBATTLEOFSEXES",
    "KXATPDOUBLES",
    "KXWTADOUBLES",
]

# Keywords that indicate a series is for tennis match outcomes.
# Applied against series ticker + title (uppercased).
_TENNIS_KEYWORDS = {"ATP", "WTA", "TENNIS", "CHALLENGER", "ITF", "DAVIS CUP", "UNITED CUP"}
# Exclude series that match keywords but aren't match-outcome markets.
_EXCLUDE_PATTERNS = {"GAME", "SPREAD", "TOTAL", "EXACT", "SET", "RANK", "SLAM FIELD",
                     "GRAND SLAM", "FINALS", "TABLE TENNIS", "DOUBLES", "NEXT GEN",
                     "ESPYTENNIS", "GOLFTENNIS", "MAJORDJO"}


@dataclass
class MarketMatch:
    """A matched Kalshi market for a live tennis match."""
    ticker: str           # the specific market ticker (for the player we model as p1)
    event_ticker: str     # groups both sides of the match
    title: str
    p1_is_yes: bool       # True: buying YES on this ticker = betting on player1
    prices: MarketPrices  # current prices from the market listing


class MarketFinder:
    """Resolves a live match to its Kalshi market ticker.

    On first use, discovers all tennis-related series tickers from the
    Kalshi /series endpoint, then queries each for active markets.
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        series_tickers: list[str] | None = None,
        refresh_secs: int = 60,
    ) -> None:
        self._kalshi = kalshi
        self._explicit_series = series_tickers
        self._discovered_series: list[str] | None = None
        self._refresh_secs = refresh_secs
        self._cache: list[dict] = []
        self._last_refresh: float = 0.0

    @property
    def _series(self) -> list[str]:
        if self._explicit_series:
            return self._explicit_series
        if self._discovered_series is not None:
            return self._discovered_series
        return _FALLBACK_SERIES

    async def discover_series(self) -> list[str]:
        """Scan the /series endpoint for tennis match series tickers.

        Returns the discovered list and caches it for future use.
        Falls back to _FALLBACK_SERIES on error.
        """
        try:
            all_series = await self._kalshi.get_series(limit=200)
            tennis = []
            for s in all_series:
                ticker = s.get("ticker", "")
                title = s.get("title", "")
                category = s.get("category", "")
                if category != "Sports":
                    continue
                combined = f"{ticker} {title}".upper()
                # Must contain a tennis keyword
                if not any(kw in combined for kw in _TENNIS_KEYWORDS):
                    continue
                # Must contain MATCH in the ticker (we want match-outcome markets)
                if "MATCH" not in ticker.upper():
                    continue
                # Exclude non-match-outcome series
                if any(exc in combined for exc in _EXCLUDE_PATTERNS):
                    continue
                tennis.append(ticker)
            if tennis:
                self._discovered_series = tennis
                log.info("MarketFinder: discovered %d tennis series: %s", len(tennis), tennis)
                return tennis
        except Exception as e:
            log.warning("MarketFinder: series discovery failed: %s", e)

        self._discovered_series = list(_FALLBACK_SERIES)
        log.info("MarketFinder: using fallback series list")
        return self._discovered_series

    async def _refresh_if_stale(self) -> None:
        now = time.time()
        if now - self._last_refresh < self._refresh_secs:
            return

        # Discover series on first refresh
        if self._discovered_series is None and self._explicit_series is None:
            await self.discover_series()

        all_markets: list[dict] = []
        for series in self._series:
            try:
                markets = await self._kalshi.get_markets(series, status="open")
                all_markets.extend(markets)
            except Exception as e:
                log.warning("MarketFinder: failed to fetch %s: %s", series, e)
        self._cache = all_markets
        self._last_refresh = now
        log.info("MarketFinder: refreshed — %d active tennis markets across %d series",
                 len(self._cache), len(self._series))

    async def find(
        self,
        player1_name: str,
        player2_name: str,
    ) -> Optional[MarketMatch]:
        """Find the Kalshi market for a match between two players.

        Returns the market where YES = player1 wins.
        Returns None if no market found.
        """
        await self._refresh_if_stale()

        last1 = _extract_last_name(player1_name)
        last2 = _extract_last_name(player2_name)
        if not last1 or not last2:
            return None

        # Group markets by event_ticker (two markets per match)
        events: dict[str, list[dict]] = {}
        for mkt in self._cache:
            et = mkt.get("event_ticker", "")
            if et:
                events.setdefault(et, []).append(mkt)

        # Search events for a match containing both player last names
        for event_ticker, event_markets in events.items():
            titles_combined = " ".join(m.get("title", "") for m in event_markets).lower()

            if last1 in titles_combined and last2 in titles_combined:
                # Found the match — find which market is for player1
                p1_market = None
                for m in event_markets:
                    title = m.get("title", "")
                    player_in_title = _extract_player_from_title(title)
                    if player_in_title and last1 in player_in_title.lower():
                        p1_market = m
                        break

                if p1_market is None:
                    p1_market = event_markets[0]
                    log.warning("MarketFinder: couldn't determine p1 market, using %s",
                                p1_market.get("ticker"))

                prices = self._kalshi.extract_prices(p1_market)

                result = MarketMatch(
                    ticker=p1_market.get("ticker", ""),
                    event_ticker=event_ticker,
                    title=p1_market.get("title", ""),
                    p1_is_yes=True,
                    prices=prices,
                )
                log.info("MarketFinder: %s vs %s → %s (yes=%s)",
                         player1_name, player2_name, result.ticker,
                         _extract_player_from_title(result.title))
                return result

        log.debug("MarketFinder: no market for %s vs %s", player1_name, player2_name)
        return None

    async def get_all_active(self) -> list[MarketMatch]:
        """Return all active tennis matches as MarketMatch objects."""
        await self._refresh_if_stale()

        events: dict[str, list[dict]] = {}
        for mkt in self._cache:
            et = mkt.get("event_ticker", "")
            if et:
                events.setdefault(et, []).append(mkt)

        results = []
        for event_ticker, event_markets in events.items():
            if len(event_markets) < 2:
                continue
            m = event_markets[0]
            prices = self._kalshi.extract_prices(m)
            results.append(MarketMatch(
                ticker=m.get("ticker", ""),
                event_ticker=event_ticker,
                title=m.get("title", ""),
                p1_is_yes=True,
                prices=prices,
            ))
        return results


def _extract_last_name(name: str) -> str:
    """Extract lowercase last name from a player name."""
    parts = name.strip().split()
    if not parts:
        return ""
    return parts[-1].lower()


def _extract_player_from_title(title: str) -> Optional[str]:
    """Extract player name from 'Will [Name] win the ...' title format."""
    match = re.match(r"Will\s+(.+?)\s+win\s+", title, re.IGNORECASE)
    if match:
        return match.group(1)
    return None
