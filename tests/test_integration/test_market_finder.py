"""Tests for market finder with mocked Kalshi client."""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from integration.kalshi_client import MarketPrices
from integration.market_finder import (
    MarketFinder,
    _extract_last_name,
    _extract_player_from_title,
    _FALLBACK_SERIES,
)


def _make_market(ticker, title, event_ticker, status="active"):
    return {
        "ticker": ticker,
        "title": title,
        "event_ticker": event_ticker,
        "status": status,
        "yes_ask_dollars": "0.5500",
        "yes_bid_dollars": "0.5300",
        "no_ask_dollars": "0.4700",
        "no_bid_dollars": "0.4500",
        "last_price_dollars": "0.5400",
    }


def _mock_prices(m):
    return MarketPrices(
        ticker=m["ticker"],
        yes_ask=0.55, yes_bid=0.53,
        no_ask=0.47, no_bid=0.45,
        last_price=0.54,
        timestamp=time.time(),
    )


class TestExtractors:
    def test_last_name(self):
        assert _extract_last_name("Jannik Sinner") == "sinner"
        assert _extract_last_name("Carlos Alcaraz") == "alcaraz"
        assert _extract_last_name("Sinner") == "sinner"

    def test_player_from_title(self):
        assert _extract_player_from_title(
            "Will Jannik Sinner win the Sinner vs Alcaraz : Final match?"
        ) == "Jannik Sinner"
        assert _extract_player_from_title("No match here") is None


class TestMarketFinder:
    @pytest.fixture
    def mock_kalshi(self):
        client = AsyncMock()
        client.get_markets = AsyncMock(return_value=[
            _make_market(
                "KXATPMATCH-26MAR12SINALC-SIN",
                "Will Jannik Sinner win the Sinner vs Alcaraz : Final match?",
                "KXATPMATCH-26MAR12SINALC",
            ),
            _make_market(
                "KXATPMATCH-26MAR12SINALC-ALC",
                "Will Carlos Alcaraz win the Sinner vs Alcaraz : Final match?",
                "KXATPMATCH-26MAR12SINALC",
            ),
            _make_market(
                "KXATPMATCH-26MAR12DJOMED-DJO",
                "Will Novak Djokovic win the Djokovic vs Medvedev : Semifinal match?",
                "KXATPMATCH-26MAR12DJOMED",
            ),
            _make_market(
                "KXATPMATCH-26MAR12DJOMED-MED",
                "Will Daniil Medvedev win the Djokovic vs Medvedev : Semifinal match?",
                "KXATPMATCH-26MAR12DJOMED",
            ),
        ])
        client.extract_prices = MagicMock(side_effect=_mock_prices)
        return client

    @pytest.mark.asyncio
    async def test_find_match_p1_yes(self, mock_kalshi):
        # Pass explicit series to skip discovery
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        result = await finder.find("Jannik Sinner", "Carlos Alcaraz")
        assert result is not None
        assert "SIN" in result.ticker
        assert result.p1_is_yes is True
        assert result.event_ticker == "KXATPMATCH-26MAR12SINALC"

    @pytest.mark.asyncio
    async def test_find_match_reversed_still_finds_p1(self, mock_kalshi):
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        result = await finder.find("Carlos Alcaraz", "Jannik Sinner")
        assert result is not None
        assert "ALC" in result.ticker
        assert result.p1_is_yes is True

    @pytest.mark.asyncio
    async def test_no_match(self, mock_kalshi):
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        result = await finder.find("Roger Federer", "Andy Murray")
        assert result is None

    @pytest.mark.asyncio
    async def test_prices_populated(self, mock_kalshi):
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        result = await finder.find("Novak Djokovic", "Daniil Medvedev")
        assert result is not None
        assert result.prices.best_ask_cents == 55
        assert result.prices.best_bid_cents == 53

    @pytest.mark.asyncio
    async def test_cache_reuse(self, mock_kalshi):
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH", "KXWTAMATCH"], refresh_secs=300)
        await finder.find("Jannik Sinner", "Carlos Alcaraz")
        await finder.find("Novak Djokovic", "Daniil Medvedev")
        # Two series queried on first call, none on second (cached)
        assert mock_kalshi.get_markets.call_count == 2

    @pytest.mark.asyncio
    async def test_no_markets_returns_none(self, mock_kalshi):
        mock_kalshi.get_markets = AsyncMock(return_value=[])
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        result = await finder.find("Jannik Sinner", "Carlos Alcaraz")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_active(self, mock_kalshi):
        finder = MarketFinder(mock_kalshi, series_tickers=["KXATPMATCH"], refresh_secs=0)
        active = await finder.get_all_active()
        # Two matches (events), each returns one MarketMatch
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_challenger_series_discovered(self, mock_kalshi):
        """Verify discover_series finds Challenger series from API response."""
        mock_kalshi._request = AsyncMock(return_value={
            "series": [
                {"ticker": "KXATPMATCH", "title": "ATP Tennis Match", "category": "Sports"},
                {"ticker": "KXWTAMATCH", "title": "WTA Tennis Match", "category": "Sports"},
                {"ticker": "KXATPCHALLENGERMATCH", "title": "Challenger ATP", "category": "Sports"},
                {"ticker": "KXWTACHALLENGERMATCH", "title": "Challenger WTA", "category": "Sports"},
                {"ticker": "KXITFMATCH", "title": "ITF Match", "category": "Sports"},
                {"ticker": "KXATPGAME", "title": "ATP Tennis Winner", "category": "Sports"},
                {"ticker": "KXNBAGAME", "title": "NBA Game", "category": "Sports"},
                {"ticker": "KXTABLETENNIS", "title": "Table Tennis Match", "category": "Sports"},
            ]
        })
        finder = MarketFinder(mock_kalshi, refresh_secs=0)
        discovered = await finder.discover_series()
        assert "KXATPMATCH" in discovered
        assert "KXWTAMATCH" in discovered
        assert "KXATPCHALLENGERMATCH" in discovered
        assert "KXWTACHALLENGERMATCH" in discovered
        assert "KXITFMATCH" in discovered
        # Excluded: no MATCH in ticker, table tennis, non-sports
        assert "KXATPGAME" not in discovered
        assert "KXNBAGAME" not in discovered
        assert "KXTABLETENNIS" not in discovered

    @pytest.mark.asyncio
    async def test_discovery_fallback_on_error(self, mock_kalshi):
        """Falls back to hardcoded list if API fails."""
        mock_kalshi._request = AsyncMock(side_effect=Exception("network error"))
        finder = MarketFinder(mock_kalshi, refresh_secs=0)
        discovered = await finder.discover_series()
        assert discovered == list(_FALLBACK_SERIES)
