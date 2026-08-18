"""Tests for the Live Tennis API feed mapping and deduplication."""
import pytest

from integration.live_tennis_feed import (
    LiveTennisApiFeed,
    _normalize_tour_level,
    _point_index,
)
from integration.tennis_feed import PointEvent


def _match(match_id="123", server=1, points=None, sets=None, games=None, **overrides):
    match = {
        "id": match_id,
        "tournament": "Test Open",
        "tour": "atp",
        "surface": "hard",
        "status": "live",
        "players": {"p1": {"name": "Novak Djokovic"}, "p2": {"name": "Carlos Alcaraz"}},
        "score": {
            "sets": sets if sets is not None else [1, 0],
            "games": games if games is not None else [[6, 4], [3, 2]],
            "points": points if points is not None else ["30", "40"],
            "server": server,
            "is_tiebreak": False,
        },
    }
    match.update(overrides)
    return match


class TestPointIndex:
    def test_basic(self):
        assert _point_index("0") == 0
        assert _point_index("15") == 1
        assert _point_index("30") == 2
        assert _point_index("40") == 3

    def test_ad(self):
        assert _point_index("AD") == 4
        assert _point_index("A") == 4

    def test_none_and_garbage(self):
        assert _point_index(None) == 0
        assert _point_index("garbage") == 0


class TestNormalizeTourLevel:
    def test_challenger(self):
        assert _normalize_tour_level("challenger") == "C"

    def test_default(self):
        assert _normalize_tour_level("atp") == "A"
        assert _normalize_tour_level(None) == "A"


class TestCurrentGames:
    def test_uses_last_set(self):
        # games = [games_p1_per_set, games_p2_per_set]; current set is the last.
        assert LiveTennisApiFeed._current_games([[6, 4], [3, 2]], 0) == 4
        assert LiveTennisApiFeed._current_games([[6, 4], [3, 2]], 1) == 2

    def test_empty(self):
        assert LiveTennisApiFeed._current_games([], 0) == 0
        assert LiveTennisApiFeed._current_games([[], []], 1) == 0


class TestMapping:
    @pytest.mark.asyncio
    async def test_maps_score_to_pointevent(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = LiveTennisApiFeed(api_key="test", on_point=on_point)
        await feed._process_response({"data": [_match(server=2)]})

        assert len(events) == 1
        ev: PointEvent = events[0]
        assert ev.match_id == "123"
        assert ev.player1_name == "Novak Djokovic"
        assert ev.sets_p1 == 1 and ev.sets_p2 == 0
        assert ev.games_p1 == 4 and ev.games_p2 == 2
        assert ev.points_p1 == 2 and ev.points_p2 == 3  # "30","40"
        assert ev.server == 1  # server=2 in the feed maps to index 1
        assert ev.surface == "Hard"
        assert ev.tournament_level == "A"
        assert ev.server_won_last_point is None


class TestDedup:
    @pytest.mark.asyncio
    async def test_dedup_skips_same_score(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = LiveTennisApiFeed(api_key="test", on_point=on_point)

        payload = {"data": [_match(points=["15", "0"])]}
        await feed._process_response(payload)
        assert len(events) == 1

        # Same score again — deduped.
        await feed._process_response(payload)
        assert len(events) == 1

        # Score advances — new event.
        await feed._process_response({"data": [_match(points=["30", "0"])]})
        assert len(events) == 2


class TestFilters:
    @pytest.mark.asyncio
    async def test_skips_doubles(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = LiveTennisApiFeed(api_key="test", on_point=on_point)
        doubles = _match()
        doubles["players"] = {
            "p1": {"name": "Player A / Player B"},
            "p2": {"name": "Player C / Player D"},
        }
        await feed._process_response({"data": [doubles]})
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_skips_completed(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = LiveTennisApiFeed(api_key="test", on_point=on_point)
        await feed._process_response({"data": [_match(status="completed")]})
        assert len(events) == 0
