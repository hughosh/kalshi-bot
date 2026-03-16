"""Tests for tennis feed score parsing and deduplication."""
import asyncio

import pytest

from integration.tennis_feed import (
    PointEvent,
    TennisFeed,
    _normalize_surface,
    _normalize_tournament_level,
    _parse_score_field,
    _server_won_from_scores,
)


class TestParseScoreField:
    def test_basic(self):
        assert _parse_score_field("15 - 30") == (1, 2)

    def test_love(self):
        assert _parse_score_field("0 - 0") == (0, 0)

    def test_deuce(self):
        assert _parse_score_field("40 - 40") == (3, 3)

    def test_ad(self):
        assert _parse_score_field("AD - 40") == (4, 3)

    def test_a_variant(self):
        assert _parse_score_field("A - 40") == (4, 3)
        assert _parse_score_field("40 - A") == (3, 4)

    def test_invalid(self):
        assert _parse_score_field("garbage") == (-1, -1)


class TestServerWon:
    def test_first_player_serves_and_wins(self):
        # First player serves, score goes 0-0 → 15-0 (first player scored)
        assert _server_won_from_scores("0 - 0", "15 - 0", server_is_first=True) is True

    def test_first_player_serves_and_loses(self):
        assert _server_won_from_scores("0 - 0", "0 - 15", server_is_first=True) is False

    def test_second_player_serves_and_wins(self):
        # Second player serves, score goes 0-0 → 0-15 (second player scored)
        assert _server_won_from_scores("0 - 0", "0 - 15", server_is_first=False) is True

    def test_second_player_serves_and_loses(self):
        assert _server_won_from_scores("0 - 0", "15 - 0", server_is_first=False) is False

    def test_first_point(self):
        assert _server_won_from_scores(None, "15 - 0", server_is_first=True) is None

    def test_ambiguous(self):
        assert _server_won_from_scores("40 - 30", "0 - 0", server_is_first=True) is None


class TestNormalizeSurface:
    def test_clay(self):
        assert _normalize_surface("Red Clay") == "Clay"

    def test_grass(self):
        assert _normalize_surface("Grass") == "Grass"

    def test_default_hard(self):
        assert _normalize_surface("outdoor hard") == "Hard"
        assert _normalize_surface("unknown") == "Hard"


class TestNormalizeTournamentLevel:
    def test_slam(self):
        assert _normalize_tournament_level("Grand Slam") == "G"

    def test_masters(self):
        assert _normalize_tournament_level("ATP Masters 1000") == "M"

    def test_challenger(self):
        assert _normalize_tournament_level("Challenger Men Singles") == "C"

    def test_default(self):
        assert _normalize_tournament_level("ATP 250") == "A"


class TestTennisFeedDedup:
    @pytest.mark.asyncio
    async def test_dedup_skips_same_score(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = TennisFeed(api_key="test", on_point=on_point)

        match_data = {
            "success": 1,
            "result": [{
                "event_key": "123",
                "event_first_player": "Player A",
                "event_second_player": "Player B",
                "event_serve": "First Player",
                "event_game_result": "15 - 0",
                "event_final_result": "0 - 0",
                "event_status": "Set 1",
                "event_type_type": "ATP Men Singles",
                "scores": [{"score_first": "1", "score_second": "0", "score_set": "1"}],
                "pointbypoint": [],
            }],
        }

        await feed._process_response(match_data)
        assert len(events) == 1

        # Same score — should be deduped
        await feed._process_response(match_data)
        assert len(events) == 1

        # Score changed — new event
        match_data["result"][0]["event_game_result"] = "30 - 0"
        await feed._process_response(match_data)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_skips_doubles(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = TennisFeed(api_key="test", on_point=on_point)

        match_data = {
            "result": [{
                "event_key": "456",
                "event_first_player": "Player A / Player B",
                "event_second_player": "Player C / Player D",
                "event_serve": "First Player",
                "event_game_result": "15 - 0",
                "event_status": "Set 1",
                "event_type_type": "ATP Men Singles",
                "scores": [],
                "pointbypoint": [],
            }],
        }

        await feed._process_response(match_data)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_skips_finished(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = TennisFeed(api_key="test", on_point=on_point)

        match_data = {
            "result": [{
                "event_key": "789",
                "event_first_player": "Player A",
                "event_second_player": "Player B",
                "event_serve": "First Player",
                "event_game_result": "0 - 0",
                "event_status": "Finished",
                "event_type_type": "ATP Men Singles",
                "scores": [],
                "pointbypoint": [],
            }],
        }

        await feed._process_response(match_data)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_pbp_emits_new_points(self):
        events = []

        async def on_point(e):
            events.append(e)

        feed = TennisFeed(api_key="test", on_point=on_point)

        match_data = {
            "result": [{
                "event_key": "100",
                "event_first_player": "Player A",
                "event_second_player": "Player B",
                "event_serve": "First Player",
                "event_game_result": "15 - 0",
                "event_final_result": "0 - 0",
                "event_status": "Set 1",
                "event_type_type": "Challenger Men Singles",
                "scores": [{"score_first": "0", "score_second": "0", "score_set": "1"}],
                "pointbypoint": [{
                    "set_number": "Set 1",
                    "number_game": "1",
                    "player_served": "First Player",
                    "score": "0 - 0",
                    "points": [
                        {"number_point": "1", "score": "15 - 0"},
                        {"number_point": "2", "score": "30 - 0"},
                    ],
                }],
            }],
        }

        await feed._process_response(match_data)
        assert len(events) == 2
        assert events[0].server == 0
        assert events[1].points_p1 == 2  # 30 mapped to index 2

        # Same PBP — no new events
        await feed._process_response(match_data)
        assert len(events) == 2

        # Add a point
        match_data["result"][0]["pointbypoint"][0]["points"].append(
            {"number_point": "3", "score": "40 - 0"}
        )
        await feed._process_response(match_data)
        assert len(events) == 3


class TestParseScoresArray:
    def test_single_set_in_progress(self):
        match = {
            "scores": [
                {"score_first": "3", "score_second": "2", "score_set": "1"},
            ]
        }
        s1, s2, g1, g2 = TennisFeed._parse_scores_array(match)
        assert (s1, s2, g1, g2) == (0, 0, 3, 2)

    def test_completed_first_set(self):
        match = {
            "scores": [
                {"score_first": "6", "score_second": "3", "score_set": "1"},
                {"score_first": "2", "score_second": "1", "score_set": "2"},
            ]
        }
        s1, s2, g1, g2 = TennisFeed._parse_scores_array(match)
        assert (s1, s2) == (1, 0)  # p1 won first set
        assert (g1, g2) == (2, 1)  # current set games

    def test_tiebreak_score(self):
        match = {
            "scores": [
                {"score_first": "6.2", "score_second": "7.7", "score_set": "1"},
                {"score_first": "4", "score_second": "3", "score_set": "2"},
            ]
        }
        s1, s2, g1, g2 = TennisFeed._parse_scores_array(match)
        assert (s1, s2) == (0, 1)  # p2 won tiebreak set (7 > 6)
        assert (g1, g2) == (4, 3)

    def test_fallback_to_final_result(self):
        match = {"event_final_result": "1 - 2"}
        s1, s2, g1, g2 = TennisFeed._parse_scores_array(match)
        assert (s1, s2) == (1, 2)
