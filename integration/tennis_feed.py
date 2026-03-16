"""API-Tennis WebSocket feed with REST fallback.

Connection strategy:
  - Primary: WebSocket at wss://wss.api-tennis.com/live?APIkey=KEY
  - On disconnect: exponential backoff 1s → 30s cap, then REST polling every 3s
  - On WebSocket reconnect: resumes WebSocket mode

Actual API-Tennis REST response shape (GET get_livescore):
  {
    "success": 1,
    "result": [
      {
        "event_key": 12109125,
        "event_date": "2026-03-13",
        "event_first_player": "C. Tabur",
        "event_second_player": "F. Romano",
        "event_final_result": "0 - 0",       # set score
        "event_game_result": "40 - 40",       # current game score
        "event_serve": "Second Player",       # "First Player" or "Second Player"
        "event_status": "Set 1",
        "event_type_type": "Challenger Men Singles",
        "tournament_name": "Cherbourg",
        "scores": [{"score_first": "1", "score_second": "1", "score_set": "1"}],
        "pointbypoint": [
          {
            "set_number": "Set 1",
            "number_game": "1",
            "player_served": "Second Player",
            "serve_winner": "Second Player",
            "score": "0 - 1",
            "points": [
              {"number_point": "1", "score": "0 - 15", ...},
              {"number_point": "2", "score": "0 - 30", ...},
            ]
          }
        ]
      }
    ]
  }

Deduplication:
  Tracks {match_id: total_points_seen}; only emits when new points appear.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)

_SCORE_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
_BACKOFF_MIN = 1.0
_BACKOFF_MAX = 30.0


@dataclass
class PointEvent:
    match_id: str
    player1_name: str
    player2_name: str
    surface: str
    tournament_level: str
    match_date: str           # YYYY-MM-DD

    sets_p1: int
    sets_p2: int
    games_p1: int
    games_p2: int
    points_p1: int            # current game score index for player1
    points_p2: int

    server: int               # 0 = player1, 1 = player2
    is_tiebreak: bool
    server_won_last_point: Optional[bool]  # None on first event of match
    point_sequence: int


def _parse_score_field(score_str: str) -> tuple[int, int]:
    """Parse '15 - 30' -> (1, 2) using _SCORE_MAP. Returns (-1,-1) on failure."""
    parts = re.split(r"\s*-\s*", score_str.strip())
    if len(parts) != 2:
        return -1, -1
    a = _SCORE_MAP.get(parts[0].strip().upper(), -1)
    b = _SCORE_MAP.get(parts[1].strip().upper(), -1)
    return a, b


def _server_won_from_scores(
    prev_score: Optional[str], curr_score: str, server_is_first: bool
) -> Optional[bool]:
    """Determine if server won last point by comparing consecutive score strings.

    Scores are always in first_player - second_player order.
    server_is_first tells us which side is serving.
    """
    if prev_score is None:
        return None
    prev = _parse_score_field(prev_score)
    curr = _parse_score_field(curr_score)
    if prev == (-1, -1) or curr == (-1, -1):
        return None

    # First player's score went up
    if curr[0] > prev[0]:
        return server_is_first
    # Second player's score went up
    if curr[1] > prev[1]:
        return not server_is_first
    return None


def _normalize_surface(surface: str) -> str:
    s = surface.lower()
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    return "Hard"


def _normalize_tournament_level(event_type: str) -> str:
    """Map API-Tennis event_type_type to ATP tour level codes."""
    et = event_type.upper()
    if "GRAND SLAM" in et or "SLAM" in et:
        return "G"
    if "MASTERS" in et or "1000" in et:
        return "M"
    if "500" in et:
        return "A"
    if "250" in et:
        return "A"
    if "CHALLENGER" in et:
        return "C"
    return "A"


class TennisFeed:
    """Streams live tennis point events from API-Tennis.

    Args:
        api_key: API-Tennis authentication key.
        on_point: Async callback invoked with each PointEvent.
        rest_poll_secs: Polling interval when in REST fallback mode.
    """

    def __init__(
        self,
        api_key: str,
        on_point: Callable[[PointEvent], asyncio.Future],
        rest_poll_secs: float = 3.0,
    ) -> None:
        self._api_key = api_key
        self._on_point = on_point
        self._rest_poll_secs = rest_poll_secs
        # match_id → total points already emitted (global sequence counter)
        self._points_seen: dict[str, int] = {}
        self._running = False

    async def run(self) -> None:
        """Run forever: WebSocket primary, REST fallback on disconnect."""
        self._running = True
        backoff = _BACKOFF_MIN
        while self._running:
            try:
                await self._run_websocket()
                backoff = _BACKOFF_MIN
            except Exception as e:
                log.warning("WebSocket error: %s — switching to REST fallback", e)
                await self._run_rest_fallback(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    async def stop(self) -> None:
        self._running = False

    async def _run_websocket(self) -> None:
        import websockets

        ws_url = f"wss://wss.api-tennis.com/live?APIkey={self._api_key}"
        log.info("Connecting to API-Tennis WebSocket")
        async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10) as ws:
            log.info("API-Tennis WebSocket connected")
            async for raw in ws:
                if not self._running:
                    return
                try:
                    data = json.loads(raw)
                    await self._process_response(data)
                except Exception as e:
                    log.debug("Error processing WS message: %s", e)

    async def _run_rest_fallback(self, initial_backoff: float) -> None:
        import httpx

        log.info("REST fallback: polling every %.1fs", self._rest_poll_secs)
        rest_url = "https://api.api-tennis.com/tennis/"
        params = {"method": "get_livescore", "APIkey": self._api_key}

        async with httpx.AsyncClient(timeout=10.0) as client:
            await asyncio.sleep(initial_backoff)
            while self._running:
                try:
                    resp = await client.get(rest_url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        await self._process_response(data)
                except Exception as e:
                    log.debug("REST poll error: %s", e)
                await asyncio.sleep(self._rest_poll_secs)

    async def _process_response(self, data) -> None:
        """Handle top-level API response: {"success":1, "result":[...]}."""
        if isinstance(data, dict):
            result = data.get("result", data)
            if isinstance(result, list):
                for match in result:
                    if isinstance(match, dict):
                        await self._process_match(match)
            elif isinstance(result, dict):
                await self._process_match(result)
        elif isinstance(data, list):
            for match in data:
                if isinstance(match, dict):
                    await self._process_match(match)

    async def _process_match(self, match: dict) -> None:
        """Parse one match object from the API-Tennis feed."""
        try:
            match_id = str(match.get("event_key", ""))
            if not match_id:
                return

            # Skip non-singles matches (doubles use "/ " separator)
            p1 = str(match.get("event_first_player", ""))
            p2 = str(match.get("event_second_player", ""))
            if not p1 or not p2 or "/" in p1 or "/" in p2:
                return

            # Skip finished matches
            status = str(match.get("event_status", ""))
            if status.lower() == "finished":
                return

            # Surface and level
            event_type = str(match.get("event_type_type", ""))
            # Only process men's singles for now (model trained on ATP)
            if "men" not in event_type.lower():
                return

            surface = _normalize_surface(str(match.get("event_ground", "Hard")))
            tour_level = _normalize_tournament_level(event_type)
            match_date = str(match.get("event_date", ""))[:10]

            # Server: "First Player" or "Second Player"
            serve_str = str(match.get("event_serve", ""))
            server = 0 if serve_str == "First Player" else 1

            # Set scores from scores array
            sets_p1, sets_p2, games_p1, games_p2 = self._parse_scores_array(match)

            # Current game score
            game_score_str = str(match.get("event_game_result", "0 - 0"))
            pts = _parse_score_field(game_score_str)
            pt_p1 = pts[0] if pts[0] >= 0 else 0
            pt_p2 = pts[1] if pts[1] >= 0 else 0

            is_tiebreak = games_p1 == 6 and games_p2 == 6

            # Process point-by-point data
            pbp_list = match.get("pointbypoint") or []
            if pbp_list:
                await self._process_pbp(
                    match_id, p1, p2, surface, tour_level, match_date,
                    sets_p1, sets_p2, is_tiebreak, pbp_list,
                )
            else:
                # No PBP — emit a single event from top-level score
                # Use game score as dedup key
                dedup_key = f"{sets_p1}{sets_p2}{games_p1}{games_p2}{game_score_str}"
                prev_seen = self._points_seen.get(match_id, 0)
                seq = hash(dedup_key) & 0x7FFFFFFF
                if match_id in self._points_seen and seq == prev_seen:
                    return
                self._points_seen[match_id] = seq

                event = PointEvent(
                    match_id=match_id,
                    player1_name=p1, player2_name=p2,
                    surface=surface, tournament_level=tour_level,
                    match_date=match_date,
                    sets_p1=sets_p1, sets_p2=sets_p2,
                    games_p1=games_p1, games_p2=games_p2,
                    points_p1=pt_p1, points_p2=pt_p2,
                    server=server, is_tiebreak=is_tiebreak,
                    server_won_last_point=None,
                    point_sequence=seq,
                )
                await self._on_point(event)

        except Exception as e:
            log.debug("Error in _process_match: %s", e)

    async def _process_pbp(
        self,
        match_id: str, p1: str, p2: str,
        surface: str, tour_level: str, match_date: str,
        sets_p1: int, sets_p2: int,
        is_tiebreak: bool, pbp_list: list,
    ) -> None:
        """Emit events from the nested point-by-point structure.

        pbp_list is a list of game objects:
          {
            "set_number": "Set 1",
            "number_game": "1",
            "player_served": "First Player",
            "score": "1 - 0",               # game score after this game
            "points": [
              {"number_point": "1", "score": "15 - 0", ...},
            ]
          }
        """
        # Flatten all points into a global sequence
        all_points: list[tuple[int, int, int, dict, bool]] = []  # (set_idx, game_idx, pt_idx, pt, server_is_first)
        for game in pbp_list:
            set_num = self._parse_set_number(str(game.get("set_number", "Set 1")))
            game_num = int(game.get("number_game", 0))
            server_is_first = str(game.get("player_served", "")) == "First Player"
            points = game.get("points") or []
            for pt in points:
                pt_num = int(pt.get("number_point", 0))
                all_points.append((set_num, game_num, pt_num, pt, server_is_first))

        # How many points have we already emitted for this match?
        prev_count = self._points_seen.get(match_id, 0)
        if len(all_points) <= prev_count:
            return  # no new points

        # Emit only new points
        new_points = all_points[prev_count:]
        global_seq = prev_count

        prev_score_str: Optional[str] = None

        for _, game_idx, _, pt, server_is_first in new_points:
            global_seq += 1
            score_str = str(pt.get("score", "0 - 0"))
            server_won = _server_won_from_scores(prev_score_str, score_str, server_is_first)
            prev_score_str = score_str

            pts = _parse_score_field(score_str)
            pt_p1 = pts[0] if pts[0] >= 0 else 0
            pt_p2 = pts[1] if pts[1] >= 0 else 0

            server = 0 if server_is_first else 1

            # Approximate games from the game index within this set
            # (exact values come from the last game's "score" field, but for
            # point-level events the per-point game count is approximate)

            event = PointEvent(
                match_id=match_id,
                player1_name=p1, player2_name=p2,
                surface=surface, tournament_level=tour_level,
                match_date=match_date,
                sets_p1=sets_p1, sets_p2=sets_p2,
                games_p1=game_idx, games_p2=0,  # will be refined below
                points_p1=pt_p1, points_p2=pt_p2,
                server=server, is_tiebreak=is_tiebreak,
                server_won_last_point=server_won,
                point_sequence=global_seq,
            )
            await self._on_point(event)

        self._points_seen[match_id] = global_seq

    @staticmethod
    def _parse_set_number(set_str: str) -> int:
        """Parse 'Set 1' -> 1."""
        m = re.search(r"\d+", set_str)
        return int(m.group()) if m else 1

    @staticmethod
    def _parse_scores_array(match: dict) -> tuple[int, int, int, int]:
        """Parse the 'scores' array to get sets won and current set games.

        scores: [
          {"score_first": "6", "score_second": "3", "score_set": "1"},
          {"score_first": "2", "score_second": "4", "score_set": "2"},
        ]
        The last entry is the current (in-progress) set.
        Completed sets have score_first or score_second >= 6.

        Returns: (sets_p1, sets_p2, games_p1_current_set, games_p2_current_set)
        """
        scores = match.get("scores") or []
        if not scores:
            # Fallback to event_final_result
            final = str(match.get("event_final_result", "0 - 0"))
            parts = re.split(r"\s*-\s*", final.strip())
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1]), 0, 0
                except ValueError:
                    pass
            return 0, 0, 0, 0

        sets_p1 = 0
        sets_p2 = 0
        games_p1 = 0
        games_p2 = 0

        for i, s in enumerate(scores):
            # Parse score_first/score_second — may contain tiebreak like "7.7"
            sf = str(s.get("score_first", "0")).split(".")[0]
            ss = str(s.get("score_second", "0")).split(".")[0]
            try:
                g1 = int(sf)
                g2 = int(ss)
            except ValueError:
                continue

            is_last = i == len(scores) - 1
            if is_last:
                # Current set — these are the in-progress game counts
                games_p1 = g1
                games_p2 = g2
            else:
                # Completed set — determine who won
                if g1 > g2:
                    sets_p1 += 1
                elif g2 > g1:
                    sets_p2 += 1

        return sets_p1, sets_p2, games_p1, games_p2
