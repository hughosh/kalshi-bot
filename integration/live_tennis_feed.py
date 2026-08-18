"""Live Tennis API feed — a drop-in alternative to the API-Tennis TennisFeed.

Vendor note: this adapter is contributed by the Live Tennis API team
(https://livetennisapi.com). Live Tennis API is a live-tennis DATA provider, not
a market or execution venue; this only produces the same ``PointEvent`` objects
``tennis_feed.TennisFeed`` produces, so the rest of the pipeline (MatchTracker →
Edge/Risk/Guard → Kalshi) is untouched. Judge accordingly.

It is opt-in: the bot keeps using API-Tennis unless ``TENNIS_FEED_PROVIDER`` is
set to ``live_tennis`` (see ``config.py``). The two feeds share the ``run()`` /
``stop()`` / ``on_point`` contract so either can drive the bot.

Source (base https://api.livetennisapi.com/api/public/v1, auth via the
``X-API-Key`` header; free key https://livetennisapi.com/subscribe/free is
30 req/min and 100 req/day — fine for testing or low-cadence polling, not
continuous fast polling):

  GET /matches?status=live   FREE   live score, current server, retired/walkover

The free tier exposes the current-score snapshot per match. This feed emits one
``PointEvent`` per NEW snapshot (deduplicated by a per-match score signature,
the same idea as ``TennisFeed``'s no-point-by-point branch), with
``server_won_last_point=None``. Per-point granularity (``server_won_last_point``)
would use the PRO-tier ``GET /matches/{id}/events`` stream; that is left as an
optional enhancement so the adapter works on the free tier as-is.

Live Tennis API Score shape (only the fields used here):
  {
    "data": [
      {
        "id": 12345, "tournament": "...", "tour": "atp",
        "surface": "hard", "status": "live", "event_status": null,
        "players": {"p1": {"name": "Novak Djokovic"}, "p2": {"name": "..."}},
        "score": {
          "sets": [1, 0],                 # [sets_p1, sets_p2]
          "games": [[6, 3], [4, 2]],       # [games_p1_per_set, games_p2_per_set]
          "points": ["30", "40"],          # ["0","15","30","40","AD"] or null
          "server": 1,                     # 1 = p1, 2 = p2, or null
          "is_tiebreak": false
        }
      }
    ]
  }
"""
import asyncio
import logging
from typing import Callable, Optional

from integration.tennis_feed import PointEvent, _normalize_surface

log = logging.getLogger(__name__)

_POINT_INDEX = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
_BACKOFF_MIN = 1.0
_BACKOFF_MAX = 30.0

_DEFAULT_BASE_URL = "https://api.livetennisapi.com/api/public/v1"


def _point_index(point: Optional[str]) -> int:
    """Map a tennis point string to the same 0..4 index TennisFeed uses."""
    if point is None:
        return 0
    return _POINT_INDEX.get(str(point).strip().upper(), 0)


def _normalize_tour_level(tour: Optional[str]) -> str:
    """Map the Live Tennis API tour to the tour-level codes the model expects.

    The /matches feed does not expose 250/500/Masters granularity, so this is a
    conservative approximation: challenger -> 'C', everything else -> 'A'.
    """
    if tour and "challenger" in str(tour).lower():
        return "C"
    return "A"


class LiveTennisApiFeed:
    """Streams live tennis point events from the Live Tennis API.

    Drop-in alternative to ``tennis_feed.TennisFeed`` — same constructor and
    ``run()`` / ``stop()`` / ``on_point`` contract.

    Args:
        api_key: Live Tennis API key (X-API-Key header).
        on_point: Async callback invoked with each PointEvent.
        rest_poll_secs: Polling interval for GET /matches?status=live.
        base_url: API base URL (override for testing).
    """

    def __init__(
        self,
        api_key: str,
        on_point: Callable[[PointEvent], asyncio.Future],
        rest_poll_secs: float = 3.0,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._on_point = on_point
        self._rest_poll_secs = rest_poll_secs
        self._base_url = base_url.rstrip("/")
        # match_id -> last emitted score signature
        self._points_seen: dict[str, int] = {}
        self._running = False

    async def run(self) -> None:
        """Poll the live-matches endpoint forever, with exponential backoff on error."""
        import httpx

        self._running = True
        backoff = _BACKOFF_MIN
        url = f"{self._base_url}/matches"
        params = {"status": "live"}
        headers = {"X-API-Key": self._api_key}

        log.info("Live Tennis API feed: polling every %.1fs", self._rest_poll_secs)
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            while self._running:
                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        await self._process_response(resp.json())
                        backoff = _BACKOFF_MIN
                    else:
                        log.warning("Live Tennis API returned HTTP %d", resp.status_code)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, _BACKOFF_MAX)
                        continue
                except Exception as e:
                    log.warning("Live Tennis API poll error: %s", e)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX)
                    continue
                await asyncio.sleep(self._rest_poll_secs)

    async def stop(self) -> None:
        self._running = False

    async def _process_response(self, data) -> None:
        """Handle the {"data": [...matches...]} response shape."""
        if isinstance(data, dict):
            matches = data.get("data", data)
        else:
            matches = data
        if isinstance(matches, list):
            for match in matches:
                if isinstance(match, dict):
                    await self._process_match(match)

    async def _process_match(self, match: dict) -> None:
        """Parse one Live Tennis API match into a PointEvent and emit if new."""
        try:
            match_id = str(match.get("id", ""))
            if not match_id:
                return

            status = str(match.get("status", "")).lower()
            if status in ("completed", "cancelled"):
                return

            players = match.get("players") or {}
            p1 = str((players.get("p1") or {}).get("name", ""))
            p2 = str((players.get("p2") or {}).get("name", ""))
            if not p1 or not p2 or "/" in p1 or "/" in p2:
                return

            score = match.get("score") or {}

            sets = score.get("sets") or []
            sets_p1 = int(sets[0]) if len(sets) > 0 and sets[0] is not None else 0
            sets_p2 = int(sets[1]) if len(sets) > 1 and sets[1] is not None else 0

            games = score.get("games") or []
            games_p1 = self._current_games(games, 0)
            games_p2 = self._current_games(games, 1)

            points = score.get("points") or []
            pt_p1 = _point_index(points[0]) if len(points) > 0 else 0
            pt_p2 = _point_index(points[1]) if len(points) > 1 else 0

            server_raw = score.get("server")
            server = 0 if server_raw == 1 else (1 if server_raw == 2 else 0)

            is_tiebreak = bool(score.get("is_tiebreak", False))

            surface = _normalize_surface(str(match.get("surface", "Hard") or "Hard"))
            tour_level = _normalize_tour_level(match.get("tour"))

            # Dedup by a stable per-match score signature (no PBP on the free tier).
            signature = hash((sets_p1, sets_p2, games_p1, games_p2, pt_p1, pt_p2, server)) & 0x7FFFFFFF
            if self._points_seen.get(match_id) == signature:
                return
            self._points_seen[match_id] = signature

            event = PointEvent(
                match_id=match_id,
                player1_name=p1,
                player2_name=p2,
                surface=surface,
                tournament_level=tour_level,
                match_date="",
                sets_p1=sets_p1,
                sets_p2=sets_p2,
                games_p1=games_p1,
                games_p2=games_p2,
                points_p1=pt_p1,
                points_p2=pt_p2,
                server=server,
                is_tiebreak=is_tiebreak,
                server_won_last_point=None,
                point_sequence=signature,
            )
            await self._on_point(event)

        except Exception as e:
            log.debug("Error in _process_match: %s", e)

    @staticmethod
    def _current_games(games: list, player_index: int) -> int:
        """Return the current-set game count for a player from the games arrays.

        games is [games_p1_per_set, games_p2_per_set]; the current set is the
        last entry of each per-player list.
        """
        try:
            per_set = games[player_index]
            if isinstance(per_set, list) and per_set:
                return int(per_set[-1])
        except (IndexError, TypeError, ValueError):
            pass
        return 0
