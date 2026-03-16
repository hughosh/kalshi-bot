"""Per-match engine session manager.

Each live match gets its own PredictionEngine instance (stateful Bayesian updater).
Shared artifacts (model, calibrator, elo_tracker, player_stats_df, h2h_df) are loaded
once and passed to each engine via PredictionEngine.from_components().

The MatchTracker:
  1. Receives PointEvents from the TennisFeed
  2. Translates them into MatchState updates
  3. Calls engine.record_point() + engine.predict()
  4. Emits PredictionResults keyed by match_id
"""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from integration.player_resolver import PlayerResolver
from integration.tennis_feed import PointEvent
from tennis.engine import PredictionEngine
from tennis.types import MatchFormat, MatchState, PredictionResult

log = logging.getLogger(__name__)


@dataclass
class MatchSession:
    """Live session for a single match."""
    match_id: str
    engine: PredictionEngine
    state: MatchState
    player1_name: str
    player2_name: str
    player1_id: str
    player2_id: str
    points_observed: int = 0
    initialized: bool = False


class MatchTracker:
    """Manages per-match PredictionEngine instances.

    Args:
        model: Trained GBDTPointWinModel (read-only, shared).
        calibrator: Fitted IsotonicCalibrator (read-only, shared).
        elo_tracker: EloTracker with loaded ratings (read-only, shared).
        player_stats_df: Player stats DataFrame (read-only, shared).
        h2h_df: Head-to-head DataFrame (read-only, shared).
        resolver: PlayerResolver for name → ATP ID mapping.
        n_mc_samples: Monte Carlo samples per prediction.
    """

    def __init__(
        self,
        model,
        calibrator,
        elo_tracker,
        player_stats_df: Optional[pd.DataFrame],
        h2h_df: Optional[pd.DataFrame],
        resolver: PlayerResolver,
        n_mc_samples: int = 200,
    ) -> None:
        self._model = model
        self._calibrator = calibrator
        self._elo_tracker = elo_tracker
        self._player_stats = player_stats_df
        self._h2h = h2h_df
        self._resolver = resolver
        self._n_mc = n_mc_samples
        self._sessions: dict[str, MatchSession] = {}

    def _get_or_create_session(self, event: PointEvent) -> Optional[MatchSession]:
        """Get existing session or create one for a new match."""
        if event.match_id in self._sessions:
            return self._sessions[event.match_id]

        # Resolve player IDs
        p1_id = self._resolver.resolve(event.player1_name)
        p2_id = self._resolver.resolve(event.player2_name)
        if p1_id is None or p2_id is None:
            log.warning("Cannot resolve players: %s (%s) vs %s (%s) — skipping match %s",
                        event.player1_name, p1_id, event.player2_name, p2_id, event.match_id)
            return None

        # Determine match format from tournament level
        if event.tournament_level == "G":
            match_format = MatchFormat.best_of_5_tiebreak()
        else:
            match_format = MatchFormat.best_of_3_tiebreak()

        # Build initial state
        state = MatchState(
            player1_id=p1_id,
            player2_id=p2_id,
            surface=event.surface,
            tournament_level=event.tournament_level,
            match_date=event.match_date,
            match_format=match_format,
            server=event.server,
        )

        # Create engine
        engine = PredictionEngine.from_components(
            model=self._model,
            calibrator=self._calibrator,
            elo_tracker=self._elo_tracker,
            player_stats_df=self._player_stats,
            h2h_df=self._h2h,
            n_mc_samples=self._n_mc,
        )

        session = MatchSession(
            match_id=event.match_id,
            engine=engine,
            state=state,
            player1_name=event.player1_name,
            player2_name=event.player2_name,
            player1_id=p1_id,
            player2_id=p2_id,
        )
        self._sessions[event.match_id] = session
        log.info("New match session: %s — %s (%s) vs %s (%s)",
                 event.match_id, event.player1_name, p1_id, event.player2_name, p2_id)
        return session

    def process_point(self, event: PointEvent) -> Optional[PredictionResult]:
        """Process a point event and return updated prediction.

        Returns None if the match can't be tracked (unresolved players, etc.).
        """
        session = self._get_or_create_session(event)
        if session is None:
            return None

        # Initialize engine on first point
        if not session.initialized:
            session.engine.new_match(session.state)
            session.initialized = True

        # Record point outcome (if we know who won it)
        if event.server_won_last_point is not None:
            session.engine.record_point(event.server_won_last_point)

            # Update in-match stats on the state
            if event.server == 0:
                session.state.serve_pts_played_p1 += 1
                if event.server_won_last_point:
                    session.state.serve_pts_won_p1 += 1
            else:
                session.state.serve_pts_played_p2 += 1
                if event.server_won_last_point:
                    session.state.serve_pts_won_p2 += 1

            session.state.recent_points.append(1 if event.server_won_last_point else 0)
            if len(session.state.recent_points) > 20:
                session.state.recent_points = session.state.recent_points[-20:]

            session.points_observed += 1

        # Update score state from feed
        session.state.sets_p1 = event.sets_p1
        session.state.sets_p2 = event.sets_p2
        session.state.games_p1 = event.games_p1
        session.state.games_p2 = event.games_p2
        session.state.points_p1 = event.points_p1
        session.state.points_p2 = event.points_p2
        session.state.server = event.server
        session.state.is_tiebreak = event.is_tiebreak

        # Predict
        try:
            result = session.engine.predict(session.state)
            return result
        except Exception as e:
            log.warning("Prediction error for match %s: %s", event.match_id, e)
            return None

    def get_session(self, match_id: str) -> Optional[MatchSession]:
        return self._sessions.get(match_id)

    def remove_session(self, match_id: str) -> None:
        if match_id in self._sessions:
            del self._sessions[match_id]
            log.info("Removed match session: %s", match_id)

    @property
    def active_sessions(self) -> dict[str, MatchSession]:
        return dict(self._sessions)
