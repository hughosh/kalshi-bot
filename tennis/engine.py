"""PredictionEngine — the live integration seam.

This is the ONLY class that downstream integrations (live feed, Kalshi adapter)
need to import. Everything else in the tennis/ package is internal.

Usage pattern (live match):
    engine = PredictionEngine.load("data/artifacts/")

    # At match start:
    engine.new_match(initial_state)
    result = engine.predict(initial_state)
    print(result.win_prob_p1, result.confidence)

    # After each point:
    engine.record_point(server_won=True)
    new_state = <update state from live feed>
    result = engine.predict(new_state)
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from tennis.types import MatchState, PredictionResult
from tennis.markov import match_win_prob, match_win_prob_batch
from tennis.updater import BayesianUpdater
from tennis.features import (
    build_pre_match_features,
    build_inplay_features,
    combine_features,
    TOUR_AVERAGES,
)

# Lazy imports to avoid hard-dep at module load time
_model_cls = None
_calibrator_cls = None
_elo_cls = None


def _get_model_cls():
    global _model_cls
    if _model_cls is None:
        from tennis.model import PointWinModel
        _model_cls = PointWinModel
    return _model_cls


def _get_calibrator_cls():
    global _calibrator_cls
    if _calibrator_cls is None:
        from tennis.calibration import MatchWinCalibrator
        _calibrator_cls = MatchWinCalibrator
    return _calibrator_cls


def _get_elo_cls():
    global _elo_cls
    if _elo_cls is None:
        from tennis.elo import EloTracker
        _elo_cls = EloTracker
    return _elo_cls


class PredictionEngine:
    """Real-time tennis win probability engine.

    The engine maintains per-match state (BayesianUpdater) and holds the
    pre-trained GBDT and calibrator as read-only artifacts. It is stateful
    with respect to the current match but stateless otherwise.

    Attributes:
        n_mc_samples: Number of Monte Carlo samples for CI estimation.
                      500 is the default; increase for tighter CIs at the cost
                      of ~2x latency (still fast: ~4ms for 1000 samples).
    """

    def __init__(
        self,
        model,
        calibrator,
        elo_tracker,
        player_stats_df: Optional[pd.DataFrame],
        h2h_df: Optional[pd.DataFrame] = None,
        n_mc_samples: int = 500,
        rng_seed: Optional[int] = None,
    ):
        self._model = model
        self._calibrator = calibrator
        self._elo = elo_tracker
        self._player_stats = player_stats_df
        self._h2h = h2h_df
        self.n_mc_samples = n_mc_samples
        self._rng = np.random.default_rng(rng_seed)

        # Per-match state (reset by new_match())
        self._updater: Optional[BayesianUpdater] = None
        self._pre_match_features: Optional[dict] = None
        self._current_server: Optional[int] = None  # server at match start

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def new_match(self, state: MatchState) -> None:
        """Initialise the engine for a new match.

        Must be called once at the start of each match before any predict()
        or record_point() calls.

        Args:
            state: Initial MatchState (typically sets/games/points all at 0).
        """
        # Compute pre-match features (server = player who serves first)
        server_id = state.player1_id if state.server == 0 else state.player2_id
        returner_id = state.player2_id if state.server == 0 else state.player1_id

        self._pre_match_features = build_pre_match_features(
            server_id=server_id,
            returner_id=returner_id,
            surface=state.surface,
            tournament_level=state.tournament_level,
            match_date=state.match_date,
            player_stats_df=self._player_stats,
            elo_tracker=self._elo,
            h2h_df=self._h2h,
        )

        # Get GBDT prior for p_serve
        p_prior = self._estimate_p_serve_prior(state)

        # Initialise Bayesian updater with GBDT prior
        self._updater = BayesianUpdater(p_serve_prior=p_prior)
        self._current_server = state.server

    def record_point(self, server_won: bool) -> None:
        """Update the Bayesian updater with the outcome of the last point.

        Call this AFTER predict() for each point, before updating the match
        state and calling predict() again.

        Args:
            server_won: True if the server won the point, False otherwise.
        """
        if self._updater is None:
            raise RuntimeError("Call new_match() before record_point().")
        self._updater.update(server_won)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, state: MatchState) -> PredictionResult:
        """Return the current match win probability and confidence estimate.

        This is the primary method called by the live integration layer.

        Args:
            state: Current MatchState. Should reflect the score AFTER the most
                   recent point (and after calling record_point()).

        Returns:
            PredictionResult with win_prob_p1, confidence, CI bounds, and
            the current p_serve estimate.
        """
        if self._updater is None:
            # Auto-initialise if new_match was not called (convenience for testing)
            self.new_match(state)

        # 1. Get posterior p_serve estimate from Bayesian updater.
        #    The updater already handles the transition from prior (GBDT) to data
        #    via its Beta distribution + optional decay. No separate blend needed.
        #    Per Klaassen & Magnus (2001), the updater's decay mechanism adapts
        #    to within-match performance shifts more cleanly than a linear blend.
        p_serve_posterior = self._updater.posterior_mean
        p_serve_posterior = float(np.clip(p_serve_posterior, 0.01, 0.99))

        # 3. Monte Carlo: sample p_serve from posterior, compute match win prob
        p_samples = self._updater.sample(n=self.n_mc_samples, rng=self._rng)
        win_probs = match_win_prob_batch(state, p_samples)

        # 4. Summarise MC distribution
        win_prob_raw = float(np.mean(win_probs))
        ci_lower = float(np.percentile(win_probs, 2.5))
        ci_upper = float(np.percentile(win_probs, 97.5))
        confidence = float(1.0 - (ci_upper - ci_lower))

        # 5. Calibrate the mean win probability
        win_prob_cal = self._calibrator.calibrate(win_prob_raw)
        # Recalibrate CI bounds consistently
        ci_lower_cal = self._calibrator.calibrate(ci_lower)
        ci_upper_cal = self._calibrator.calibrate(ci_upper)

        win_prob_cal = float(np.clip(win_prob_cal, 1e-6, 1 - 1e-6))
        ci_lower_cal = float(np.clip(ci_lower_cal, 0.0, 1.0))
        ci_upper_cal = float(np.clip(ci_upper_cal, 0.0, 1.0))

        return PredictionResult(
            win_prob_p1=win_prob_cal,
            win_prob_p2=1.0 - win_prob_cal,
            confidence=confidence,
            ci_lower=ci_lower_cal,
            ci_upper=ci_upper_cal,
            p_serve_estimate=p_serve_posterior,
            points_observed=self._updater.n_observed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_p_serve_prior(self, state: MatchState) -> float:
        """Get initial p_serve estimate using pre-match GBDT features."""
        if self._model is None or self._pre_match_features is None:
            return TOUR_AVERAGES["srv_total_won_pct"]

        # Use zero in-play features for the prior (match hasn't started)
        inplay_zero = {
            "match_srv_pts_won_pct": TOUR_AVERAGES["srv_total_won_pct"],
            "match_rtn_pts_won_pct": TOUR_AVERAGES["rtn_won_pct"],
            "momentum_last5": 2.5,
            "momentum_last10": 5.0,
            "sets_server": 0.0,
            "sets_returner": 0.0,
            "games_server": 0.0,
            "games_returner": 0.0,
            "points_played_total": 0.0,
            "is_tiebreak": 0.0,
            "is_final_set": 0.0,
        }
        x = combine_features(self._pre_match_features, inplay_zero)
        return float(np.clip(self._model.predict_single(x), 0.01, 0.99))

    def _estimate_p_serve(self, state: MatchState) -> float:
        """Get current p_serve estimate combining pre-match and in-play features."""
        if self._model is None or self._pre_match_features is None:
            return TOUR_AVERAGES["srv_total_won_pct"]

        inplay = build_inplay_features(state)
        x = combine_features(self._pre_match_features, inplay)
        return float(np.clip(self._model.predict_single(x), 0.01, 0.99))

    # ------------------------------------------------------------------
    # Loading from artifacts directory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifacts_dir: str, n_mc_samples: int = 500) -> PredictionEngine:
        """Load a trained PredictionEngine from saved artifacts.

        Expected files in artifacts_dir:
            gbdt_model.pkl         — trained PointWinModel
            calibrator.pkl         — trained MatchWinCalibrator
            player_elo.parquet     — EloTracker history
            player_stats.parquet   — rolling player stats (optional)
            h2h.parquet            — head-to-head records (optional)

        Args:
            artifacts_dir: Path to directory containing model artifacts.
            n_mc_samples: Number of Monte Carlo samples for CI (default 500).

        Returns:
            Loaded PredictionEngine ready for inference.
        """
        ModelCls = _get_model_cls()
        CalCls = _get_calibrator_cls()
        EloCls = _get_elo_cls()

        model_path = os.path.join(artifacts_dir, "gbdt_model.pkl")
        cal_path = os.path.join(artifacts_dir, "calibrator.pkl")
        elo_path = os.path.join(artifacts_dir, "player_elo.parquet")
        stats_path = os.path.join(artifacts_dir, "player_stats.parquet")
        h2h_path = os.path.join(artifacts_dir, "h2h.parquet")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GBDT model not found: {model_path}")
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"Calibrator not found: {cal_path}")

        model = ModelCls.load(model_path)
        calibrator = CalCls.load(cal_path)

        elo_tracker = EloCls.load(elo_path) if os.path.exists(elo_path) else EloCls()

        player_stats = (
            pd.read_parquet(stats_path) if os.path.exists(stats_path) else None
        )
        h2h = pd.read_parquet(h2h_path) if os.path.exists(h2h_path) else None

        return cls(
            model=model,
            calibrator=calibrator,
            elo_tracker=elo_tracker,
            player_stats_df=player_stats,
            h2h_df=h2h,
            n_mc_samples=n_mc_samples,
        )

    @classmethod
    def from_components(
        cls,
        model,
        calibrator,
        elo_tracker=None,
        player_stats_df=None,
        h2h_df=None,
        n_mc_samples: int = 500,
    ) -> PredictionEngine:
        """Construct engine from already-loaded components (useful for testing)."""
        if elo_tracker is None:
            EloCls = _get_elo_cls()
            elo_tracker = EloCls()
        return cls(
            model=model,
            calibrator=calibrator,
            elo_tracker=elo_tracker,
            player_stats_df=player_stats_df,
            h2h_df=h2h_df,
            n_mc_samples=n_mc_samples,
        )
