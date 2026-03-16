"""Surface-specific ELO rating tracker.

Maintains separate ELO ratings per player per surface (Hard/Clay/Grass).
Processes matches in chronological order to avoid data leakage.

K-factors follow common practice in tennis ELO literature:
    Grand Slams / Masters: K=40 (higher weight)
    ATP 500/250:           K=32
    Challenger/Other:      K=20
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional

import pandas as pd


INITIAL_RATING = 1500.0
DEFAULT_K = 32.0

K_FACTOR = {
    "G": 40.0,  # Grand Slam
    "M": 40.0,  # Masters 1000
    "A": 32.0,  # ATP 500 / 250
    "F": 32.0,  # Tour Finals
    "C": 20.0,  # Challenger
    "S": 20.0,  # Satellite / ITF
    "O": 20.0,  # Other
}

SURFACES = ("Hard", "Clay", "Grass", "Carpet")


class EloTracker:
    """Maintains surface-specific ELO ratings for all players.

    Usage:
        tracker = EloTracker()
        tracker.process_matches(matches_df)   # train in date order
        rating = tracker.get_rating("player_id", "Hard")
    """

    def __init__(self):
        # ratings[surface][player_id] = current ELO
        self.ratings: Dict[str, Dict[str, float]] = {s: defaultdict(lambda: INITIAL_RATING) for s in SURFACES}
        # Store a snapshot per (tourney_date, player_id, surface) for look-up
        # at any historical date.  Key: (surface, player_id) → list of (date_str, rating)
        self._history: Dict[tuple, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def process_matches(self, matches_df: pd.DataFrame) -> None:
        """Process a DataFrame of matches in chronological order.

        Expected columns (JeffSackmann tennis_atp format):
            tourney_date  : int YYYYMMDD or str
            tourney_level : str ("G", "M", "A", "F", "C", "S")
            surface       : str ("Hard", "Clay", "Grass", "Carpet")
            winner_id     : int or str
            loser_id      : int or str
        """
        df = matches_df.copy()
        df["tourney_date"] = df["tourney_date"].astype(str)
        df = df.sort_values("tourney_date").reset_index(drop=True)

        for _, row in df.iterrows():
            surface = str(row.get("surface", "Hard"))
            if surface not in SURFACES:
                surface = "Hard"

            w_id = str(row["winner_id"])
            l_id = str(row["loser_id"])
            level = str(row.get("tourney_level", "A"))
            date = str(row["tourney_date"])

            k = K_FACTOR.get(level, DEFAULT_K)
            self._update_elo(w_id, l_id, surface, k, date)

    def _update_elo(
        self, winner_id: str, loser_id: str, surface: str, k: float, date: str
    ) -> None:
        r_w = self.ratings[surface][winner_id]
        r_l = self.ratings[surface][loser_id]

        expected_w = _expected(r_w, r_l)
        expected_l = 1.0 - expected_w

        new_r_w = r_w + k * (1.0 - expected_w)
        new_r_l = r_l + k * (0.0 - expected_l)

        self.ratings[surface][winner_id] = new_r_w
        self.ratings[surface][loser_id] = new_r_l

        # Record snapshot (date, rating) after each update
        self._history[(surface, winner_id)].append((date, new_r_w))
        self._history[(surface, loser_id)].append((date, new_r_l))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_rating(
        self,
        player_id: str,
        surface: str,
        as_of_date: Optional[str] = None,
    ) -> float:
        """Return ELO rating for a player on a given surface.

        Args:
            player_id: Player identifier (str).
            surface: One of "Hard", "Clay", "Grass", "Carpet".
            as_of_date: ISO date string "YYYYMMDD" or "YYYY-MM-DD".
                        If None, returns the most recent rating.

        Returns:
            ELO rating (float). Returns INITIAL_RATING for unknown players.
        """
        if surface not in SURFACES:
            surface = "Hard"

        player_id = str(player_id)

        if as_of_date is None:
            return self.ratings[surface].get(player_id, INITIAL_RATING)

        # Normalise date to 8-digit YYYYMMDD string
        as_of = as_of_date.replace("-", "")

        history = self._history.get((surface, player_id))
        if not history:
            return INITIAL_RATING

        # Binary search for last entry before or on as_of_date
        lo, hi = 0, len(history) - 1
        result = INITIAL_RATING
        while lo <= hi:
            mid = (lo + hi) // 2
            d, r = history[mid]
            if d <= as_of:
                result = r
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def get_rating_diff(
        self,
        player1_id: str,
        player2_id: str,
        surface: str,
        as_of_date: Optional[str] = None,
    ) -> float:
        """ELO difference: player1 rating - player2 rating."""
        r1 = self.get_rating(player1_id, surface, as_of_date)
        r2 = self.get_rating(player2_id, surface, as_of_date)
        return r1 - r2

    def win_probability_from_elo(
        self,
        player1_id: str,
        player2_id: str,
        surface: str,
        as_of_date: Optional[str] = None,
    ) -> float:
        """ELO-based pre-match win probability for player 1."""
        r1 = self.get_rating(player1_id, surface, as_of_date)
        r2 = self.get_rating(player2_id, surface, as_of_date)
        return _expected(r1, r2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export full rating history as a DataFrame for saving to parquet."""
        rows = []
        for (surface, player_id), history in self._history.items():
            for date, rating in history:
                rows.append({
                    "surface": surface,
                    "player_id": player_id,
                    "date": date,
                    "elo": rating,
                })
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save rating history to parquet."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> EloTracker:
        """Load from saved parquet and rebuild tracker state."""
        df = pd.read_parquet(path)
        tracker = cls()
        for _, row in df.iterrows():
            surface = row["surface"]
            player_id = str(row["player_id"])
            date = str(row["date"])
            elo = float(row["elo"])
            if surface not in tracker.ratings:
                continue
            tracker._history[(surface, player_id)].append((date, elo))

        # Rebuild current ratings from latest history entry
        for (surface, player_id), history in tracker._history.items():
            if history:
                tracker.ratings[surface][player_id] = history[-1][1]

        return tracker


# ---------------------------------------------------------------------------
# Baseline: pure ELO match win probability (used as a benchmark in backtest)
# ---------------------------------------------------------------------------

def elo_baseline_win_prob(elo_diff: float) -> float:
    """Convert ELO difference to win probability (standard logistic).

    This is the naive ELO baseline for benchmarking purposes.
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _expected(r1: float, r2: float) -> float:
    """Standard ELO expected score for player 1."""
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
