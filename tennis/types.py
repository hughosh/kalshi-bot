"""Core data types for the tennis win probability engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Surface(str, Enum):
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"


class TourneyLevel(str, Enum):
    SLAM = "G"        # Grand Slam
    MASTERS = "M"     # ATP Masters 1000
    ATP500 = "A"      # ATP 500 / 250
    CHALLENGER = "C"  # Challenger
    OTHER = "O"


@dataclass
class MatchFormat:
    """Describes the rules governing how many sets/games constitute a win."""
    best_of: int = 3                 # 3 or 5
    final_set_tiebreak: bool = True  # True = super-tiebreak at 6-6 in final set
    final_set_advantage: bool = False  # True = no tiebreak in final set (Wimbledon pre-2019)

    # Common presets
    @classmethod
    def best_of_3_tiebreak(cls) -> MatchFormat:
        """Standard ATP best-of-3 with final-set tiebreak."""
        return cls(best_of=3, final_set_tiebreak=True, final_set_advantage=False)

    @classmethod
    def best_of_5_tiebreak(cls) -> MatchFormat:
        """Grand Slam best-of-5 with final-set tiebreak (US Open / Australian Open style)."""
        return cls(best_of=5, final_set_tiebreak=True, final_set_advantage=False)

    @classmethod
    def best_of_5_advantage(cls) -> MatchFormat:
        """Grand Slam best-of-5 with advantage final set (Wimbledon/Roland Garros classic)."""
        return cls(best_of=5, final_set_tiebreak=False, final_set_advantage=True)


@dataclass
class MatchState:
    """Complete snapshot of a tennis match at a given moment.

    The match state is sufficient to compute a win probability — no external
    match history is needed at prediction time (pre-match context is embedded
    in the player IDs and surface, which the engine uses to look up features).
    """
    # --- Score ---
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0     # games won in the current set
    games_p2: int = 0
    points_p1: int = 0    # points in the current game (or tiebreak)
    points_p2: int = 0
    server: int = 0       # 0 = player 1 is serving, 1 = player 2 is serving
    is_tiebreak: bool = False
    match_format: MatchFormat = field(default_factory=MatchFormat.best_of_3_tiebreak)

    # --- Running in-match stats (updated after each point) ---
    serve_pts_played_p1: int = 0
    serve_pts_won_p1: int = 0
    serve_pts_played_p2: int = 0
    serve_pts_won_p2: int = 0

    # Last N point outcomes for momentum calculation.
    # Each element is 1 if the *server at that point* won, else 0.
    recent_points: list = field(default_factory=list)

    # --- Pre-match context (static; used for GBDT prior, doesn't change) ---
    player1_id: str = ""
    player2_id: str = ""
    surface: str = Surface.HARD   # str for JSON-friendliness
    tournament_level: str = TourneyLevel.ATP500
    match_date: str = ""          # ISO date "YYYY-MM-DD"

    def is_final_set(self) -> bool:
        sets_needed = (self.match_format.best_of + 1) // 2
        return (self.sets_p1 + self.sets_p2) == (self.match_format.best_of - 1)

    def sets_needed(self) -> int:
        return (self.match_format.best_of + 1) // 2

    def server_serve_pct(self) -> Optional[float]:
        """Fraction of serve points won by current server in this match."""
        if self.server == 0:
            played = self.serve_pts_played_p1
            won = self.serve_pts_won_p1
        else:
            played = self.serve_pts_played_p2
            won = self.serve_pts_won_p2
        if played == 0:
            return None
        return won / played

    def returner_return_pct(self) -> Optional[float]:
        """Fraction of return points won by current returner in this match.
        (= 1 - server's serve pct from the returner's perspective)
        """
        if self.server == 0:
            # returner is p2; p2's serve stats when they were serving
            played = self.serve_pts_played_p2
            won = self.serve_pts_played_p2 - self.serve_pts_won_p2
        else:
            played = self.serve_pts_played_p1
            won = self.serve_pts_played_p1 - self.serve_pts_won_p1
        if played == 0:
            return None
        return won / played

    def total_points_played(self) -> int:
        return self.serve_pts_played_p1 + self.serve_pts_played_p2


@dataclass
class PredictionResult:
    """Output of the prediction engine for a single match state.

    This is the contract the live integration layer reads from.
    """
    win_prob_p1: float       # P(player 1 wins match) ∈ [0, 1]
    win_prob_p2: float       # = 1 - win_prob_p1
    confidence: float        # ∈ [0, 1], higher = more certain (1 - CI_width)
    ci_lower: float          # 95% CI lower bound on win_prob_p1
    ci_upper: float          # 95% CI upper bound on win_prob_p1
    p_serve_estimate: float  # posterior mean of server's point win probability
    points_observed: int     # total points played so far in match

    def __repr__(self) -> str:
        return (
            f"PredictionResult("
            f"p1={self.win_prob_p1:.3f} "
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}] "
            f"conf={self.confidence:.3f} "
            f"p_serve={self.p_serve_estimate:.3f} "
            f"n={self.points_observed})"
        )
