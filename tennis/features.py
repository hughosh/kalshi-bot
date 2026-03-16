"""Feature engineering for the tennis point-win probability model.

Features are split into:
  - Pre-match features: computed from historical data up to match_date.
                        Static throughout a match. Cached per match.
  - In-play features:   derived from current MatchState (score, running stats).
                        Updated after every point.

FEATURE_COLUMNS defines the exact ordered list of columns fed to the GBDT model.
The same ordering must be used for both training and inference.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from tennis.elo import EloTracker, INITIAL_RATING
from tennis.types import MatchState, Surface, TourneyLevel


# ---------------------------------------------------------------------------
# Feature column definition (must stay stable across train/infer)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # ELO
    "elo_server",
    "elo_returner",
    "elo_diff",
    # Rankings
    "ranking_server",
    "ranking_returner",
    "ranking_diff",
    # Serve stats (rolling 52-week)
    "srv_1st_in_pct_server",
    "srv_1st_won_pct_server",
    "srv_2nd_won_pct_server",
    "srv_total_won_pct_server",
    "srv_1st_in_pct_returner",
    "srv_1st_won_pct_returner",
    "srv_2nd_won_pct_returner",
    "srv_total_won_pct_returner",
    # Return stats (rolling 52-week)
    "rtn_won_pct_server",
    "rtn_won_pct_returner",
    # Head-to-head
    "h2h_srv_win_rate",          # server H2H win rate vs returner
    "h2h_matches",               # number of H2H matches
    # Demographics
    "age_server",
    "age_returner",
    "age_diff",
    # Surface one-hot
    "surface_hard",
    "surface_clay",
    "surface_grass",
    # Tournament level one-hot
    "tourney_level_slam",
    "tourney_level_masters",
    # In-play dynamic features
    "match_srv_pts_won_pct",     # server's serve pts won % in this match
    "match_rtn_pts_won_pct",     # returner's return pts won % in this match
    "momentum_last5",            # server pts won in last 5 points (0-5)
    "momentum_last10",           # server pts won in last 10 points (0-10)
    "sets_server",               # sets won by server so far
    "sets_returner",
    "games_server",              # games won in current set
    "games_returner",
    "points_played_total",       # total points played in match
    "is_tiebreak",
    "is_final_set",
]

N_FEATURES = len(FEATURE_COLUMNS)

# Tour-average fallback values (imputed when player history is insufficient)
TOUR_AVERAGES = {
    "srv_1st_in_pct": 0.62,
    "srv_1st_won_pct": 0.73,
    "srv_2nd_won_pct": 0.54,
    "srv_total_won_pct": 0.63,
    "rtn_won_pct": 0.37,
    "ranking": 200,
    "age": 26.0,
}


# ---------------------------------------------------------------------------
# Player stats helper — built from JeffSackmann tennis_atp match CSV data
# ---------------------------------------------------------------------------

def build_player_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling per-player match stats from JeffSackmann ATP match data.

    Returns a long-form DataFrame with columns:
        player_id, date, surface,
        srv_1st_in_pct, srv_1st_won_pct, srv_2nd_won_pct, srv_total_won_pct,
        rtn_won_pct, ranking, age

    Each row represents a snapshot of a player's rolling 52-week stats as of
    that match date (computed from matches *before* that date).

    Args:
        matches_df: Combined ATP match data with columns:
            tourney_date, surface, winner_id, loser_id,
            winner_age, loser_age, winner_rank, loser_rank,
            w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms,
            l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms,
            w_bpSaved, l_bpSaved (optional)
    """
    df = matches_df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # --- Build per-player per-match records ---
    records = []

    def add_records(row, prefix_w, prefix_l):
        for role, opp_role in [("w", "l"), ("l", "w")]:
            try:
                svpt = float(row.get(f"{role}_svpt", 0) or 0)
                if svpt == 0:
                    continue
                first_in = float(row.get(f"{role}_1stIn", 0) or 0)
                first_won = float(row.get(f"{role}_1stWon", 0) or 0)
                second_won = float(row.get(f"{role}_2ndWon", 0) or 0)
                opp_svpt = float(row.get(f"{opp_role}_svpt", 0) or 0)
                opp_first_in = float(row.get(f"{opp_role}_1stIn", 0) or 0)
                opp_first_won = float(row.get(f"{opp_role}_1stWon", 0) or 0)
                opp_second_won = float(row.get(f"{opp_role}_2ndWon", 0) or 0)

                second_in = svpt - first_in
                second_in_opp = opp_svpt - opp_first_in

                records.append({
                    "player_id": str(row[f"{role}inner_id" if role == "w" else f"{role}oser_id"]),
                    "date": row["tourney_date"],
                    "surface": str(row.get("surface", "Hard")),
                    "ranking": float(row.get(f"{role}inner_rank" if role == "w" else f"{role}oser_rank", 200) or 200),
                    "age": float(row.get(f"{role}inner_age" if role == "w" else f"{role}oser_age", 26.0) or 26.0),
                    "svpt": svpt,
                    "first_in": first_in,
                    "first_won": first_won,
                    "second_in": max(second_in, 0),
                    "second_won": second_won,
                    "rtn_pts_won": opp_svpt - opp_first_won - opp_second_won,
                    "opp_svpt": opp_svpt,
                })
            except (KeyError, TypeError, ValueError):
                continue

    for _, row in df.iterrows():
        add_records(row, "w", "l")

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return stats_df

    stats_df = stats_df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # --- Compute rolling 52-week aggregated stats ---
    # For each match, we need stats from the preceding 52 weeks.
    # We do this by grouping per player and using a rolling window.
    result_rows = []
    for player_id, group in stats_df.groupby("player_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for i, row in group.iterrows():
            cutoff = row["date"] - pd.Timedelta(weeks=52)
            # Include matches strictly before this match
            window = group[(group["date"] >= cutoff) & (group["date"] < row["date"])]
            if len(window) < 5:
                # Not enough history — use tour averages
                result_rows.append({
                    "player_id": player_id,
                    "date": row["date"],
                    "surface": row["surface"],
                    "ranking": row["ranking"],
                    "age": row["age"],
                    "srv_1st_in_pct": TOUR_AVERAGES["srv_1st_in_pct"],
                    "srv_1st_won_pct": TOUR_AVERAGES["srv_1st_won_pct"],
                    "srv_2nd_won_pct": TOUR_AVERAGES["srv_2nd_won_pct"],
                    "srv_total_won_pct": TOUR_AVERAGES["srv_total_won_pct"],
                    "rtn_won_pct": TOUR_AVERAGES["rtn_won_pct"],
                })
            else:
                total_svpt = window["svpt"].sum()
                total_first_in = window["first_in"].sum()
                total_first_won = window["first_won"].sum()
                total_second_in = window["second_in"].sum()
                total_second_won = window["second_won"].sum()
                total_rtn_pts_won = window["rtn_pts_won"].sum()
                total_opp_svpt = window["opp_svpt"].sum()

                first_in_pct = total_first_in / total_svpt if total_svpt > 0 else TOUR_AVERAGES["srv_1st_in_pct"]
                first_won_pct = total_first_won / total_first_in if total_first_in > 0 else TOUR_AVERAGES["srv_1st_won_pct"]
                second_won_pct = total_second_won / total_second_in if total_second_in > 0 else TOUR_AVERAGES["srv_2nd_won_pct"]
                total_won_pct = (total_first_won + total_second_won) / total_svpt if total_svpt > 0 else TOUR_AVERAGES["srv_total_won_pct"]
                rtn_won_pct = total_rtn_pts_won / total_opp_svpt if total_opp_svpt > 0 else TOUR_AVERAGES["rtn_won_pct"]

                result_rows.append({
                    "player_id": player_id,
                    "date": row["date"],
                    "surface": row["surface"],
                    "ranking": row["ranking"],
                    "age": row["age"],
                    "srv_1st_in_pct": first_in_pct,
                    "srv_1st_won_pct": first_won_pct,
                    "srv_2nd_won_pct": second_won_pct,
                    "srv_total_won_pct": total_won_pct,
                    "rtn_won_pct": rtn_won_pct,
                })

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_pre_match_features(
    server_id: str,
    returner_id: str,
    surface: str,
    tournament_level: str,
    match_date: str,  # YYYY-MM-DD
    player_stats_df: pd.DataFrame,  # output of build_player_stats
    elo_tracker: EloTracker,
    h2h_df: Optional[pd.DataFrame] = None,  # head-to-head records
) -> dict:
    """Compute static pre-match feature dict for a given server/returner pair.

    All features reflect knowledge available strictly before match_date.
    """
    date_ts = pd.Timestamp(match_date)
    date_8d = match_date.replace("-", "")

    def get_stats(player_id: str) -> dict:
        if player_stats_df is None or player_stats_df.empty:
            return {}
        # Most recent row for this player before match_date
        mask = (
            (player_stats_df["player_id"] == str(player_id))
            & (player_stats_df["date"] < date_ts)
        )
        rows = player_stats_df[mask]
        if rows.empty:
            return {}
        return rows.sort_values("date").iloc[-1].to_dict()

    srv_stats = get_stats(server_id)
    rtn_stats = get_stats(returner_id)

    # ELO ratings
    elo_srv = elo_tracker.get_rating(server_id, surface, date_8d)
    elo_rtn = elo_tracker.get_rating(returner_id, surface, date_8d)

    # Surface one-hot
    surf_norm = str(surface).capitalize()
    surface_hard = int(surf_norm == "Hard")
    surface_clay = int(surf_norm == "Clay")
    surface_grass = int(surf_norm == "Grass")

    # Tournament level one-hot
    level_slam = int(str(tournament_level) == "G")
    level_masters = int(str(tournament_level) == "M")

    # H2H
    h2h_win_rate = 0.5  # neutral default
    h2h_matches = 0
    if h2h_df is not None and not h2h_df.empty:
        h2h = h2h_df[
            (h2h_df["player1_id"] == str(server_id))
            & (h2h_df["player2_id"] == str(returner_id))
            & (h2h_df["date"] < date_ts)
        ]
        if not h2h.empty:
            latest = h2h.sort_values("date").iloc[-1]
            h2h_win_rate = float(latest.get("srv_win_rate", 0.5))
            h2h_matches = int(latest.get("n_matches", 0))

    return {
        "elo_server": elo_srv,
        "elo_returner": elo_rtn,
        "elo_diff": elo_srv - elo_rtn,
        "ranking_server": float(srv_stats.get("ranking", TOUR_AVERAGES["ranking"])),
        "ranking_returner": float(rtn_stats.get("ranking", TOUR_AVERAGES["ranking"])),
        "ranking_diff": (
            float(srv_stats.get("ranking", TOUR_AVERAGES["ranking"]))
            - float(rtn_stats.get("ranking", TOUR_AVERAGES["ranking"]))
        ),
        "srv_1st_in_pct_server": float(srv_stats.get("srv_1st_in_pct", TOUR_AVERAGES["srv_1st_in_pct"])),
        "srv_1st_won_pct_server": float(srv_stats.get("srv_1st_won_pct", TOUR_AVERAGES["srv_1st_won_pct"])),
        "srv_2nd_won_pct_server": float(srv_stats.get("srv_2nd_won_pct", TOUR_AVERAGES["srv_2nd_won_pct"])),
        "srv_total_won_pct_server": float(srv_stats.get("srv_total_won_pct", TOUR_AVERAGES["srv_total_won_pct"])),
        "srv_1st_in_pct_returner": float(rtn_stats.get("srv_1st_in_pct", TOUR_AVERAGES["srv_1st_in_pct"])),
        "srv_1st_won_pct_returner": float(rtn_stats.get("srv_1st_won_pct", TOUR_AVERAGES["srv_1st_won_pct"])),
        "srv_2nd_won_pct_returner": float(rtn_stats.get("srv_2nd_won_pct", TOUR_AVERAGES["srv_2nd_won_pct"])),
        "srv_total_won_pct_returner": float(rtn_stats.get("srv_total_won_pct", TOUR_AVERAGES["srv_total_won_pct"])),
        "rtn_won_pct_server": float(srv_stats.get("rtn_won_pct", TOUR_AVERAGES["rtn_won_pct"])),
        "rtn_won_pct_returner": float(rtn_stats.get("rtn_won_pct", TOUR_AVERAGES["rtn_won_pct"])),
        "h2h_srv_win_rate": h2h_win_rate,
        "h2h_matches": float(h2h_matches),
        "age_server": float(srv_stats.get("age", TOUR_AVERAGES["age"])),
        "age_returner": float(rtn_stats.get("age", TOUR_AVERAGES["age"])),
        "age_diff": (
            float(srv_stats.get("age", TOUR_AVERAGES["age"]))
            - float(rtn_stats.get("age", TOUR_AVERAGES["age"]))
        ),
        "surface_hard": surface_hard,
        "surface_clay": surface_clay,
        "surface_grass": surface_grass,
        "tourney_level_slam": level_slam,
        "tourney_level_masters": level_masters,
    }


def build_inplay_features(state: MatchState) -> dict:
    """Compute dynamic in-play features from current MatchState."""
    # Serve win percentages this match
    if state.server == 0:
        srv_played = state.serve_pts_played_p1
        srv_won = state.serve_pts_won_p1
        rtn_played = state.serve_pts_played_p2  # returner (p2) has served these
        rtn_won = state.serve_pts_played_p2 - state.serve_pts_won_p2  # p1 won these returns
    else:
        srv_played = state.serve_pts_played_p2
        srv_won = state.serve_pts_won_p2
        rtn_played = state.serve_pts_played_p1
        rtn_won = state.serve_pts_played_p1 - state.serve_pts_won_p1

    match_srv_pct = srv_won / srv_played if srv_played > 0 else TOUR_AVERAGES["srv_total_won_pct"]
    match_rtn_pct = rtn_won / rtn_played if rtn_played > 0 else TOUR_AVERAGES["rtn_won_pct"]

    # Momentum: recent N points (server perspective)
    recent = state.recent_points
    momentum5 = float(sum(recent[-5:])) if len(recent) >= 1 else 2.5
    momentum10 = float(sum(recent[-10:])) if len(recent) >= 1 else 5.0

    # Set/game context
    if state.server == 0:
        sets_srv = state.sets_p1
        sets_rtn = state.sets_p2
        games_srv = state.games_p1
        games_rtn = state.games_p2
    else:
        sets_srv = state.sets_p2
        sets_rtn = state.sets_p1
        games_srv = state.games_p2
        games_rtn = state.games_p1

    return {
        "match_srv_pts_won_pct": match_srv_pct,
        "match_rtn_pts_won_pct": match_rtn_pct,
        "momentum_last5": momentum5,
        "momentum_last10": momentum10,
        "sets_server": float(sets_srv),
        "sets_returner": float(sets_rtn),
        "games_server": float(games_srv),
        "games_returner": float(games_rtn),
        "points_played_total": float(state.total_points_played()),
        "is_tiebreak": float(state.is_tiebreak),
        "is_final_set": float(state.is_final_set()),
    }


def combine_features(pre_match: dict, inplay: dict) -> np.ndarray:
    """Merge pre-match and in-play feature dicts into an ordered numpy array.

    The order follows FEATURE_COLUMNS exactly.
    """
    merged = {**pre_match, **inplay}
    return np.array([merged.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32)


def features_to_dataframe(feature_vectors: list[np.ndarray]) -> pd.DataFrame:
    """Convert a list of feature vectors to a DataFrame with named columns."""
    return pd.DataFrame(feature_vectors, columns=FEATURE_COLUMNS)
