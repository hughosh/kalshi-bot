#!/usr/bin/env python3
"""Build training datasets from raw JeffSackmann data.

Steps:
1. Load ATP match-level CSVs (tennis_atp/atp_matches_YYYY.csv)
2. Build ELO tracker from match history (temporal order)
3. Compute rolling player stats (serve%, return%, ranking)
4. Load point-by-point data from two sources:
   a) tennis_slam_pointbypoint — rich point-level data (Grand Slams 2011+)
   b) tennis_pointbypoint — PBP string data (ATP main tour)
5. For each point in each match, emit a row with:
      - Pre-match features (ELO, serve%, ranking, surface, etc.)
      - In-play features at that point in time
      - Target: server_won (0/1)
      - match_date (for temporal split)
6. Save as train/val/test parquet files

Usage:
    python scripts/build_features.py [--atp-dir DATA_DIR] [--out-dir OUT_DIR]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.elo import EloTracker
from tennis.features import (
    build_player_stats,
    build_pre_match_features,
    combine_features,
    FEATURE_COLUMNS,
    TOUR_AVERAGES,
)

TRAIN_CUTOFF = "2022-01-01"
VAL_CUTOFF = "2023-01-01"

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

SLAM_SURFACE = {
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard",
}


def load_atp_matches(atp_dir: Path, start_year: int = 2003) -> pd.DataFrame:
    """Load all atp_matches_YYYY.csv files and concatenate."""
    pattern = str(atp_dir / "atp_matches_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ATP match files found in {atp_dir}")

    dfs = []
    for f in files:
        stem = os.path.basename(f).replace("atp_matches_", "").replace(".csv", "")
        try:
            year = int(stem)
        except ValueError:
            continue  # skip non-year files like atp_matches_amateur.csv
        if year < start_year:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")

    print(f"  Loaded {len(dfs)} ATP match files ({sum(len(d) for d in dfs):,} matches)")
    return pd.concat(dfs, ignore_index=True)


def build_name_to_id(matches_df: pd.DataFrame) -> dict:
    """Build player name → ATP numeric ID mapping from match data."""
    mapping = {}
    for col_id, col_name in [("winner_id", "winner_name"), ("loser_id", "loser_name")]:
        for _, row in matches_df[[col_id, col_name]].drop_duplicates().iterrows():
            name = str(row[col_name]).strip()
            pid = str(int(row[col_id])) if pd.notna(row[col_id]) else ""
            if name and pid and name != "nan":
                mapping[name] = pid
    print(f"  Name→ID mapping: {len(mapping)} players")
    return mapping


def parse_pbp_sequence(pbp: str) -> list:
    """Parse a PBP sequence string to a list of 0/1 (1=server won point).

    The JeffSackmann format uses:
        S = server wins point
        R = returner wins point
    Other characters (set/game separators) are skipped.
    """
    result = []
    for c in str(pbp):
        if c == "S":
            result.append(1)
        elif c == "R":
            result.append(0)
    return result


def _assign_server_to_points(points: list[int]) -> list[tuple[int, int]]:
    """Assign correct server identity to each point in a PBP sequence.

    In the S/R format, S=server won, R=returner won. The server identity
    alternates each game. Player 1 serves first.

    Args:
        points: list of 0/1 (1 = current server won point)

    Returns:
        list of (server, server_won) where server is 1 or 2.
    """
    result: list[tuple[int, int]] = []
    current_server = 1  # player 1 serves first
    srv_pts = 0  # server's points in current game
    rtn_pts = 0  # returner's points in current game
    games_p1 = 0  # games won by p1 in current set
    games_p2 = 0  # games won by p2 in current set
    in_tiebreak = False
    tb_points_played = 0

    for server_won in points:
        result.append((current_server, server_won))

        if in_tiebreak:
            # Tiebreak scoring
            if server_won:
                srv_pts += 1
            else:
                rtn_pts += 1
            tb_points_played += 1

            # Check tiebreak end: first to 7 with 2+ lead
            if (srv_pts >= 7 or rtn_pts >= 7) and abs(srv_pts - rtn_pts) >= 2:
                # Tiebreak over — determine who won
                if srv_pts > rtn_pts:
                    # Server won tiebreak = won the set
                    if current_server == 1:
                        games_p1 = 0
                        games_p2 = 0
                    else:
                        games_p1 = 0
                        games_p2 = 0
                else:
                    games_p1 = 0
                    games_p2 = 0

                # After tiebreak, the OTHER player serves first in the new set
                # (the player who received first in the tiebreak)
                current_server = 2 if current_server == 1 else 1
                srv_pts = 0
                rtn_pts = 0
                in_tiebreak = False
                tb_points_played = 0
            else:
                # Tiebreak server rotation: first point by initial server,
                # then alternate every 2 points
                # Server changes after points 1, 3, 5, 7, ...
                if tb_points_played % 2 == 1:
                    current_server = 2 if current_server == 1 else 1
                    srv_pts, rtn_pts = rtn_pts, srv_pts
        else:
            # Standard game scoring
            if server_won:
                srv_pts += 1
            else:
                rtn_pts += 1

            # Check if game is over
            game_over = False
            if srv_pts >= 4 and srv_pts - rtn_pts >= 2:
                game_over = True
                # Server won the game
                if current_server == 1:
                    games_p1 += 1
                else:
                    games_p2 += 1
            elif rtn_pts >= 4 and rtn_pts - srv_pts >= 2:
                game_over = True
                # Returner won the game
                if current_server == 1:
                    games_p2 += 1
                else:
                    games_p1 += 1

            if game_over:
                srv_pts = 0
                rtn_pts = 0

                # Check for tiebreak (6-6)
                if games_p1 == 6 and games_p2 == 6:
                    in_tiebreak = True
                    tb_points_played = 0
                    # Server alternates (the player who would normally serve)
                    current_server = 2 if current_server == 1 else 1
                elif games_p1 >= 6 and games_p1 - games_p2 >= 2:
                    # P1 won the set — reset games, alternate serve
                    games_p1 = 0
                    games_p2 = 0
                    current_server = 2 if current_server == 1 else 1
                elif games_p2 >= 6 and games_p2 - games_p1 >= 2:
                    # P2 won the set — reset games, alternate serve
                    games_p1 = 0
                    games_p2 = 0
                    current_server = 2 if current_server == 1 else 1
                else:
                    # Normal game change — alternate serve
                    current_server = 2 if current_server == 1 else 1

    return result


# ---------------------------------------------------------------------------
# Slam PBP loading (tennis_slam_pointbypoint)
# ---------------------------------------------------------------------------

def load_slam_pbp(slam_dir: Path, name_to_id: dict) -> list:
    """Load Grand Slam point-by-point data and return list of match dicts.

    Each match dict contains:
        match_id, player1_id, player2_id, surface, tourney_level,
        match_date, winner (1 or 2), points: list of (server: 1/2, winner: 1/2)
    """
    # Exclude doubles and mixed files (they have -doubles.csv or -mixed.csv suffixes)
    matches_files = sorted(
        f for f in glob.glob(str(slam_dir / "*-matches.csv"))
        if not f.endswith(("-doubles.csv", "-mixed.csv"))
    )
    points_files = sorted(
        f for f in glob.glob(str(slam_dir / "*-points.csv"))
        if not f.endswith(("-doubles.csv", "-mixed.csv"))
    )

    if not matches_files:
        print("  No slam match files found")
        return []

    # Load all matches
    match_dfs = []
    for f in matches_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Extract slam name from filename (e.g., "2011-ausopen-matches.csv")
            fname = os.path.basename(f)
            parts = fname.replace("-matches.csv", "").split("-", 1)
            if len(parts) == 2:
                df["_year"] = parts[0]
                df["_slam"] = parts[1]
            match_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: {f}: {e}")

    matches = pd.concat(match_dfs, ignore_index=True) if match_dfs else pd.DataFrame()

    # Load all points
    point_dfs = []
    for f in points_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            point_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: {f}: {e}")

    points = pd.concat(point_dfs, ignore_index=True) if point_dfs else pd.DataFrame()

    if matches.empty or points.empty:
        print("  No slam data loaded")
        return []

    # Filter to men's singles only (where event_name is available)
    # Non-suffixed files (without -doubles/-mixed) are already men's singles
    if "event_name" in matches.columns:
        has_event = matches["event_name"].notna() & (matches["event_name"] != "")
        matches = matches[~has_event | matches["event_name"].str.contains("Men", na=False, case=False)]

    print(f"  Slam matches: {len(matches):,}, slam points: {len(points):,}")

    # Group points by match_id
    points_grouped = points.groupby("match_id")

    result = []
    skipped = 0
    for _, mrow in matches.iterrows():
        mid = mrow.get("match_id", "")
        if mid not in points_grouped.groups:
            skipped += 1
            continue

        match_points = points_grouped.get_group(mid).sort_values("PointNumber")
        if len(match_points) < 10:
            skipped += 1
            continue

        # Map player names to ATP IDs
        p1_name = str(mrow.get("player1", "")).strip()
        p2_name = str(mrow.get("player2", "")).strip()
        p1_id = name_to_id.get(p1_name, p1_name)
        p2_id = name_to_id.get(p2_name, p2_name)

        slam = str(mrow.get("_slam", mrow.get("slam", "")))
        surface = SLAM_SURFACE.get(slam, "Hard")
        year = str(mrow.get("_year", mrow.get("year", "")))
        winner_val = mrow.get("winner", 0)
        winner = int(winner_val) if pd.notna(winner_val) and winner_val != 0 else 0

        # Infer winner from SetWinner if not available in match metadata
        if winner == 0 and "SetWinner" in match_points.columns:
            set_winners = match_points.groupby("SetNo")["SetWinner"].last()
            p1_sets = int((set_winners == 1).sum())
            p2_sets = int((set_winners == 2).sum())
            if p1_sets > p2_sets:
                winner = 1
            elif p2_sets > p1_sets:
                winner = 2

        # Build match date from year + slam approximate month
        slam_months = {"ausopen": "01", "frenchopen": "06", "wimbledon": "07", "usopen": "09"}
        month = slam_months.get(slam, "06")
        match_date = f"{year}-{month}-15"

        # Extract point sequence: (PointServer, PointWinner)
        point_list = []
        for _, pt in match_points.iterrows():
            ps = pt.get("PointServer", 0)
            pw = pt.get("PointWinner", 0)
            if ps in (1, 2) and pw in (1, 2):
                server_won = 1 if ps == pw else 0
                point_list.append((int(ps), server_won))

        if len(point_list) < 10:
            skipped += 1
            continue

        result.append({
            "match_id": str(mid),
            "player1_id": p1_id,
            "player2_id": p2_id,
            "surface": surface,
            "tourney_level": "G",
            "match_date": match_date,
            "winner": winner,
            "points": point_list,  # list of (server: 1/2, server_won: 0/1)
        })

    print(f"  Slam matches loaded: {len(result):,} ({skipped} skipped)")
    return result


# ---------------------------------------------------------------------------
# Main tour PBP loading (tennis_pointbypoint)
# ---------------------------------------------------------------------------

def load_main_pbp(pbp_dir: Path, matches_df: pd.DataFrame, name_to_id: dict) -> list:
    """Load main-tour PBP data (S/R string format) and return list of match dicts.

    Joins with ATP match data on player names + approximate date to get
    surface, tourney_level, and player IDs.
    """
    patterns = [
        str(pbp_dir / "pbp_matches_atp_main_*.csv"),
        str(pbp_dir / "pbp_matches_atp_qual_*.csv"),
    ]
    dfs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                print(f"  Loaded {os.path.basename(f)}: {len(df):,} matches")
            except Exception as e:
                print(f"  WARNING: {f}: {e}")

    if not dfs:
        return []

    pbp_df = pd.concat(dfs, ignore_index=True)

    # Parse PBP date to a standard format
    # Format is "DD Mon YY" (e.g., "01 Jan 17")
    pbp_df["_parsed_date"] = pd.to_datetime(pbp_df["date"], format="mixed", dayfirst=True, errors="coerce")

    # Build match lookup from ATP matches: (winner_name, loser_name, approx_date) → match row
    # We'll use a name-pair → list of matches lookup for fuzzy date matching
    matches_df = matches_df.copy()
    matches_df["_date"] = pd.to_datetime(matches_df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    matches_df["_wname"] = matches_df["winner_name"].str.strip()
    matches_df["_lname"] = matches_df["loser_name"].str.strip()

    # Build lookup: frozenset(name1, name2) → list of match rows
    match_lookup = {}
    for idx, row in matches_df.iterrows():
        wn = str(row["_wname"])
        ln = str(row["_lname"])
        key = frozenset([wn, ln])
        if key not in match_lookup:
            match_lookup[key] = []
        match_lookup[key].append(row)

    result = []
    matched = 0
    unmatched = 0
    skipped = 0

    for _, pbp_row in pbp_df.iterrows():
        pbp_str = str(pbp_row.get("pbp", ""))
        if not pbp_str or pbp_str == "nan":
            skipped += 1
            continue

        points = parse_pbp_sequence(pbp_str)
        if len(points) < 10:
            skipped += 1
            continue

        s1 = str(pbp_row.get("server1", "")).strip()
        s2 = str(pbp_row.get("server2", "")).strip()
        pbp_date = pbp_row.get("_parsed_date")

        if pd.isna(pbp_date):
            skipped += 1
            continue

        # Try to find matching ATP match
        key = frozenset([s1, s2])
        candidates = match_lookup.get(key, [])

        best_match = None
        best_delta = pd.Timedelta(days=999)
        for c in candidates:
            if pd.notna(c["_date"]):
                delta = abs(pbp_date - c["_date"])
                if delta < best_delta:
                    best_delta = delta
                    best_match = c

        if best_match is not None and best_delta <= pd.Timedelta(days=7):
            matched += 1
            winner_id = str(int(best_match["winner_id"]))
            loser_id = str(int(best_match["loser_id"]))
            surface = str(best_match.get("surface", "Hard"))
            tourney_level = str(best_match.get("tourney_level", "A"))
            match_date_raw = str(best_match.get("tourney_date", ""))
            if len(match_date_raw) == 8:
                match_date = f"{match_date_raw[:4]}-{match_date_raw[4:6]}-{match_date_raw[6:8]}"
            else:
                match_date = str(pbp_date.date())

            # Determine who is server1 (= first server)
            winner_name = str(best_match["_wname"])
            if s1 == winner_name:
                p1_id, p2_id = winner_id, loser_id
            else:
                p1_id, p2_id = loser_id, winner_id

            # Convert S/R string to (server, server_won) format
            # Assign correct server identity based on game boundaries
            point_list = _assign_server_to_points(points)  # server=1 means "initial server"

            result.append({
                "match_id": str(pbp_row.get("pbp_id", "")),
                "player1_id": p1_id,
                "player2_id": p2_id,
                "surface": surface,
                "tourney_level": tourney_level,
                "match_date": match_date,
                "winner": 1 if s1 == winner_name else 2,
                "points": point_list,
            })
        else:
            unmatched += 1

    print(f"  Main PBP: {matched:,} matched, {unmatched:,} unmatched, {skipped:,} skipped")
    return result


# ---------------------------------------------------------------------------
# Feature row generation
# ---------------------------------------------------------------------------

def build_training_rows_from_matches(
    match_list: list,
    player_stats_df: pd.DataFrame,
    elo_tracker: EloTracker,
    h2h_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate one training row per point from processed match dicts."""
    rows = []
    total_points = 0
    skipped = 0

    for match in match_list:
        p1_id = match["player1_id"]
        p2_id = match["player2_id"]
        surface = match["surface"]
        tourney_level = match["tourney_level"]
        match_date = match["match_date"]
        point_list = match["points"]
        match_winner = match.get("winner", 0)  # 1=player1 won, 2=player2 won

        # Pre-match features (computed once per match, using initial server = p1)
        try:
            pre_match = build_pre_match_features(
                server_id=p1_id,
                returner_id=p2_id,
                surface=surface,
                tournament_level=tourney_level,
                match_date=match_date,
                player_stats_df=player_stats_df,
                elo_tracker=elo_tracker,
                h2h_df=h2h_df if h2h_df is not None and not h2h_df.empty else None,
            )
        except Exception:
            skipped += 1
            continue

        # Track running stats
        serve_pts_played_p1 = 0
        serve_pts_won_p1 = 0
        serve_pts_played_p2 = 0
        serve_pts_won_p2 = 0
        recent_points = []
        sets_p1 = 0
        sets_p2 = 0
        games_p1 = 0
        games_p2 = 0

        for pt_idx, (server, server_won) in enumerate(point_list):
            # Current server stats for in-play features
            if server == 1:
                sp_played = serve_pts_played_p1
                sp_won = serve_pts_won_p1
                rp_played = serve_pts_played_p2
                rp_won = serve_pts_won_p2
            else:
                sp_played = serve_pts_played_p2
                sp_won = serve_pts_won_p2
                rp_played = serve_pts_played_p1
                rp_won = serve_pts_won_p1

            inplay = {
                "match_srv_pts_won_pct": (
                    sp_won / sp_played if sp_played > 0
                    else TOUR_AVERAGES["srv_total_won_pct"]
                ),
                "match_rtn_pts_won_pct": (
                    (rp_played - rp_won) / rp_played if rp_played > 0
                    else TOUR_AVERAGES["rtn_won_pct"]
                ),
                "momentum_last5": float(sum(recent_points[-5:])) if recent_points else 2.5,
                "momentum_last10": float(sum(recent_points[-10:])) if recent_points else 5.0,
                "sets_server": float(sets_p1 if server == 1 else sets_p2),
                "sets_returner": float(sets_p2 if server == 1 else sets_p1),
                "games_server": float(games_p1 if server == 1 else games_p2),
                "games_returner": float(games_p2 if server == 1 else games_p1),
                "points_played_total": float(
                    serve_pts_played_p1 + serve_pts_played_p2
                ),
                "is_tiebreak": 0.0,
                "is_final_set": float(
                    (sets_p1 + sets_p2) >= 4  # rough heuristic for slams
                ),
            }

            feature_vec = combine_features(pre_match, inplay)
            row_data = dict(zip(FEATURE_COLUMNS, feature_vec))
            row_data["server_won"] = int(server_won)
            row_data["match_date"] = match_date
            row_data["match_id"] = match["match_id"]
            row_data["match_winner"] = match_winner
            row_data["point_idx"] = pt_idx
            rows.append(row_data)

            # Update running stats
            if server == 1:
                serve_pts_played_p1 += 1
                serve_pts_won_p1 += server_won
            else:
                serve_pts_played_p2 += 1
                serve_pts_won_p2 += server_won

            recent_points.append(server_won)
            if len(recent_points) > 20:
                recent_points = recent_points[-20:]

            total_points += 1

    print(f"  Generated {total_points:,} training points ({skipped} matches skipped)")
    return pd.DataFrame(rows)


def build_h2h(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head win rates for each player pair."""
    df = matches_df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    records = []
    pair_stats: dict = {}

    for _, row in df.iterrows():
        w = str(row["winner_id"])
        l = str(row["loser_id"])
        p1, p2 = min(w, l), max(w, l)
        key = (p1, p2)

        stats = pair_stats.get(key, {"n": 0, "p1_wins": 0})
        n = stats["n"]
        p1_wins = stats["p1_wins"]
        if n > 0:
            records.append({
                "player1_id": p1,
                "player2_id": p2,
                "date": row["tourney_date"],
                "n_matches": n,
                "srv_win_rate": p1_wins / n,
            })

        stats["n"] += 1
        stats["p1_wins"] += int(w == p1)
        pair_stats[key] = stats

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Build training features from raw data")
    parser.add_argument("--atp-dir", default=str(RAW_DIR / "tennis_atp"))
    parser.add_argument("--pbp-dir", default=str(RAW_DIR / "tennis_pointbypoint"))
    parser.add_argument("--slam-dir", default=str(RAW_DIR / "tennis_slam_pointbypoint"))
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--start-year", type=int, default=2003)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=== Step 1: Loading ATP match data ===")
    matches_df = load_atp_matches(Path(args.atp_dir), start_year=args.start_year)

    print("\n=== Step 2: Building ELO tracker ===")
    elo = EloTracker()
    elo.process_matches(matches_df)
    elo.save(os.path.join(args.artifacts_dir, "player_elo.parquet"))
    print("  ELO tracker saved.")

    print("\n=== Step 3: Building player stats ===")
    player_stats = build_player_stats(matches_df)
    player_stats.to_parquet(
        os.path.join(args.artifacts_dir, "player_stats.parquet"), index=False
    )
    print(f"  Player stats: {len(player_stats):,} rows")

    print("\n=== Step 4: Building H2H records ===")
    h2h = build_h2h(matches_df)
    h2h.to_parquet(os.path.join(args.artifacts_dir, "h2h.parquet"), index=False)
    print(f"  H2H records: {len(h2h):,} rows")

    print("\n=== Step 5: Building name→ID mapping ===")
    name_to_id = build_name_to_id(matches_df)

    print("\n=== Step 6: Loading slam PBP data ===")
    slam_matches = load_slam_pbp(Path(args.slam_dir), name_to_id)

    print("\n=== Step 7: Loading main tour PBP data ===")
    main_matches = load_main_pbp(Path(args.pbp_dir), matches_df, name_to_id)

    all_matches = slam_matches + main_matches
    print(f"\n  Total matches with PBP: {len(all_matches):,}")

    if not all_matches:
        print("  ERROR: No matches with point-by-point data found.")
        sys.exit(1)

    print("\n=== Step 8: Building feature dataset ===")
    features_df = build_training_rows_from_matches(
        all_matches, player_stats, elo, h2h
    )

    if features_df.empty:
        print("  ERROR: No training rows generated.")
        sys.exit(1)

    features_df["match_date"] = pd.to_datetime(features_df["match_date"])

    print("\n=== Step 9: Time-based split ===")
    train = features_df[features_df["match_date"] < TRAIN_CUTOFF]
    val = features_df[
        (features_df["match_date"] >= TRAIN_CUTOFF)
        & (features_df["match_date"] < VAL_CUTOFF)
    ]
    test = features_df[features_df["match_date"] >= VAL_CUTOFF]

    print(f"  Train: {len(train):,} points")
    if not train.empty:
        print(f"         ({train['match_date'].min()} – {train['match_date'].max()})")
    print(f"  Val:   {len(val):,} points")
    if not val.empty:
        print(f"         ({val['match_date'].min()} – {val['match_date'].max()})")
    print(f"  Test:  {len(test):,} points")
    if not test.empty:
        print(f"         ({test['match_date'].min()} – {test['match_date'].max()})")

    train.to_parquet(os.path.join(args.out_dir, "features_train.parquet"), index=False)
    val.to_parquet(os.path.join(args.out_dir, "features_val.parquet"), index=False)
    test.to_parquet(os.path.join(args.out_dir, "features_test.parquet"), index=False)
    print(f"\n  Saved to {args.out_dir}/")
    print("\nDone. Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
