#!/usr/bin/env python3
"""Tune BayesianUpdater parameters for optimal point-to-point trading.

Grid searches over virtual_sample_size and decay_rate, evaluating
match-level log-loss at multiple checkpoints during each match.

Also simulates point-level edge tracking to assess the model's
suitability for high-frequency trading.

Usage:
    python scripts/tune_updater.py [--artifacts-dir PATH] [--val-data PATH]
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.features import FEATURE_COLUMNS, build_pre_match_features, combine_features, TOUR_AVERAGES
from tennis.model import PointWinModel
from tennis.calibration import MatchWinCalibrator
from tennis.markov import match_win_prob
from tennis.updater import BayesianUpdater
from tennis.types import MatchFormat, MatchState

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"

CHECKPOINTS = [10, 20, 40, 60, 80, 100]  # evaluate after this many points
VSS_VALUES = [5, 10, 15, 20, 30, 50]
DECAY_VALUES = [0.99, 0.995, 0.999, 1.0]


def simulate_match(
    model: PointWinModel,
    calibrator: MatchWinCalibrator,
    match_df: pd.DataFrame,
    match_format: MatchFormat,
    vss: int,
    decay: float,
) -> dict:
    """Simulate the engine pipeline on a single match.

    Returns dict mapping checkpoint → match_win_prob at that point count.
    """
    match_df = match_df.sort_values("point_idx")
    if len(match_df) < 5:
        return {}

    # Get initial p_serve from GBDT (first point features)
    X_first = match_df[FEATURE_COLUMNS].values[:1].astype(np.float32)
    p_prior = float(np.clip(model.predict_proba(X_first)[0], 0.01, 0.99))

    updater = BayesianUpdater(p_serve_prior=p_prior, virtual_sample_size=vss, decay_rate=decay)

    state = MatchState(
        sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
        points_p1=0, points_p2=0, server=0,
        match_format=match_format,
    )

    results = {}
    for pt_idx, (_, row) in enumerate(match_df.iterrows()):
        server_won = bool(row["server_won"])
        updater.update(server_won)

        n_obs = pt_idx + 1
        if n_obs in CHECKPOINTS:
            p_serve = float(np.clip(updater.posterior_mean, 0.01, 0.99))
            raw_prob = match_win_prob(state, p_serve)
            cal_prob = float(np.clip(calibrator.calibrate(raw_prob), 1e-6, 1 - 1e-6))
            results[n_obs] = cal_prob

    return results


def evaluate_edge_profile(
    model: PointWinModel,
    calibrator: MatchWinCalibrator,
    match_df: pd.DataFrame,
    match_format: MatchFormat,
    vss: int,
    decay: float,
    true_outcome: int,
) -> list[float]:
    """Simulate the engine and compute edge (model_prob - fair_price) at each point.

    For edge analysis: assumes fair market price = true probability.
    Returns list of edge values (positive = model correctly sees value).
    """
    match_df = match_df.sort_values("point_idx")
    if len(match_df) < 10:
        return []

    X_first = match_df[FEATURE_COLUMNS].values[:1].astype(np.float32)
    p_prior = float(np.clip(model.predict_proba(X_first)[0], 0.01, 0.99))

    updater = BayesianUpdater(p_serve_prior=p_prior, virtual_sample_size=vss, decay_rate=decay)

    state = MatchState(
        sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
        points_p1=0, points_p2=0, server=0,
        match_format=match_format,
    )

    edges = []
    for _, row in match_df.iterrows():
        server_won = bool(row["server_won"])
        updater.update(server_won)

        p_serve = float(np.clip(updater.posterior_mean, 0.01, 0.99))
        raw_prob = match_win_prob(state, p_serve)
        cal_prob = float(np.clip(calibrator.calibrate(raw_prob), 1e-6, 1 - 1e-6))

        # Edge: how much the model favors the correct side
        if true_outcome == 1:
            edge = cal_prob - 0.5  # positive if model correctly favors p1
        else:
            edge = (1 - cal_prob) - 0.5

        edges.append(edge)

    return edges


def main():
    parser = argparse.ArgumentParser(description="Tune BayesianUpdater parameters")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--val-data", default=str(PROCESSED_DIR / "features_val.parquet"))
    parser.add_argument("--max-matches", type=int, default=200, help="Limit matches for speed")
    args = parser.parse_args()

    print("=== Loading artifacts ===")
    model = PointWinModel.load(os.path.join(args.artifacts_dir, "gbdt_model.pkl"))
    calibrator = MatchWinCalibrator.load(os.path.join(args.artifacts_dir, "calibrator.pkl"))

    print("=== Loading validation data ===")
    val_df = pd.read_parquet(args.val_data)
    val_df["match_date"] = pd.to_datetime(val_df["match_date"])
    print(f"  {len(val_df):,} points")

    if "match_id" not in val_df.columns:
        print("ERROR: match_id required")
        sys.exit(1)

    # Group by match, limit for speed
    match_groups = {mid: g for mid, g in val_df.groupby("match_id")}
    match_ids = list(match_groups.keys())[:args.max_matches]
    print(f"  Evaluating {len(match_ids)} matches")

    # Determine outcomes
    match_outcomes = {}
    for mid in match_ids:
        g = match_groups[mid]
        if "match_winner" in g.columns:
            mw = g["match_winner"].iloc[0]
            if mw == 0:
                continue
            match_outcomes[mid] = int(mw == 1)
        else:
            match_outcomes[mid] = int(g["server_won"].mean() > 0.5)

    match_ids = [mid for mid in match_ids if mid in match_outcomes]
    match_format = MatchFormat.best_of_3_tiebreak()

    # Grid search
    print(f"\n=== Grid Search: {len(VSS_VALUES)} x {len(DECAY_VALUES)} = {len(VSS_VALUES)*len(DECAY_VALUES)} combinations ===")
    results = []

    for vss, decay in itertools.product(VSS_VALUES, DECAY_VALUES):
        checkpoint_probs = {cp: [] for cp in CHECKPOINTS}
        checkpoint_outcomes = {cp: [] for cp in CHECKPOINTS}

        for mid in match_ids:
            g = match_groups[mid]
            cp_results = simulate_match(model, calibrator, g, match_format, vss, decay)
            outcome = match_outcomes[mid]
            for cp, prob in cp_results.items():
                checkpoint_probs[cp].append(prob)
                checkpoint_outcomes[cp].append(outcome)

        row = {"vss": vss, "decay": decay}
        total_ll = 0.0
        n_checkpoints = 0
        for cp in CHECKPOINTS:
            if len(checkpoint_probs[cp]) >= 10:
                ll = log_loss(checkpoint_outcomes[cp], checkpoint_probs[cp])
                row[f"logloss_at_{cp}"] = ll
                total_ll += ll
                n_checkpoints += 1

        row["avg_logloss"] = total_ll / max(n_checkpoints, 1)
        results.append(row)
        print(f"  vss={vss:3d}  decay={decay:.3f}  avg_logloss={row['avg_logloss']:.4f}")

    # Find best
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["avg_logloss"].idxmin()]
    print(f"\n=== Best Parameters ===")
    print(f"  virtual_sample_size = {int(best['vss'])}")
    print(f"  decay_rate = {best['decay']}")
    print(f"  avg log-loss = {best['avg_logloss']:.4f}")

    # Edge profile analysis with best parameters
    print(f"\n=== Edge Profile Analysis (best params) ===")
    best_vss = int(best["vss"])
    best_decay = float(best["decay"])

    all_edges = []
    for mid in match_ids[:100]:
        g = match_groups[mid]
        outcome = match_outcomes[mid]
        edges = evaluate_edge_profile(model, calibrator, g, match_format, best_vss, best_decay, outcome)
        all_edges.extend(edges)

    if all_edges:
        edges_arr = np.array(all_edges)
        print(f"  Total edge observations: {len(edges_arr):,}")
        print(f"  Mean edge: {edges_arr.mean():.4f}")
        print(f"  Std edge: {edges_arr.std():.4f}")
        print(f"  Edge > 0 (correct direction): {(edges_arr > 0).mean()*100:.1f}%")
        print(f"  Sharpe-like (mean/std): {edges_arr.mean() / max(edges_arr.std(), 1e-6):.3f}")

    # Save
    csv_path = os.path.join(args.artifacts_dir, "updater_tuning.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
