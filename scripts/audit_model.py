#!/usr/bin/env python3
"""Audit the trained model for overfitting and data leakage.

Reports:
  - Train/val/test gap analysis (point-level and match-level)
  - Feature importance ranking with leakage flags
  - Permutation importance for in-play features on match-level prediction
  - Calibration curve diagnostics

Usage:
    python scripts/audit_model.py [--artifacts-dir PATH]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.features import FEATURE_COLUMNS
from tennis.model import PointWinModel
from tennis.calibration import MatchWinCalibrator
from tennis.markov import match_win_prob
from tennis.types import MatchFormat, MatchState

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"

# Features that are computed from in-match point outcomes.
# If these dominate match-level importance, it signals leakage.
INPLAY_FEATURES = {
    "match_srv_pts_won_pct",
    "match_rtn_pts_won_pct",
    "momentum_last5",
    "momentum_last10",
    "sets_server",
    "sets_returner",
    "games_server",
    "games_returner",
    "points_played_total",
    "is_tiebreak",
    "is_final_set",
}


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def point_level_metrics(model: PointWinModel, df: pd.DataFrame, label: str) -> dict:
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["server_won"].values.astype(np.float32)
    p = model.predict_proba(X)
    ll = log_loss(y, p)
    bs = brier_score_loss(y, p)
    print(f"  {label:12s}  log-loss={ll:.4f}  brier={bs:.4f}  (n={len(y):,})")
    return {f"{label}_point_logloss": ll, f"{label}_point_brier": bs}


def match_level_metrics(
    model: PointWinModel,
    calibrator: MatchWinCalibrator,
    df: pd.DataFrame,
    label: str,
) -> dict:
    if "match_id" not in df.columns:
        print(f"  {label}: no match_id column, skipping match-level")
        return {}

    match_format = MatchFormat.best_of_3_tiebreak()
    probs = []
    outcomes = []

    for match_id, group in df.groupby("match_id"):
        group = group.sort_values("point_idx")
        if len(group) < 5:
            continue

        X_first = group[FEATURE_COLUMNS].values[:1].astype(np.float32)
        p_serve = float(model.predict_proba(X_first)[0])

        state = MatchState(
            sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
            points_p1=0, points_p2=0, server=0,
            match_format=match_format,
        )
        raw_prob = match_win_prob(state, p_serve)
        probs.append(raw_prob)

        if "match_winner" in group.columns:
            mw = group["match_winner"].iloc[0]
            if mw == 0:
                outcomes.append(-1)  # unknown
            else:
                outcomes.append(int(mw == 1))
        else:
            outcomes.append(int(group["server_won"].mean() > 0.5))

    probs = np.array(probs)
    outcomes = np.array(outcomes)

    # Filter unknown outcomes
    valid = outcomes >= 0
    probs = probs[valid]
    outcomes = outcomes[valid]

    if len(probs) == 0:
        return {}

    cal_probs = calibrator.calibrate_batch(probs)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    cal_probs = np.clip(cal_probs, 1e-6, 1 - 1e-6)

    ll_raw = log_loss(outcomes, probs)
    ll_cal = log_loss(outcomes, cal_probs)
    bs_raw = brier_score_loss(outcomes, probs)
    bs_cal = brier_score_loss(outcomes, cal_probs)
    ece = MatchWinCalibrator().expected_calibration_error(cal_probs, outcomes)

    print(f"  {label:12s}  match log-loss raw={ll_raw:.4f} cal={ll_cal:.4f}  brier raw={bs_raw:.4f} cal={bs_cal:.4f}  ECE={ece:.4f}  (n={len(probs)})")
    return {
        f"{label}_match_logloss_raw": ll_raw,
        f"{label}_match_logloss_cal": ll_cal,
        f"{label}_match_brier_raw": bs_raw,
        f"{label}_match_brier_cal": bs_cal,
        f"{label}_match_ece": ece,
        f"{label}_n_matches": int(len(probs)),
    }


def feature_importance_analysis(model: PointWinModel) -> dict:
    print("\n=== Feature Importance (GBDT gain) ===")
    imp = model.feature_importance()
    if not imp:
        print("  No feature importance available")
        return {}

    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in sorted_imp) or 1.0

    inplay_gain = 0
    prematch_gain = 0

    for i, (name, gain) in enumerate(sorted_imp):
        pct = gain / total_gain * 100
        is_inplay = name in INPLAY_FEATURES
        marker = " [IN-PLAY]" if is_inplay else ""
        if i < 15:
            print(f"  {i+1:2d}. {name:40s} {gain:10.1f} ({pct:5.1f}%){marker}")

        if is_inplay:
            inplay_gain += gain
        else:
            prematch_gain += gain

    inplay_pct = inplay_gain / total_gain * 100
    prematch_pct = prematch_gain / total_gain * 100
    print(f"\n  In-play features: {inplay_pct:.1f}% of total gain")
    print(f"  Pre-match features: {prematch_pct:.1f}% of total gain")

    if inplay_pct > 60:
        print("  WARNING: In-play features dominate. This is expected for point-level")
        print("           prediction but may indicate leakage if match-level calibration is suspiciously good.")

    return {"inplay_gain_pct": inplay_pct, "prematch_gain_pct": prematch_pct}


def permutation_importance_test(
    model: PointWinModel,
    calibrator: MatchWinCalibrator,
    df: pd.DataFrame,
) -> dict:
    """Shuffle in-play features and measure impact on match-level metrics."""
    print("\n=== Permutation Importance (match-level impact) ===")

    if "match_id" not in df.columns:
        print("  Skipping — no match_id")
        return {}

    # Baseline match-level metrics
    baseline = match_level_metrics(model, calibrator, df, "baseline")
    if not baseline:
        return {}

    baseline_ll = baseline.get("baseline_match_logloss_cal", 999)

    features_to_test = ["match_srv_pts_won_pct", "momentum_last5", "momentum_last10"]
    results = {"baseline_match_logloss_cal": baseline_ll}

    for feat in features_to_test:
        if feat not in df.columns:
            continue
        df_shuffled = df.copy()
        rng = np.random.default_rng(42)
        df_shuffled[feat] = rng.permutation(df_shuffled[feat].values)
        shuffled = match_level_metrics(model, calibrator, df_shuffled, f"shuf_{feat[:15]}")
        shuffled_ll = shuffled.get(f"shuf_{feat[:15]}_match_logloss_cal", 999)
        delta = shuffled_ll - baseline_ll
        print(f"  Shuffle {feat}: delta log-loss = {delta:+.4f}")
        results[f"permute_{feat}_delta_ll"] = delta

        if delta > 0.05:
            print(f"    LEAKAGE SIGNAL: Shuffling {feat} significantly degrades match calibration")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit model for overfitting and leakage")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    print("=== Loading model artifacts ===")
    model = PointWinModel.load(os.path.join(args.artifacts_dir, "gbdt_model.pkl"))
    calibrator = MatchWinCalibrator.load(os.path.join(args.artifacts_dir, "calibrator.pkl"))

    all_metrics = {}

    # Load datasets
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(args.processed_dir, f"features_{name}.parquet")
        if os.path.exists(path):
            splits[name] = load_split(path)
            print(f"  {name}: {len(splits[name]):,} points")
        else:
            print(f"  {name}: NOT FOUND")

    # Point-level gap analysis
    print("\n=== Point-Level Gap Analysis ===")
    for name, df in splits.items():
        metrics = point_level_metrics(model, df, name)
        all_metrics.update(metrics)

    if "train" in splits and "val" in splits:
        gap = all_metrics.get("val_point_logloss", 0) - all_metrics.get("train_point_logloss", 0)
        print(f"\n  Train-Val gap: {gap:+.4f}")
        if gap > 0.05:
            print("  WARNING: Significant train-val gap suggests overfitting")

    # Match-level gap analysis
    print("\n=== Match-Level Gap Analysis ===")
    for name, df in splits.items():
        metrics = match_level_metrics(model, calibrator, df, name)
        all_metrics.update(metrics)

    # Feature importance
    imp_metrics = feature_importance_analysis(model)
    all_metrics.update(imp_metrics)

    # Permutation importance (on test set if available, else val)
    perm_df = splits.get("test", splits.get("val"))
    if perm_df is not None:
        perm_metrics = permutation_importance_test(model, calibrator, perm_df)
        all_metrics.update(perm_metrics)

    # Save report
    report_path = os.path.join(args.artifacts_dir, "audit_report.json")
    with open(report_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n  Audit report saved: {report_path}")


if __name__ == "__main__":
    main()
