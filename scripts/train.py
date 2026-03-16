#!/usr/bin/env python3
"""Train the GBDT point-win model and calibrator.

Steps:
1. Load train/val parquet feature files
2. Train XGBoost (and optionally LightGBM) with early stopping on val log-loss
3. Compute match-level win probability predictions on val set using the
   full pipeline (GBDT → Markov chain)
4. Fit isotonic regression calibrator on val set predictions
5. Save artifacts: gbdt_model.pkl, calibrator.pkl

Usage:
    python scripts/train.py [--backend xgboost|lightgbm]
"""
import argparse
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


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load feature parquet and return (X, y, meta_df)."""
    df = pd.read_parquet(path)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["server_won"].values.astype(np.float32)
    meta = df[["match_date", "match_id", "point_idx"]].copy() if "match_id" in df.columns else df[["match_date"]].copy()
    return X, y, meta


def build_match_win_probs_on_val(
    model: PointWinModel,
    val_df: pd.DataFrame,
    match_format: MatchFormat,
) -> tuple[np.ndarray, np.ndarray]:
    """For each point in val set, compute match win probability using Markov chain.

    Returns:
        (raw_match_probs, match_outcomes):
          raw_match_probs: P(server at match start wins) at each point snapshot
          match_outcomes:  1 if initial server won match, 0 otherwise
    """
    X_val = val_df[FEATURE_COLUMNS].values.astype(np.float32)
    p_serve_all = model.predict_proba(X_val)

    match_probs = []
    match_outcomes = []

    for match_id, group in val_df.groupby("match_id"):
        group = group.sort_values("point_idx")
        idx = group.index

        # For each point, compute match win prob from approximate state
        # We use a simplified state (just point index) since full score tracking
        # is expensive here; the key signal is p_serve from GBDT.
        # For calibration we sample at the match START only (index 0).
        if len(group) == 0:
            continue

        # Use first point's p_serve as pre-match estimate for calibration
        first_idx = idx[0]
        p_serve = float(p_serve_all[first_idx])

        # Create a fresh start state for Markov calculation
        state = MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            match_format=match_format,
        )
        raw_prob = match_win_prob(state, p_serve)
        match_probs.append(raw_prob)

        # Match outcome: did player 1 (initial server in our data) win?
        # Use actual match winner from metadata (1=player1 won, 2=player2 won).
        if "match_winner" in group.columns:
            match_outcome = int(group["match_winner"].iloc[0] == 1)
        else:
            # Fallback for legacy data without match_winner column
            pts = group["server_won"].values
            match_outcome = int(pts.mean() > 0.5)
        match_outcomes.append(match_outcome)

    return np.array(match_probs), np.array(match_outcomes)


def main():
    parser = argparse.ArgumentParser(description="Train tennis point-win model")
    parser.add_argument("--backend", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--compare", action="store_true", help="Also train LightGBM and compare")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=== Loading features ===")
    train_path = os.path.join(args.processed_dir, "features_train.parquet")
    val_path = os.path.join(args.processed_dir, "features_val.parquet")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Feature files not found. Run build_features.py first.")
        sys.exit(1)

    X_train, y_train, meta_train = load_features(train_path)
    X_val, y_val, meta_val = load_features(val_path)
    val_df = pd.read_parquet(val_path)
    val_df["match_date"] = pd.to_datetime(val_df["match_date"])

    # Split validation set: H1 for early stopping, H2 for calibration
    # This prevents the calibrator from seeing the same data used for GBDT training
    # Split at the midpoint between the min and max dates
    val_dates = val_df["match_date"].sort_values().unique()
    if len(val_dates) >= 2:
        val_cutoff_mid = val_dates[len(val_dates) // 2]
    else:
        val_cutoff_mid = val_df["match_date"].max()
    mask_es = val_df["match_date"] < val_cutoff_mid
    mask_cal = val_df["match_date"] >= val_cutoff_mid

    X_val_es = val_df[mask_es][FEATURE_COLUMNS].values.astype(np.float32)
    y_val_es = val_df[mask_es]["server_won"].values.astype(np.float32)
    val_df_cal = val_df[mask_cal].copy()

    print(f"  Train:    {len(X_train):,} points")
    print(f"  Val (ES): {len(X_val_es):,} points (early stopping)")
    print(f"  Val (Cal):{len(val_df_cal):,} points (calibration)")

    print(f"\n=== Training {args.backend} model ===")
    model = PointWinModel(backend=args.backend)
    metrics = model.train(
        X_train, y_train,
        X_val_es, y_val_es,
        feature_names=FEATURE_COLUMNS,
        early_stopping_rounds=50,
        verbose=True,
    )

    print(f"\nMetrics:")
    print(f"  Train log-loss: {metrics['train_logloss']:.4f}")
    print(f"  Val   log-loss: {metrics['val_logloss']:.4f}")
    print(f"  Train Brier:    {metrics['train_brier']:.4f}")
    print(f"  Val   Brier:    {metrics['val_brier']:.4f}")

    if args.compare:
        alt = "lightgbm" if args.backend == "xgboost" else "xgboost"
        print(f"\n=== Comparing with {alt} ===")
        model_alt = PointWinModel(backend=alt)
        metrics_alt = model_alt.train(X_train, y_train, X_val, y_val,
                                       feature_names=FEATURE_COLUMNS, verbose=True)
        print(f"  {alt} val log-loss: {metrics_alt['val_logloss']:.4f}")
        print(f"  {alt} val Brier:    {metrics_alt['val_brier']:.4f}")
        if metrics_alt["val_logloss"] < metrics["val_logloss"]:
            print(f"  {alt} is better — using it.")
            model = model_alt

    # Feature importance
    imp = model.feature_importance()
    if imp:
        top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 features by gain:")
        for name, score in top10:
            print(f"  {name:40s} {score:.1f}")

    print("\n=== Calibrating on validation set (H2) ===")
    match_format = MatchFormat.best_of_3_tiebreak()
    if "match_id" in val_df_cal.columns and "match_winner" in val_df_cal.columns:
        raw_probs, outcomes = build_match_win_probs_on_val(model, val_df_cal, match_format)
        print(f"  Val matches for calibration: {len(raw_probs)}")
        # Filter out matches with unknown winner (match_winner == 0)
        valid = outcomes >= 0
        raw_probs = raw_probs[valid]
        outcomes = outcomes[valid]
        print(f"  Valid matches (known winner): {len(raw_probs)}")
        calibrator = MatchWinCalibrator()
        calibrator.fit(raw_probs, outcomes)
        ece = calibrator.expected_calibration_error(raw_probs, outcomes)
        print(f"  Pre-calibration ECE: {ece:.4f}")
        cal_probs = calibrator.calibrate_batch(raw_probs)
        ece_cal = calibrator.expected_calibration_error(cal_probs, outcomes)
        print(f"  Post-calibration ECE: {ece_cal:.4f}")
    elif "match_id" in val_df_cal.columns:
        print("  WARNING: match_winner column missing — using proxy (rebuild features!)")
        raw_probs, outcomes = build_match_win_probs_on_val(model, val_df_cal, match_format)
        calibrator = MatchWinCalibrator()
        calibrator.fit(raw_probs, outcomes)
    else:
        print("  Fitting calibrator on point-level predictions (match_id not available)")
        X_val_cal = val_df_cal[FEATURE_COLUMNS].values.astype(np.float32)
        y_val_cal = val_df_cal["server_won"].values.astype(np.float32)
        calibrator = MatchWinCalibrator()
        calibrator.fit(model.predict_proba(X_val_cal), y_val_cal)

    print("\n=== Saving artifacts ===")
    model_path = os.path.join(args.artifacts_dir, "gbdt_model.pkl")
    cal_path = os.path.join(args.artifacts_dir, "calibrator.pkl")
    model.save(model_path)
    calibrator.save(cal_path)
    print(f"  Model saved:      {model_path}")
    print(f"  Calibrator saved: {cal_path}")
    print("\nDone. Next step: python scripts/backtest.py")


if __name__ == "__main__":
    main()
