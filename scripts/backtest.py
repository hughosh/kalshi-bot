#!/usr/bin/env python3
"""Backtest the trained prediction engine on held-out test data.

Evaluates on data/processed/features_test.parquet (matches from 2023+).

Metrics reported:
  Point-level (GBDT):
    - Log-loss, Brier score
    - Calibration curve (reliability diagram)

  Match-level (full pipeline):
    - Log-loss, Brier score
    - Calibration curve
    - ECE (Expected Calibration Error)

  Baseline comparisons:
    - ELO baseline (pre-match only, no in-play)
    - Constant p_serve=0.63 Markov chain (serve-only baseline)

  Inference latency:
    - p50, p95, p99 for engine.predict()

Usage:
    python scripts/backtest.py [--artifacts-dir PATH] [--test-data PATH]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.features import FEATURE_COLUMNS
from tennis.model import PointWinModel
from tennis.calibration import MatchWinCalibrator
from tennis.elo import EloTracker, elo_baseline_win_prob
from tennis.markov import match_win_prob
from tennis.types import MatchFormat, MatchState
from tennis.engine import PredictionEngine
from tennis.updater import BayesianUpdater

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"


def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def evaluate_point_level(
    model: PointWinModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    calibrator: MatchWinCalibrator,
) -> dict:
    """Evaluate GBDT point-win predictions."""
    p_raw = model.predict_proba(X_test)
    p_cal = calibrator.calibrate_batch(p_raw)

    ll_raw = log_loss(y_test, p_raw)
    ll_cal = log_loss(y_test, p_cal)
    bs_raw = brier_score_loss(y_test, p_raw)
    bs_cal = brier_score_loss(y_test, p_cal)

    print(f"\n--- Point-level metrics (n={len(y_test):,}) ---")
    print(f"  Log-loss (raw):      {ll_raw:.4f}")
    print(f"  Log-loss (cal):      {ll_cal:.4f}")
    print(f"  Brier score (raw):   {bs_raw:.4f}")
    print(f"  Brier score (cal):   {bs_cal:.4f}")

    return {
        "point_logloss_raw": ll_raw,
        "point_logloss_cal": ll_cal,
        "point_brier_raw": bs_raw,
        "point_brier_cal": bs_cal,
    }


def evaluate_match_level(
    model: PointWinModel,
    calibrator: MatchWinCalibrator,
    test_df: pd.DataFrame,
    match_format: MatchFormat,
    elo_tracker: EloTracker,
) -> dict:
    """Evaluate match-level predictions on test set."""
    if "match_id" not in test_df.columns:
        print("  Skipping match-level evaluation (no match_id column)")
        return {}

    match_probs_raw = []
    match_probs_elo = []
    match_probs_constant = []
    match_outcomes = []

    for match_id, group in test_df.groupby("match_id"):
        group = group.sort_values("point_idx")
        if len(group) < 5:
            continue

        # Match outcome: use actual winner if available (check first to avoid orphan probs)
        if "match_winner" in group.columns:
            mw = group["match_winner"].iloc[0]
            if mw == 0:
                continue  # unknown winner, skip
            outcome = int(mw == 1)
        else:
            outcome = int(group["server_won"].mean() > 0.5)

        # GBDT + Markov chain prediction at match start
        X_first = group[FEATURE_COLUMNS].values[:1].astype(np.float32)
        p_serve_gbdt = float(model.predict_proba(X_first)[0])

        state = MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            match_format=match_format,
        )
        raw_prob = match_win_prob(state, p_serve_gbdt)
        match_probs_raw.append(raw_prob)

        # ELO baseline: use elo_diff feature
        elo_diff = float(group["elo_diff"].iloc[0])
        elo_prob = elo_baseline_win_prob(elo_diff)
        match_probs_elo.append(elo_prob)

        # Constant p=0.63 baseline
        const_prob = match_win_prob(state, 0.63)
        match_probs_constant.append(const_prob)

        match_outcomes.append(outcome)

    if not match_probs_raw:
        print("  No matches to evaluate.")
        return {}

    raw = np.array(match_probs_raw)
    elo = np.array(match_probs_elo)
    const = np.array(match_probs_constant)
    outcomes = np.array(match_outcomes)
    cal = calibrator.calibrate_batch(raw)

    # Clamp probabilities to avoid infinite log-loss
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    cal = np.clip(cal, 1e-6, 1 - 1e-6)
    elo = np.clip(elo, 1e-6, 1 - 1e-6)
    const = np.clip(const, 1e-6, 1 - 1e-6)

    ll_raw = log_loss(outcomes, raw)
    ll_cal = log_loss(outcomes, cal)
    ll_elo = log_loss(outcomes, elo)
    ll_const = log_loss(outcomes, const)

    bs_raw = brier_score_loss(outcomes, raw)
    bs_cal = brier_score_loss(outcomes, cal)
    bs_elo = brier_score_loss(outcomes, elo)
    bs_const = brier_score_loss(outcomes, const)

    ece_cal = MatchWinCalibrator().expected_calibration_error(cal, outcomes)

    n_matches = len(match_outcomes)
    print(f"\n--- Match-level metrics (n={n_matches:,} matches) ---")
    print(f"{'Model':<25} {'Log-loss':>10} {'Brier':>10}")
    print(f"  {'GBDT+Markov (raw)':<23} {ll_raw:>10.4f} {bs_raw:>10.4f}")
    print(f"  {'GBDT+Markov (cal)':<23} {ll_cal:>10.4f} {bs_cal:>10.4f}")
    print(f"  {'ELO baseline':<23} {ll_elo:>10.4f} {bs_elo:>10.4f}")
    print(f"  {'Constant p=0.63':<23} {ll_const:>10.4f} {bs_const:>10.4f}")
    print(f"\n  ECE (calibrated): {ece_cal:.4f}")

    if ll_cal < ll_elo:
        print(f"  ✓ Model outperforms ELO baseline on log-loss ({ll_cal:.4f} < {ll_elo:.4f})")
    else:
        print(f"  ✗ ELO baseline is better on log-loss ({ll_elo:.4f} < {ll_cal:.4f})")

    return {
        "match_logloss_raw": ll_raw,
        "match_logloss_cal": ll_cal,
        "match_logloss_elo": ll_elo,
        "match_brier_raw": bs_raw,
        "match_brier_cal": bs_cal,
        "match_brier_elo": bs_elo,
        "match_ece_cal": ece_cal,
        "n_matches": n_matches,
        "probs_cal": cal,
        "outcomes": outcomes,
    }


def evaluate_inference_latency(engine: PredictionEngine) -> dict:
    """Measure engine.predict() latency over 1000 random states."""
    rng = np.random.default_rng(42)
    states = []
    for _ in range(1000):
        state = MatchState(
            sets_p1=int(rng.integers(0, 2)),
            sets_p2=int(rng.integers(0, 2)),
            games_p1=int(rng.integers(0, 6)),
            games_p2=int(rng.integers(0, 6)),
            points_p1=int(rng.integers(0, 4)),
            points_p2=int(rng.integers(0, 4)),
            server=int(rng.integers(0, 2)),
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        states.append(state)

    # Warm up
    engine.new_match(states[0])
    engine.predict(states[0])

    latencies = []
    for state in states:
        engine.new_match(state)
        t0 = time.perf_counter()
        engine.predict(state)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies) * 1000  # ms
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\n--- Inference latency (n=1000 states) ---")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")
    if p95 < 10.0:
        print(f"  ✓ p95 latency {p95:.2f}ms < 10ms target")
    else:
        print(f"  ✗ p95 latency {p95:.2f}ms exceeds 10ms target")

    return {"latency_p50": p50, "latency_p95": p95, "latency_p99": p99}


def plot_calibration(probs: np.ndarray, outcomes: np.ndarray, title: str, save_path: str) -> None:
    """Save a reliability diagram (calibration curve)."""
    cal = MatchWinCalibrator()
    mean_pred, frac_pos = cal.calibration_curve(probs, outcomes, n_bins=20)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "b-o", markersize=4, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Calibration curve saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest tennis prediction engine")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--test-data", default=str(PROCESSED_DIR / "features_test.parquet"))
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}")
        print("Run build_features.py first.")
        sys.exit(1)

    print("=== Loading artifacts ===")
    try:
        engine = PredictionEngine.load(args.artifacts_dir)
        model = engine._model
        calibrator = engine._calibrator
        elo_tracker = engine._elo
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run train.py first.")
        sys.exit(1)

    print("=== Loading test data ===")
    test_df = load_test_data(args.test_data)
    print(f"  Test set: {len(test_df):,} points, {test_df['match_date'].min()} – {test_df['match_date'].max()}")

    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = test_df["server_won"].values.astype(np.float32)

    print("\n=== Evaluating point-level metrics ===")
    point_metrics = evaluate_point_level(model, X_test, y_test, calibrator)

    print("\n=== Evaluating match-level metrics ===")
    match_format = MatchFormat.best_of_3_tiebreak()
    match_metrics = evaluate_match_level(model, calibrator, test_df, match_format, elo_tracker)

    print("\n=== Measuring inference latency ===")
    latency_metrics = evaluate_inference_latency(engine)

    if "probs_cal" in match_metrics:
        print("\n=== Generating calibration plots ===")
        plot_calibration(
            match_metrics["probs_cal"],
            match_metrics["outcomes"],
            "Match Win Probability Calibration (Test Set)",
            os.path.join(args.plots_dir, "calibration_curve.png"),
        )

    print("\n=== Summary ===")
    all_metrics = {**point_metrics, **{k: v for k, v in match_metrics.items()
                                         if not isinstance(v, np.ndarray)}, **latency_metrics}
    for k, v in sorted(all_metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k}: {v}")

    # Save metrics to CSV
    metrics_path = os.path.join(args.plots_dir, "backtest_metrics.csv")
    pd.DataFrame([all_metrics]).to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
