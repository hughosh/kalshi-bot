"""Probability calibration for the tennis win probability engine.

Applies isotonic regression to correct systematic over/under-confidence in
the raw Markov chain match win probability outputs.

The calibrator is fitted on the validation set: for each point in each
validation match, we record the model's predicted match win probability and
the actual match outcome. Isotonic regression then learns a monotone mapping
from raw probability to calibrated probability.

References:
    Zadrozny & Elkan (2002). "Transforming classifier scores into accurate
    multiclass probability estimates." KDD.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class MatchWinCalibrator:
    """Isotonic regression calibrator for match win probabilities.

    Fits on (raw_prob, match_outcome) pairs from the validation set.
    At inference time, maps raw Markov chain probabilities to calibrated ones.
    """

    def __init__(self):
        self._calibrator: Optional[IsotonicRegression] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit calibration model on held-out (raw_prob, outcome) pairs.

        Args:
            probs: Array of raw match win probabilities ∈ [0, 1].
                   These should come from the validation set only.
            outcomes: Binary match outcomes (1 if player 1 won, 0 if player 2).
                      Must have the same length as probs.
        """
        probs = np.asarray(probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(probs) != len(outcomes):
            raise ValueError(f"probs ({len(probs)}) and outcomes ({len(outcomes)}) must have equal length.")

        # Isotonic regression: y_min=0, y_max=1, increasing=True
        # out_of_bounds="clip" ensures extrapolation stays in [0, 1]
        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self._calibrator.fit(probs, outcomes)
        self._fitted = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, prob: float) -> float:
        """Map a raw probability to a calibrated probability.

        Args:
            prob: Raw match win probability from the Markov chain.

        Returns:
            Calibrated probability ∈ [0, 1].
        """
        if not self._fitted:
            return prob  # No-op if not fitted yet
        return float(self._calibrator.predict([prob])[0])

    def calibrate_batch(self, probs: np.ndarray) -> np.ndarray:
        """Batch calibration for arrays of probabilities."""
        if not self._fitted:
            return np.asarray(probs, dtype=np.float64)
        return self._calibrator.predict(probs).astype(np.float64)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Expected Calibration Error (ECE) using equal-frequency bins.

        Lower is better. A perfectly calibrated model has ECE = 0.

        Args:
            probs: Predicted probabilities (calibrated or uncalibrated).
            outcomes: True binary outcomes.
            n_bins: Number of bins.

        Returns:
            ECE ∈ [0, 1].
        """
        probs = np.asarray(probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        n = len(probs)

        # Equal-frequency binning
        sorted_idx = np.argsort(probs)
        bin_size = n // n_bins

        ece = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else n
            if end <= start:
                continue
            idx = sorted_idx[start:end]
            bin_prob = probs[idx].mean()
            bin_outcome = outcomes[idx].mean()
            ece += (end - start) / n * abs(bin_prob - bin_outcome)

        return float(ece)

    def calibration_curve(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve (mean predicted prob vs actual frequency).

        Returns:
            (mean_predicted, fraction_of_positives) arrays of length n_bins.
        """
        from sklearn.calibration import calibration_curve as sk_cal_curve

        fraction_of_positives, mean_predicted = sk_cal_curve(
            outcomes, probs, n_bins=n_bins, strategy="quantile"
        )
        return mean_predicted, fraction_of_positives

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save calibrator to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"calibrator": self._calibrator, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> MatchWinCalibrator:
        """Load calibrator from pickle file."""
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._calibrator = data["calibrator"]
        obj._fitted = data["fitted"]
        return obj
