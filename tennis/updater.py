"""Bayesian in-match p_serve updater using Beta-Binomial conjugate model.

The prior over p_serve (server's point win probability) is Beta(α, β), where
α and β are set such that the prior mean equals the GBDT's pre-match estimate,
and α + β equals the virtual sample size (controls prior strength).

After observing each point, we update with optional exponential decay:
    # Decay old evidence (fading memory)
    α = α * decay
    β = β * decay
    # Then incorporate new observation
    if server won:    α += 1
    if server lost:   β += 1

The posterior mean provides the updated p_serve estimate. The posterior
standard deviation quantifies uncertainty and drives Monte Carlo confidence
intervals in the engine.

References:
    Ingram (2019) — "A point-based Bayesian hierarchical model to predict
    the outcome of tennis matches." Validates this conjugate approach.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


class BayesianUpdater:
    """Beta-Binomial conjugate updater for p_serve.

    Attributes:
        VIRTUAL_SAMPLE_SIZE: Prior strength in "pseudo-points". Higher values
            mean the prior dominates longer before observations take over.
            20 is reasonable: roughly equivalent to having watched 20 points
            at the prior rate.
    """

    VIRTUAL_SAMPLE_SIZE: int = 20

    def __init__(
        self,
        p_serve_prior: float,
        virtual_sample_size: int | None = None,
        decay_rate: float = 1.0,
    ):
        """Initialise updater.

        Args:
            p_serve_prior: Prior mean of p_serve, typically the GBDT estimate.
            virtual_sample_size: Override the default prior strength.
            decay_rate: Exponential decay factor applied to α,β before each
                update. 1.0 = no decay (standard conjugate). 0.99 = ~1% decay
                per point, giving an effective window of ~100 points.
                Bounded from below so α+β never drops below virtual_sample_size.
        """
        if virtual_sample_size is None:
            virtual_sample_size = self.VIRTUAL_SAMPLE_SIZE

        p_serve_prior = float(np.clip(p_serve_prior, 1e-4, 1 - 1e-4))
        vss = float(max(virtual_sample_size, 2))

        # Parameterise Beta so that prior mean = p_serve_prior
        self._alpha = p_serve_prior * vss
        self._beta = (1.0 - p_serve_prior) * vss
        self._n_observed = 0
        self._decay_rate = float(np.clip(decay_rate, 0.9, 1.0))
        self._min_effective_n = vss  # floor for α+β

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def update(self, server_won: bool) -> None:
        """Incorporate a new observed point outcome.

        If decay_rate < 1.0, older observations are discounted first,
        keeping effective sample size bounded for faster adaptation.

        Args:
            server_won: True if the current server won the point.
        """
        if self._decay_rate < 1.0:
            # Apply decay, but don't let effective sample size fall below minimum
            new_alpha = self._alpha * self._decay_rate
            new_beta = self._beta * self._decay_rate
            if (new_alpha + new_beta) >= self._min_effective_n:
                self._alpha = new_alpha
                self._beta = new_beta

        if server_won:
            self._alpha += 1.0
        else:
            self._beta += 1.0
        self._n_observed += 1

    def reset_prior(self, new_p_serve: float, strength: float | None = None) -> None:
        """Soft-reset the prior to a new p_serve estimate.

        Useful for periodic re-estimation (e.g., at set changes) based on
        fresh GBDT features without discarding all observed data.

        Args:
            new_p_serve: New prior mean for p_serve.
            strength: How many pseudo-observations the new prior counts as.
                      Defaults to half the minimum effective sample size.
        """
        new_p = float(np.clip(new_p_serve, 1e-4, 1 - 1e-4))
        if strength is None:
            strength = self._min_effective_n / 2.0
        strength = max(strength, 2.0)

        # Blend: keep current posterior but inject new prior evidence
        total = self._alpha + self._beta
        current_weight = max(total - strength, self._min_effective_n)
        current_mean = self._alpha / total

        blended_mean = (current_weight * current_mean + strength * new_p) / (current_weight + strength)
        new_total = current_weight + strength

        self._alpha = blended_mean * new_total
        self._beta = (1.0 - blended_mean) * new_total

    # ------------------------------------------------------------------
    # Posterior quantities
    # ------------------------------------------------------------------

    @property
    def posterior_mean(self) -> float:
        """Posterior mean of p_serve: α / (α + β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def posterior_std(self) -> float:
        """Posterior standard deviation of p_serve."""
        a, b = self._alpha, self._beta
        n = a + b
        return math.sqrt(a * b / (n * n * (n + 1.0)))

    @property
    def posterior_mode(self) -> float:
        """Posterior mode: (α-1)/(α+β-2), defined for α,β > 1."""
        a, b = self._alpha, self._beta
        if a > 1.0 and b > 1.0:
            return (a - 1.0) / (a + b - 2.0)
        return self.posterior_mean

    @property
    def n_observed(self) -> int:
        """Number of points observed so far."""
        return self._n_observed

    @property
    def effective_sample_size(self) -> float:
        """Current effective sample size (α + β)."""
        return self._alpha + self._beta

    def credible_interval(self, coverage: float = 0.95) -> tuple[float, float]:
        """Bayesian credible interval for p_serve.

        Args:
            coverage: Coverage probability (default 95%).

        Returns:
            (lower, upper) bounds.
        """
        alpha_tail = (1.0 - coverage) / 2.0
        lower = stats.beta.ppf(alpha_tail, self._alpha, self._beta)
        upper = stats.beta.ppf(1.0 - alpha_tail, self._alpha, self._beta)
        return float(lower), float(upper)

    def sample(self, n: int = 500, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw n samples from the posterior Beta distribution.

        Used by the engine for Monte Carlo propagation of uncertainty through
        the Markov chain.

        Args:
            n: Number of samples.
            rng: Optional numpy random generator for reproducibility.

        Returns:
            Array of shape (n,) with p_serve samples ∈ (0, 1).
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.beta(self._alpha, self._beta, size=n).astype(np.float64)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianUpdater("
            f"mean={self.posterior_mean:.4f}, "
            f"std={self.posterior_std:.4f}, "
            f"alpha={self._alpha:.2f}, beta={self._beta:.2f}, "
            f"n={self._n_observed}, "
            f"decay={self._decay_rate})"
        )
