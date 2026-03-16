"""Tests for the Bayesian in-match p_serve updater."""
import pytest
import numpy as np

from tennis.updater import BayesianUpdater


class TestBayesianUpdater:
    def test_prior_mean(self):
        """Posterior mean before any observations equals the prior."""
        u = BayesianUpdater(p_serve_prior=0.65)
        assert abs(u.posterior_mean - 0.65) < 1e-6

    def test_prior_std_positive(self):
        u = BayesianUpdater(p_serve_prior=0.65)
        assert u.posterior_std > 0

    def test_update_server_wins_increases_mean(self):
        """Server winning points should push posterior mean up."""
        u = BayesianUpdater(p_serve_prior=0.5)
        for _ in range(10):
            u.update(server_won=True)
        assert u.posterior_mean > 0.5

    def test_update_server_loses_decreases_mean(self):
        """Returner winning points should pull posterior mean down."""
        u = BayesianUpdater(p_serve_prior=0.5)
        for _ in range(10):
            u.update(server_won=False)
        assert u.posterior_mean < 0.5

    def test_observations_counted(self):
        u = BayesianUpdater(p_serve_prior=0.6)
        assert u.n_observed == 0
        u.update(True)
        u.update(False)
        u.update(True)
        assert u.n_observed == 3

    def test_posterior_std_decreases_with_data(self):
        """More observations → less uncertainty."""
        u = BayesianUpdater(p_serve_prior=0.63)
        std_before = u.posterior_std
        for _ in range(50):
            u.update(True)
        assert u.posterior_std < std_before

    def test_extreme_prior(self):
        """Edge case: prior near 0 or 1."""
        u = BayesianUpdater(p_serve_prior=0.99)
        assert u.posterior_mean > 0.9
        u2 = BayesianUpdater(p_serve_prior=0.01)
        assert u2.posterior_mean < 0.1

    def test_sample_shape(self):
        u = BayesianUpdater(p_serve_prior=0.63)
        samples = u.sample(n=500)
        assert samples.shape == (500,)

    def test_sample_in_range(self):
        u = BayesianUpdater(p_serve_prior=0.63)
        for _ in range(20):
            u.update(True)
        samples = u.sample(n=1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_mean_close_to_posterior_mean(self):
        u = BayesianUpdater(p_serve_prior=0.65)
        for _ in range(30):
            u.update(True)
        for _ in range(10):
            u.update(False)
        samples = u.sample(n=10000, rng=np.random.default_rng(42))
        assert abs(samples.mean() - u.posterior_mean) < 0.005

    def test_credible_interval_coverage(self):
        u = BayesianUpdater(p_serve_prior=0.6)
        for _ in range(40):
            u.update(True)
        lo, hi = u.credible_interval(0.95)
        assert lo < u.posterior_mean < hi
        assert hi - lo < 0.3  # should be reasonably tight

    def test_large_virtual_sample_size(self):
        """Large virtual sample size makes prior dominate more."""
        u_strong = BayesianUpdater(p_serve_prior=0.7, virtual_sample_size=200)
        u_weak = BayesianUpdater(p_serve_prior=0.7, virtual_sample_size=5)
        # After 10 observations all returns, strong prior should still be higher
        for _ in range(10):
            u_strong.update(False)
            u_weak.update(False)
        assert u_strong.posterior_mean > u_weak.posterior_mean

    def test_repr(self):
        u = BayesianUpdater(p_serve_prior=0.63)
        r = repr(u)
        assert "BayesianUpdater" in r
        assert "mean=" in r
