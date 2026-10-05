"""Unabhängiges Orakel für den MCD: auf Mini-Instanzen die Vollaufzählung aller h-Teilmengen (kleinste Determinante der Kovarianz), dazu Konsistenz-Korrektur und Neugewichtung explizit mit numpy.linalg.inv
und scipy nachgerechnet (nur ein anderer Rechenweg, nicht derselbe Code) und der Mahalanobis-Abstand gegen die Inverse."""

import itertools

import numpy as np
import pytest
from scipy.stats import chi2

import ee_algorithm as alg


def _best_logdet(X, h):
    p = X.shape[1]
    best = np.inf
    for subset in itertools.combinations(range(len(X)), h):
        Y = X[list(subset)]
        Yc = Y - Y.mean(axis=0)
        sign, ld = np.linalg.slogdet((Yc.T @ Yc / (h - 1)).reshape(p, p))
        if sign > 0:
            best = min(best, ld)
    return best


def test_fit_mcd_finds_the_global_minimum_determinant_on_mini_instances():
    rng = np.random.default_rng(3)
    checked = 0
    for i in range(24):
        p = int(rng.integers(1, 4))
        n = int(rng.integers(p + 5, 11))
        X = rng.standard_normal((n, p))
        if i % 3 == 1:
            X[: max(1, n // 4)] += 6.0                                                    # Ausreißer
        support = float(rng.choice([0.5, 0.6, 0.75, 1.0]))
        h = alg.subset_size(n, p, support)
        best = _best_logdet(X, h)
        fit = alg.fit_mcd(X, support, seed=i, reweight=False)
        assert fit.logdet_history[-1] == pytest.approx(best, abs=1e-6)
        checked += 1
    assert checked == 24


def test_consistency_correction_and_reweighting_equal_an_explicit_recomputation():
    rng = np.random.default_rng(4)
    for i in range(25):
        p = int(rng.integers(1, 5))
        n = int(rng.integers(25, 100))
        X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + 2
        X[: n // 6] += rng.uniform(4, 9)
        raw = alg.fit_mcd(X, 0.5, i, reweight=False)
        full = alg.fit_mcd(X, 0.5, i, reweight=True)
        mu = X[raw.support].mean(axis=0)
        cov = np.cov(X[raw.support].T).reshape(p, p)
        d2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(cov), X - mu)
        cov = cov * np.median(d2) / chi2.ppf(0.5, p)
        d2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(cov), X - mu)
        assert np.allclose(raw.location, mu) and np.allclose(raw.covariance, cov, rtol=1e-8) and np.allclose(raw.d2, d2, rtol=1e-8)
        q = chi2.ppf(0.975, p)
        keep = d2 < q
        assert keep.sum() > p
        mu2 = X[keep].mean(axis=0)
        cov2 = np.cov(X[keep].T).reshape(p, p) * chi2.cdf(q, p) / chi2.cdf(q, p + 2)
        d22 = np.einsum("ij,jk,ik->i", X - mu2, np.linalg.inv(cov2), X - mu2)
        assert np.allclose(full.location, mu2) and np.allclose(full.covariance, cov2, rtol=1e-8) and np.allclose(full.d2, d22, rtol=1e-6)


def test_mahalanobis_equals_the_explicit_inverse_for_every_scale():
    rng = np.random.default_rng(5)
    for _ in range(60):
        p = int(rng.integers(1, 6))
        n = int(rng.integers(p + 2, 40))
        X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) * rng.choice([1e-3, 1.0, 1e3]) + rng.choice([0.0, 1e3])
        mu, cov = X.mean(axis=0), np.cov(X.T).reshape(p, p)
        ref = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(cov), X - mu)
        assert np.allclose(alg.mahalanobis_sq(X, mu, cov), ref, rtol=1e-6, atol=1e-8)
