"""Kern: chi²-Verteilung gegen scipy, Mahalanobis-Abstand mit Handinstanzen, FastMCD gegen scikit-learn, C-Schritte, Konsistenz-Korrektur, Ellipsen."""

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

import ee_algorithm as alg
import ee_constants as C


def _gauss_with_outliers(n=300, p=4, n_out=45, shift=8.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    normal = rng.standard_normal((n - n_out, p)) @ A + 3.0
    out = rng.standard_normal((n_out, p)) * 0.4 + shift
    X = np.concatenate([normal, out])
    return X, np.arange(n) >= n - n_out


@pytest.mark.parametrize("dof", [1, 2, 3, 5, 12, 30])
@pytest.mark.parametrize("q", [0.01, 0.5, 0.9, 0.975, 0.999])
def test_chi2_matches_scipy(dof, q):
    assert alg.chi2_ppf(q, dof) == pytest.approx(chi2.ppf(q, dof), rel=1e-7)
    x = chi2.ppf(q, dof)
    assert alg.chi2_cdf(x, dof) == pytest.approx(q, abs=1e-9)


def test_chi2_edge_cases():
    assert alg.chi2_cdf(0.0, 4) == 0.0 and alg.chi2_cdf(-1.0, 4) == 0.0
    assert alg.chi2_cdf(1e4, 4) == pytest.approx(1.0)
    assert alg.chi2_ppf(0.5, 2) == pytest.approx(2 * np.log(2), rel=1e-9)                       # chi²(2): Exponentialverteilung mit Mittel 2


def test_mahalanobis_hand_instance():
    mu = np.array([1.0, 2.0])
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    X = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 5.0], [3.0, 3.0]])
    assert np.allclose(alg.mahalanobis_sq(X, mu, cov), [0.0, 1.0, 9.0, 2.0])
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(alg.mahalanobis_sq(X @ rot.T, mu @ rot.T, rot @ cov @ rot.T), [0.0, 1.0, 9.0, 2.0])                 # invariant gegen Drehung


def test_mahalanobis_uses_the_rank_for_singular_covariance():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    fit = alg.fit_classical(X)
    assert fit.dof == 1 and np.isfinite(fit.d2).all()
    assert fit.d2.sum() == pytest.approx((len(X) - 1) * fit.dof)                                 # Summe der Abstände im Datensatz = (n - 1) mal Rang


def test_classical_fit_equals_numpy_and_the_distance_sum_is_n_minus_one_times_p():
    X, _ = _gauss_with_outliers()
    fit = alg.fit_classical(X)
    assert np.allclose(fit.location, X.mean(axis=0)) and np.allclose(fit.covariance, np.cov(X.T))
    assert fit.d2.sum() == pytest.approx((len(X) - 1) * X.shape[1], rel=1e-9) and fit.d2.max() <= (len(X) - 1) ** 2 / len(X) + 1e-9


def test_subset_size():
    assert alg.subset_size(300, 12, 0.5) == 156 and alg.subset_size(300, 12, 0.75) == 225 and alg.subset_size(300, 12, 1.0) == 300
    assert alg.subset_size(20, 30, 0.5) == 20 and alg.subset_size(31, 2, 0.5) == 17                      # höchstens n; mindestens (n + p + 1) / 2


def test_mcd_recovers_the_normal_parameters_where_the_classical_fit_fails():
    X, is_out = _gauss_with_outliers()
    normal = X[~is_out]
    truth_mu, truth_cov = normal.mean(axis=0), np.cov(normal.T)
    robust, classical = alg.fit_mcd(X, 0.5, 0), alg.fit_classical(X)
    err = lambda f: np.linalg.norm(f.covariance - truth_cov) / np.linalg.norm(truth_cov)
    assert np.linalg.norm(robust.location - truth_mu) < 0.15 * np.linalg.norm(classical.location - truth_mu)
    assert err(robust) < 0.2 and err(classical) > 1.0
    thr = alg.threshold(robust, 0.975)
    assert (robust.d2[is_out] > thr).all()                                                        # alle Ausreißer gefunden
    assert (classical.d2[is_out] > alg.threshold(classical, 0.975)).mean() < 0.5                   # klassisch maskiert


def test_mcd_close_to_scikit_learn():
    X, is_out = _gauss_with_outliers(seed=2)
    ours = alg.fit_mcd(X, 0.5, 0, reweight=True)
    sk = MinCovDet(random_state=0).fit(X)
    scale = np.sqrt(np.diag(np.cov(X[~is_out].T)))
    assert np.linalg.norm((ours.location - sk.location_) / scale) < 0.1
    assert np.linalg.norm(ours.covariance - sk.covariance_) / np.linalg.norm(sk.covariance_) < 0.15
    sk_flag = sk.mahalanobis(X) > chi2.ppf(0.975, X.shape[1])
    ours_flag = ours.d2 > alg.threshold(ours, 0.975)
    assert (sk_flag == ours_flag).mean() > 0.95


def test_c_steps_never_increase_the_determinant():
    X, _ = _gauss_with_outliers(seed=1)
    fit = alg.fit_mcd(X, 0.5, 4, reweight=False)
    h = np.array(fit.logdet_history)
    assert len(h) >= 2 and (np.diff(h) <= 1e-9 * np.abs(h[:-1])).all()
    assert all(len(s) == fit.h for s in fit.subsets) and fit.n_converged >= 1
    assert set(fit.support) == set(fit.subsets[-1])


def test_more_starts_never_give_a_larger_determinant():
    X, _ = _gauss_with_outliers(seed=3)
    few = alg.fit_mcd(X, 0.5, 5, n_starts=2, reweight=False)
    many = alg.fit_mcd(X, 0.5, 5, n_starts=30, reweight=False)
    assert many.logdet_history[-1] <= few.logdet_history[-1] + 1e-9


def test_full_support_without_reweighting_is_the_classical_fit_up_to_the_consistency_factor():
    X, _ = _gauss_with_outliers(seed=4)
    mcd = alg.fit_mcd(X, 1.0, 0, reweight=False)
    cl = alg.fit_classical(X)
    assert np.allclose(mcd.location, cl.location)
    ratio = mcd.covariance / cl.covariance
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-6)                                            # nur ein Skalenfaktor (Konsistenz-Korrektur)


def test_consistency_correction_puts_the_median_at_the_chi2_median():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 5))
    fit = alg.fit_mcd(X, 0.5, 0, reweight=False)
    assert np.median(fit.d2) == pytest.approx(chi2.ppf(0.5, 5), rel=1e-6)
    assert np.linalg.norm(fit.covariance - np.eye(5)) / np.sqrt(5) < 0.25


def test_reweighting_gives_the_expected_false_alarm_rate_on_clean_gaussian_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3000, 6))
    fit = alg.fit_mcd(X, 0.5, 0, reweight=True)
    assert fit.method == "mcd-reweighted"
    assert (fit.d2 > alg.threshold(fit, 0.975)).mean() == pytest.approx(0.025, abs=0.01)


def test_mcd_is_deterministic_and_the_start_seed_is_separate():
    X, _ = _gauss_with_outliers()
    a, b, c = alg.fit_mcd(X, 0.5, 3), alg.fit_mcd(X, 0.5, 3), alg.fit_mcd(X, 0.5, 4)
    assert np.array_equal(a.location, b.location) and np.array_equal(a.covariance, b.covariance)
    assert np.allclose(a.location, c.location, atol=0.2)                                         # anderer Start, dieselbe Lösung (Kern gefunden)


def test_mcd_affine_equivariance():
    X, _ = _gauss_with_outliers()
    A = np.random.default_rng(5).standard_normal((4, 4))
    f1, f2 = alg.fit_mcd(X, 0.5, 0), alg.fit_mcd(X @ A.T + 2.0, 0.5, 0)
    assert np.allclose(f1.d2, f2.d2, rtol=1e-5, atol=1e-6)


def test_projection_axes_and_ellipse_points():
    cov = np.diag([9.0, 4.0, 1.0])
    axes = alg.projection_axes(cov)
    assert axes.shape == (3, 2) and np.allclose(np.abs(axes[:, 0]), [1, 0, 0]) and np.allclose(np.abs(axes[:, 1]), [0, 1, 0])
    pts = alg.ellipse_points(np.array([1.0, 2.0, 3.0]), cov, 1.0, axes)
    assert np.allclose(pts.mean(axis=0), [1.0, 2.0], atol=0.05)
    assert (pts[:, 0].max() - pts[:, 0].min()) / 2 == pytest.approx(3.0, abs=0.01) and (pts[:, 1].max() - pts[:, 1].min()) / 2 == pytest.approx(2.0, abs=0.01)
    level = alg.chi2_ppf(0.975, 3)
    pts = alg.ellipse_points(np.zeros(3), cov, level, axes)
    inside = alg.mahalanobis_sq(np.concatenate([pts, np.zeros((len(pts), 1))], axis=1), np.zeros(3), cov)                  # Randpunkte mit dritter Koordinate 0 liegen innerhalb der Ellipse
    assert (inside <= level + 1e-9).all()


def test_projected_ellipse_is_the_shadow_of_the_ellipsoid():
    """Die Projektion des Ellipsoids {d² <= level} auf eine Ebene ist die Ellipse mit der projizierten Kovarianz (größte Ausdehnung geprüft)."""
    rng = np.random.default_rng(2)
    B = rng.standard_normal((4, 4))
    cov = B @ B.T
    axes = alg.projection_axes(cov)
    level = 6.0
    pts = alg.ellipse_points(np.zeros(4), cov, level, axes)
    # Punkte auf dem Ellipsoidrand, projiziert, müssen innerhalb der Projektions-Ellipse liegen
    samples = rng.standard_normal((2000, 4))
    boundary = samples / np.sqrt(alg.mahalanobis_sq(samples, np.zeros(4), cov))[:, None] * np.sqrt(level)
    proj = boundary @ axes
    Sp = np.linalg.inv(axes.T @ cov @ axes)
    assert (np.einsum("ij,jk,ik->i", proj, Sp, proj) <= level + 1e-6).all()
    assert abs(np.abs(pts[:, 0]).max() - np.sqrt(level * np.linalg.eigvalsh(axes.T @ cov @ axes).max())) < 0.05 * np.sqrt(level * np.linalg.eigvalsh(axes.T @ cov @ axes).max()) + 0.5
