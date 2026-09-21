"""Szenario (Normale Zeilen wie in der PCA-Demo, Betriebsarten, Krümmung, Anomalien, Zusatzmerkmale), Kennzahlen mit Handinstanzen, Urteil, Analyse."""

import numpy as np
import pytest

import ee_algorithm as alg
import ee_constants as C
import ee_evaluation as ev
import ee_scenario as sc

# Zeilensummen der ersten acht Zeilen der PCA-Demo (generate_dataset(300, 2, 0.0, 0.25, 0, 7)): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_demo_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)
    assert ds.X[0, 0] == pytest.approx(38459.09501056836, rel=1e-12)                                # Distanz der ersten Tour


def test_default_dataset_is_frozen():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (45, 45, 20), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_the_same_tours_with_and_without_more_anomalies():
    a, b = ev.make_dataset(contamination=5), ev.make_dataset(contamination=20)
    both_normal = ~a.anomaly & ~b.anomaly
    assert both_normal.sum() > 200 and np.array_equal(a.X[both_normal], b.X[both_normal])            # feste Ziehungsreihenfolge
    assert a.anomaly[a.anomaly].sum() < b.anomaly.sum() and (b.anomaly[a.anomaly]).all()             # die Anomalien wachsen mit dem Anteil (kleinste Lose)


@pytest.mark.parametrize("p", [2, 5, 12, 13, 20, 30])
def test_feature_count_and_names(p):
    ds = ev.make_dataset(p=p)
    assert ds.X.shape == (300, p) and len(ds.names) == p
    if p >= 13:
        assert ds.names[12] == "Zusatzmerkmal 13" and np.array_equal(ds.X[:, :12], ev.make_dataset(p=12).X)      # die ersten 12 bleiben gleich


def test_first_two_features_load_on_both_latent_factors():
    ds = ev.make_dataset(p=2, contamination=1, noise=0.0)
    assert ds.names == ["Distanz", "Zeitfenster-Enge"]
    z = ds.z[~ds.anomaly]
    r = np.corrcoef(np.c_[z, ds.X[~ds.anomaly]].T)[:2, 2:]
    assert abs(r[0, 0]) > 0.9 and abs(r[1, 1]) > 0.9


def test_modes_form_separated_groups_and_the_gap_anomalies_sit_between_them():
    ds = ev.make_dataset(n_modes=2, kind="gap", contamination=10)
    centers = sc.mode_centers(2)
    normal = ~ds.anomaly
    assert np.allclose(centers, [[2.2, 0.0], [-2.2, 0.0]], atol=1e-9)
    for m in (0, 1):
        assert np.linalg.norm(ds.z[normal & (ds.mode == m)].mean(axis=0) - centers[m]) < 0.15
    assert np.abs(ds.z[ds.anomaly]).max() < 1.2                                               # in der Lücke um den Ursprung
    assert (np.linalg.norm(ds.z[normal][:, None, :] - centers[None], axis=2).min(axis=1) < 2.0).all()
    assert set(np.unique(ds.mode)) == {0, 1}
    assert np.allclose(sc.mode_centers(3).sum(axis=0), 0.0, atol=1e-9) and np.allclose(np.linalg.norm(sc.mode_centers(3), axis=1), C.MODE_RADIUS)


def test_gap_falls_back_to_scattered_with_one_mode():
    assert ev.make_dataset(kind="gap").kind == "scattered" and np.array_equal(ev.make_dataset(kind="gap").X, ev.make_dataset(kind="scattered").X)


def test_anomaly_kinds_have_the_documented_geometry():
    sca = ev.make_dataset(kind="scattered", strength=6.0, contamination=20)
    r = np.linalg.norm(sca.z[sca.anomaly], axis=1)
    assert 2.5 < r.min() and r.max() < 12.0 and abs(r.mean() - 6.0) < 0.8                    # 6 Faktor-σ mit 20 % Streuung
    ang = np.arctan2(sca.z[sca.anomaly, 1], sca.z[sca.anomaly, 0])
    assert ang.max() - ang.min() > 4.0                                                          # in alle Richtungen
    clu = ev.make_dataset(kind="cluster", strength=6.0, contamination=20)
    zc = clu.z[clu.anomaly]
    assert np.linalg.norm(zc.mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15 and zc.std(axis=0).max() < 0.45


def test_curvature_zero_changes_nothing_and_positive_bends_the_normal_surface():
    a, b = ev.make_dataset(curvature=0.0), ev.make_dataset()
    assert np.array_equal(a.X, b.X)
    c = ev.make_dataset(curvature=1.0)
    assert not np.allclose(c.X, a.X) and np.array_equal(c.z, a.z)


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3).X, ev.make_dataset(seed=3).X) and not np.array_equal(ev.make_dataset(seed=3).X, ev.make_dataset(seed=4).X)


# --- Kennzahlen: Handinstanzen ---------------------------------------------------------------------------------------------------------


def test_roc_auc_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([4, 3, 2, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5                                # Bindungen zählen halb
    assert ev.roc_auc(np.array([1, 3, 2, 4.0]), np.array([0, 1, 0, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert ev.roc_auc(s, y) == pytest.approx(brute)


def test_average_precision_hand_instance():
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 1, 0, 0], bool)) == 1.0
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)
    assert np.isnan(ev.average_precision(np.array([1.0, 2.0]), np.array([False, False])))


def test_roc_curve_endpoints_and_area():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and (np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all()
    assert np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)


def test_flag_metrics_hand_instance():
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["n_flagged"] == 3
    assert m["f1"] == pytest.approx(0.4)
    none = ev.flag_metrics(np.zeros(4, bool), np.array([1, 0, 0, 0], bool))
    assert none["recall"] == 0.0 and none["f1"] == 0.0 and none["precision"] == 0.0 and none["false_alarm"] == 0.0
    assert np.isnan(ev.flag_metrics(np.array([1, 0], bool), np.zeros(2, bool))["f1"])


# --- Analyse ----------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_and_reference():
    a = ev.analyse_for((300, 12, 1, 0.0, 0.25, 10, "scattered", 6.0, 7))
    assert set(a.scores) == {"classical", "robust"} and a.robust.h == 156 and a.classical.dof == 12 == a.robust.dof
    normal = a.ds.X[~a.ds.anomaly]
    assert np.allclose(a.reference_center, normal.mean(axis=0)) and np.allclose(a.reference_cov, np.cov(normal.T))
    for det in ("classical", "robust"):
        s = a.scores[det]
        assert s["threshold"] == pytest.approx(alg.chi2_ppf(0.975, 12)) and 0 <= s["auc"] <= 1 and s["n_flagged"] == int(ev.flagged(a, det).sum())
    assert a.scores["robust"]["shape_error"] < a.scores["classical"]["shape_error"] and a.scores["robust"]["center_error"] < a.scores["classical"]["center_error"]


def test_analysis_is_deterministic_and_the_start_seed_is_separate_from_the_data_seed():
    p = (300, 12, 1, 0.0, 0.25, 10, "scattered", 6.0, 7)
    a, b = ev.analyse_for(p), ev.analyse_for(p)
    assert a.scores == b.scores
    c = ev.analyse_for(p, ev.Settings(start=5))
    assert np.array_equal(a.ds.X, c.ds.X) and abs(a.scores["robust"]["auc"] - c.scores["robust"]["auc"]) < 0.02


def test_sweep_rows_carry_mean_std_and_range():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for det in ev.DETECTORS:
            assert r[f"{det}_auc_min"] <= r[f"{det}_auc"] <= r[f"{det}_auc_max"] and r[f"{det}_auc_std"] >= 0
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    s = ev.sweep("quantile", values=(0.9, 0.999))
    assert s[0]["robust_false_alarm"] > s[1]["robust_false_alarm"] and s[0]["classical_auc"] == s[1]["classical_auc"]           # nur die Schwelle ändert sich, nicht die Rangfolge


def test_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.support, s.quantile, s.reweight) == (C.DEFAULT_SUPPORT, C.DEFAULT_QUANTILE, C.DEFAULT_REWEIGHT) and C.SWEEP_SEEDS == tuple(range(100000, 100005))
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS)


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _code(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    settings = ev.Settings(quantile=p.pop("quantile", C.DEFAULT_QUANTILE))
    a = ev.analyse_for((p["n"], p["p"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], 7), settings)
    return ev.verdict(a)


def test_verdict_codes_for_every_case():
    assert _code()[:2] == ("success", "masked")
    assert _code(kind="cluster", contamination=20)[:2] == ("success", "masked")
    assert _code(kind="cluster", contamination=40)[:2] == ("warning", "beyond_breakdown")
    assert _code(n_modes=2, kind="gap")[:2] == ("warning", "gap")
    assert _code(curvature=0.75)[:2] == ("warning", "not_convex")
    assert _code(n=30)[:2] == ("warning", "too_few_samples")
    assert _code(n_modes=3)[:2] == ("warning", "multi_modal")
    assert _code(contamination=3, strength=12.0, n=600)[:2][0] == "success"
    assert _code(strength=3.0, contamination=2, n=600)[1] in ("comparable", "masked", "threshold_off")


def test_comparable_and_threshold_off_are_reachable():
    kinds = {_code(contamination=c, strength=s, n=n, p=p, quantile=q)[1] for c, s, n, p, q in ((1, 12.0, 600, 2, 0.975), (2, 12.0, 300, 2, 0.975), (2, 12.0, 600, 12, 0.9), (10, 12.0, 300, 5, 0.6 + 0.3))}
    assert "comparable" in kinds
    seen = {_code(n=n, p=p, contamination=2)[1] for n in (100, 150) for p in (8, 10)}
    assert seen & {"threshold_off", "too_few_samples"}


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _code(n_modes=3)
    assert code == "multi_modal" and 0 <= data["reference_recall"] <= 1 and data["reference_recall"] - data["robust_recall"] > ev.MODE_RECALL_DROP
    for key in ("classical_recall", "robust_recall", "classical_auc", "robust_auc", "classical_shape_error", "robust_shape_error", "robust_false_alarm", "nominal", "n_anomalies", "p", "n"):
        assert key in data
    kind, code, data = _code(n=30)
    assert data["n"] == 30 and data["classical_recall"] == 0.0 and data["robust_false_alarm"] > 0.1
