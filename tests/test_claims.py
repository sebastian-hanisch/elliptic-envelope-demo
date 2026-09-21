"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo der robuste Detektor verliert, steht das hier ebenso als Test wie dort, wo er gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import ee_algorithm as alg
import ee_constants as C
import ee_evaluation as ev

TOL = 0.02


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev.analyse(ev.make_dataset(seed=s, **dict(items)), settings) for s in C.SWEEP_SEEDS)


def runs(settings=ev.Settings(), **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=ev.Settings(), **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Seitenleiste: Touren und Merkmale ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,fa,cla_recall", [(20, 0.16, 0.00), (30, 0.24, 0.00), (50, 0.29, 0.20), (100, 0.21, 0.42), (200, 0.06, 0.44)])
def test_tours_sweep(n, fa, cla_recall):
    near(m("robust", "false_alarm", n=n), fa)
    near(m("classical", "recall", n=n), cla_recall)


def test_in_sample_distance_bound_makes_the_classical_detector_blind_at_20_tours():
    bound = (20 - 1) ** 2 / 20
    near(bound, 18.05, 0.01)
    near(alg.chi2_ppf(0.975, 12), 23.3, 0.05)
    assert bound < alg.chi2_ppf(0.975, 12)
    for a in runs(n=20):
        assert a.classical.d2.max() <= bound + 1e-9                          # der Abstand einer Tour im Datensatz ist höchstens (n-1)^2/n
        assert a.scores["classical"]["recall"] == 0.0


@pytest.mark.parametrize("p,fa,cla_recall", [(2, 0.03, 0.84), (5, 0.02, 0.73), (8, 0.03, 0.57), (12, 0.04, 0.43), (20, 0.08, 0.28), (30, 0.16, 0.21)])
def test_feature_count_sweep(p, fa, cla_recall):
    near(m("robust", "false_alarm", p=p), fa)
    near(m("classical", "recall", p=p), cla_recall)


def test_no_more_tours_than_features_means_nothing_is_flagged_and_the_ranking_is_random():
    for n, p in ((20, 20), (20, 30), (30, 30)):
        for det in ("classical", "robust"):
            assert m(det, "false_alarm", n=n, p=p) == 0.0 and m(det, "recall", n=n, p=p) == 0.0
            aucs = [x.scores[det]["auc"] for x in runs(n=n, p=p)]
            assert 0.25 <= np.mean(aucs) <= 0.75                                                       # Rangfolge im Mittel zufällig
    everything = [x.scores[det]["auc"] for det in ("classical", "robust") for n, p in ((20, 20), (20, 30), (30, 30)) for x in runs(n=n, p=p)]
    assert min(everything) < 0.25 and max(everything) > 0.75                                           # Streuung von unter 0.25 bis über 0.75 (Einzelwerte hängen bei singulärer Kovarianz an Rundungsfehlern der Plattform)


# --- Seitenleiste: Betriebsarten, Krümmung, Rauschen ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("modes,rob_recall,rob_auc,cla_recall", [(1, 0.97, 1.00, 0.43), (2, 0.77, 0.95, 0.39), (3, 0.47, 0.85, 0.17)])
def test_modes_sweep(modes, rob_recall, rob_auc, cla_recall):
    near(m("robust", "recall", n_modes=modes), rob_recall)
    near(m("robust", "auc", n_modes=modes), rob_auc)
    near(m("classical", "recall", n_modes=modes), cla_recall)


@pytest.mark.parametrize("curv,fa,f1,cla_f1", [(0.0, 0.04, 0.84, 0.57), (0.25, 0.24, 0.49, 0.90), (0.5, 0.32, 0.41, 0.91), (0.75, 0.35, 0.39, 0.91), (1.0, 0.36, 0.38, 0.91)])
def test_curvature_sweep_the_robust_ellipse_overflags_while_the_ranking_stays_perfect(curv, fa, f1, cla_f1):
    near(m("robust", "false_alarm", curvature=curv), fa)
    near(m("robust", "f1", curvature=curv), f1)
    near(m("classical", "f1", curvature=curv), cla_f1)
    if curv > 0:
        near(m("robust", "auc", curvature=curv), 1.00, 0.01)
        assert m("classical", "f1", curvature=curv) > m("robust", "f1", curvature=curv) + 0.3


@pytest.mark.parametrize("noise,cla,rob", [(0.0, 0.87, 1.00), (0.25, 0.43, 0.97), (0.5, 0.42, 0.96), (1.0, 0.38, 0.92)])
def test_noise_sweep(noise, cla, rob):
    near(m("classical", "recall", noise=noise), cla)
    near(m("robust", "recall", noise=noise), rob)


# --- Seitenleiste: Anomalien ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c,cla,rob", [(2, 0.93, 1.00), (5, 0.75, 0.99), (10, 0.43, 0.97), (20, 0.14, 0.95), (30, 0.06, 0.93), (40, 0.04, 0.76)])
def test_contamination_sweep_classical_is_masked_robust_holds(c, cla, rob):
    near(m("classical", "recall", contamination=c), cla)
    near(m("robust", "recall", contamination=c), rob)


def test_robust_breaks_down_at_45_percent_scattered():
    near(m("robust", "recall", contamination=45), 0.40)
    near(m("robust", "auc", contamination=45), 0.87)
    near(m("robust", "auc", contamination=40), 0.97)


def test_dense_group_masks_more_and_breaks_the_robust_estimate_earlier():
    near(m("classical", "recall", kind="cluster"), 0.12)
    near(m("classical", "auc", kind="cluster"), 0.83)
    near(m("robust", "recall", kind="cluster"), 1.00)
    near(m("robust", "auc", kind="cluster"), 1.00)
    near(m("classical", "recall", kind="cluster", contamination=20), 0.03)
    near(m("classical", "auc", kind="cluster", contamination=20), 0.62)
    near(m("robust", "recall", kind="cluster", contamination=20), 1.00)
    near(m("robust", "auc", kind="cluster", contamination=40), 0.39)
    near(m("robust", "auc", kind="cluster", contamination=30), 0.49, 0.03)
    near(m("robust", "auc", kind="cluster", contamination=25), 1.00, 0.01)
    near(m("robust", "recall", kind="cluster", contamination=40), 0.04)
    near(m("classical", "auc", kind="cluster", contamination=40), 0.44)
    assert m("robust", "auc", kind="cluster", contamination=40) < 0.5 and m("classical", "auc", kind="cluster", contamination=40) < 0.5       # schlechter als Raten
    assert m("robust", "auc", kind="cluster", contamination=30) < m("robust", "auc", contamination=30) - 0.3


def test_anomalies_in_the_gap_are_never_found_by_either_detector():
    for n_modes in (2, 3):
        for det in ("classical", "robust"):
            assert 0.3 <= m(det, "auc", n_modes=n_modes, kind="gap") <= 0.5                       # unter 0.5: die Anomalien wirken normaler als die Normalen
    near(m("robust", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("classical", "auc", n_modes=2, kind="gap"), 0.40)
    assert m("robust", "recall", n_modes=2, kind="gap") < 0.1


@pytest.mark.parametrize("strength,rob,cla", [(3.0, 0.23, 0.12), (4.0, 0.62, 0.24), (6.0, 0.97, 0.43), (9.0, 1.00, 0.62), (12.0, 1.00, 0.68)])
def test_strength_sweep(strength, rob, cla):
    near(m("robust", "recall", strength=strength), rob)
    near(m("classical", "recall", strength=strength), cla)


# --- Robuste Schätzung: Stützanteil, Schwelle, Neugewichtung ----------------------------------------------------------------------------------


def test_support_fraction_at_10_and_20_percent():
    for h, r10 in ((0.5, 0.97), (0.6, 0.96), (0.75, 0.96), (0.9, 0.92), (1.0, 0.79)):
        near(m("robust", "recall", ev.Settings(support=h)), r10)
    for h, r20 in ((0.5, 0.95), (0.6, 0.96), (0.75, 0.94), (0.9, 0.45), (1.0, 0.20)):
        near(m("robust", "recall", ev.Settings(support=h), contamination=20), r20)


@pytest.mark.parametrize("q,fa,rec,f1", [(0.9, 0.11, 0.99, 0.67), (0.95, 0.06, 0.99, 0.77), (0.975, 0.04, 0.97, 0.84), (0.99, 0.02, 0.94, 0.89), (0.999, 0.00, 0.85, 0.90)])
def test_quantile_sweep(q, fa, rec, f1):
    s = ev.Settings(quantile=q)
    near(m("robust", "false_alarm", s), fa, 0.015)
    near(m("robust", "recall", s), rec)
    near(m("robust", "f1", s), f1)


def test_reweighting_lowers_false_alarms_and_the_covariance_error():
    on, off = ev.Settings(reweight=True), ev.Settings(reweight=False)
    near(m("robust", "false_alarm", on), 0.039, 0.006)
    near(m("robust", "false_alarm", off), 0.065, 0.006)
    near(m("robust", "f1", on), 0.84)
    near(m("robust", "f1", off), 0.76)
    near(m("robust", "shape_error", on), 0.06, 0.02)
    near(m("robust", "shape_error", off), 0.12, 0.03)


# --- Presets und Grenzen-Tabelle --------------------------------------------------------------------------------------------------------------


def test_preset_help_numbers():
    near(m("classical", "shape_error"), 1.7, 0.1)
    near(m("robust", "shape_error"), 0.06, 0.02)
    near(m("classical", "auc"), 0.95, 0.01)
    near(m("robust", "auc"), 1.00, 0.01)
    near(m("classical", "recall"), 0.43)
    near(m("robust", "recall"), 0.97)
    near(m("robust", "auc", n_modes=2, kind="gap"), 0.40)                                            # Lücke
    near(m("robust", "false_alarm", curvature=0.75), 0.35)                                             # gekrümmt
    near(m("robust", "f1", curvature=0.75), 0.39)
    near(m("classical", "f1", curvature=0.75), 0.91)
    near(m("classical", "recall", n=30), 0.00)                                                         # wenige Touren
    near(m("robust", "recall", n=30), 0.80)
    near(m("robust", "false_alarm", n=30), 0.24)


def test_limits_table_numbers():
    near(m("robust", "recall", n_modes=3), 0.47)
    near(m("robust", "false_alarm"), 0.04, 0.01)                                                        # ein Modus: 4 % statt nominell 2.5 %
    assert m("robust", "false_alarm") > 1.3 * (1 - C.DEFAULT_QUANTILE)
    near(m("robust", "false_alarm", curvature=0.75), 0.35)                                              # gut ein Drittel
    near(m("robust", "false_alarm", n=30), 0.24)
    near(m("robust", "false_alarm", p=30), 0.16)
    near(m("robust", "auc", contamination=5), 1.00, 0.01)                                              # Rangfolge perfekt ...
    near(m("robust", "f1", contamination=5), 0.71, 0.03)                                               # ... aber F1 nur 0.71 mit dem Standard-Quantil
    near(m("robust", "f1", ev.Settings(quantile=0.999)), 0.90)


def test_share_threshold_beats_the_chi2_quantile_at_few_anomalies():
    t = ev.threshold_table()
    by = {o["contamination"]: o for o in t["oracle"]}
    near(by[5]["chi2_f1"], 0.71, 0.03)
    near(by[5]["share_f1"], 0.91, 0.03)
    near(by[10]["chi2_f1"], 0.84, 0.03)
    near(by[20]["chi2_f1"], 0.92, 0.03)
    assert by[5]["share_f1"] - by[5]["chi2_f1"] > 0.15 > by[20]["share_f1"] - by[20]["chi2_f1"]


def test_breakdown_table_matches_the_chart_texts():
    b = ev.breakdown_table()
    by = {r["x"]: r for r in b["kinds"]["scattered"]}
    near(by[40]["robust_auc"], 0.97)
    near(by[45]["robust_recall"], 0.40)
    assert by[45]["classical_auc"] < 0.8
    cl = {r["x"]: r for r in b["kinds"]["cluster"]}
    assert cl[20]["robust_auc"] > 0.99 and cl[25]["robust_auc"] > 0.99 and cl[30]["robust_auc"] < 0.55 and cl[40]["robust_auc"] < 0.5
    sup = {r["x"]: r for r in b["support"]}
    assert sup[0.5]["robust_recall"] > 0.9 > sup[0.9]["robust_recall"] > sup[1.0]["robust_recall"]
    assert sup[1.0]["robust_center_error"] > 3 * sup[0.6]["robust_center_error"]


def test_dimension_table_has_the_expected_extremes():
    cells = {(c["n"], c["p"]): c for c in ev.dimension_table()}
    assert max(c["robust_false_alarm"] for c in cells.values()) < 0.32 and max(c["robust_false_alarm"] for c in cells.values()) > 0.28
    assert cells[(400, 2)]["robust_false_alarm"] < 0.04 and cells[(20, 20)]["classical_recall"] == 0.0 and cells[(100, 2)]["classical_recall"] > 0.85


def test_modes_table_covers_eight_cells_and_gap_needs_two_modes():
    rows = ev.modes_table()
    assert len(rows) == 8 and not any(r["n_modes"] == 1 and r["kind"] == "gap" for r in rows)


def test_affine_equivariance_units_do_not_matter():
    """Die Abstände sind gegen Einheiten und Verschiebung unempfindlich - anders als die PCA der Vorgänger-Demo."""
    ds = ev.make_dataset()
    base = ev.analyse(ds)
    rng = np.random.default_rng(3)
    A = np.diag(rng.uniform(1e-3, 1e3, ds.p))
    ds2 = type(ds)(X=ds.X @ A + 7.0, z=ds.z, anomaly=ds.anomaly, mode=ds.mode, n_modes=ds.n_modes, kind=ds.kind, p=ds.p)
    other = ev.analyse(ds2)
    assert np.allclose(other.classical.d2, base.classical.d2, rtol=1e-6, atol=1e-6)
    assert np.allclose(other.robust.d2, base.robust.d2, rtol=1e-5, atol=1e-5)
    assert other.scores["robust"]["auc"] == pytest.approx(base.scores["robust"]["auc"]) and other.scores["robust"]["recall"] == base.scores["robust"]["recall"]


def test_two_modes_still_find_far_dense_groups_and_three_modes_gap_is_below_chance():
    near(m("robust", "recall", n_modes=2, kind="cluster"), 1.00, 0.02)
    near(m("robust", "auc", n_modes=3, kind="gap"), 0.38)
    near(m("classical", "auc", n_modes=3, kind="gap"), 0.38)


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, contamination=45))
    assert a.seconds["robust"] < 3.0 and a.seconds["classical"] < 0.5
