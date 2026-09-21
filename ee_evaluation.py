"""Auswertung der Elliptic-Envelope-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate), Genauigkeit der Schätzung (Zentrum, Kovarianz),
Analyse einer Aufnahme für den klassischen und den robusten Schätzer, Sweeps und Experimente auf Abruf, Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import ee_algorithm as alg
import ee_constants as C
import ee_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


def covariance_error(cov, reference):
    return float(np.linalg.norm(cov - reference) / np.linalg.norm(reference))


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    support: float = C.DEFAULT_SUPPORT
    quantile: float = C.DEFAULT_QUANTILE
    reweight: bool = C.DEFAULT_REWEIGHT
    start: int = 0                              # Seed der zufälligen MCD-Starts (entkoppelt vom Seed der Aufnahme)


DATA_KEYS = ("n", "p", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE, contamination=C.DEFAULT_CONTAMINATION,
                    kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE, contamination=C.DEFAULT_CONTAMINATION,
                 kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_modes, curvature, noise, contamination, kind, strength, seed)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # "classical"/"robust" -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold, center_error, shape_error)
    reference_center: np.ndarray  # Mittel der wahren Normalen
    reference_cov: np.ndarray     # Kovarianz der wahren Normalen
    seconds: dict


def _detector_scores(fit, ds, settings, mu_ref, cov_ref):
    thr = alg.threshold(fit, settings.quantile)
    m = flag_metrics(fit.d2 > thr, ds.anomaly)
    scale = float(np.sqrt(np.trace(cov_ref) / len(cov_ref)))
    m.update(auc=roc_auc(fit.d2, ds.anomaly), ap=average_precision(fit.d2, ds.anomaly), threshold=thr,
             center_error=float(np.linalg.norm(fit.location - mu_ref) / scale), shape_error=covariance_error(fit.covariance, cov_ref))
    return m


def analyse(ds, settings=Settings(), params=None):
    secs = {}
    t0 = time.perf_counter()
    classical = alg.fit_classical(ds.X)
    secs["classical"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    robust = alg.fit_mcd(ds.X, settings.support, settings.start, reweight=settings.reweight)
    secs["robust"] = time.perf_counter() - t0
    normal = ds.X[~ds.anomaly]
    mu_ref, cov_ref = alg._mean_cov(normal)
    scores = {"classical": _detector_scores(classical, ds, settings, mu_ref, cov_ref), "robust": _detector_scores(robust, ds, settings, mu_ref, cov_ref)}
    return Analysis(ds, settings, params, classical, robust, scores, mu_ref, cov_ref, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params)


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "support": (0.5, 0.6, 0.75, 0.9, 1.0),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "support": "Stützanteil h/n der robusten Schätzung", "quantile": "chi²-Quantil der Schwelle"}
SETTING_PARAMETERS = ("support", "quantile")
DETECTORS = ("classical", "robust")
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm", "center_error", "shape_error")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(analyse(make_dataset(seed=s, **{**DEFAULT_DATA, **kw}), settings)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen beider Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


BREAKDOWN_CONTAMINATION = (2, 5, 10, 15, 20, 25, 30, 35, 40, 45)
BREAKDOWN_SUPPORTS = (0.5, 0.6, 0.75, 0.9, 1.0)


def breakdown_table(settings=Settings(), **base):
    """Bruchpunkt: AUC und Recall beider Detektoren über den Anteil der Anomalien, für verstreute Ausreißer und eine dichte Gruppe abseits; dazu der Einfluss des Stützanteils h/n bei 20 % Anomalien."""
    out = {"contamination": BREAKDOWN_CONTAMINATION, "kinds": {}, "support": []}
    for kind in ("scattered", "cluster"):
        rows = []
        for c in BREAKDOWN_CONTAMINATION:
            row = _mean_over_seeds(settings, **{**base, "kind": kind, "contamination": c, "n_modes": 1})
            row["x"] = c
            rows.append(row)
        out["kinds"][kind] = rows
    for h in BREAKDOWN_SUPPORTS:
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "support": h}), **{**base, "kind": "scattered", "contamination": 20, "n_modes": 1})
        row["x"] = h
        out["support"].append(row)
    return out


def modes_table(settings=Settings(), **base):
    """Betriebsarten × Art der Anomalien: AUC beider Detektoren (die Art 'Lücke' gibt es erst ab zwei Betriebsarten)."""
    rows = []
    for n_modes in (1, 2, 3):
        for kind in C.KINDS:
            if kind == "gap" and n_modes == 1:
                continue
            row = _mean_over_seeds(settings, **{**base, "n_modes": n_modes, "kind": kind})
            row.update(n_modes=n_modes, kind=kind)
            rows.append(row)
    return rows


THRESHOLD_QUANTILES = (0.9, 0.95, 0.975, 0.99, 0.999)
THRESHOLD_CONTAMINATIONS = (5, 10, 20)


def threshold_table(settings=Settings(), **base):
    """Schwelle: Precision, Recall, Fehlalarmrate und F1 beider Detektoren je chi²-Quantil; dazu der F1 der robusten Schätzung mit dem chi²-Quantil gegen eine Schwelle, die den wahren Anomalieanteil kennt."""
    rows = sweep("quantile", THRESHOLD_QUANTILES, settings, **base)
    oracle = []
    for c in THRESHOLD_CONTAMINATIONS:
        chi_f1, share_f1 = [], []
        for s in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=s, **{**DEFAULT_DATA, **base, "contamination": c}), settings)
            k = int(a.ds.anomaly.sum())
            top = np.zeros(a.ds.n, bool)
            top[np.argsort(-a.robust.d2)[:k]] = True
            share_f1.append(flag_metrics(top, a.ds.anomaly)["f1"])
            chi_f1.append(a.scores["robust"]["f1"])
        oracle.append({"contamination": c, "chi2_f1": float(np.mean(chi_f1)), "share_f1": float(np.mean(share_f1))})
    return {"rows": rows, "oracle": oracle}


DIMENSION_N = (20, 30, 50, 100, 200, 400)
DIMENSION_P = (2, 5, 12, 20, 30)


def dimension_table(settings=Settings(), **base):
    """Hohe Dimension: robuste Fehlalarmrate und klassischer Recall über die Tourenzahl n und die Merkmalszahl p."""
    cells = []
    for n in DIMENSION_N:
        for p in DIMENSION_P:
            row = _mean_over_seeds(settings, **{**base, "n": n, "p": p})
            row.update(n=n, p=p)
            cells.append(row)
    return cells


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

BREAKDOWN_MIN_CONTAMINATION = 25       # Anteil [%], ab dem die robuste Schätzung ihren Bruchpunkt erreichen kann
BREAKDOWN_AUC = 0.9
GAP_AUC = 0.6
MODE_RECALL_DROP = 0.15
CURVED_FALSE_ALARM = 0.15
CURVED_MIN = 0.2
SAMPLES_PER_FEATURE = 10
THRESHOLD_FALSE_ALARM_FACTOR = 3.0     # Fehlalarmrate über dem Dreifachen der nominellen (1 - Quantil) plus Puffer
THRESHOLD_FALSE_ALARM_BUFFER = 0.02
MASKED_RECALL_GAP = 0.2


def verdict(a):
    """(Art, Code, Kennzahlen): der erste zutreffende Befund in der Reihenfolge Bruchpunkt, Lücke, mehrere Betriebsarten, gekrümmter Normalbereich, zu wenige Touren, falsche Schwelle, Maskierung, sonst gleichauf."""
    ds, s = a.ds, a.settings
    c, r = a.scores["classical"], a.scores["robust"]
    n_anom = int(ds.anomaly.sum())
    contamination = 100.0 * n_anom / ds.n
    nominal = 1.0 - s.quantile
    data = {"n": ds.n, "p": ds.p, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": contamination, "n_anomalies": n_anom, "quantile": s.quantile, "nominal": nominal,
            **{f"classical_{k}": v for k, v in c.items()}, **{f"robust_{k}": v for k, v in r.items()}}
    if contamination >= BREAKDOWN_MIN_CONTAMINATION and r["auc"] < BREAKDOWN_AUC:
        return "warning", "beyond_breakdown", data
    if ds.kind == "gap" and r["auc"] < GAP_AUC:
        return "warning", "gap", data
    if ds.n_modes >= 2 and a.params is not None:
        ref = list(a.params)
        ref[2] = 1
        ref_recall = analyse_for(tuple(ref), s).scores["robust"]["recall"]
        data["reference_recall"] = ref_recall
        if ref_recall - r["recall"] > MODE_RECALL_DROP:
            return "warning", "multi_modal", data
    if a.params is not None and a.params[3] >= CURVED_MIN and r["false_alarm"] > CURVED_FALSE_ALARM:
        return "warning", "not_convex", data
    if ds.n < SAMPLES_PER_FEATURE * ds.p:
        return "warning", "too_few_samples", data
    if r["false_alarm"] > THRESHOLD_FALSE_ALARM_FACTOR * nominal + THRESHOLD_FALSE_ALARM_BUFFER:
        return "warning", "threshold_off", data
    if r["recall"] - c["recall"] >= MASKED_RECALL_GAP:
        return "success", "masked", data
    return "success", "comparable", data


# --- Ansichten für die App ---------------------------------------------------------------------------------------------------------------


def flagged(a, detector):
    """Boolesche Markierung der Touren als Anomalie beim gewählten Schwellenquantil."""
    fit = a.classical if detector == "classical" else a.robust
    return fit.d2 > a.scores[detector]["threshold"]
