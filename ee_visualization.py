"""Plotly-Visualisierungen der Elliptic-Envelope-Demo: Touren in der Ebene der größten Streuung mit den Ellipsen der beiden Schätzer, C-Schritte der robusten Schätzung, Verteilung der
Mahalanobis-Abstände, ROC-Kurven, Kennzahlen-Balken, Sweeps, Bruchpunkt, Betriebsarten, Schwellen und die Tabelle über Tourenzahl und Merkmalszahl. Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ee_algorithm as alg
import ee_constants as C

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f"
DETECTOR_COLORS = {"classical": ORANGE, "robust": TEAL}
DETECTOR_NAMES = {"classical": "klassisch", "robust": "robust (MCD)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Projektion -----------------------------------------------------------------------------------------------------------------


def standardise(X, *fits):
    """Standardisierte Koordinaten (Mittelwert 0, Streuung 1 je Merkmal) und die Schätzer darin - die Mahalanobis-Abstände ändern sich dadurch nicht (affine Äquivarianz), die Einheiten der Merkmale spielen keine Rolle."""
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    Z = (X - m) / s
    out = [(f.location - m) / s for f in fits], [f.covariance / np.outer(s, s) for f in fits]
    return Z, out[0], out[1]


def projection(X, classical, robust):
    """Punkte und beide Schätzer in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (in standardisierten Merkmalen): (Punkte n x 2, Achsen, Zentren, Kovarianzen)."""
    Z, mus, covs = standardise(X, classical, robust)
    axes = alg.projection_axes(covs[1])
    return Z @ axes, axes, mus, covs


def _scatter_traces(P, anomaly, flagged=None, subset=None):
    traces = []
    normal = ~anomaly
    traces.append(go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"))
    traces.append(go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)"))
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip",
                                 name="als Anomalie markiert"))
    if subset is not None:
        traces.append(go.Scatter(x=P[subset, 0], y=P[subset, 1], mode="markers", marker=dict(size=11, color=TEAL, line=dict(width=2), symbol="circle-open"), hoverinfo="skip",
                                 name="Teilmenge h"))
    return traces


def build_scatter(P, anomaly, ellipses=(), flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert); `ellipses` = [(Punkte, Farbe, Name, gestrichelt), ...]."""
    fig = go.Figure(_scatter_traces(P, anomaly, flagged))
    for pts, color, name, dash in ellipses:
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, width=3, dash="dash" if dash else "solid"), hoverinfo="skip", name=name))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen) - was die Anomalien in den Kennzahlen selbst tun."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_scatter_traces(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_csteps(P, anomaly, subsets, history):
    """C-Schritte der besten Startlösung: links die Determinante der Kovarianz der Teilmenge je Schritt (sinkt monoton), rechts die gewählte Teilmenge (offene Kreise) beim ersten, mittleren und letzten Schritt."""
    picks = sorted({0, len(subsets) // 2, len(subsets) - 1})
    fig = make_subplots(rows=1, cols=1 + len(picks), column_widths=[0.28] + [0.72 / len(picks)] * len(picks), subplot_titles=["log det je C-Schritt"] + [f"C-Schritt {k + 1}" for k in picks], horizontal_spacing=0.06)
    fig.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=history, mode="lines+markers", line=dict(color=TEAL, width=3), hoverinfo="skip", showlegend=False), row=1, col=1)
    for c, k in enumerate(picks):
        for tr in _scatter_traces(P, anomaly, subset=subsets[k]):
            tr.showlegend = False
            fig.add_trace(tr, row=1, col=c + 2)
    fig.update_xaxes(title_text="Schritt", row=1, col=1, dtick=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_distances(d2, anomaly, thr, name):
    """Histogramm der Mahalanobis-Abstände (Wurzel von d²) der normalen und der Sonderfahrten; senkrechte Linie = Schwelle (chi²-Quantil)."""
    dist = np.sqrt(d2)
    hi = float(max(dist.max(), np.sqrt(thr) * 1.2))
    bins = dict(start=0.0, end=hi, size=hi / 40.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dist[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.7, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=dist[anomaly], xbins=bins, marker_color=RED, opacity=0.8, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(np.sqrt(thr)), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=340, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=f"Mahalanobis-Abstand ({name})"), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) beider Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen beider Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in ("classical", "robust"):
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=["–" if np.isnan(v) else f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


def _band(fig, xs, row, key, color, col):
    y, sd = np.array([r[key] for r in row]), np.array([r[key + "_std"] for r in row])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC beider Detektoren (mit Streuung über die Sweep-Datensätze), rechts Trefferquote (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge der Abstände)", "Recall und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in ("classical", "robust"):
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_recall"] for r in rows], mode="lines+markers", name=f"Recall {DETECTOR_NAMES[det]}", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", name=f"Fehlalarm {DETECTOR_NAMES[det]}", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip",
                                 showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_shape_sweep(rows, xlabel):
    """Fehler der geschätzten Kovarianz gegen die der wahren Normalen (relative Frobenius-Norm, logarithmisch) für beide Schätzer."""
    fig = go.Figure()
    xs = [r["x"] for r in rows]
    for det in ("classical", "robust"):
        fig.add_trace(go.Scatter(x=xs, y=[max(r[f"{det}_shape_error"], 1e-3) for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=DETECTOR_COLORS[det], width=3), hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=xlabel), yaxis=dict(title="relativer Fehler der Kovarianz", type="log"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_breakdown(table):
    """Bruchpunkt: AUC (durchgezogen) und Recall (gestrichelt) über den Anteil der Anomalien für verstreute Ausreißer (links) und eine dichte Gruppe abseits (rechts)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("verstreute Ausreißer", "dichte Gruppe abseits"), horizontal_spacing=0.1, shared_yaxes=True)
    for col, kind in ((1, "scattered"), (2, "cluster")):
        rows = table["kinds"][kind]
        xs = [r["x"] for r in rows]
        for det in ("classical", "robust"):
            color = DETECTOR_COLORS[det]
            fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=f"AUC {DETECTOR_NAMES[det]}", line=dict(color=color, width=3), hoverinfo="skip", showlegend=col == 1), row=1, col=col)
            fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_recall"] for r in rows], mode="lines+markers", name=f"Recall {DETECTOR_NAMES[det]}", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip",
                                     showlegend=col == 1), row=1, col=col)
        fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=col)
    fig.update_xaxes(title="Anteil der Anomalien [%]")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_support(rows):
    """Stützanteil h/n bei 20 % verstreuten Anomalien: Recall und Zentrumsfehler der robusten Schätzung (h/n = 1: alle Punkte, nur die Neugewichtung wirkt)."""
    xs = [r["x"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Recall der robusten Schätzung", "Fehler des geschätzten Zentrums"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["robust_recall"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["robust_center_error"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="Stützanteil h/n")
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_modes(rows):
    """AUC beider Detektoren je Anzahl Betriebsarten und Art der Anomalien (Balken; die Linie bei 0.5 ist Raten)."""
    labels = [f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}<br>{KIND_NAMES[r['kind']]}" for r in rows]
    fig = go.Figure()
    for det in ("classical", "robust"):
        y = [r[f"{det}_auc"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=380, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="AUC", range=[0, 1.15]), legend=dict(orientation="h", y=-0.45))
    return lock_axes(fig)


def build_threshold(rows):
    """Precision, Recall, Fehlalarmrate und F1 der robusten Schätzung über das chi²-Quantil der Schwelle."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("robust_precision", "Precision", GREEN, "solid"), ("robust_recall", "Recall", TEAL, "solid"), ("robust_false_alarm", "Fehlalarmrate", RED, "dash"), ("robust_f1", "F1", PURPLE, "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3 if key == "robust_f1" else 2, dash=dash), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="chi²-Quantil der Schwelle", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_oracle(oracle):
    """F1 der robusten Schätzung mit dem chi²-Quantil (Standard) gegen eine Schwelle, die den wahren Anomalieanteil kennt (die k größten Abstände)."""
    xs = [f"{o['contamination']} % Anomalien" for o in oracle]
    fig = go.Figure()
    for key, name, color in (("chi2_f1", "chi²-Quantil", TEAL), ("share_f1", "wahrer Anteil bekannt", PURPLE)):
        y = [o[key] for o in oracle]
        fig.add_trace(go.Bar(x=xs, y=y, name=name, marker_color=color, text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="F1", range=[0, 1.15]), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_dimension(cells):
    """Links die Fehlalarmrate der robusten Schätzung, rechts der Recall der klassischen über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten); dunkler = größerer Wert."""
    ns = sorted({c["n"] for c in cells})
    ps = sorted({c["p"] for c in cells})
    grid = {(c["n"], c["p"]): c for c in cells}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Fehlalarmrate (robust)", "Recall (klassisch)"), horizontal_spacing=0.14)
    for col, key, scale in ((1, "robust_false_alarm", "Reds"), (2, "classical_recall", "Blues")):
        z = [[grid[(n, p)][key] for p in ps] for n in ns]
        fig.add_trace(go.Heatmap(z=z, x=[f"p = {p}" for p in ps], y=[f"n = {n}" for n in ns], colorscale=scale, zmin=0, zmax=1, text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", showscale=False,
                                 hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)
