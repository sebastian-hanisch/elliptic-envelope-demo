"""Elliptic Envelope - Anomalien als Punkte außerhalb einer robusten Ellipse - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Elliptic Envelope (robuste Gauß-Anpassung per Minimum Covariance Determinant) -
und lässt stattdessen das Beispiel wachsen. Erstes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: die einfache Baseline, an deren Schwächen (ein einziger elliptischer Normalbereich,
Gauß'sche Daten, viele Touren im Verhältnis zu den Merkmalen) die späteren Stücke ansetzen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ee_algorithm as alg
import ee_constants as C
from ee_evaluation import SWEEP_LABELS, Settings, analyse_for, breakdown_table, dimension_table, flagged, modes_table, roc_curve, sweep, threshold_table, verdict
from ee_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ee_visualization import (
    DETECTOR_COLORS,
    build_breakdown,
    build_csteps,
    build_dimension,
    build_distances,
    build_features,
    build_method_bars,
    build_modes,
    build_oracle,
    build_roc,
    build_scatter,
    build_shape_sweep,
    build_support,
    build_sweep,
    build_threshold,
    projection,
)

st.set_page_config(page_title="Elliptic Envelope – Sebastian Hanisch", layout="wide")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    return sweep(parameter, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _breakdown(base, settings):
    return breakdown_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _modes(base, settings):
    return modes_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _dimension(base, settings):
    return dimension_table(settings, **dict(base))


st.title("📏 Elliptic Envelope – Anomalien außerhalb einer robusten Ellipse")
st.markdown(
    """
Wie erkennt man Sonderfahrten, ohne vorher zu wissen, wie sie aussehen? Der **Elliptic Envelope** ist die einfachste Antwort: die normalen Touren werden als **eine Gauß'sche Wolke** beschrieben - Mittelwert $\\mu$ und Kovarianz $\\Sigma$ -,
und eine Tour gilt als Anomalie, wenn ihr **Mahalanobis-Abstand** $d^2 = (x-\\mu)^\\top \\Sigma^{-1} (x-\\mu)$ über einem $\\chi^2$-Quantil liegt, also **außerhalb der Ellipse**.
Der Haken: Mittelwert und Kovarianz werden aus denselben Daten geschätzt, in denen die Anomalien stecken - viele Ausreißer verzerren die Ellipse so, dass sie sich selbst verstecken (**Maskierung**).
Die robuste Antwort ist der **Minimum Covariance Determinant (MCD)**: die Ellipse wird nur aus der dichtesten Hälfte der Touren geschätzt.
Was das bringt, wo es endet und welche Annahme dahinter steckt (**ein** elliptischer Normalbereich), misst diese Demo - mit Siegen und Niederlagen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - erstes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Linie hat keinen Konvergenzpunkt: die nächsten Stücke (lokale Dichte, Kernel-Grenze, zufällige Bäume, ...) beheben jeweils **eine** Schwäche dieser Wurzel auf einem anderen Weg. "
    "Die Touren sind dieselben Lieferrouten-Kennzahlen wie in der PCA-Demo; dort drehen Sonderfahrten die erste Hauptachse, hier sollen sie gefunden werden."
)

with st.expander("So funktioniert der Elliptic Envelope", expanded=True):
    st.markdown(
        """
1. **Klassisch.** Mittelwert und Kovarianz aller Touren, Abstand $d^2$ jeder Tour. Sind die normalen Touren Gauß'sch, ist $d^2$ wie $\\chi^2$ mit $p$ Freiheitsgraden verteilt - die Schwelle ist das gewählte Quantil (z. B. 97,5 %), und bei sauberen Daten werden dann nominell 2,5 % der normalen Touren als Anomalie markiert.
2. **Maskierung.** Mit 10 % oder 20 % Anomalien werden $\\mu$ und $\\Sigma$ selbst verschoben und aufgebläht: die Anomalien sind jetzt Teil der "normalen" Wolke und fallen nicht mehr auf.
3. **Robust (MCD).** Gesucht ist die Teilmenge von $h$ Touren (mindestens die Hälfte), deren Kovarianzmatrix die **kleinste Determinante** hat. FastMCD findet sie über **C-Schritte**: aus einer zufälligen Startmenge die Abstände aller Touren berechnen, die $h$ kleinsten nehmen, neu schätzen - die Determinante sinkt in jedem Schritt.
   500 zufällige Starts mit je zwei C-Schritten, die besten zehn laufen weiter, die beste Lösung zählt; danach eine **Konsistenz-Korrektur** (Gauß'sche Daten) und eine **Neugewichtung** mit den Touren innerhalb der Schwelle.
4. **Was vorausgesetzt wird:** ein einziger elliptischer, konvexer Normalbereich, ungefähr Gauß'sche Verteilung, deutlich mehr Touren als Merkmale und weniger als die Hälfte Anomalien. Was **nicht** vorausgesetzt wird: gleiche Einheiten - der Abstand ist **affin äquivariant**, Meter und Minuten zählen gleich (anders als bei der PCA-Einheitenfalle).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren (Zeilen). Bei 10 % Anomalien und 12 Merkmalen ist die Fehlalarmrate der robusten Schätzung 0.16 / 0.24 / 0.29 / 0.21 / 0.06 bei 20 / 30 / 50 / 100 / 200 Touren (nominell 0.025); "
             "der klassische Recall ist 0.00 / 0.00 / 0.20 / 0.42 / 0.44. Der klassische Abstand kann bei n Touren höchstens (n−1)²/n erreichen: bei 20 Touren 18.05 - unter der Schwelle 23.3 (12 Merkmale).",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). Bei 300 Touren und 10 % Anomalien: Fehlalarmrate der robusten Schätzung 0.03 / 0.02 / 0.03 / 0.04 / 0.08 / 0.16 bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen, "
             "klassischer Recall 0.84 / 0.73 / 0.57 / 0.43 / 0.28 / 0.21 - je mehr Merkmale, desto schlechter schätzbar die Kovarianz.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. Eine Ellipse passt nur zu einer: bei 1 / 2 / 3 Betriebsarten (10 % verstreute Anomalien) findet die robuste Schätzung 97 % / 77 % / 47 % der Anomalien "
             "(AUC 1.00 / 0.95 / 0.85), die klassische 43 % / 39 % / 17 %.",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 bleibt die Rangfolge perfekt (AUC 1.00), aber die Fehlalarmrate der robusten Schätzung steigt auf 0.04 / 0.24 / 0.32 / 0.35 / 0.36 "
             "und ihr F1 fällt auf 0.84 / 0.49 / 0.41 / 0.39 / 0.38 - die klassische, breitere Ellipse kommt auf F1 0.57 / 0.90 / 0.91 / 0.91 / 0.91.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen (relativ zu den versteckten Faktoren). Bei 0 / 0.25 / 0.5 / 1.0: Recall klassisch 0.87 / 0.43 / 0.42 / 0.38, robust 1.00 / 0.97 / 0.96 / 0.92.",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind (mindestens eine). Bei verstreuten Ausreißern: Recall klassisch 0.93 / 0.75 / 0.43 / 0.14 / 0.06 / 0.04 bei 2 / 5 / 10 / 20 / 30 / 40 %, robust 1.00 / 0.99 / 0.97 / 0.95 / 0.93 / 0.76; "
             "bei 45 % bricht auch die robuste Schätzung ein (Recall 0.40, AUC 0.87).",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe: alle beieinander - das maskiert am stärksten (bei 10 %: Recall klassisch 0.12, AUC 0.83; robust 1.00 / 1.00). "
             "In der Lücke (ab zwei Betriebsarten): zwischen den Betriebsarten, wo die eine Ellipse alles für normal hält - AUC 0.40 für beide Detektoren.",
    )
    if kind != "gap":
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind (in Standardabweichungen der versteckten Faktoren). Bei 3 / 4 / 6 / 9 / 12: Recall robust 0.23 / 0.62 / 0.97 / 1.00 / 1.00, klassisch 0.12 / 0.24 / 0.43 / 0.62 / 0.68 - "
                 "je näher, desto schwerer für beide.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**Robuste Schätzung**")
    support = st.slider(
        "Stützanteil h/n", *bounds("support_slider"), key="support_slider", step=0.05,
        help="Welcher Anteil der Touren die Ellipse bestimmt (0.5 = größter Bruchpunkt, 1 = alle: dann wirkt nur noch die Neugewichtung). Bei 10 % Anomalien: Recall 0.97 / 0.96 / 0.96 / 0.92 / 0.79 für h/n 0.5 / 0.6 / 0.75 / 0.9 / 1.0, "
             "bei 20 %: 0.95 / 0.96 / 0.94 / 0.45 / 0.20 - weniger Stützpunkte sind robuster.",
    )
    quantile = st.slider(
        "Schwelle: chi²-Quantil", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
        help="Ab welchem Anteil der chi²-Verteilung eine Tour als Anomalie gilt. Bei 10 % Anomalien: Fehlalarmrate der robusten Schätzung 0.11 / 0.06 / 0.04 / 0.02 / 0.00, Recall 0.99 / 0.99 / 0.97 / 0.94 / 0.85 bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999; "
             "F1 0.67 / 0.77 / 0.84 / 0.89 / 0.90. Die robuste Schätzung markiert mehr Normale als nominell - eine höhere Schwelle gleicht das aus.",
    )
    reweight = st.toggle(
        "Neugewichtung", key="reweight_toggle",
        help="Nach dem MCD werden Mittelwert und Kovarianz noch einmal aus allen Touren innerhalb der Schwelle geschätzt. An: Fehlalarmrate 0.039, F1 0.84, Fehler der Kovarianz 0.06; aus: 0.065, 0.76, 0.12 (10 % Anomalien, Mittel über fünf Aufnahmen).",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "support_slider": float(support), "quantile_slider": float(quantile),
    "reweight_toggle": int(bool(reweight)), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(support=float(round(support, 2)), quantile=float(quantile), reweight=bool(reweight))
with st.spinner("Schätze..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
cs, rs = a.scores["classical"], a.scores["robust"]
n_anom = int(ds.anomaly.sum())
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)), "contamination": int(contamination),
                          "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- Elliptic Envelope in Aktion --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Elliptic Envelope in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Klassische Ellipse", 3: "3 · MCD: C-Schritte", 4: "4 · Robuste Ellipse", 5: "5 · Ergebnis"}
if "ee_step" not in st.session_state or st.session_state.get("ee_step_owner") != data_key:
    st.session_state["ee_step"] = 1
    st.session_state["ee_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ee_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes, mus, covs = projection(ds.X, a.classical, a.robust)
ell_classical = alg.ellipse_points(mus[0], covs[0], a.scores["classical"]["threshold"], axes)
ell_robust = alg.ellipse_points(mus[1], covs[1], a.scores["robust"]["threshold"], axes)
flag_c, flag_r = flagged(a, "classical"), flagged(a, "robust")


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {ds.names[0]} gegen {ds.names[min(1, ds.p - 1)]}** (Einheiten wie gemessen)")
            c2.plotly_chart(build_features(ds.X, ds.anomaly, ds.names, 0, 1), width="stretch", key="step_features")
        elif current_step == 2:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Klassische Ellipse** (Mittelwert und Kovarianz aller Touren, Schwelle = {settings.quantile:.1%}-Quantil; Kreise = markiert)")
            c1.plotly_chart(build_scatter(P, ds.anomaly, [(ell_classical, DETECTOR_COLORS["classical"], "klassisch", False)], flag_c), width="stretch", key="step_classical")
            c2.markdown("**Abstände der klassischen Schätzung**")
            c2.plotly_chart(build_distances(a.classical.d2, ds.anomaly, cs["threshold"], "klassisch"), width="stretch", key="step_classical_hist")
        elif current_step == 3:
            st.markdown(f"**C-Schritte der besten von {C.MCD_STARTS} Startlösungen (die besten {C.MCD_KEEP} laufen bis zur Konvergenz)** (h = {a.robust.h} von {ds.n} Touren; offene Kreise = die Teilmenge, aus der die Ellipse geschätzt wird)")
            st.plotly_chart(build_csteps(P, ds.anomaly, a.robust.subsets, a.robust.logdet_history), width="stretch", key="step_csteps")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            c1.markdown("**Robuste Ellipse** (durchgezogen) gegen die klassische (gestrichelt)")
            c1.plotly_chart(build_scatter(P, ds.anomaly, [(ell_classical, DETECTOR_COLORS["classical"], "klassisch", True), (ell_robust, DETECTOR_COLORS["robust"], "robust (MCD)", False)], flag_r), width="stretch",
                            key="step_robust")
            c2.markdown("**Abstände der robusten Schätzung**")
            c2.plotly_chart(build_distances(a.robust.d2, ds.anomaly, rs["threshold"], "robust"), width="stretch", key="step_robust_hist")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(x.d2, ds.anomaly), a.scores[d]["auc"]) for d, x in (("classical", a.classical), ("robust", a.robust))}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {ds.p} Kennzahlen, davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), {ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. "
               "Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung in standardisierten Kennzahlen; der Detektor selbst sieht alle Kennzahlen.")
elif step == 2:
    st.caption(f"Die Ellipse wird aus allen Touren geschätzt. Sie findet {_pct(cs['recall'])} der Sonderfahrten (AUC {cs['auc']:.2f}) bei einer Fehlalarmrate von {cs['false_alarm']:.1%} (nominell {1 - settings.quantile:.1%}); "
               f"der Fehler ihrer Kovarianz gegen die der wahren Normalen ist {cs['shape_error']:.2f} - die Anomalien haben sie aufgebläht.")
elif step == 3:
    st.caption(f"{a.robust.n_converged} der {C.MCD_KEEP} weiterverfolgten Starts sind konvergiert; die Determinante der Teilmengen-Kovarianz sinkt in jedem C-Schritt, bis sich die Teilmenge nicht mehr ändert. "
               "Die Teilmenge liegt im dichten Kern - die Sonderfahrten sind nicht dabei, deshalb beeinflussen sie die Ellipse nicht.")
elif step == 4:
    st.caption(f"Robust: Recall {_pct(rs['recall'])}, AUC {rs['auc']:.2f}, Fehlalarmrate {rs['false_alarm']:.1%} (nominell {1 - settings.quantile:.1%}); Fehler der Kovarianz {rs['shape_error']:.2f} "
               f"(klassisch {cs['shape_error']:.2f}). Schwelle für d²: {rs['threshold']:.1f} ({a.robust.dof} Freiheitsgrade).")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge der Abstände (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge (AUC 1.00) kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Detektor gefunden hat – klassisch gegen robust")
st.caption(
    "Anomalie = Tour mit d² über dem chi²-Quantil der Schwelle. **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Abstand hat als eine zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). "
    "**Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (robust)", f"{rs['auc']:.2f}", delta=f"klassisch {cs['auc']:.2f}", delta_color="off", help="Rangfolge der Mahalanobis-Abstände; im Delta der klassische Detektor.")
m2.metric("Recall (robust)", _pct(rs["recall"]), delta=f"klassisch {_pct(cs['recall'])}", delta_color="off", help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m3.metric("Fehlalarmrate (robust)", f"{rs['false_alarm']:.1%}", delta=f"nominell {1 - settings.quantile:.1%}", delta_color="off",
          help="Anteil der normalen Touren, die als Anomalie markiert werden; nominell (1 − Quantil), wenn die Normalen Gauß'sch und die Kovarianz gut geschätzt sind.")
m4.metric("Fehler der Kovarianz (robust)", f"{rs['shape_error']:.2f}", delta=f"klassisch {cs['shape_error']:.2f}", delta_color="off",
          help="Relative Frobenius-Norm der Abweichung von der Kovarianz der wahren normalen Touren (Wahrheit aus dem Generator).")

_t = vd
if code == "masked":
    st.success(f"✅ Maskierung sichtbar: der klassische Detektor findet {_pct(_t['classical_recall'])} der {_t['n_anomalies']} Sonderfahrten, der robuste {_pct(_t['robust_recall'])} (AUC {_t['classical_auc']:.2f} gegen {_t['robust_auc']:.2f}). "
               f"Die Sonderfahrten verzerren Mittelwert und Kovarianz der klassischen Schätzung (Fehler {_t['classical_shape_error']:.2f} gegen {_t['robust_shape_error']:.2f}) und verstecken sich damit selbst; der MCD stützt sich auf die dichteste Hälfte. "
               f"Bedenken: die robuste Schätzung markiert {_t['robust_false_alarm']:.1%} der normalen Touren, nominell wären {_t['nominal']:.1%}.")
elif code == "comparable":
    st.success(f"✅ Beide Detektoren finden die Sonderfahrten ähnlich gut (Recall klassisch {_pct(_t['classical_recall'])}, robust {_pct(_t['robust_recall'])}). Bei wenigen oder weit entfernten Anomalien lohnt die robuste Schätzung kaum - "
               "sie wirkt erst, wenn die Anomalien die klassische Ellipse verzerren.")
elif code == "beyond_breakdown":
    st.warning(f"⚠️ Über dem Bruchpunkt: {_t['contamination']:.0f} % der Touren sind Anomalien. Die robuste Schätzung stützt sich auf h = {a.robust.h} Touren und kann die Anomalien nicht mehr überstimmen - AUC {_t['robust_auc']:.2f} (unter 0.5: "
               f"die Anomalien wirken normaler als die Normalen), Recall {_pct(_t['robust_recall'])}. Klassisch: AUC {_t['classical_auc']:.2f}. Bei verstreuten Ausreißern hält sie länger, bei einer dichten Gruppe bricht sie früher ein.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['robust_auc']:.2f} (robust) und {_t['classical_auc']:.2f} (klassisch) - sie haben kleinere Abstände als die normalen Touren, weil die eine Ellipse beide Betriebsarten "
               "und die Lücke umschließt. Robuste Schätzung hilft hier nicht: verletzt ist die Annahme eines einzigen elliptischen Normalbereichs.")
elif code == "multi_modal":
    st.warning(f"⚠️ Mehrere Betriebsarten: die robuste Schätzung findet {_pct(_t['robust_recall'])} der Anomalien, mit einer Betriebsart wären es {_pct(_t['reference_recall'])}. Die Ellipse muss beide Gruppen und die Lücke umschließen und wird dadurch "
               "so weit, dass nahe Anomalien darin liegen.")
elif code == "not_convex":
    st.warning(f"⚠️ Gekrümmter Normalbereich: die Rangfolge stimmt (AUC {_t['robust_auc']:.2f}), aber die robuste Ellipse umschließt nur den dichten Kern - {_t['robust_false_alarm']:.0%} der normalen Touren liegen außerhalb der Schwelle "
               f"(nominell {_t['nominal']:.1%}), F1 {_t['robust_f1']:.2f}. Die klassische, weitere Ellipse hat F1 {_t['classical_f1']:.2f}. Die Gauß-Annahme hinter dem chi²-Quantil ist verletzt.")
elif code == "too_few_samples":
    st.warning(f"⚠️ Zu wenige Touren für {_t['p']} Merkmale ({_t['n']} < {10 * _t['p']}): der klassische Detektor findet {_pct(_t['classical_recall'])} (sein Abstand kann bei {_t['n']} Touren höchstens {(_t['n'] - 1) ** 2 / _t['n']:.1f} erreichen), "
               f"der robuste {_pct(_t['robust_recall'])}, markiert aber {_t['robust_false_alarm']:.0%} der normalen Touren (nominell {_t['nominal']:.1%}) - die Kovarianz ist nicht zuverlässig schätzbar.")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['robust_auc']:.2f}), aber {_t['robust_false_alarm']:.0%} der normalen Touren werden markiert (nominell {_t['nominal']:.1%}). "
               "Ein höheres Quantil oder eine Schwelle, die den erwarteten Anteil kennt, würde helfen.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Fehler des Zentrums (in Streuungen)", "center_error", "{:.2f}"), ("Fehler der Kovarianz", "shape_error", "{:.2f}"), ("Schwelle für d²", "threshold", "{:.1f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "klassisch": [r[2].format(cs[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was geschätzt wurde**")
    st.table({"": ["Rechenzeit", "Stützpunkte h", "Freiheitsgrade", "Neugewichtung", "konvergierte Starts (von den besten)"],
              "klassisch": [f"{a.seconds['classical'] * 1000:.1f} ms", f"{ds.n}", f"{a.classical.dof}", "–", "–"],
              "robust (MCD)": [f"{a.seconds['robust'] * 1000:.0f} ms", f"{a.robust.h}", f"{a.robust.dof}", "ja" if a.robust.method == "mcd-reweighted" else "nein", f"{a.robust.n_converged} / {C.MCD_KEEP}"]})
    st.caption("Bei n < p + 1 ist die Kovarianz nicht vollen Ranges (dann zählt der Rang als Freiheitsgrad). Der Abstand ist affin äquivariant: andere Einheiten oder Skalen der Merkmale ändern die Abstände nicht.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Touren, Merkmalen, Anomalien und Schwelle ab?")
sweep_options = [k for k in SWEEP_LABELS if not (kind == "gap" and k in ("strength", "n_modes"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination), "strength": float(strength),
           "support": float(support), "quantile": float(quantile)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.markdown("**Fehler der geschätzten Kovarianz gegen die der wahren normalen Touren**")
    st.plotly_chart(build_shape_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_shape")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Der Fehler der Kovarianz ist ein anderes Maß als die Erkennung: "
               "er zeigt, wie gut die Ellipse die normalen Touren beschreibt, unabhängig von der Schwelle.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Bruchpunkt: wie viele Anomalien verträgt die robuste Schätzung?")
if st.button("Anteil der Anomalien von 2 bis 45 % durchfahren (dauert etwa 15 Sekunden)", key="breakdown_start"):
    st.session_state["breakdown_on"] = True
if st.session_state.get("breakdown_on"):
    with st.spinner("Rechne 10 Anteile × 2 Arten × 5 Datensätze..."):
        bd = _breakdown(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_breakdown(bd), width="stretch", key="breakdown_chart")
    st.markdown("**Stützanteil h/n bei 20 % verstreuten Anomalien**")
    st.plotly_chart(build_support(bd["support"]), width="stretch", key="support_chart")
    st.caption("Mittel über 5 feste Datensätze, eine Betriebsart. Der Bruchpunkt des MCD ist höchstens 50 % (h = n/2): darunter hält die Schätzung, darüber kann sie nicht mehr entscheiden, welche Gruppe die normale ist. "
               "Eine **dichte** Gruppe bricht die Schätzung früher als verstreute Ausreißer (AUC unter 0.5 heißt: die Anomalien wirken normaler als die Normalen).")

st.markdown("---")

st.subheader("🔬 Mehrere Betriebsarten: wenn eine Ellipse nicht reicht")
if st.button("Betriebsarten × Art der Anomalien vergleichen (dauert etwa 5 Sekunden)", key="modes_start"):
    st.session_state["modes_on"] = True
if st.session_state.get("modes_on"):
    with st.spinner("Vergleiche 1-3 Betriebsarten × 3 Arten × 5 Datensätze..."):
        mt = _modes(tuple(kv for kv in base_data if kv[0] not in ("kind", "n_modes")), settings)
    st.plotly_chart(build_modes(mt), width="stretch", key="modes_chart")
    st.caption("AUC beider Detektoren (Mittel über 5 feste Datensätze). Weit entfernte Anomalien werden auch mit mehreren Betriebsarten gefunden - solange sie außerhalb der einen großen Ellipse liegen; "
               "Anomalien **in der Lücke** liegen darin und werden nie gefunden (AUC unter 0.5). Der robuste Schätzer ändert daran nichts: er passt eine Ellipse, nicht zwei.")

st.markdown("---")

st.subheader("🔬 Die Schwelle: chi²-Quantil gegen bekannten Anteil")
if st.button("Schwellen vergleichen (dauert etwa 5 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 5 Quantile × 5 Datensätze und 3 Anomalieanteile..."):
        tt = _threshold(tuple(kv for kv in base_data if kv[0] != "contamination") + (("contamination", int(contamination)),), settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Kennzahlen der robusten Schätzung je Quantil**")
    c1.plotly_chart(build_threshold(tt["rows"]), width="stretch", key="threshold_chart")
    c2.markdown("**F1: chi²-Quantil gegen Schwelle mit bekanntem Anteil**")
    c2.plotly_chart(build_oracle(tt["oracle"]), width="stretch", key="oracle_chart")
    st.caption("Links: ein höheres Quantil senkt Fehlalarme und kostet Recall; der klassische Detektor verliert mit der Schwelle viel schneller. Rechts: eine Schwelle, die den wahren Anteil kennt (die k größten Abstände), ist bei wenigen Anomalien deutlich besser als das "
               "chi²-Quantil - dessen nominelle Fehlalarmrate stimmt für die robuste Schätzung nicht (sie markiert mehr Normale). Der wahre Anteil ist in der Praxis unbekannt.")

st.markdown("---")

st.subheader("🔬 Viele Merkmale, wenige Touren")
if st.button("Tourenzahl × Merkmalszahl durchfahren (dauert etwa 15 Sekunden)", key="dimension_start"):
    st.session_state["dimension_on"] = True
if st.session_state.get("dimension_on"):
    with st.spinner("Rechne 6 Tourenzahlen × 5 Merkmalszahlen × 5 Datensätze..."):
        dt = _dimension(tuple(kv for kv in base_data if kv[0] not in ("n", "p")), settings)
    st.plotly_chart(build_dimension(dt), width="stretch", key="dimension_chart")
    st.caption("Mittel über 5 feste Datensätze. Ist n nicht deutlich größer als p, ist die Kovarianz schlecht geschätzt: die robuste Schätzung markiert bis zu gut einem Viertel der normalen Touren, der klassische Detektor findet fast nichts. "
               "Ist n nicht größer als p, hat die Kovarianz nur Rang n − 1, die Abstände sind nach oben durch (n−1)²/n begrenzt und sagen kaum noch etwas über die einzelne Tour: beide Detektoren markieren dann nichts (Fehlalarmrate 0.00), und die Rangfolge ist zufällig (AUC je Datensatz von unter 0.25 bis über 0.75).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein einziger elliptischer, konvexer Normalbereich** | Zwei Betriebsarten mit Anomalien in der Lücke: AUC **0.40** für klassisch und robust (schlechter als Raten); bei drei Betriebsarten findet die robuste Schätzung nur 47 % der verstreuten Anomalien. Gekrümmter Normalbereich: Fehlalarmrate 0.34 statt nominell 0.025. | **LOF** (lokale Dichte), **One-Class SVM** (gelernte Grenze), **Isolation Forest** (Zufallsbäume) |
| **Gauß'sche Normalverteilung (chi²-Schwelle)** | Auch bei einem Modus markiert die robuste Schätzung 4 % statt 2,5 % der normalen Touren; bei Krümmung ein Drittel. Eine Schwelle mit bekanntem Anteil erreicht bei 5 % Anomalien F1 0.91 statt 0.71. | **ECOD** (verteilungsfrei), Isolation Forest |
| **Deutlich mehr Touren als Merkmale** | Bei 30 Touren und 12 Merkmalen markiert die robuste Schätzung 24 % der Normalen, der klassische Detektor findet nichts; bei 300 Touren und 30 Merkmalen 16 %. | **Feature Bagging** (Ensembles über Merkmalsteilmengen), Isolation Forest |
| **Weniger als die Hälfte Anomalien** | Bei 30 % dichter Gruppe kippt die Schätzung (AUC **0.49**), bei 40 % wird die Gruppe zum Normalbereich (AUC **0.39**); verstreute Ausreißer hält der MCD bis 40 % (AUC 0.97), bei 45 % bricht er ein (Recall 0.40). | (keine Rettung innerhalb der Linie: das ist die Grenze jedes robusten Schätzers) |
| **Die Schwelle passt** | Rangfolge und Schwelle sind zwei Fragen: bei 5 % Anomalien hat die robuste Schätzung AUC 1.00, aber F1 nur 0.71 mit dem Standard-Quantil (bei 10 % Anomalien: 0.84 mit 0.975, 0.90 mit 0.999). | Kalibrierung; parameterfreie Verfahren (ECOD) |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie (noch nicht gebaut): ECOD (verteilungsfrei), LOF (lokale Dichte) und Feature Bagging, One-Class SVM und Deep SVDD, Isolation Forest und Extended Isolation Forest sowie ein Autoencoder. "
    "Die Wurzel ist bewusst die einfachste Antwort: eine Ellipse."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Normale Touren $x_i \sim \mathcal{N}(\mu, \Sigma)$ im $\mathbb{R}^p$, Anomalien stammen aus einer anderen Verteilung. Mahalanobis-Abstand $d^2(x) = (x-\mu)^\top \Sigma^{-1}(x-\mu)$; unter dem Modell ist $d^2 \sim \chi^2_p$.
**Schwelle:** $x$ ist Anomalie, wenn $d^2(x) > \chi^2_{p,\,1-\alpha}$ (hier $1-\alpha$ = gewähltes Quantil); nominelle Fehlalarmrate $\alpha$.

**Klassisch.** $\hat\mu = \bar x$, $\hat\Sigma = \frac{1}{n-1}\sum (x_i-\bar x)(x_i-\bar x)^\top$. Kovarianz mit Rang $r < p$ (bei $n \le p$): Abstände über die von null verschiedenen Eigenwerte, Freiheitsgrade $r$. Der Abstand einer Tour im Datensatz ist höchstens $(n-1)^2/n$.

**MCD.** $\min_{H \subset \{1..n\},\ |H| = h} \det \hat\Sigma_H$ mit $h \ge \lceil (n+p+1)/2 \rceil$ (größter Bruchpunkt, $\approx \frac{n-p}{2n}$). **FastMCD:** aus einer zufälligen Startmenge $H_0$ der Größe $p+1$ wiederholt $(\hat\mu_H, \hat\Sigma_H) \to d^2_i \to H' = $ die $h$ kleinsten $d^2_i$;
$\det \hat\Sigma_{H'} \le \det \hat\Sigma_H$ (C-Schritt, Rousseeuw und Van Driessen). 500 zufällige Starts mit je zwei C-Schritten, die besten zehn laufen bis zur Konvergenz. **Konsistenz-Korrektur:** $\hat\Sigma \leftarrow \hat\Sigma \cdot \mathrm{med}_i(d^2_i) / \chi^2_{p,0.5}$.
**Neugewichtung:** $w_i = \mathbb{1}[d^2_i \le \chi^2_{p,0.975}]$, $\hat\mu$, $\hat\Sigma$ aus den Touren mit $w_i = 1$, mit Korrektur der abgeschnittenen Verteilung $\hat\Sigma \leftarrow \hat\Sigma\, F_{\chi^2_p}(q)/F_{\chi^2_{p+2}}(q)$.

**Kennzahlen.** AUC $= P(d^2_{\text{Anomalie}} > d^2_{\text{normal}})$ (Rangsumme, Bindungen halb); mittlere Präzision (Summe über die Treffer der Präzision an jeder Stelle); Recall, Precision, F1, Fehlalarmrate bei der Schwelle; Fehler der Kovarianz $\lVert \hat\Sigma - \Sigma_{\text{normal}} \rVert_F / \lVert \Sigma_{\text{normal}} \rVert_F$ (die Kovarianz der wahren normalen Touren dieses Datensatzes).

**Affine Äquivarianz.** Für $x \mapsto Ax + b$ ändern sich weder klassische noch MCD-Abstände - Einheiten und Skalen der Merkmale sind gleichgültig (per Test).

**Grenzen.** (1) Ein Normalbereich: eine Ellipse umschließt mehrere Gruppen samt Lücke. (2) Gauß-Annahme: das chi²-Quantil ist nur dafür kalibriert; der MCD markiert bei endlichem $n$ mehr Normale als nominell. (3) $n$ nicht deutlich größer als $p$: die Kovarianz ist instabil. (4) Bruchpunkt: über etwa der Hälfte Anomalien kann der MCD nicht mehr entscheiden, was normal ist.

Implementiert in `ee_algorithm.py` (chi²-Verteilung, klassisch, FastMCD, Ellipsen), `ee_scenario.py` (Touren mit Betriebsarten, Krümmung und Anomalien), `ee_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
