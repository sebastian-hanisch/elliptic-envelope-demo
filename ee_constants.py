"""Defaults, Slider-Grenzen und Presets für die Elliptic-Envelope-Demo (robuste Anomalie-Erkennung an Lieferrouten-Kennzahlen)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))

# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION, kind=DEFAULT_KIND,
                strength=DEFAULT_STRENGTH, support=DEFAULT_SUPPORT, quantile=DEFAULT_QUANTILE, reweight=DEFAULT_REWEIGHT, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Verstreute Ausreißer (10 %)": _preset(),
    "Dichte Gruppe abseits (20 %)": _preset(kind="cluster", contamination=20),
    "Über dem Bruchpunkt (40 %)": _preset(kind="cluster", contamination=40),
    "Anomalien in der Lücke": _preset(n_modes=2, kind="gap"),
    "Gekrümmter Normalbereich": _preset(curvature=0.75),
    "Wenige Touren": _preset(n=30),
}
PRESET_HELP = {
    "Verstreute Ausreißer (10 %)": "Im Mittel über fünf Aufnahmen: 10 % Sonderfahrten, jede in eine andere Richtung, 6 Faktor-σ entfernt. Sie verzerren die klassische Schätzung selbst (Fehler der Kovarianz 1.7 gegen 0.06 bei der robusten), "
                                   "und der klassische Detektor findet nur 43 % von ihnen, der robuste 97 % (AUC 0.95 gegen 1.00) - das ist Maskierung.",
    "Dichte Gruppe abseits (20 %)": "Im Mittel über fünf Aufnahmen: 20 % der Touren bilden eine dichte Gruppe weit abseits. Sie zieht Mittelwert und Kovarianz der klassischen Schätzung zu sich - die Gruppe gilt als normal "
                                    "(Recall 0.03, AUC 0.62); die robuste Schätzung stützt sich auf die dichtere Hälfte und findet alle (Recall 1.00, AUC 1.00).",
    "Über dem Bruchpunkt (40 %)": "Im Mittel über fünf Aufnahmen: die dichte Gruppe umfasst 40 % der Touren - fast so viele wie die Normalen. Die robuste Schätzung mit h = n/2 kann sie nicht mehr überstimmen und wählt die Gruppe als Normalbereich: "
                                  "AUC 0.39 (schlechter als Raten), Recall 0.04; klassisch 0.44. Bei verstreuten Ausreißern hält sie länger (AUC 0.97 bei 40 %, 0.87 bei 45 %).",
    "Anomalien in der Lücke": "Im Mittel über fünf Aufnahmen: zwei Betriebsarten, 10 % Anomalien in der Lücke dazwischen. Die eine Ellipse umschließt beide Betriebsarten und die Lücke - die Anomalien wirken normaler als die normalen Touren: "
                              "AUC 0.40 für den klassischen wie den robusten Detektor. Kein Schätzer hilft, denn verletzt ist die Annahme eines einzigen elliptischen Normalbereichs (Ansatzpunkt der nächsten Stücke der Linie).",
    "Gekrümmter Normalbereich": "Im Mittel über fünf Aufnahmen: der Normalbereich ist gebogen (Krümmung 0.75). Die Rangfolge der Abstände bleibt perfekt (AUC 1.00), aber die robuste Ellipse umschließt nur den dichten Kern: "
                                "35 % der normalen Touren liegen außerhalb der Schwelle (nominell 2.5 %), F1 0.39 - die breitere klassische Ellipse kommt auf F1 0.91.",
    "Wenige Touren": "Im Mittel über fünf Aufnahmen: 30 Touren bei 12 Merkmalen. Die klassische Schätzung markiert nichts (Recall 0.00), die robuste findet 80 % der Anomalien, markiert aber 24 % der normalen Touren als Anomalie "
                     "(nominell 2.5 %): die Kovarianz ist mit so wenig Touren nicht zuverlässig schätzbar.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen des klassischen (classical_*) und robusten (robust_*) Detektors und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Verstreute Ausreißer (10 %)": {"classical_recall": (0.2, 0.8), "robust_recall": (0.85, 1.0), "robust_auc": (0.95, 1.0), "verdict": ("masked",)},
    "Dichte Gruppe abseits (20 %)": {"classical_recall": (0.0, 0.2), "robust_recall": (0.9, 1.0), "robust_auc": (0.95, 1.0), "verdict": ("masked",)},
    "Über dem Bruchpunkt (40 %)": {"robust_auc": (0.0, 0.6), "robust_recall": (0.0, 0.3), "verdict": ("beyond_breakdown",)},
    "Anomalien in der Lücke": {"robust_auc": (0.0, 0.6), "classical_auc": (0.0, 0.6), "robust_recall": (0.0, 0.3), "verdict": ("gap",)},
    "Gekrümmter Normalbereich": {"robust_auc": (0.95, 1.0), "robust_false_alarm": (0.2, 0.6), "robust_f1": (0.2, 0.6), "verdict": ("not_convex",)},
    "Wenige Touren": {"classical_recall": (0.0, 0.2), "robust_false_alarm": (0.1, 0.5), "verdict": ("too_few_samples",)},
}
