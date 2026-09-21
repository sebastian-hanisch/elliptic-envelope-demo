# Elliptic Envelope – Anomalien außerhalb einer robusten Ellipse – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-elliptic-envelope-demo.streamlit.app/)**

Erstes Stück (**Wurzel**) der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Elliptic Envelope** (robuste Gauß-Anpassung per **Minimum Covariance Determinant**, MCD) – an einem wachsenden Beispiel,
mit der **klassischen** Schätzung (Mittelwert und Kovarianz aller Touren) als Gegenstück. Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in der [pca-demo](../pca-demo) (12 Kennzahlen aus zwei versteckten Faktoren, dieselben festen Matrizen; die normalen Zeilen sind dort dieselben, per Test geprüft) –
dort drehen Sonderfahrten die erste Hauptachse, hier sollen sie gefunden werden.

**Einordnung in die Reihe (die Kanten des Graphen):** die Wurzel ist bewusst die **einfachste** Antwort auf "was ist ungewöhnlich?": die normalen Touren sind **eine Gauß'sche Wolke**, eine Tour ist Anomalie, wenn ihr Mahalanobis-Abstand über einem χ²-Quantil liegt.
Ihre Schwächen sind die Ansatzpunkte der späteren Stücke: nur **ein** elliptischer, konvexer Normalbereich (→ LOF, One-Class SVM, Isolation Forest), Gauß-Annahme und viele Touren im Verhältnis zu den Merkmalen (→ ECOD, Feature Bagging).
Die Linie hat **keinen Konvergenzpunkt** – drei unabhängige Äste, jeweils mit einer Fortsetzung, dazu ein Kontrast (ECOD) und ein Autoencoder-Ast; bisher gebaut: die Wurzel, [isolation-forest-demo](../isolation-forest-demo), [extended-isolation-forest-demo](../extended-isolation-forest-demo), [lof-demo](../lof-demo), [feature-bagging-demo](../feature-bagging-demo) und [ecod-demo](../ecod-demo).
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ecod-demo                  (Kontrast: verteilungsfrei)                       [gebaut]
  ├─ lof-demo → Feature Bagging (lokale Dichte; Ensembles gegen viele Merkmale)   [beide gebaut]
  ├─ One-Class SVM → Deep SVDD  (gelernte Grenze)                                 [nicht gebaut]
  ├─ isolation-forest-demo → Extended Isolation Forest (Zufallsbäume)             [beide gebaut]
  └─ Autoencoder                (Rekonstruktionsfehler)                           [nicht gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25, Schwelle 97,5 %-Quantil; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall | ✅ **Maskierung**: der klassische Detektor findet **43 %** der Anomalien, der robuste **97 %** (AUC 0.95 gegen 1.00); Fehler der geschätzten Kovarianz gegen die der wahren Normalen **1.7 gegen 0.06** |
| Anteil der Anomalien | ✅ Recall klassisch 0.93 / 0.75 / 0.43 / 0.14 / 0.06 / 0.04 bei 2 / 5 / 10 / 20 / 30 / 40 %, robust 1.00 / 0.99 / 0.97 / 0.95 / 0.93 / 0.76; ❌ bei **45 %** bricht auch der MCD ein (Recall 0.40, AUC 0.87) |
| Dichte Gruppe abseits | ✅ maskiert noch stärker: bei 20 % klassisch Recall 0.03 (AUC 0.62), robust 1.00; ❌ **bricht den MCD früher**: AUC 1.00 bis 25 %, **0.49 bei 30 %, 0.39 bei 40 %** – unter 0.5: die Anomalien wirken normaler als die Normalen, die Gruppe *ist* der Normalbereich geworden |
| Anomalien in der Lücke | ❌ zwei Betriebsarten, Anomalien dazwischen: AUC **0.40** für klassisch und robust (drei Betriebsarten 0.38) – die eine Ellipse umschließt beide Gruppen und die Lücke |
| Mehrere Betriebsarten | ⚠️ 1 / 2 / 3 Betriebsarten (verstreute Anomalien): Recall robust 0.97 / 0.77 / 0.47, AUC 1.00 / 0.95 / 0.85; weit entfernte *dichte Gruppen* werden mit zwei Betriebsarten weiter gefunden (Recall 1.00) |
| Gekrümmter Normalbereich | ❌ Rangfolge bleibt perfekt (AUC 1.00), aber die robuste Ellipse hüllt nur den dichten Kern: Fehlalarmrate **0.24 / 0.32 / 0.35 / 0.36** bei Krümmung 0.25 / 0.5 / 0.75 / 1 (nominell 0.025), F1 0.49 – 0.38; die breitere klassische Ellipse kommt auf F1 **0.90 – 0.91** |
| Wenige Touren | ❌ Fehlalarmrate der robusten Schätzung 0.16 / 0.24 / 0.29 / 0.21 / 0.06 bei 20 / 30 / 50 / 100 / 200 Touren; der klassische Detektor findet bei 20 und 30 Touren **nichts** (Abstand im Datensatz höchstens (n−1)²/n = 18.05 bei 20 Touren, Schwelle 23.3) |
| Viele Merkmale | ❌ bei 300 Touren Fehlalarmrate 0.03 / 0.02 / 0.03 / 0.04 / 0.08 / 0.16 bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen; klassischer Recall 0.84 / 0.73 / 0.57 / 0.43 / 0.28 / 0.21. Bei **n ≤ p** markiert keiner der beiden etwas, die Rangfolge ist zufällig (AUC je Datensatz von unter 0.25 bis über 0.75) |
| Schwelle | ⚠️ die robuste Schätzung markiert **4 %** statt nominell 2,5 % der Normalen; F1 0.67 / 0.77 / 0.84 / 0.89 / 0.90 bei Quantil 0.9 / 0.95 / 0.975 / 0.99 / 0.999. Bei 5 % Anomalien: **F1 0.71 mit dem χ²-Quantil, 0.91 mit einer Schwelle, die den wahren Anteil kennt** |
| Stützanteil h/n | ✅ 0.5 / 0.6 / 0.75 / 0.9 / 1.0: Recall bei 20 % Anomalien 0.95 / 0.96 / 0.94 / **0.45 / 0.20** – weniger Stützpunkte sind robuster |
| Neugewichtung | ✅ an gegen aus: Fehlalarmrate 0.039 / 0.065, F1 0.84 / 0.76, Fehler der Kovarianz 0.06 / 0.12 |
| Einheiten | ✅ die Abstände sind **affin äquivariant**: Meter, Minuten und Skalen ändern nichts (per Test) – anders als die Einheitenfalle der PCA-Demo |
| Rechenzeit | ✅ 0.12 s für die ganze Analyse im Standardfall, 0.3 s im Extremfall (600 Touren, 30 Merkmale, 45 % Anomalien) |

## Was die Demo zeigt

1. **Elliptic Envelope in Aktion** (Schritt-Slider + Abspielen): **Touren** (in der Ebene der zwei größten Streuungsrichtungen, daneben zwei Rohmerkmale) → **Klassische Ellipse** (mit den Abständen) → **MCD: C-Schritte** (Determinante je Schritt, die gewählte Teilmenge h)
   → **Robuste Ellipse** gegen die klassische → **Ergebnis** (Kennzahlen bei der Schwelle und ROC-Kurven).
2. **Was der Detektor gefunden hat – klassisch gegen robust:** AUC, Recall, Fehlalarmrate, Fehler der Kovarianz mit Urteil
   (Codes: über dem Bruchpunkt → Lücke → mehrere Betriebsarten → gekrümmter Normalbereich → zu wenige Touren → falsche Schwelle → Maskierung → gleichauf); Tabelle mit AUC, mittlerer Präzision, Precision, Recall, F1, Fehlalarmrate, Fehler von Zentrum und Kovarianz, Schwelle.
3. **📐 Sweeps** über Touren, Merkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, Stützanteil und Schwelle (feste Datensätze ab 100000, Streuung; dazu der Fehler der Kovarianz).
4. **🔬 Bruchpunkt** (2–45 % für verstreute Ausreißer und dichte Gruppe, dazu der Stützanteil), **🔬 Betriebsarten** (× Art der Anomalien), **🔬 Schwelle** (Kennzahlen je Quantil; χ²-Quantil gegen bekannten Anteil) und **🔬 viele Merkmale, wenige Touren** (Tourenzahl × Merkmalszahl) – Experimente auf Abruf.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (ein Normalbereich, Gauß, viele Touren, weniger als die Hälfte Anomalien, die Schwelle).

Regler: Touren (20–600), Merkmale (2–30; ab 13 Zusatzmerkmale), Betriebsarten (1–3), Krümmung (0–1), Rauschen, Anteil der Anomalien (1–45 %, exakt, mindestens eine), Art der Anomalien (verstreut / dichte Gruppe / in der Lücke – letztere erst ab zwei Betriebsarten),
Abstand der Anomalien (3–12 Faktor-σ; bei "Lücke" ausgeblendet, Wert bleibt erhalten), Stützanteil h/n (0.5–1), χ²-Quantil der Schwelle (0.9–0.999), Neugewichtung.

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC klassisch | Recall klassisch | AUC robust | Recall robust | Fehlalarm robust | Urteil |
|---|---|---|---|---|---|---|
| Verstreute Ausreißer (10 %) | 0.95 | 0.40 | 1.00 | 1.00 | 0.04 | Maskierung sichtbar |
| Dichte Gruppe abseits (20 %) | 0.56 | 0.00 | 1.00 | 1.00 | 0.03 | Maskierung sichtbar |
| Über dem Bruchpunkt (40 %) | 0.41 | 0.02 | 0.32 | 0.02 | 0.15 | über dem Bruchpunkt |
| Anomalien in der Lücke | 0.35 | 0.00 | 0.35 | 0.00 | 0.05 | Anomalien in der Lücke |
| Gekrümmter Normalbereich (0.75) | 1.00 | 0.83 | 1.00 | 1.00 | 0.35 | gekrümmter Normalbereich |
| Wenige Touren (30) | 0.94 | 0.00 | 0.88 | 0.67 | 0.26 | zu wenige Touren |

## Modell und Verfahren

- **Szenario** (`ee_scenario.py`): die 12 Kennzahlen der PCA-Demo aus **zwei** versteckten Faktoren (dieselben Ladungsmatrizen und Ziehungsreihenfolge; die normalen Zeilen sind bei einer Betriebsart, ohne Krümmung und bei p = 12 dieselben, Spalten umsortiert – per Test gegen eingefrorene Zeilensummen).
  Neu: **Betriebsarten** (1–3 Gruppen im Faktorraum: Punkte auf einem Kreis), **Krümmung** (die nichtlinearen Terme der PCA-Demo), **Anomalien** in drei Arten – verstreut (jede in eine andere Richtung im Abstand s), dichte Gruppe (alle beieinander, Streuung 0.3) und in der Lücke (um den Ursprung zwischen den Betriebsarten) –
  mit **exaktem** Anteil (die Touren mit den kleinsten Losen, mindestens eine) und **Merkmalszahl** bis 30 (Merkmale 13 ... p: feste Mischungen der Faktoren plus eigenes Rauschen), damit auch n < p vorkommt.
- **Klassisch** (`ee_algorithm.py`, numpy von Grund auf): Mittelwert, Kovarianz, quadrierter Mahalanobis-Abstand über die Eigenzerlegung der **Korrelationsmatrix** (einheitenunabhängig; verschwindende Eigenwerte werden abgeschnitten, der Rang zählt als Freiheitsgrad), Schwelle = χ²-Quantil (eigene Implementierung der unvollständigen Gammafunktion, gegen scipy geprüft).
- **MCD** (FastMCD): 500 zufällige Startmengen aus p + 1 Touren mit je **zwei C-Schritten** (Abstände, die h kleinsten wählen, neu schätzen), die besten zehn laufen bis die Determinante nicht mehr sinkt; h ≥ (n + p + 1)/2, wählbar bis n; **Konsistenz-Korrektur** (Median der Abstände gegen den χ²-Median) und optionale **Neugewichtung** (Touren innerhalb des Quantils, mit Korrektur der abgeschnittenen Verteilung).
- **Auswertung** (`ee_evaluation.py`): AUC (Rangsumme, Bindungen halb), mittlere Präzision, Precision, Recall, F1, Fehlalarmrate, Fehler des Zentrums und der Kovarianz gegen die der wahren normalen Touren, Sweeps, Bruchpunkt-, Betriebsarten-, Schwellen- und Dimensions-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Die erste MCD-Fassung fand bei 20 % dichter Gruppe die falsche Lösung.** Zwei Fehler zusammen: die Regularisierung der Determinante war am Spurbetrag skaliert, bei Merkmalen in Metern und Anteilen (Faktor 10⁵) wurde die Determinante dadurch verzerrt, und dieselbe Eigenwertschranke ließ in Rohdaten **eine Dimension verschwinden** (Rang 11 statt 12, falsche χ²-Schwelle);
  außerdem liefen 30 Starts alle bis zur Konvergenz – und blieben alle in gemischten lokalen Minima mit Teilen der Gruppe (log det 34.24 gegen 33.60 für den Normalkern). Erst die einheitenunabhängige Zerlegung (Test: Skalierung der Merkmale ändert die Abstände nicht) und das echte FastMCD-Schema mit 500 Starts fanden den Kern.
- **Die robuste Schätzung ist nicht "die bessere":** sie markiert 4 % statt 2,5 % der Normalen, bei gekrümmtem Normalbereich ein Drittel und ihr F1 liegt dort weit unter dem der klassischen Ellipse (0.39 gegen 0.91). Vor dem Bau war die Vermutung, dass der MCD "bis fast 50 %" hält; gemessen gilt das nur für verstreute Ausreißer (Recall 0.76 bei 40 %), eine **dichte Gruppe** bricht ihn schon bei 30 % (AUC 0.49).
- **Mehrere Betriebsarten:** Vermutung war, dass die AUC beider Detektoren bei zwei bis drei Betriebsarten stark fällt – das stimmt für Anomalien **in der Lücke** (0.40, unter Raten), nicht für weit entfernte (dichte Gruppen bei zwei Betriebsarten: Recall 1.00). Kein Schätzer hilft in der Lücke: er passt eine Ellipse, nicht zwei.
- **Die Schwelle ist eine zweite Fehlerquelle:** die Rangfolge (AUC 1.00) und die Entscheidung sind zwei Fragen. Bei 5 % Anomalien hat der robuste Detektor F1 0.71 mit dem Standard-Quantil und 0.91, wenn der wahre Anteil bekannt wäre – in der Praxis ist er es nicht.
- **n nicht deutlich größer als p:** der klassische Abstand kann im Datensatz höchstens (n−1)²/n erreichen; bei n ≤ p ist die Kovarianz nicht vollen Ranges und die Abstände sagen kaum etwas (Summe der Abstände = (n−1)·Rang).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie. Literatur nur mit Namen: Mahalanobis (Abstand), Rousseeuw und Van Driessen (MCD, FastMCD). Der Elliptic Envelope in scikit-learn ist ein MCD mit Schwelle; diese Demo rechnet ihn selbst und prüft ihn gegen scikit-learn.

## Verifikation

- Kern: χ²-Verteilung gegen scipy (Quantile, Verteilungsfunktion, Sonderfälle); Mahalanobis mit Handinstanzen (auch drehungsinvariant, singuläre Kovarianz, Summe der Abstände = (n−1)·Rang, Obergrenze (n−1)²/n); **MCD gegen `sklearn.covariance.MinCovDet`** (Zentrum, Kovarianz, markierte Touren);
  C-Schritte senken die Determinante nie; mehr Starts nie schlechter; volle Stützmenge = klassisch bis auf den Skalenfaktor; Konsistenz-Korrektur (Median = χ²-Median); Fehlalarmrate 2,5 % auf sauberen Gauß'schen Daten; Determinismus und vom Datensatz entkoppelter Start-Seed; **affine Äquivarianz** (Skalierung, Drehung); Ellipsen (Projektion eines Ellipsoids).
- Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, exakter Anomalie-Anteil, feste Ziehungsreihenfolge, Merkmalszahl, Betriebsarten, Lücke, Geometrie der Anomalie-Arten. Kennzahlen mit Handinstanzen (AUC gegen Brute-Force, mittlere Präzision, ROC-Fläche, Precision/Recall/F1), Analyse, Urteilscodes.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Bruchpunkt-, Betriebsarten-, Schwellen- und Dimensions-Tabellen; jeweils Mittel über die festen Sweep-Datensätze mit engen Toleranzen, positive **und** negative Aussagen);
  alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, mehr Merkmale als Touren, Randgrößen, ausgeblendeter Abstandsregler behält den Wert, Art "Lücke" nur ab zwei Betriebsarten, Sweep-Optionen, Experimente auf Abruf), Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Bruchpunkt / Betriebsarten / Schwelle / viele Merkmale, 🚧 Grenzen, Mathe |
| `ee_algorithm.py` | χ²-Verteilung, klassische Schätzung, FastMCD, Schwelle, Ellipsen |
| `ee_scenario.py`, `ee_constants.py` | Touren mit Betriebsarten, Krümmung und Anomalien; Konstanten, Presets |
| `ee_evaluation.py` | AUC, mittlere Präzision, Precision/Recall/F1, Analyse, Sweeps, Experimente, Urteil |
| `ee_presets.py`, `ee_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Kern (scipy- und scikit-learn-Kreuzprüfung), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
