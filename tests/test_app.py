"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ee_constants as C
from ee_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"Verstreute Ausreißer (10 %)": "success", "Dichte Gruppe abseits (20 %)": "success", "Über dem Bruchpunkt (40 %)": "warning", "Anomalien in der Lücke": "warning",
                 "Gekrümmter Normalbereich": "warning", "Wenige Touren": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    """Erst die Betriebsarten setzen und laufen lassen (die Art "Lücke" gibt es erst ab zwei; AppTest prüft gegen die Optionen des vorigen Laufs), dann alles andere."""
    at.session_state["n_modes_slider"] = p["n_modes"]
    at.run()
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.toggle)}


def test_default_renders_without_exception():
    at = _run()
    assert any("Elliptic Envelope in Aktion" in m.value for m in at.markdown)
    assert not at.error and len(at.success) == 1 and not at.warning                    # Standardfall: Maskierung sichtbar, robust gewinnt


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["p_slider"] = C.P_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MIN
        at.session_state["support_slider"] = C.SUPPORT_MAX
        at.session_state["quantile_slider"] = C.QUANTILE_MAX
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["p_slider"] = C.P_MIN
        at.session_state["n_modes_slider"] = C.N_MODES_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MAX
        at.session_state["kind_select"] = "cluster"
        at.session_state["strength_slider"] = C.STRENGTH_MIN
        at.session_state["quantile_slider"] = C.QUANTILE_MIN
        at.session_state["reweight_toggle"] = False
    _run(large)

    at = _run(lambda a: (a.session_state.__setitem__("p_slider", 2), a.session_state.__setitem__("n_modes_slider", 3)))
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 12, 30])
def test_every_step_renders(step, p):
    def setup(at):
        at.session_state["p_slider"] = p
        at.session_state["ee_step"] = step
    _run(setup)


def test_more_features_than_tours_renders_every_step():
    def setup(at):
        at.session_state["n_tours_slider"] = 20
        at.session_state["p_slider"] = 30
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["ee_step"] = step
        at.run()
        assert not at.exception


def test_strength_slider_is_hidden_for_gap_anomalies_and_its_value_is_kept():
    at = _run(lambda a: (a.session_state.__setitem__("strength_slider", 9.0)))
    assert "Abstand der Anomalien (Faktor-σ)" in _labels(at)
    at.session_state["n_modes_slider"] = 2
    at.run()
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and "Abstand der Anomalien (Faktor-σ)" not in _labels(at)
    at.session_state["kind_select"] = "scattered"
    at.run()
    assert not at.exception and [s for s in at.sidebar.slider if s.label.startswith("Abstand")][0].value == 9.0


def test_gap_kind_needs_two_modes_and_falls_back_when_modes_drop():
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    at.session_state["n_modes_slider"] = 1
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] not in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    assert at.session_state["kind_select"] == "scattered"


def test_sweep_options_drop_strength_and_modes_for_gap_anomalies():
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    from ee_evaluation import SWEEP_LABELS
    assert not ({SWEEP_LABELS["strength"], SWEEP_LABELS["n_modes"]} & set(options)) and SWEEP_LABELS["contamination"] in options


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["ee_step"] = 4
    at.run()
    assert not at.exception and at.session_state["ee_step"] == 4
    at.session_state["contamination_slider"] = 20
    at.run()
    assert not at.exception and at.session_state["ee_step"] == 1


def test_experiments_run_on_demand_and_sweeps_run():
    at = _run()
    for parameter in ("contamination", "p", "quantile", "support"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        [b for b in at.button if b.key == "sweep_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    for key in ("breakdown_start", "modes_start", "threshold_start", "dimension_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert all(at.session_state[k] for k in ("breakdown_on", "modes_on", "threshold_on", "dimension_on"))


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "ee_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_unique_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    parts = source.split("plotly_chart(")[1:]
    keys = [re.search(r'key="([a-z_]+)"', part.split("plotly_chart(")[0]) for part in parts]
    assert len(parts) >= 15 and all(keys)
    names = [k.group(1) for k in keys]
    assert len(set(names)) == len(names)


def test_app_has_no_dead_file_links_and_the_verbatim_footer():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.findall(r"\]\([a-z_]+\.py\)", source)
    assert "https://sebastianhanisch.net/kontakt.html" in source and "Interesse an einer maßgeschneiderten Lösung für" in source
