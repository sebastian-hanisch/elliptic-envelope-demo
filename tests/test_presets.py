"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import ee_constants as C
from ee_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["n"], p["p"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], p["seed"])
    a = analyse_for(params, Settings(p["support"], p["quantile"], p["reweight"]))
    out = {"verdict": verdict(a)[1]}
    for det in ("classical", "robust"):
        for key, value in a.scores[det].items():
            out[f"{det}_{key}"] = value
    return out


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_most_presets_show_a_negative_result():
    negative = [n for n, b in C.PRESET_EXPECTED_BANDS.items() if b["verdict"][0] in ("beyond_breakdown", "gap", "not_convex", "too_few_samples")]
    assert len(negative) == 4


def test_the_default_setting_is_the_first_preset():
    assert C.PRESETS["Verstreute Ausreißer (10 %)"] == C._preset() and next(iter(C.PRESETS.values()))["contamination"] == C.DEFAULT_CONTAMINATION


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n"] <= C.N_TOURS_MAX and p["n"] % 10 == 0 and C.P_MIN <= p["p"] <= C.P_MAX and C.N_MODES_MIN <= p["n_modes"] <= C.N_MODES_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and abs(p["curvature"] * 4 - round(p["curvature"] * 4)) < 1e-9
        assert C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.CONTAMINATION_MIN <= p["contamination"] <= C.CONTAMINATION_MAX
        assert p["kind"] in C.KINDS and (p["kind"] != "gap" or p["n_modes"] >= 2)                          # 'Lücke' nur mit mindestens zwei Betriebsarten
        assert C.STRENGTH_MIN <= p["strength"] <= C.STRENGTH_MAX and abs(p["strength"] * 2 - round(p["strength"] * 2)) < 1e-9
        assert C.SUPPORT_MIN <= p["support"] <= C.SUPPORT_MAX and C.QUANTILE_MIN <= p["quantile"] <= C.QUANTILE_MAX and isinstance(p["reweight"], bool)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if key == "verdict":
            assert value in expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
