"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe nm_presets.py in nmf-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ee_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _bool(value):
    value = str(value).lower()
    if value not in ("1", "0", "true", "false"):
        raise ValueError(value)
    return value in ("1", "true")


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "n_modes_slider": SettingSpec("modes", int, C.DEFAULT_N_MODES, C.N_MODES_MIN, C.N_MODES_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "contamination_slider": SettingSpec("cont", int, C.DEFAULT_CONTAMINATION, C.CONTAMINATION_MIN, C.CONTAMINATION_MAX),
    "kind_select": SettingSpec("kind", _choice(C.KINDS), C.DEFAULT_KIND),
    "strength_slider": SettingSpec("str", float, C.DEFAULT_STRENGTH, C.STRENGTH_MIN, C.STRENGTH_MAX),
    "support_slider": SettingSpec("h", float, C.DEFAULT_SUPPORT, C.SUPPORT_MIN, C.SUPPORT_MAX),
    "quantile_slider": SettingSpec("q", float, C.DEFAULT_QUANTILE, C.QUANTILE_MIN, C.QUANTILE_MAX),
    "reweight_toggle": SettingSpec("rw", _bool, C.DEFAULT_REWEIGHT),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n": "n_tours_slider", "p": "p_slider", "n_modes": "n_modes_slider", "curvature": "curvature_slider", "noise": "noise_slider", "contamination": "contamination_slider",
               "kind": "kind_select", "strength": "strength_slider", "support": "support_slider", "quantile": "quantile_slider", "reweight": "reweight_toggle", "seed": "seed_input"}
KEPT = {"strength_slider": "_strength_kept"}          # bei Anomalien "in der Lücke" ausgeblendet (dort gibt es keinen Abstand): der Wert bleibt hier erhalten


def kind_options(n_modes):
    """Die Art 'in der Lücke' gibt es erst ab zwei Betriebsarten."""
    return tuple(k for k in C.KINDS if k != "gap" or n_modes >= 2)


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; der Abstand der Anomalien (bei 'Lücke' ausgeblendet, sonst vom Widget-Zustand gelöscht) kehrt zum zuletzt gewählten Wert zurück; eine ungültige Art wird zurückgesetzt."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default
    if st.session_state["kind_select"] not in kind_options(st.session_state["n_modes_slider"]):
        st.session_state["kind_select"] = C.DEFAULT_KIND


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("curvature_slider", 4), ("noise_slider", 20), ("strength_slider", 2), ("support_slider", 20)):
        st.session_state[key] = round(st.session_state.get(key, SETTING_SPECS[key].default) * step) / step
    st.session_state["quantile_slider"] = round(st.session_state.get("quantile_slider", C.DEFAULT_QUANTILE) * 1000) / 1000
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_strength_kept"] = C.PRESETS[name]["strength"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
