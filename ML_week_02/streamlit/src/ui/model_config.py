"""
model_config.py – Hyperparameter configuration widgets.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.models.registry import ModelInfo, get_models_for_task


def render_model_selector(task: str) -> List[str]:
    """Render checkboxes for model selection; return list of selected keys."""
    models = get_models_for_task(task)
    st.subheader("🤖 Select Models to Train")

    selected: List[str] = []
    cols = st.columns(min(len(models), 3))
    for i, m in enumerate(models):
        with cols[i % len(cols)]:
            if st.checkbox(m.display_name, value=True, key=f"sel_{m.key}"):
                selected.append(m.key)

    return selected


def render_hyperparams(model_key: str, model_info: ModelInfo) -> Dict[str, Any]:
    """Render Streamlit widgets for each hyper-parameter in the spec."""
    params: Dict[str, Any] = {}
    spec = model_info.param_spec

    if not spec:
        st.caption("No tuneable hyper-parameters.")
        return params

    for pname, pconf in spec.items():
        ptype = pconf.get("type", "float")

        if ptype == "int":
            params[pname] = st.slider(
                pname,
                min_value=pconf["min"],
                max_value=pconf["max"],
                value=pconf.get("default", pconf["min"]),
                key=f"hp_{model_key}_{pname}",
            )
        elif ptype == "float":
            if pconf.get("log_scale"):
                import math
                log_min = math.log10(pconf["min"])
                log_max = math.log10(pconf["max"])
                log_default = math.log10(pconf.get("default", pconf["min"]))
                log_val = st.slider(
                    f"{pname} (log₁₀)",
                    min_value=log_min,
                    max_value=log_max,
                    value=log_default,
                    step=0.01,
                    key=f"hp_{model_key}_{pname}",
                )
                params[pname] = round(10 ** log_val, 6)
            else:
                params[pname] = st.slider(
                    pname,
                    min_value=pconf["min"],
                    max_value=pconf["max"],
                    value=pconf.get("default", pconf["min"]),
                    step=0.01,
                    key=f"hp_{model_key}_{pname}",
                )
        elif ptype == "select":
            options = pconf.get("options", [])
            default_idx = options.index(pconf["default"]) if pconf.get("default") in options else 0
            params[pname] = st.selectbox(
                pname,
                options,
                index=default_idx,
                key=f"hp_{model_key}_{pname}",
            )

    return params


def render_all_hyperparams(selected_keys: List[str], task: str) -> Dict[str, Dict[str, Any]]:
    """Render parameter widgets for all selected models, grouped in expanders."""
    from src.models.registry import get_model

    st.subheader("⚙️ Hyperparameters")
    use_defaults = st.checkbox("Use default hyperparameters", value=True, key="use_defaults")

    params_map: Dict[str, Dict[str, Any]] = {}
    if use_defaults:
        return params_map

    for key in selected_keys:
        info = get_model(key)
        with st.expander(f"🔧 {info.display_name}", expanded=False):
            params_map[key] = render_hyperparams(key, info)

    return params_map
