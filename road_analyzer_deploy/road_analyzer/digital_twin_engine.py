"""
digital_twin_engine.py
-----------------------
Pure-Python "digital twin" data generator for the web dashboard (IRC:106 LOS-based).

Why this exists
----------------
The dashboard's JS (static/app.js: dtRender / dtDrawCapChart / dtDrawSpdChart /
dtAnimateRoads) already knows how to draw the twin — capacity-over-time charts,
speed-over-time charts, and an animated ideal-road-vs-defect-road comparison.
It was built to be fed by app.py's /api/digital-twin/status endpoint, which in
turn expected a road_analyzer/digital_twin_bridge.py module that would launch
real MATLAB/Simulink (matlab_twin/roadtwin.m) and poll it for results.

That bridge module was never actually added to the repo, and — more
importantly — a cloud host like Render doesn't have a licensed MATLAB
installation available to run headless anyway. Requiring MATLAB locally to
see the twin defeats the point of a web dashboard.

This module produces the same *shape* of result using the same underlying
traffic-flow theory your MATLAB script and core.py already use elsewhere in
this project (Greenshields speed-density model, referenced throughout
core.py's pothole-penalty comments), computed directly in Python from the
numbers analyse_image() already calculated. No MATLAB, no subprocess, no
external service — it runs in milliseconds as part of the normal request.

If you later get a licensed MATLAB instance connected (e.g. via MATLAB
Engine API for Python, or a self-hosted matlab_video_service), you can swap
the call site in app.py back to that bridge without touching the frontend —
the JSON shape returned here is intentionally identical to what dtRender()
expects.
"""

import math
from typing import Optional

# Display settings for the dashboard time series only. They do not change
# any capacity, V/C, LOS or speed value, which all come from core.py.
SIM_DURATION_S = 60
SIM_STEP_S     = 2
RAMP_TAU_S     = 12.0

_LATEST = {"status": "idle", "twin_data": None, "error": None}


def _approach(start: float, final: float, t: float, tau: float = RAMP_TAU_S) -> float:
    """First-order approach from start to final (display smoothing only)."""
    return final + (start - final) * math.exp(-t / tau)


def generate_twin_data(analysis_result: dict) -> dict:
    """
    Re-expresses core.py's IRC-based result as a short time series for the
    dashboard. Throughput = min(demand, capacity) for the ideal road (C0)
    and the obstructed road (C_eff); speeds are the IRC:106 cl. 5.3 speeds
    at the resulting LOS. Replaces the earlier Greenshields relation, whose
    speed rose as capacity fell.
    """
    tr = analysis_result.get("traffic_regime", {}) or {}
    v_free     = float(analysis_result.get("operating_speed_kmh")
                       or analysis_result.get("free_flow_speed_kmh") or 50.0)
    ideal_cap  = float(analysis_result.get("original_capacity_pcu_hr") or 0.0)
    defect_cap = float(analysis_result.get("reduced_capacity_pcu_hr") or ideal_cap)
    demand     = float(analysis_result.get("demand_pcu_hr")
                       or analysis_result.get("design_service_volume_pcu_hr") or ideal_cap)
    v_ideal    = float(tr.get("ideal_speed_kmh") or v_free)
    v_defect   = float(tr.get("congested_speed_kmh") or v_ideal)
    loss_pct   = float(analysis_result.get("capacity_loss_pct") or 0.0)

    q_ideal  = min(demand, ideal_cap)
    q_defect = min(demand, defect_cap)

    sim_time, ideal_series, defect_series, speed_series = [], [], [], []
    t = 0.0
    while t <= SIM_DURATION_S:
        sim_time.append(round(t, 1))
        ideal_series.append(round(_approach(0.0, q_ideal, t), 1))
        defect_series.append(round(_approach(0.0, q_defect, t), 1))
        speed_series.append(round(_approach(v_free, v_defect, t), 1))
        t += SIM_STEP_S

    speed_reduction_pct = round(max(0.0, (v_ideal - v_defect) / v_ideal * 100) if v_ideal > 0 else 0.0, 1)
    has_pothole = "pothole" in (analysis_result.get("per_defect") or {})
    return {
        "engine": "python-irc106-los",
        "summary": {
            "ideal_capacity_pcu_hr":    round(ideal_cap, 1),
            "defect_capacity_pcu_hr":   round(defect_cap, 1),
            "ideal_volume_design_pcu":  round(q_ideal, 1),
            "defect_volume_design_pcu": round(q_defect, 1),
            "demand_pcu_hr":            round(demand, 1),
            "steady_state_speed_kmh":   round(v_defect, 1),
            "ideal_speed_kmh":          round(v_ideal, 1),
            "capacity_loss_pct":        round(loss_pct, 1),
            "speed_reduction_pct":      speed_reduction_pct,
            "pothole_speed_impact_pct": speed_reduction_pct if has_pothole else 0.0,
            "los_ideal":                analysis_result.get("level_of_service_ideal"),
            "los":                      analysis_result.get("level_of_service"),
            "vc_ratio":                 analysis_result.get("vc_ratio"),
        },
        "simulation_time_s":      sim_time,
        "ideal_capacity_pcu_hr":  ideal_series,
        "defect_capacity_pcu_hr": defect_series,
        "vehicle_speed_kmh":      speed_series,
    }


def run_and_store(analysis_result: dict) -> dict:
    """Generate twin data and stash it as 'the latest' for the poll endpoints."""
    try:
        twin_data = generate_twin_data(analysis_result)
        _LATEST["status"]    = "done"
        _LATEST["twin_data"] = twin_data
        _LATEST["error"]     = None
        return twin_data
    except Exception as e:  # pragma: no cover - defensive, twin is a bonus view
        _LATEST["status"]    = "error"
        _LATEST["twin_data"] = None
        _LATEST["error"]     = str(e)
        raise


def get_twin_status() -> dict:
    if _LATEST["status"] == "error":
        return {"status": "error", "error": _LATEST["error"]}
    if _LATEST["status"] == "done":
        return {"status": "done", "twin_data": _LATEST["twin_data"]}
    return {"status": "idle"}


def get_latest_twin_data() -> Optional[dict]:
    return _LATEST["twin_data"]
