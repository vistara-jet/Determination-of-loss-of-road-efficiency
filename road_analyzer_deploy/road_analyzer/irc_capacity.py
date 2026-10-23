"""
road_analyzer.irc_capacity
==========================
Pure capacity / level-of-service formulas (no image or model code here, so
every number can be unit-tested and traced to its source).

Sources used
------------
IRC:106-1990  Guidelines for Capacity of Urban Roads in Plain Areas
              - Table 1: PCU equivalency factors (by each vehicle type's own
                share of the traffic stream: 5 % and 10 % or more)
              - Table 2: Design Service Volumes (DSV), urban roads
              - Design service volumes correspond to LOS C, i.e. V/C = 0.70
              - Cl. 5.3: average travel speed at each LOS as a share of
                free-flow speed
IRC:106 (2022 draft revision) Guidelines for Capacity of Urban Roads in
              Plain Terrain
              - Eq. 5: lane capacity from operating speed,
                C_lane = 20.11 * V_os + 210.8   (divided roads)
              - Eq. 6: C = C0 * F_OP * F_BS * F_AC * F_PD
              - Tables 12-16: on-street parking, bus stop, access point and
                pedestrian-crossing factors
              - Tables 8/9: LOS by V/C (undivided / divided)
              - Table 3: operating speeds by road class
IRC:82-2015   Code of Practice for Maintenance of Bituminous Roads
              - Cl. 7.5.3.4: pothole classes (small / medium / large)
              - Table 5.3: urban road condition rating from pothole area %

Anything that is NOT from these sources is listed in ASSUMPTION_DEFAULTS
and reported back in every result under "assumptions", so the user can see
exactly what was measured, what was entered, and what was assumed.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

# =====================================================================
# IRC:106-1990 Table 2 -- Design Service Volume (PCU/h), urban roads
# =====================================================================
IRC106_DSV: Dict[str, Dict[str, Optional[int]]] = {
    "2lane_oneway":    {"arterial": 2400, "sub_arterial": 1900, "collector": 1400},
    "2lane_twoway":    {"arterial": 1500, "sub_arterial": 1200, "collector":  900},
    "3lane_oneway":    {"arterial": 3600, "sub_arterial": 2900, "collector": 2200},
    "4lane_undivided": {"arterial": 3000, "sub_arterial": 2400, "collector": 1800},
    "4lane_divided":   {"arterial": 3600, "sub_arterial": 2900, "collector": None},
    "6lane_undivided": {"arterial": 4800, "sub_arterial": 3800, "collector": None},
    "6lane_divided":   {"arterial": 5400, "sub_arterial": 4300, "collector": None},
    "8lane_divided":   {"arterial": 7200, "sub_arterial": None, "collector": None},
}

IRC106_DSV_LABELS: Dict[str, str] = {
    "2lane_oneway":    "2-Lane One-Way",
    "2lane_twoway":    "2-Lane Two-Way",
    "3lane_oneway":    "3-Lane One-Way",
    "4lane_undivided": "4-Lane Undivided",
    "4lane_divided":   "4-Lane Divided",
    "6lane_undivided": "6-Lane Undivided",
    "6lane_divided":   "6-Lane Divided",
    "8lane_divided":   "8-Lane Divided",
}

FRINGE_CONDITION_DESC: Dict[str, str] = {
    "arterial":     "No frontage access, no standing vehicles, very little cross traffic",
    "sub_arterial": "Frontage development, side roads, bus stops, no standing vehicles",
    "collector":    "Free frontage access, parked vehicles, bus stops, heavy cross traffic",
}

# Which side-friction effects are ALREADY built into each IRC:106-1990
# traffic-condition column. Applying a draft-IRC:106 factor on top of a DSV
# that already includes the same effect would count it twice.
FRINGE_ALREADY_INCLUDES: Dict[str, set] = {
    "arterial":     set(),
    "sub_arterial": {"access", "bus_stop"},
    "collector":    {"access", "bus_stop", "parking"},
}

# IRC:106 design service volumes are set at LOS C. The upper V/C limit of
# LOS C is 0.70, so capacity = DSV / 0.70.
VC_AT_DESIGN_SERVICE_VOLUME = 0.70


def is_divided_like(carriageway_key: str) -> bool:
    """One-way and divided carriageways have no opposing flow in the
    photographed stream; undivided two-way roads do."""
    return carriageway_key.endswith("_divided") or carriageway_key.endswith("_oneway")


def is_divided(carriageway_key: str) -> bool:
    return carriageway_key.endswith("_divided")


def get_irc106_dsv(carriageway_key: str, fringe_condition: str) -> float:
    row = IRC106_DSV.get(carriageway_key)
    if row is None:
        raise ValueError(f"Unknown carriageway type: '{carriageway_key}'")
    val = row.get(fringe_condition)
    if val is None:
        raise ValueError(
            f"IRC:106 Table 2 has no DSV for '{carriageway_key}' "
            f"under '{fringe_condition}' fringe conditions."
        )
    return float(val)


# =====================================================================
# Level of service -- draft IRC:106 (2022) Tables 8 (undivided) / 9 (divided)
# Upper V/C bound of each level.
# =====================================================================
LOS_ORDER = ["A", "B", "C", "D", "E", "F"]
LOS_VC_UNDIVIDED: List[Tuple[float, str]] = [
    (0.35, "A"), (0.55, "B"), (0.70, "C"), (0.85, "D"), (1.00, "E"), (math.inf, "F")]
LOS_VC_DIVIDED: List[Tuple[float, str]] = [
    (0.30, "A"), (0.45, "B"), (0.70, "C"), (0.85, "D"), (1.00, "E"), (math.inf, "F")]

# IRC:106-1990 cl. 5.3 -- average travel speed as a share of free-flow speed.
# Level F is given as 25-33 %; the midpoint 0.29 is used.
LOS_SPEED_FRACTION: Dict[str, float] = {
    "A": 0.90, "B": 0.70, "C": 0.50, "D": 0.40, "E": 0.33, "F": 0.29}

LOS_GUIDANCE: Dict[str, Tuple[str, str]] = {
    "A": ("Adequate (LOS A)", "Design traffic flows freely. Routine monitoring only."),
    "B": ("Adequate (LOS B)", "Design traffic flows freely. Routine monitoring only."),
    "C": ("Adequate (LOS C)", "Road still serves its IRC design volume. Clear defects at the next maintenance cycle."),
    "D": ("Approaching capacity (LOS D)", "Design traffic is near capacity. Schedule removal of the listed obstructions within 1-2 weeks."),
    "E": ("At capacity (LOS E)", "Road is at capacity under design traffic. Treat URGENT items as priority works and re-survey within 1 week."),
    "F": ("Over capacity (LOS F)", "Road can no longer carry its IRC design traffic. Escalate to the traffic and municipal authority immediately."),
}


def level_of_service(vc: float, carriageway_key: str) -> str:
    table = LOS_VC_DIVIDED if is_divided_like(carriageway_key) else LOS_VC_UNDIVIDED
    x = round(vc, 3) if math.isfinite(vc) else vc
    for upper, los in table:
        if x <= upper:
            return los
    return "F"


def los_steps(better: str, worse: str) -> int:
    return LOS_ORDER.index(worse) - LOS_ORDER.index(better)


# =====================================================================
# Operating speed -- draft IRC:106 (2022) Table 3 (km/h)
# =====================================================================
OPERATING_SPEED_KMH: Dict[str, float] = {
    "arterial": 50.0, "sub_arterial": 40.0, "collector": 30.0}


def lane_capacity_from_speed(v_os_kmh: float) -> float:
    """Draft IRC:106 (2022) Eq. 5 (divided roads): C_lane = 20.11 V_os + 210.8."""
    return 20.11 * float(v_os_kmh) + 210.8


# =====================================================================
# IRC:106-1990 Table 1 -- PCU factors
# (value when the type is 5 % of traffic, value when 10 % or more)
# =====================================================================
IRC106_PCU_TABLE1: Dict[str, Tuple[float, float, str]] = {
    "two_wheeler":     (0.50, 0.75, "Two-wheeler (motorcycle, scooter)"),
    "car":             (1.00, 1.00, "Passenger car, pick-up van"),
    "auto_rickshaw":   (1.20, 2.00, "Auto-rickshaw"),
    "lcv":             (1.40, 2.00, "Light commercial vehicle"),
    "truck_bus":       (2.20, 3.70, "Truck or bus"),
    "tractor_trailer": (4.00, 5.00, "Agricultural tractor-trailer"),
    "cycle":           (0.40, 0.50, "Pedal cycle"),
    "cycle_rickshaw":  (1.50, 2.00, "Cycle rickshaw"),
    "tonga":           (1.50, 2.00, "Tonga (horse-drawn)"),
    "hand_cart":       (2.00, 3.00, "Hand cart"),
}

# Used only when the user gives no classified count.
DEFAULT_COMPOSITION: Dict[str, float] = {
    "two_wheeler": 0.40, "car": 0.30, "auto_rickshaw": 0.10, "lcv": 0.10,
    "truck_bus": 0.08, "tractor_trailer": 0.01, "cycle": 0.01,
}


def pcu_for_share(vehicle_type: str, share: float) -> float:
    """IRC:106 Table 1 value for a vehicle type at its own share of the
    stream; linear interpolation between 5 % and 10 %."""
    lo, hi, _ = IRC106_PCU_TABLE1[vehicle_type]
    if share <= 0.05:
        return lo
    if share >= 0.10:
        return hi
    return lo + (hi - lo) * (share - 0.05) / 0.05


def normalise_composition(comp: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not comp:
        return dict(DEFAULT_COMPOSITION)
    clean: Dict[str, float] = {}
    for k, v in comp.items():
        if k not in IRC106_PCU_TABLE1:
            raise ValueError(f"Unknown vehicle type '{k}'. Valid: {list(IRC106_PCU_TABLE1)}")
        v = float(v)
        if v < 0:
            raise ValueError(f"Share for '{k}' cannot be negative.")
        if v > 0:
            clean[k] = v
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Traffic composition shares must add up to more than zero.")
    # Accept either fractions (sum 1) or percentages / counts -> normalise.
    return {k: v / total for k, v in clean.items()}


def average_pcu(comp: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Stream-average PCU per vehicle: P = sum(p_i * PCU_i)."""
    per_type = {k: round(pcu_for_share(k, s), 3) for k, s in comp.items()}
    avg = sum(comp[k] * pcu_for_share(k, comp[k]) for k in comp)
    return avg, per_type


# =====================================================================
# Draft IRC:106 (2022) Eq. 6 adjustment factors
# =====================================================================
def parking_factor(occupancy_pct: Optional[float]) -> Optional[float]:
    """Table 12 -- share of segment length occupied by on-street parking."""
    if occupancy_pct is None:
        return None
    if occupancy_pct <= 0:
        return 1.0
    if occupancy_pct < 25:
        return 0.70
    if occupancy_pct <= 50:
        return 0.65
    return 0.40


def bus_stop_factor(stop_type: Optional[str], buses_per_hr: Optional[float]) -> float:
    """Table 14."""
    if not stop_type or stop_type == "none":
        return 1.0
    high = (buses_per_hr or 0) > 30
    if stop_type == "kerbside":
        return 0.60 if high else 0.80
    if stop_type == "bay":
        return 0.80 if high else 0.90
    raise ValueError("bus_stop_type must be none, kerbside or bay")


def access_factor(movements_per_hr: Optional[float]) -> float:
    """Table 15 -- vehicle entries + exits per hour at access points."""
    if not movements_per_hr or movements_per_hr <= 0:
        return 1.0
    if movements_per_hr < 200:
        return 0.90
    if movements_per_hr <= 500:
        return 0.80
    return 0.70


def pedestrian_factor(peds_per_hr: Optional[float]) -> float:
    """Table 16 -- pedestrians crossing mid-block per hour (table stops at
    1500/h; the last value is held above that)."""
    if not peds_per_hr or peds_per_hr <= 0:
        return 1.0
    if peds_per_hr < 250:
        return 0.90
    if peds_per_hr <= 500:
        return 0.78
    if peds_per_hr <= 1000:
        return 0.72
    return 0.68


# =====================================================================
# IRC:82-2015 potholes
# =====================================================================
POTHOLE_CLASSES = ["small", "medium", "large"]


def pothole_class_from_width(width_m: float) -> str:
    """Cl. 7.5.3.4: small ~200 mm wide, medium ~500 mm, large > 500 mm."""
    if width_m <= 0.20:
        return "small"
    if width_m <= 0.50:
        return "medium"
    return "large"


def pothole_class_from_depth(depth_mm: float) -> str:
    """Cl. 7.5.3.4: small ~25 mm deep, medium 25-50 mm, large > 50 mm."""
    if depth_mm <= 25:
        return "small"
    if depth_mm <= 50:
        return "medium"
    return "large"


def worse_class(a: str, b: str) -> str:
    return a if POTHOLE_CLASSES.index(a) >= POTHOLE_CLASSES.index(b) else b


def urban_pothole_rating(area_pct: Optional[float]) -> Optional[str]:
    """IRC:82-2015 Table 5.3 (urban roads): nil = Good, >0-0.5 % = Fair,
    > 0.5 % = Poor."""
    if area_pct is None:
        return None
    if area_pct <= 0:
        return "Good"
    if area_pct <= 0.5:
        return "Fair"
    return "Poor"


# Speed at a pothole as a share of the normal operating speed. NOT from
# IRC -- used only when no measured speed is supplied, and reported as an
# assumption in every result.
DEFAULT_POTHOLE_SPEED_RATIO: Dict[str, float] = {"small": 0.90, "medium": 0.80, "large": 0.60}


def pothole_capacity_factor(v_normal_kmh: float, v_pothole_kmh: float) -> float:
    """f_p = C_lane(V_pothole) / C_lane(V_normal), both from draft IRC:106 Eq. 5."""
    f = lane_capacity_from_speed(v_pothole_kmh) / lane_capacity_from_speed(v_normal_kmh)
    return max(0.0, min(1.0, f))


# =====================================================================
# Geometry helpers on the lateral (across-road) axis
# =====================================================================
def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ivs = sorted((l, r) for l, r in intervals if r > l)
    merged: List[Tuple[float, float]] = []
    for l, r in ivs:
        if merged and l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))
    return merged


def union_length(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(r - l for l, r in merge_intervals(intervals))


def free_gaps(blocked: Iterable[Tuple[float, float]], width_m: float) -> List[Tuple[float, float]]:
    gaps, cursor = [], 0.0
    for l, r in merge_intervals(blocked):
        l, r = max(0.0, l), min(width_m, r)
        if l > cursor:
            gaps.append((cursor, l))
        cursor = max(cursor, r)
    if cursor < width_m:
        gaps.append((cursor, width_m))
    return gaps


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


# =====================================================================
# Main capacity calculation
# =====================================================================
WIDTH_CLASSES = ("barricade", "street_vendor", "cart", "garbage", "tree_on_road")
PARKING_CLASS = "illegal_parking"
POTHOLE_CLASS = "pothole"


def compute_capacity(defects: List[dict], p: dict) -> dict:
    """
    defects: list of {"cls", "lat_l", "lat_r", "f_p" (potholes only)}
             in metres across the photographed carriageway.
    p: prepared parameters (see prepare_params()).

    Capacity model
    --------------
      C0      = DSV_scope / 0.70                         (IRC:106, LOS C)
      C_lane  = C0 / n                                   (n lanes in photo)
      u_i     = usable free width in lane i / lane width (gaps < g_min unusable)
      p_i     = worst pothole factor among potholes touching lane i
      C_geom  = sum_i C_lane * u_i * p_i
      C_eff   = C_geom * F_OP * F_BS * F_AC * F_PD       (draft IRC:106 Eq. 6)
    """
    W, n, g_min = p["width_m"], p["lanes_in_photo"], p["min_usable_gap_m"]
    lane_w = W / n
    C0 = p["capacity_c0"]
    c_lane = C0 / n

    blocked = [(d["lat_l"], d["lat_r"]) for d in defects if d["cls"] in WIDTH_CLASSES]
    usable_gaps = [g for g in free_gaps(blocked, W) if (g[1] - g[0]) >= g_min - 1e-9]
    potholes = [d for d in defects if d["cls"] == POTHOLE_CLASS]

    lanes = []
    c_geom = 0.0
    for i in range(n):
        lane = (i * lane_w, (i + 1) * lane_w)
        u = sum(overlap(g, lane) for g in usable_gaps) / lane_w
        fps = [d["f_p"] for d in potholes if overlap((d["lat_l"], d["lat_r"]), lane) > 0]
        pf = min(fps) if fps else 1.0
        c = c_lane * u * pf
        c_geom += c
        lanes.append({"lane": i + 1, "from_m": round(lane[0], 2), "to_m": round(lane[1], 2),
                      "usable_fraction": round(u, 3), "pothole_factor": round(pf, 3),
                      "capacity_pcu_hr": round(c, 1)})

    has_parking = any(d["cls"] == PARKING_CLASS for d in defects)
    f_op = 1.0
    if has_parking or p.get("parking_occupancy_pct") is not None:
        if "parking" in p["fringe_includes"]:
            f_op = 1.0
        else:
            occ = p.get("parking_occupancy_pct")
            f_op = parking_factor(occ) if occ is not None else 0.70
    if not p.get("include_parking", True):
        f_op = 1.0

    f_bs = 1.0 if "bus_stop" in p["fringe_includes"] else bus_stop_factor(p.get("bus_stop_type"), p.get("buses_per_hr"))
    f_ac = 1.0 if "access" in p["fringe_includes"] else access_factor(p.get("access_movements_per_hr"))
    f_pd = pedestrian_factor(p.get("ped_crossings_per_hr"))

    c_eff = max(0.0, min(C0, c_geom * f_op * f_bs * f_ac * f_pd))
    usable_width = sum(g[1] - g[0] for g in usable_gaps)
    return {
        "capacity_c0": C0,
        "c_geom": c_geom,
        "c_eff": c_eff,
        "f_op": f_op, "f_bs": f_bs, "f_ac": f_ac, "f_pd": f_pd,
        "blocked_m": min(union_length(blocked), W),
        "usable_width_m": usable_width,
        "unusable_slivers_m": max(0.0, W - min(union_length(blocked), W) - usable_width),
        "lanes": lanes,
        "has_parking": has_parking,
    }


def traffic_state(c_eff: float, p: dict) -> dict:
    """V/C, LOS and expected travel speed for a capacity value."""
    demand = p["demand_pcu_hr"]
    vc = demand / c_eff if c_eff > 0 else math.inf
    los = level_of_service(vc, p["carriageway_key"])
    speed = p["operating_speed_kmh"] * LOS_SPEED_FRACTION[los]
    return {"vc": vc, "los": los, "speed_kmh": speed}
