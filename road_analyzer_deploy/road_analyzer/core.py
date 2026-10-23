"""
road_analyzer.core
==================
Detection + geometry + capacity assembly for the Indian Road Capacity
Analyzer. All capacity / LOS formulas live in irc_capacity.py.

Pipeline for one photo
----------------------
1. YOLOv8 detection; per-class confidence thresholds (calibrated file if
   present); confidence-aware vendor/cart vs vehicle conflict resolution.
2. Geometry: if the user marked the four carriageway corners on the photo,
   a perspective (homography) transform maps every detection's ground
   contact points to metres on the road plane. Otherwise the photo width is
   taken as the carriageway width and this is flagged in the result.
   Objects that fall outside the carriageway are ignored.
3. Potholes: IRC:82-2015 class from measured width (and depth if entered);
   capacity factor from the speed drop via draft IRC:106 Eq. 5.
4. Capacity: Estimated Capacity (LOS C Assumption) = DSV / 0.70, assuming
   the DSV corresponds to LOS C operating conditions (V/C = 0.70); then
   lane-by-lane usable width and pothole factors;
   then draft IRC:106 Eq. 6 factors for parking, bus stops, access points
   and pedestrians (only those NOT already included in the chosen
   IRC:106-1990 traffic-condition column).
5. V/C of the design demand (or observed volume), LOS, expected speed,
   per-defect impact ("LOS if removed"), and an assumptions register.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from road_analyzer import irc_capacity as irc
from road_analyzer import road_edges
from road_analyzer.irc_capacity import (  # re-exported for app.py / reports
    IRC106_DSV, IRC106_DSV_LABELS, FRINGE_CONDITION_DESC, OPERATING_SPEED_KMH,
    IRC106_PCU_TABLE1, DEFAULT_COMPOSITION, get_irc106_dsv, union_length as union_of_intervals,
)

logger = logging.getLogger("road_analyzer.core")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _log_memory_usage(label: str) -> None:
    if not _HAS_PSUTIL:
        return
    try:
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        logger.info("%s: process_rss=%.0fMB system_available=%.0fMB (%.0f%% used)",
                    label, proc.memory_info().rss / 1048576, vm.available / 1048576, vm.percent)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Backwards-compatible names used by app.py / frontend
# ---------------------------------------------------------------------
IRC_FREE_FLOW_SPEED: Dict[str, Dict[str, float]] = {
    k: dict(OPERATING_SPEED_KMH) for k in IRC106_DSV}
IRC106_PCU_FACTORS = {
    "share_5pct":       {k: v[0] for k, v in IRC106_PCU_TABLE1.items()},
    "share_10pct_plus": {k: v[1] for k, v in IRC106_PCU_TABLE1.items()},
}
TRAFFIC_REGIME_DESC: Dict[str, str] = {
    "composition": "PCU from IRC:106-1990 Table 1 using each vehicle type's own share "
                   "(enter a classified count for best accuracy).",
}


def get_free_flow_speed(carriageway_key: str, fringe_condition: str) -> float:
    return OPERATING_SPEED_KMH.get(fringe_condition, 50.0)


# ---------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------
DEFAULT_CONF_THRESHOLDS: Dict[str, float] = {
    "barricade": 0.45, "pothole": 0.45, "illegal_parking": 0.50,
    "street_vendor": 0.65, "cart": 0.65, "garbage": 0.45,
    "tree_on_road": 0.50, "vehicle": 0.40,
}
DEFAULT_CONF = 0.45
CONF_THRESHOLDS = DEFAULT_CONF_THRESHOLDS  # legacy name

CLASS_NAMES = ["barricade", "pothole", "illegal_parking", "street_vendor",
               "cart", "garbage", "vehicle"]  # replaced by model.names at load time

# Objects standing on the road: only their bottom edge touches the ground.
# Flat objects (potholes) lie entirely on the road plane.
FLAT_CLASSES = {"pothole"}

INFERENCE_IMGSZ = int(os.environ.get("ROAD_IMGSZ", "640"))
VETO_IOU = float(os.environ.get("ROAD_VETO_IOU", "0.40"))
DEFAULT_MIN_USABLE_GAP_M = 1.0


def load_conf_thresholds(model_path: str) -> Tuple[Dict[str, float], bool]:
    """Use models/thresholds.json (written by tools/calibrate_thresholds.py)
    when it exists, else the built-in defaults."""
    p = Path(model_path).with_name("thresholds.json")
    if p.exists():
        try:
            data = json.loads(p.read_text())
            th = dict(DEFAULT_CONF_THRESHOLDS)
            th.update({k: float(v) for k, v in data.get("thresholds", data).items()})
            return th, True
        except Exception as e:
            logger.warning("Could not read %s (%s); using default thresholds.", p, e)
    return dict(DEFAULT_CONF_THRESHOLDS), False


def iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def apply_vehicle_veto(detections: List[dict],
                       veto_classes: Tuple[str, ...] = ("street_vendor", "cart"),
                       iou_threshold: float = VETO_IOU) -> Tuple[List[dict], int]:
    """Where a vendor/cart box and a vehicle box describe the same object
    (IoU >= threshold), keep only the more confident label. Previously the
    vendor/cart was always dropped, even when the model was surer of it."""
    drop = set()
    for i, d in enumerate(detections):
        if d["cls_name"] not in veto_classes:
            continue
        for j, v in enumerate(detections):
            if v["cls_name"] != "vehicle" or j in drop:
                continue
            if iou(d["xyxy"], v["xyxy"]) >= iou_threshold:
                drop.add(i if v["conf"] >= d["conf"] else j)
                if i in drop:
                    break
    kept = [d for k, d in enumerate(detections) if k not in drop]
    vetoed = sum(1 for k in drop if detections[k]["cls_name"] in veto_classes)
    return kept, vetoed


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
class RoadGeometry:
    """Maps image pixels to road-plane metres.

    Calibrated: road_edge_points = 4 [x, y] pixel points on the ORIGINAL
    photo in the order near-left, near-right, far-right, far-left (the
    corners of a stretch of carriageway). A homography maps them to a
    rectangle W x L. Lateral position is exact for any L; longitudinal
    position is only reported when calib_length_m (the real distance
    between the near and far lines) is given.

    Uncalibrated: the photo width is assumed to span the carriageway with
    no perspective (flagged in the result)."""

    def __init__(self, cfg: dict, img_w: int, img_h: int, scale: float):
        self.W = float(cfg["total_width_m"])
        self.L = cfg.get("calib_length_m") or None
        self.img_w, self.img_h = img_w, img_h
        self.H = None
        pts = cfg.get("road_edge_points")
        if pts:
            src = np.float32(pts) * scale
            if src.shape != (4, 2):
                raise ValueError("road_edge_points must be 4 [x, y] points.")
            if not (src[0, 1] > src[3, 1] and src[1, 1] > src[2, 1]):
                raise ValueError("road_edge_points order must be near-left, near-right, "
                                 "far-right, far-left (near points lower in the photo).")
            Ld = float(self.L) if self.L else self.W
            dst = np.float32([[0, 0], [self.W, 0], [self.W, Ld], [0, Ld]])
            self.H = cv2.getPerspectiveTransform(src, dst)
            # Sign of the homogeneous scale for points on the road (the
            # midpoint of the near edge is on the road by definition).
            mid = (src[0] + src[1]) / 2.0
            self._wsign = 1.0 if (self.H @ np.array([mid[0], mid[1], 1.0]))[2] > 0 else -1.0
        self.method = ("homography (4 marked carriageway corners)" if self.H is not None
                       else "uncalibrated: photo width = carriageway width, no perspective")
        self.src = (np.float32(pts) * scale) if pts else None

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def _map(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        v = self.H @ np.array([x, y, 1.0])
        if v[2] * self._wsign <= 1e-9:   # at/above the horizon -> not on the road plane
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    def footprint(self, xyxy, flat: bool) -> Optional[dict]:
        x1, y1, x2, y2 = xyxy
        if not self.calibrated:
            s = self.img_w / self.W
            return {"lat_l": x1 / s, "lat_r": x2 / s, "long_near": None, "long_far": None}
        pts = [(x1, y2), (x2, y2)] + ([(x1, y1), (x2, y1)] if flat else [])
        mapped = [m for m in (self._map(x, y) for x, y in pts) if m is not None]
        if len(mapped) < 2:
            return None
        lats = [m[0] for m in mapped]
        longs = [m[1] for m in mapped]
        out = {"lat_l": min(lats), "lat_r": max(lats), "long_near": None, "long_far": None}
        if self.L:
            out["long_near"], out["long_far"] = min(longs), max(longs)
        return out


# ---------------------------------------------------------------------
# Parameter preparation
# ---------------------------------------------------------------------
def prepare_params(cfg: dict, occupancy_override: Optional[float] = None) -> dict:
    key, fringe = cfg["carriageway_key"], cfg["fringe_condition"]
    W, n = float(cfg["total_width_m"]), int(cfg["num_lanes"])
    if W <= 0 or n <= 0:
        raise ValueError("total_width_m and num_lanes must be positive.")
    dsv_total = get_irc106_dsv(key, fringe)

    covers = cfg.get("photo_covers") or "whole"
    if covers == "one_direction":
        if not irc.is_divided(key):
            raise ValueError("photo_covers='one_direction' only applies to divided carriageways.")
        lanes_in_photo = max(1, n // 2)
        dsv_scope = dsv_total / 2.0
    else:
        lanes_in_photo = n
        dsv_scope = dsv_total

    comp = irc.normalise_composition(cfg.get("traffic_composition"))
    avg_pcu, pcu_per_type = irc.average_pcu(comp)

    occ = occupancy_override if occupancy_override is not None else cfg.get("parking_occupancy_pct")
    return {
        "carriageway_key": key, "fringe": fringe, "width_m": W,
        "lanes_in_photo": lanes_in_photo, "photo_covers": covers,
        "dsv_total": dsv_total, "dsv_scope": dsv_scope,
        "capacity_c0": dsv_scope / irc.VC_AT_DESIGN_SERVICE_VOLUME,
        "operating_speed_kmh": float(cfg.get("operating_speed_kmh") or OPERATING_SPEED_KMH[fringe]),
        "operating_speed_measured": bool(cfg.get("operating_speed_kmh")),
        "demand_pcu_hr": float(cfg.get("observed_volume_pcu_hr") or dsv_scope),
        "demand_is_observed": bool(cfg.get("observed_volume_pcu_hr")),
        "min_usable_gap_m": float(cfg.get("min_usable_gap_m") or DEFAULT_MIN_USABLE_GAP_M),
        "parking_occupancy_pct": occ,
        "parking_occupancy_source": ("measured from photos/frames" if occupancy_override is not None
                                     else ("entered by user" if cfg.get("parking_occupancy_pct") is not None
                                           else None)),
        "bus_stop_type": cfg.get("bus_stop_type") or "none",
        "buses_per_hr": cfg.get("buses_per_hr"),
        "access_movements_per_hr": cfg.get("access_movements_per_hr"),
        "ped_crossings_per_hr": cfg.get("ped_crossings_per_hr"),
        "fringe_includes": irc.FRINGE_ALREADY_INCLUDES[fringe],
        "composition": comp, "composition_entered": bool(cfg.get("traffic_composition")),
        "avg_pcu": avg_pcu, "pcu_per_type": pcu_per_type,
        "include_parking": True, "include_side_friction": True,
    }


def _scenario(defects: List[dict], p: dict, include_parking=True, include_side_friction=True) -> dict:
    q = dict(p)
    q["include_parking"] = include_parking
    if not include_side_friction:
        q.update(bus_stop_type="none", access_movements_per_hr=None, ped_crossings_per_hr=None)
    if not include_parking:
        q["parking_occupancy_pct"] = None
    return irc.compute_capacity(defects, q)


# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------
IRC_ACTION_RULES: Dict[str, dict] = {
    "pothole": {
        "code_ref": "IRC:82-2015 (Maintenance of Bituminous Roads) cl. 7.5.3; IRC:SP:55-2014 (work zone)",
        "MONITOR": "Log in the pavement condition register (IRC:82 Table 5.3) and check at the next routine inspection.",
        "ROUTINE": "Fill with open- or dense-graded premix per IRC:82 cl. 7.5.3.5; barricade until repaired.",
        "URGENT":  "Large pothole (IRC:82 cl. 7.5.3.4) / LOS impact: patch immediately with premix or cold mix and place warning signs (IRC:67).",
    },
    "illegal_parking": {
        "code_ref": "Motor Vehicles Act 1988 s.122; IRC:67-2012 (signs); IRC:35-2015 (markings)",
        "MONITOR": "Repaint No-Parking markings and signs.",
        "ROUTINE": "Deploy traffic wardens at peak hours; enforce No-Parking signage.",
        "URGENT":  "Immediate towing enforcement under MV Act s.122; install bollards along the stretch.",
    },
    "street_vendor": {
        "code_ref": "Street Vendors (Protection of Livelihood and Regulation of Street Vending) Act, 2014",
        "MONITOR": "Record vendor positions for Town Vending Committee review.",
        "ROUTINE": "Coordinate with the TVC to move vendors to a designated vending zone.",
        "URGENT":  "Immediate relocation drive with the municipal hawking squad and TVC.",
    },
    "cart": {
        "code_ref": "Street Vendors Act 2014; municipal bye-laws on loading/unloading zones",
        "MONITOR": "Monitor cart movement; no immediate action.",
        "ROUTINE": "Restrict carts to off-peak hours or designated zones.",
        "URGENT":  "Remove from the carriageway now; designate an alternative loading bay.",
    },
    "garbage": {
        "code_ref": "Solid Waste Management Rules, 2016",
        "MONITOR": "Clear at the next municipal collection round.",
        "ROUTINE": "Request priority clearance within 48 hours under SWM Rules 2016.",
        "URGENT":  "Immediate clearance; install a community bin to prevent recurrence.",
    },
    "barricade": {
        "code_ref": "IRC:SP:55-2014 (Traffic Management in Work Zones); IRC:67-2012",
        "MONITOR": "Verify the work zone is permitted and signed per IRC:SP:55.",
        "ROUTINE": "Reduce barricaded width to the IRC:SP:55 minimum; add diversion signs.",
        "URGENT":  "Coordinate with the executing agency to relocate now; advance warning signs per IRC:67.",
    },
    "tree_on_road": {
        "code_ref": "IRC:SP:55-2014; municipal tree authority bye-laws",
        "MONITOR": "Log for tree-authority review.",
        "ROUTINE": "Request pruning within 7 days.",
        "URGENT":  "Immediate removal/pruning with diversion signs.",
    },
}


def get_irc_action(defect_name: str, tier: str) -> dict:
    rule = IRC_ACTION_RULES.get(defect_name)
    if rule is None:
        return {"code_ref": "N/A", "severity": "INVESTIGATE",
                "action": "No standard action mapped; flag for manual inspection."}
    return {"code_ref": rule["code_ref"], "severity": tier, "action": rule[tier]}


TARGET_LOS = "C"   # IRC:106 design level of service


def restoration_plan(defects: List[dict], p: dict) -> List[dict]:
    """Smallest ordered set of defect-class removals that brings the design
    demand back to LOS C (greedy: each step removes the class that restores
    the most capacity)."""
    remaining = sorted({d["cls"] for d in defects})
    removed: List[str] = []
    plan = []
    def cap(excl):
        keep = [d for d in defects if d["cls"] not in excl]
        return _scenario(keep, p, include_parking="illegal_parking" not in excl)["c_eff"]
    current = irc.traffic_state(cap(removed), p)
    while remaining and irc.LOS_ORDER.index(current["los"]) > irc.LOS_ORDER.index(TARGET_LOS):
        best = max(remaining, key=lambda c: cap(removed + [c]))
        removed.append(best)
        remaining.remove(best)
        c_after = cap(removed)
        current = irc.traffic_state(c_after, p)
        plan.append({"step": len(plan) + 1, "remove": best, "capacity_after_pcu_hr": round(c_after, 1),
                     "vc_after": round(current["vc"], 3), "los_after": current["los"]})
    return plan


def _tier(cls: str, los_now: str, plan: List[dict], pothole_class: Optional[str] = None) -> str:
    """Defects needed to restore LOS C are URGENT when the road is at or
    over capacity (LOS E/F) and ROUTINE at LOS D; all others MONITOR.
    IRC:82 large potholes are always URGENT and medium ones at least
    ROUTINE (road-user safety, independent of capacity)."""
    needed = any(s["remove"] == cls for s in plan)
    if needed:
        t = "URGENT" if los_now in ("E", "F") else "ROUTINE"
    else:
        t = "MONITOR"
    if pothole_class == "large":
        t = "URGENT"
    elif pothole_class == "medium" and t == "MONITOR":
        t = "ROUTINE"
    return t


def get_overall_guidance_for_los(los: str) -> dict:
    band, action = irc.LOS_GUIDANCE[los]
    return {"band": band, "action": action, "los": los}


# ---------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------
class RoadAnalyzer:
    def __init__(self, model_path: str, enable_depth: bool = False):
        from ultralytics import YOLO
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.class_names = [self.model.names[i] for i in sorted(self.model.names)]
        global CLASS_NAMES
        CLASS_NAMES[:] = self.class_names
        self.conf_thresholds, self.thresholds_calibrated = load_conf_thresholds(self.model_path)
        self.enable_depth = enable_depth
        self._call_count = 0
        logger.info("RoadAnalyzer: model %s classes=%s calibrated_thresholds=%s",
                    self.model_path, self.class_names, self.thresholds_calibrated)

    # ---------------- detection ----------------
    def _detect(self, image_bgr, image_label: str = "") -> Tuple[List[dict], int]:
        self._call_count += 1
        pred = self.model.predict(image_bgr, conf=0.10, verbose=False,
                                  imgsz=INFERENCE_IMGSZ, device="cpu")[0]
        raw = []
        if pred.boxes is not None:
            for box in pred.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf < self.conf_thresholds.get(cls_name, DEFAULT_CONF):
                    continue
                raw.append({"cls_name": cls_name, "conf": conf,
                            "xyxy": tuple(float(v) for v in box.xyxy[0])})
        kept, vetoed = apply_vehicle_veto(raw)
        logger.info("detect #%d %s kept=%d vetoed=%d", self._call_count, image_label, len(kept), vetoed)
        return kept, vetoed

    # ---------------- stage 1: detection + geometry ----------------
    def extract(self, image_path: str, road_config: dict) -> dict:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h0, w0 = img.shape[:2]
        # Keep at least 2x the inference size so nothing the model can use
        # is thrown away; larger photos are reduced to save CPU time.
        max_edge = max(1600, 2 * INFERENCE_IMGSZ)
        scale = 1.0
        if max(h0, w0) > max_edge:
            scale = max_edge / max(h0, w0)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

        # Road edges: the user's marked corners win; otherwise detect them.
        edge_info = {"source": "marked" if road_config.get("road_edge_points") else "none"}
        geo_cfg = road_config
        if not road_config.get("road_edge_points") and road_config.get("auto_edges", "auto") != "off":
            det = road_edges.detect_road_corners(img)
            edge_info = {"source": "auto", "confidence": det["confidence"], "method": det["method"],
                         "reason": det["reason"], "accepted": det["points"] is not None}
            if det["points"] is not None:
                # detector returns points on this (possibly resized) image;
                # RoadGeometry expects original-photo pixels.
                geo_cfg = dict(road_config, road_edge_points=[[x / scale, y / scale] for x, y in det["points"]])
                edge_info["points_original_px"] = geo_cfg["road_edge_points"]
        geo = RoadGeometry(geo_cfg, w, h, scale)
        if edge_info["source"] == "auto" and geo.calibrated:
            geo.method = (f"homography from auto-detected road edges "
                          f"({edge_info['method']}, confidence {edge_info['confidence']:.2f})")
        detections, vetoed = self._detect(img, Path(image_path).name)
        W = geo.W

        defects, off_road, vehicles = [], [], 0
        for d in detections:
            cls = d["cls_name"]
            if cls == "vehicle":
                vehicles += 1
                continue
            fp = geo.footprint(d["xyxy"], flat=cls in FLAT_CLASSES)
            if fp is None or fp["lat_r"] <= 0 or fp["lat_l"] >= W:
                off_road.append({"cls": cls, "conf": round(d["conf"], 2),
                                 "reason": "outside the marked carriageway" if geo.calibrated else "outside photo"})
                continue
            lat_l, lat_r = max(0.0, fp["lat_l"]), min(W, fp["lat_r"])
            rec = {"cls": cls, "conf": d["conf"], "xyxy": d["xyxy"],
                   "lat_l": lat_l, "lat_r": lat_r, "width_m": lat_r - lat_l,
                   "long_near": fp["long_near"], "long_far": fp["long_far"],
                   "length_m": (fp["long_far"] - fp["long_near"]) if fp["long_near"] is not None else None}
            defects.append(rec)

        return {"image": Path(image_path).name, "image_path": str(image_path),
                "img_w": w, "img_h": h, "scale": scale,
                "geometry_method": geo.method, "calibrated": geo.calibrated,
                "calib_length_m": geo.L, "defects": defects, "off_road": off_road,
                "vehicles_detected": vehicles, "vetoed": vetoed,
                "has_parking": any(x["cls"] == "illegal_parking" for x in defects),
                "edge_detection": edge_info,
                "_img": img, "_H": geo.H, "_src": geo.src}

    # ---------------- stage 2: capacity + LOS ----------------
    def assemble(self, ex: dict, road_config: dict,
                 occupancy_override: Optional[float] = None) -> dict:
        p = prepare_params(road_config, occupancy_override)
        W = p["width_m"]
        defects = [dict(d) for d in ex["defects"]]

        # ---- potholes: IRC:82 class + speed-based capacity factor ----
        potholes = [d for d in defects if d["cls"] == "pothole"]
        depth_mm = road_config.get("pothole_depth_mm")
        v_meas = road_config.get("pothole_speed_kmh")
        widest = max(potholes, key=lambda d: d["width_m"]) if potholes else None
        for d in potholes:
            cls_w = irc.pothole_class_from_width(d["width_m"])
            c = cls_w
            if depth_mm and d is widest:
                c = irc.worse_class(cls_w, irc.pothole_class_from_depth(float(depth_mm)))
            d["irc82_class"] = c
            v_p = float(v_meas) if v_meas else p["operating_speed_kmh"] * irc.DEFAULT_POTHOLE_SPEED_RATIO[c]
            v_p = min(v_p, p["operating_speed_kmh"])
            d["speed_at_pothole_kmh"] = v_p
            d["f_p"] = irc.pothole_capacity_factor(p["operating_speed_kmh"], v_p)

        full = irc.compute_capacity(defects, p)
        C0, c_eff = full["capacity_c0"], full["c_eff"]
        ideal = irc.traffic_state(C0, p)
        now = irc.traffic_state(c_eff, p)

        # ---- per-defect impact ----
        plan = restoration_plan(defects, p)
        per_defect: Dict[str, dict] = {}
        classes = sorted({d["cls"] for d in defects})
        for cls in classes:
            mine = [d for d in defects if d["cls"] == cls]
            others = [d for d in defects if d["cls"] != cls]
            iso = _scenario(mine, p, include_parking=(cls == "illegal_parking"), include_side_friction=False)
            without = _scenario(others, p, include_parking=(cls != "illegal_parking"))
            st_without = irc.traffic_state(without["c_eff"], p)
            worst_cls = None
            if cls == "pothole":
                worst_cls = max((d["irc82_class"] for d in mine), key=irc.POTHOLE_CLASSES.index)
            tier = _tier(cls, now["los"], plan, worst_cls)
            act = get_irc_action(cls, tier)
            loss = C0 - iso["c_eff"]
            blocked = min(irc.union_length((d["lat_l"], d["lat_r"]) for d in mine), W)
            entry = {
                "count": len(mine),
                "blocked_m": round(blocked, 2),
                "capacity_loss_pcu": round(loss, 1),
                "capacity_loss_pct": round(loss / C0 * 100, 1) if C0 else 0.0,
                "width_factor": round(1 - blocked / W, 3) if cls in irc.WIDTH_CLASSES else 1.0,
                "restored_capacity_if_removed_pcu_hr": round(without["c_eff"] - c_eff, 1),
                "los_if_removed": st_without["los"],
                "vc_if_removed": round(st_without["vc"], 3),
                "severity": act["severity"], "code_ref": act["code_ref"], "action": act["action"],
                "detections": [{
                    "conf": round(d["conf"], 2), "width_m": round(d["width_m"], 2),
                    "length_m": round(d["length_m"], 2) if d["length_m"] is not None else None,
                    "height_m": None,
                    "left_m": round(d["lat_l"], 2), "right_m": round(d["lat_r"], 2),
                    "long_m": round(d["long_near"], 2) if d["long_near"] is not None else None,
                    **({"irc82_class": d["irc82_class"], "capacity_factor": round(d["f_p"], 3),
                        "speed_at_pothole_kmh": round(d["speed_at_pothole_kmh"], 1)} if cls == "pothole" else {}),
                } for d in mine],
            }
            if cls == "illegal_parking":
                entry["capacity_method"] = ("Included in IRC:106 'collector' DSV (no extra factor)"
                                            if "parking" in p["fringe_includes"]
                                            else "Draft IRC:106 Table 12 on-street parking factor F_OP")
                entry["factor_applied"] = full["f_op"]
                entry["occupancy_pct"] = p["parking_occupancy_pct"]
            if cls == "pothole":
                area_pct = None
                if ex.get("calib_length_m") and all(d["length_m"] is not None for d in mine):
                    area = sum(d["width_m"] * max(d["length_m"], 0) for d in mine)
                    area_pct = round(area / (W * float(ex["calib_length_m"])) * 100, 3)
                entry["capacity_method"] = "Draft IRC:106 Eq. 5 speed-capacity ratio, applied to affected lanes"
                entry["depth_summary"] = {
                    "worst_severity": worst_cls,
                    "classification_basis": ("width + measured depth (IRC:82 cl. 7.5.3.4)" if depth_mm
                                             else "measured width only (IRC:82 cl. 7.5.3.4); depth needs a field check"),
                    "measured_depth_mm": float(depth_mm) if depth_mm else None,
                    "avg_estimated_depth_cm": round(float(depth_mm) / 10, 1) if depth_mm else None,
                    "worst_width_m": round(max(d["width_m"] for d in mine), 2),
                    "pothole_area_pct": area_pct,
                    "irc82_condition_rating": irc.urban_pothole_rating(area_pct),
                    "penalty_applied": round(min(d["f_p"] for d in mine), 3),
                    "speed_source": "measured" if v_meas else "assumed (enter measured speed)",
                    "scored_count": len(mine), "unscored_count": 0,
                }
            per_defect[cls] = entry

        # ---- vehicles/h ----
        avg_pcu = p["avg_pcu"]
        veh = lambda pcu: round(pcu / avg_pcu, 0)

        worst_pothole = per_defect.get("pothole", {}).get("depth_summary", {}).get("worst_severity", "none")
        guidance = get_overall_guidance_for_los(now["los"])
        warnings = self._warnings(ex, p, full, road_config)

        factors_str = (f"F_OP {full['f_op']:.2f} x F_BS {full['f_bs']:.2f} x "
                       f"F_AC {full['f_ac']:.2f} x F_PD {full['f_pd']:.2f}")
        result = {
            "image": ex["image"],
            "image_size_px": {"width": ex["img_w"], "height": ex["img_h"]},
            "road_config": road_config,
            "irc_basis": {
                "base_dsv_pcu_hr": p["dsv_total"],
                "dsv_for_photographed_stream_pcu_hr": p["dsv_scope"],
                "capacity_c0_pcu_hr": round(C0, 1),
                "carriageway_key": p["carriageway_key"],
                "fringe_condition": p["fringe"],
                "fringe_desc": FRINGE_CONDITION_DESC.get(p["fringe"], ""),
                "source": "IRC:106-1990 Table 2 DSV; Estimated Capacity (LOS C Assumption) = DSV / 0.70",
            },
            "capacity_basis": {
                "label": "Estimated Capacity (LOS C Assumption)",
                "formula": "Capacity = DSV / 0.70",
                "assumption": "DSV corresponds to LOS C operating conditions (V/C = 0.70)",
                "value_pcu_hr": round(C0, 1),
            },
            "free_flow_speed_kmh": p["operating_speed_kmh"],
            "operating_speed_kmh": p["operating_speed_kmh"],
            "traffic_regime": {
                "regime": "composition",
                "desc": TRAFFIC_REGIME_DESC["composition"],
                "composition": {k: round(v, 3) for k, v in p["composition"].items()},
                "composition_source": "entered" if p["composition_entered"] else "default (assumed)",
                "avg_pcu_per_vehicle": round(avg_pcu, 3),
                "pcu_factors": p["pcu_per_type"],
                "base_vehicles_per_hr": veh(C0),
                "reduced_vehicles_per_hr": veh(c_eff),
                "free_flow_speed_kmh": p["operating_speed_kmh"],
                "ideal_speed_kmh": round(ideal["speed_kmh"], 1),
                "congested_speed_kmh": round(now["speed_kmh"], 1),
                "note": "PCU per IRC:106-1990 Table 1 at each type's own share; speeds per IRC:106-1990 cl. 5.3 at the resulting LOS.",
            },
            "capacity_calculation": {
                "base_dsv_pcu_hr": round(p["dsv_scope"], 1),
                "capacity_c0_pcu_hr": round(C0, 1),
                "base_vehicles_per_hr": veh(C0),
                "total_width_m": round(W, 2),
                "lanes_in_photo": p["lanes_in_photo"],
                "total_blocked_m": round(full["blocked_m"], 2),
                "unusable_slivers_m": round(full["unusable_slivers_m"], 2),
                "effective_width_m": round(full["usable_width_m"], 2),
                "width_factor": round(full["usable_width_m"] / W, 3),
                "geometric_capacity_pcu_hr": round(full["c_geom"], 1),
                "pothole_penalty": per_defect.get("pothole", {}).get("depth_summary", {}).get("penalty_applied", 1.0),
                "worst_pothole_depth": worst_pothole,
                "side_friction_factors": {"F_OP": full["f_op"], "F_BS": full["f_bs"],
                                          "F_AC": full["f_ac"], "F_PD": full["f_pd"]},
                "lanes": full["lanes"],
                "avg_pcu_per_vehicle": round(avg_pcu, 3),
                "formula": (f"Estimated Capacity (LOS C Assumption) = DSV / 0.70 = {p['dsv_scope']:.0f} / 0.70 = {C0:.0f} PCU/h; "
                            f"C_geom = sum(C_lane x u_i x p_i) = {full['c_geom']:.0f}; "
                            f"C_eff = C_geom x {factors_str} = {c_eff:.0f} PCU/h"),
            },
            "original_capacity_pcu_hr": round(C0, 1),
            "design_service_volume_pcu_hr": round(p["dsv_scope"], 1),
            "original_capacity_vehicles_hr": veh(C0),
            "reduced_capacity_pcu_hr": round(c_eff, 1),
            "reduced_capacity_vehicles_hr": veh(c_eff),
            "capacity_loss_pcu_hr": round(C0 - c_eff, 1),
            "capacity_loss_vehicles_hr": veh(C0) - veh(c_eff),
            "capacity_loss_pct": round((C0 - c_eff) / C0 * 100, 1) if C0 else 0.0,
            "demand_pcu_hr": round(p["demand_pcu_hr"], 1),
            "demand_basis": "observed volume (entered)" if p["demand_is_observed"] else "IRC:106 design service volume",
            "vc_ratio_ideal": round(ideal["vc"], 3),
            "vc_ratio": round(now["vc"], 3) if math.isfinite(now["vc"]) else None,
            "level_of_service_ideal": ideal["los"],
            "level_of_service": now["los"],
            "overall_guidance": guidance,
            "restoration_plan": plan,
            "effective_width_m": round(full["usable_width_m"], 2),
            "vehicle_veto_suppressed": ex["vetoed"],
            "vehicles_detected": ex["vehicles_detected"],
            "per_defect": per_defect,
            "geometry": {"method": ex["geometry_method"], "calibrated": ex["calibrated"],
                         "edge_detection": {k: v for k, v in (ex.get("edge_detection") or {}).items()},
                         "calib_length_m": ex.get("calib_length_m"),
                         "ignored_outside_carriageway": ex["off_road"]},
            "roadrunner_obstacles": [{
                "defect_type": d["cls"], "left_m": round(d["lat_l"], 3), "right_m": round(d["lat_r"], 3),
                "width_m": round(d["width_m"], 3),
                "height_m": round(d["length_m"], 3) if d["length_m"] is not None else 0.0,
                "long_m": round(d["long_near"], 3) if d["long_near"] is not None else None,
                "conf": round(d["conf"], 3)} for d in defects],
            "warnings": warnings,
        }
        result["assumptions"] = assumption_register(result, p, ex, road_config, self.thresholds_calibrated)
        return result

    def _warnings(self, ex, p, full, cfg) -> List[str]:
        w = []
        ed = ex.get("edge_detection") or {}
        if ed.get("source") == "auto" and ex["calibrated"]:
            w.append(f"Road edges detected automatically (confidence {ed['confidence']:.2f}). "
                     "Check the edge overlay; if the lines miss the carriageway, mark the corners by hand.")
        if not ex["calibrated"]:
            why = f" Automatic edge detection: {ed['reason']}" if ed.get("source") == "auto" and ed.get("reason") else ""
            w.append("Photo not calibrated: lateral positions assume the photo edges are the carriageway edges. "
                     "Mark the 4 carriageway corners for perspective-correct measurement." + why)
        if full["has_parking"] and p["parking_occupancy_pct"] is None and "parking" not in p["fringe_includes"]:
            w.append("Parking detected but its occupancy along the segment is unknown: lowest IRC band (<25 %, F_OP 0.70) used. "
                     "Analyse a batch/video of the stretch or enter the measured occupancy.")
        if full["has_parking"] and p["fringe"] in ("arterial", "sub_arterial"):
            w.append(f"IRC:106 '{p['fringe']}' conditions assume no standing vehicles; parking is handled by F_OP.")
        if any(d["cls"] == "pothole" for d in ex["defects"]) and not cfg.get("pothole_speed_kmh"):
            w.append("Pothole speed not measured: default speed reduction by IRC:82 class used for the capacity factor.")
        if full["unusable_slivers_m"] >= 0.05:
            w.append(f"{full['unusable_slivers_m']:.2f} m of free width is in gaps narrower than "
                     f"{p['min_usable_gap_m']:.1f} m and counted as unusable.")
        if ex["off_road"]:
            w.append(f"{len(ex['off_road'])} detection(s) lie outside the carriageway and were ignored.")
        return w

    # ---------------- public single-image API ----------------
    def analyse_image(self, image_path: str, road_config: dict, save_outputs: bool = True,
                      output_dir: Optional[str] = None,
                      occupancy_override: Optional[float] = None) -> dict:
        _log_memory_usage(f"analyse_image start ({Path(image_path).name})")
        ex = self.extract(image_path, road_config)
        result = self.assemble(ex, road_config, occupancy_override)
        if save_outputs:
            self._save(result, image_path, output_dir)
            self._save_views(ex, result, road_config, image_path, output_dir)
        return result

    @staticmethod
    def _save_views(ex: dict, result: dict, cfg: dict, image_path: str, output_dir: Optional[str]) -> None:
        """Edge overlay on the photo and a top-down (bird's-eye) view of the
        calibrated stretch with each detection's footprint."""
        result["edges_image"] = result["birdseye_image"] = None
        if ex.get("_H") is None:
            return
        out_dir = Path(output_dir) if output_dir else Path(image_path).parent
        stem = Path(image_path).stem
        try:
            ov = road_edges.draw_edges_overlay(ex["_img"], ex["_src"].tolist())
            cv2.imwrite(str(out_dir / f"{stem}_edges.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result["edges_image"] = f"{stem}_edges.jpg"
            W = float(cfg["total_width_m"])
            L = float(ex["calib_length_m"]) if ex.get("calib_length_m") else W
            lanes = result["capacity_calculation"]["lanes_in_photo"]
            ppm = max(20.0, min(60.0, 900.0 / max(W, L)))
            top = road_edges.birdseye_view(ex["_img"], ex["_H"], W, L, lanes, ex["defects"], px_per_m=ppm)
            cv2.imwrite(str(out_dir / f"{stem}_birdseye.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result["birdseye_image"] = f"{stem}_birdseye.jpg"
            result["birdseye_note"] = (f"Top-down view of the marked stretch, {W:g} m wide x {L:g} m long."
                                       if ex.get("calib_length_m") else
                                       f"Top-down view, {W:g} m wide. Along-road length is not to scale "
                                       f"(enter the near-far distance to scale it).")
        except Exception as e:
            logger.warning("Could not save geometry views: %s", e)

    @staticmethod
    def _save(result: dict, image_path: str, output_dir: Optional[str]) -> None:
        out_dir = Path(output_dir) if output_dir else Path(image_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        json_path = out_dir / f"{stem}_analysis.json"
        json_path.write_text(json.dumps(result, indent=2, default=float))
        csv_path = out_dir / f"{stem}_roadrunner.csv"
        with open(csv_path, "w") as f:
            f.write("defect_type,left_m,right_m,width_m,length_m,long_m,conf\n")
            for o in result["roadrunner_obstacles"]:
                f.write(f"{o['defect_type']},{o['left_m']},{o['right_m']},{o['width_m']},"
                        f"{o['height_m']},{o['long_m'] if o['long_m'] is not None else ''},{o['conf']}\n")
        result["annotated_image_filename"] = None
        result["_json_path"] = str(json_path)
        result["_csv_path"] = str(csv_path)

    def annotated_frame(self, image_path: str):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.model.predict(img, conf=0.25, verbose=False, imgsz=INFERENCE_IMGSZ,
                                  device="cpu")[0].plot()

    # ---------------- segment aggregation ----------------
    @staticmethod
    def _measured_occupancy(extracts: List[dict], road_config: dict) -> Optional[float]:
        """Share of photos/frames along the stretch that show on-street
        parking = share of segment length occupied (draft IRC:106 Table 12
        input). Needs at least 3 samples; a user-entered value wins."""
        if road_config.get("parking_occupancy_pct") is not None or len(extracts) < 3:
            return None
        return round(100.0 * sum(e["has_parking"] for e in extracts) / len(extracts), 1)

    @staticmethod
    def _segment_summary(results: List[dict], occupancy: Optional[float]) -> dict:
        if not results:
            return {"worst_capacity_loss_pct": None, "worst_image_or_frame": None,
                    "avg_capacity_loss_pct": None, "segment": None}
        losses = [r["capacity_loss_pct"] for r in results]
        worst = min(results, key=lambda r: r["reduced_capacity_pcu_hr"])
        return {
            "worst_capacity_loss_pct": worst["capacity_loss_pct"],
            "worst_image_or_frame": worst["image"],
            "avg_capacity_loss_pct": round(sum(losses) / len(losses), 1),
            "segment": {
                "rule": "Bottleneck: a road segment carries no more than its most restricted point.",
                "bottleneck_image_or_frame": worst["image"],
                "capacity_pcu_hr": worst["reduced_capacity_pcu_hr"],
                "capacity_c0_pcu_hr": worst["original_capacity_pcu_hr"],
                "capacity_loss_pct": worst["capacity_loss_pct"],
                "vc_ratio": worst["vc_ratio"],
                "level_of_service": worst["level_of_service"],
                "level_of_service_ideal": worst["level_of_service_ideal"],
                "parking_occupancy_pct_measured": occupancy,
                "samples": len(results),
            },
        }

    def analyse_batch(self, image_paths: List[str], road_config: dict, output_dir: str) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spacing = road_config.get("photo_spacing_m")
        extracts, errors = [], []
        for pth in image_paths:
            try:
                ex = self.extract(pth, road_config)
                for k in ("_img", "_H", "_src"):
                    ex.pop(k, None)          # keep memory flat across many photos
                extracts.append(ex)
            except Exception as e:
                errors.append({"image": Path(pth).name, "error": str(e)})
        occ = self._measured_occupancy(extracts, road_config)
        results = []
        for i, ex in enumerate(extracts):
            cfg_i = dict(road_config)
            if spacing:
                cfg_i["chainage_m"] = float(road_config.get("chainage_m") or 0) + i * float(spacing)
            r = self.assemble(ex, cfg_i, occ)
            self._save(r, ex["image_path"], str(out_dir))
            results.append(r)

        summary = self._segment_summary(results, occ)
        summary.update({
            "mode": "batch", "road_config": road_config,
            "num_images": len(image_paths), "num_succeeded": len(results), "errors": errors,
            "per_image": [{"image": r["image"], "capacity_loss_pct": r["capacity_loss_pct"],
                           "level_of_service": r["level_of_service"],
                           "chainage_m": r["road_config"].get("chainage_m"),
                           "defects_found": list(r["per_defect"].keys()),
                           "annotated_image_filename": r.get("annotated_image_filename")}
                          for r in results],
        })
        clean = {k: v for k, v in summary.items()}
        (out_dir / "batch_summary.json").write_text(json.dumps(clean, indent=2, default=float))
        summary["_json_path"] = str(out_dir / "batch_summary.json")
        summary["_per_image_full"] = results
        return summary

    def analyse_video(self, video_path: str, road_config: dict, output_dir: str,
                      sample_every_sec: float = 1.0, track_iou_threshold: float = 0.5,
                      max_frames: Optional[int] = None) -> dict:
        out_dir = Path(output_dir)
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # If the survey vehicle speed is known, sample by distance instead.
        survey_kmh = road_config.get("survey_speed_kmh")
        sample_dist = road_config.get("sample_every_m")
        if survey_kmh and sample_dist:
            sample_every_sec = float(sample_dist) / (float(survey_kmh) / 3.6)
        frame_interval = max(1, int(round(fps * sample_every_sec)))

        extracts = []
        frame_idx = sampled = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_interval == 0:
                    fp = frames_dir / f"frame_{sampled:05d}.jpg"
                    try:
                        cv2.imwrite(str(fp), frame)
                        ex = self.extract(str(fp), road_config)
                        for k in ("_img", "_H", "_src"):
                            ex.pop(k, None)
                        ex["timestamp_sec"], ex["frame_index"] = frame_idx / fps, frame_idx
                        extracts.append(ex)
                    except Exception as e:
                        logger.warning("Video frame %d failed: %s", frame_idx, e)
                    sampled += 1
                    if max_frames and sampled >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()

        occ = self._measured_occupancy(extracts, road_config)
        results, tracks = [], []
        for ex in extracts:
            r = self.assemble(ex, road_config, occ)
            r["timestamp_sec"], r["frame_index"] = round(ex["timestamp_sec"], 2), ex["frame_index"]
            self._save(r, ex["image_path"], str(out_dir))
            self._update_tracks(tracks, r, ex["timestamp_sec"], track_iou_threshold)
            results.append(r)

        summary = self._segment_summary(results, occ)
        summary.update({
            "mode": "video", "road_config": road_config, "video": Path(video_path).name,
            "fps": round(fps, 2), "total_frames_in_video": total_frames,
            "sampled_every_sec": round(sample_every_sec, 3), "frames_analysed": len(results),
            "unique_defect_instances": [
                {"cls_name": t["cls_name"], "first_seen_sec": round(t["first_seen_sec"], 2),
                 "last_seen_sec": round(t["last_seen_sec"], 2), "times_seen": t["hits"],
                 "max_blocked_m": round(t["max_blocked_m"], 2)} for t in tracks],
            "unique_defect_count": len(tracks),
            "frame_by_frame": [{"frame_index": r["frame_index"], "timestamp_sec": r["timestamp_sec"],
                                "capacity_loss_pct": r["capacity_loss_pct"],
                                "level_of_service": r["level_of_service"]} for r in results],
        })
        (out_dir / "video_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        summary["_json_path"] = str(out_dir / "video_summary.json")
        summary["_frame_results_full"] = results
        return summary

    @staticmethod
    def _update_tracks(tracks, r, t_sec, iou_thr):
        """Match defects across frames by class and lateral overlap
        (1-D IoU on the across-road interval)."""
        for o in r["roadrunner_obstacles"]:
            a = (o["left_m"], o["right_m"])
            best, best_s = None, 0.0
            for i, t in enumerate(tracks):
                if t["cls_name"] != o["defect_type"] or t.get("_seen") == r["frame_index"]:
                    continue
                inter = irc.overlap(a, t["last"])
                union = max(a[1], t["last"][1]) - min(a[0], t["last"][0])
                s = inter / union if union > 0 else 0.0
                if s > best_s:
                    best, best_s = i, s
            if best is not None and best_s >= iou_thr:
                t = tracks[best]
                t.update(last=a, last_seen_sec=t_sec, hits=t["hits"] + 1, _seen=r["frame_index"],
                         max_blocked_m=max(t["max_blocked_m"], o["width_m"]))
            else:
                tracks.append({"cls_name": o["defect_type"], "last": a, "first_seen_sec": t_sec,
                               "last_seen_sec": t_sec, "hits": 1, "max_blocked_m": o["width_m"],
                               "_seen": r["frame_index"]})


# ---------------------------------------------------------------------
# Assumptions register -- what was measured, entered, IRC-sourced or assumed
# ---------------------------------------------------------------------
def assumption_register(result: dict, p: dict, ex: dict, cfg: dict, thr_calibrated: bool) -> List[dict]:
    def row(item, value, basis, status):
        return {"item": item, "value": value, "basis": basis, "status": status}

    rows = [
        row("Carriageway geometry", ex["geometry_method"],
            ("Homography from auto-detected road edges" if (ex.get("edge_detection") or {}).get("source") == "auto"
             else "Homography from 4 marked corners") if ex["calibrated"] else "Photo edges = carriageway edges",
            "measured" if ex["calibrated"] else "assumed"),
        row("Estimated Capacity (LOS C Assumption)",
            f"Capacity = DSV / 0.70 = {p['dsv_scope']:.0f} / 0.70 = {p['capacity_c0']:.0f} PCU/h",
            "DSV (IRC:106-1990 Table 2) corresponds to LOS C operating conditions (V/C = 0.70)", "assumed"),
        row("Photographed stream", "one direction of a divided road" if p["photo_covers"] == "one_direction"
            else "whole carriageway",
            "Divided-road DSV taken as two-way total and halved" if p["photo_covers"] == "one_direction" else "-",
            "entered"),
        row("Usable gap", f">= {p['min_usable_gap_m']:.1f} m",
            "Narrower free gaps carry no traffic (wider than a two-wheeler with clearance)",
            "entered" if cfg.get("min_usable_gap_m") else "assumed (adjustable)"),
        row("Operating speed", f"{p['operating_speed_kmh']:.0f} km/h",
            "Measured" if p["operating_speed_measured"] else "Draft IRC:106 (2022) Table 3 by road class",
            "measured" if p["operating_speed_measured"] else "IRC (draft)"),
        row("Pothole capacity factor", "C_lane(V_pothole)/C_lane(V_normal)",
            "Draft IRC:106 Eq. 5; speed " + ("measured" if cfg.get("pothole_speed_kmh") else
                                             "assumed 90/80/60 % of operating speed for IRC:82 small/medium/large"),
            "measured" if cfg.get("pothole_speed_kmh") else "assumed speed (IRC formula)"),
        row("Pothole class", "IRC:82-2015 cl. 7.5.3.4",
            "width + measured depth" if cfg.get("pothole_depth_mm") else "measured width; depth not measurable from a photo",
            "IRC"),
        row("Side-friction factors", "F_OP, F_BS, F_AC, F_PD",
            "Draft IRC:106 (2022) Eq. 6, Tables 12-16; skipped where already in the IRC:106-1990 column",
            "IRC (draft)"),
        row("Parking occupancy", p["parking_occupancy_pct"],
            p["parking_occupancy_source"] or "unknown: lowest band used if parking is seen",
            "measured" if p["parking_occupancy_source"] == "measured from photos/frames"
            else ("entered" if p["parking_occupancy_source"] else "assumed")),
        row("Traffic composition", {k: round(v, 2) for k, v in p["composition"].items()},
            "Classified count" if p["composition_entered"] else "Typical Indian urban mix",
            "entered" if p["composition_entered"] else "assumed"),
        row("PCU factors", p["pcu_per_type"], "IRC:106-1990 Table 1, by each type's own share", "IRC"),
        row("Demand for V/C", f"{p['demand_pcu_hr']:.0f} PCU/h",
            "Observed volume" if p["demand_is_observed"] else "IRC design service volume",
            "measured" if p["demand_is_observed"] else "IRC"),
        row("Speed at each LOS", "90/70/50/40/33/29 % of free-flow speed", "IRC:106-1990 cl. 5.3", "IRC"),
        row("Detection thresholds", "per-class confidence",
            "Calibrated on a validation set" if thr_calibrated else "Built-in defaults (run tools/calibrate_thresholds.py)",
            "measured" if thr_calibrated else "assumed"),
        row("Moving vs parked vehicles", "'vehicle' = moving traffic",
            "A single photo cannot show motion; parked vehicles count only if labelled illegal_parking", "assumed"),
        row("Obstruction length", "bottleneck principle",
            "A point obstruction limits the whole segment's throughput", "traffic-flow principle"),
    ]
    return rows
