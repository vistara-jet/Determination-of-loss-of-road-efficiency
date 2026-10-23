import sys, types, math, json
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from road_analyzer import irc_capacity as irc
from road_analyzer import core

# --- PCU: IRC:106 Table 1 interpolation ---
avg, per = irc.average_pcu(irc.normalise_composition(None))
assert abs(per["truck_bus"] - 3.1) < 1e-9 and per["two_wheeler"] == 0.75 and per["cycle"] == 0.4
assert abs(avg - 1.292) < 1e-3, avg
# --- LOS boundaries ---
assert irc.level_of_service(0.70, "4lane_divided") == "C"
assert irc.level_of_service(0.701, "4lane_divided") == "D"
assert irc.level_of_service(1.53, "4lane_divided") == "F"
assert irc.level_of_service(0.30, "2lane_twoway") == "A"
# --- Eq.5 ---
assert abs(irc.lane_capacity_from_speed(50) - 1216.3) < 1e-6
assert abs(irc.pothole_capacity_factor(50, 40) - 1015.2/1216.3) < 1e-9
# --- factors ---
assert irc.parking_factor(30) == 0.65 and irc.parking_factor(10) == 0.70 and irc.parking_factor(60) == 0.40
assert irc.bus_stop_factor("kerbside", 40) == 0.60 and irc.access_factor(300) == 0.80 and irc.pedestrian_factor(600) == 0.72
# --- IRC:82 ---
assert irc.pothole_class_from_width(1.75) == "large" and irc.pothole_class_from_width(0.3) == "medium"
assert irc.pothole_class_from_depth(30) == "medium"
# --- gaps ---
assert irc.free_gaps([(0, 2.19), (12, 14)], 14) == [(2.19, 12)]

# --- full worked example: 4-lane divided arterial, W=14, vendor+garbage+parking+pothole ---
class Dummy: thresholds_calibrated = False
Dummy._warnings = core.RoadAnalyzer._warnings
dummy = Dummy()
def det(cls, l, r, conf=0.8):
    return {"cls": cls, "conf": conf, "xyxy": (0,0,1,1), "lat_l": l, "lat_r": r, "width_m": r-l,
            "long_near": None, "long_far": None, "length_m": None}
ex = {"image": "x.jpg", "image_path": "x.jpg", "img_w": 1600, "img_h": 1200, "scale": 1.0,
      "geometry_method": "uncalibrated", "calibrated": False, "calib_length_m": None,
      "defects": [det("street_vendor", 0, 1.496), det("garbage", 0.875, 2.1875),
                  det("illegal_parking", 11.996, 14.0), det("pothole", 6.125, 7.875)],
      "off_road": [], "vehicles_detected": 0, "vetoed": 0, "has_parking": True}
cfg = {"total_width_m": 14, "num_lanes": 4, "carriageway_key": "4lane_divided",
       "fringe_condition": "arterial", "usable_shoulder_m": 0, "parking_occupancy_pct": 30,
       "pothole_speed_kmh": 40}
r = core.RoadAnalyzer.assemble(dummy, ex, cfg)
cc = r["capacity_calculation"]
print(json.dumps({k: r[k] for k in ["original_capacity_pcu_hr","reduced_capacity_pcu_hr","capacity_loss_pct",
      "vc_ratio_ideal","vc_ratio","level_of_service_ideal","level_of_service"]}, indent=0))
print("lanes", cc["lanes"]); print("factors", cc["side_friction_factors"])
print("speeds", r["traffic_regime"]["ideal_speed_kmh"], "->", r["traffic_regime"]["congested_speed_kmh"])
for k, v in r["per_defect"].items():
    print(k, v["severity"], v["capacity_loss_pct"], "LOS if removed", v["los_if_removed"], v.get("capacity_method",""))
print("warnings", r["warnings"])
# hand check: lanes 3.5 m; blocked [0,2.1875]; lane1 usable (3.5-2.1875)/3.5 = 0.375 (gap 2.1875..14 is >=1 m)
C0 = 3600/0.7; cl = C0/4; fp = 1015.2/1216.3
# pothole 6.125-7.875 touches lane 2 (3.5-7.0) and lane 3 (7.0-10.5)
hand = (cl*0.375 + cl*fp + cl*fp + cl) * 0.65
assert abs(r["reduced_capacity_pcu_hr"] - round(hand,1)) < 0.2, (r["reduced_capacity_pcu_hr"], hand)
assert r["traffic_regime"]["congested_speed_kmh"] < r["traffic_regime"]["ideal_speed_kmh"]
# collector: parking already included -> F_OP 1
cfg2 = dict(cfg, carriageway_key="4lane_undivided", fringe_condition="collector")
r2 = core.RoadAnalyzer.assemble(dummy, ex, cfg2)
assert r2["capacity_calculation"]["side_friction_factors"]["F_OP"] == 1.0
# no defects -> zero loss, LOS C
ex0 = dict(ex, defects=[], has_parking=False)
r0 = core.RoadAnalyzer.assemble(dummy, ex0, {**cfg, "parking_occupancy_pct": None})
assert r0["capacity_loss_pct"] == 0.0 and r0["level_of_service"] == "C", r0["level_of_service"]
# one-direction scope
r3 = core.RoadAnalyzer.assemble(dummy, ex0, {**cfg, "total_width_m": 7, "photo_covers": "one_direction", "parking_occupancy_pct": None})
assert r3["design_service_volume_pcu_hr"] == 1800 and r3["capacity_calculation"]["lanes_in_photo"] == 2
# sliver: vendor leaving 0.5 m at left edge
ex4 = dict(ex0, defects=[det("street_vendor", 0.5, 2.0)])
r4 = core.RoadAnalyzer.assemble(dummy, ex4, {**cfg, "parking_occupancy_pct": None})
assert r4["capacity_calculation"]["unusable_slivers_m"] == 0.5
print("ALL FORMULA TESTS PASSED; hand =", round(hand,1))
