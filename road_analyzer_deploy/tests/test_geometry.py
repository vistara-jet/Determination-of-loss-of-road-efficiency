import sys, numpy as np, cv2
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from road_analyzer.core import RoadGeometry, apply_vehicle_veto
# Build a synthetic camera: road plane W=7 m, L=30 m -> image via known homography
W, L = 7.0, 30.0
img_pts = np.float32([[100, 950], [1500, 950], [950, 400], [650, 400]])     # near-left, near-right, far-right, far-left
road_pts = np.float32([[0, 0], [W, 0], [W, L], [0, L]])
H_img = cv2.getPerspectiveTransform(road_pts, img_pts)   # road -> image
def to_img(x, y):
    v = H_img @ np.array([x, y, 1.0]); return v[0]/v[2], v[1]/v[2]
cfg = {"total_width_m": W, "road_edge_points": img_pts.tolist(), "calib_length_m": L}
g = RoadGeometry(cfg, 1600, 1000, 1.0)
# a vendor standing on the road from lateral 1.0 to 2.5 m at 10 m distance
xl, yb = to_img(1.0, 10.0); xr, _ = to_img(2.5, 10.0)
fp = g.footprint((xl, yb - 120, xr, yb), flat=False)
print("vendor", {k: round(v, 3) for k, v in fp.items()})
assert abs(fp["lat_l"] - 1.0) < 1e-3 and abs(fp["lat_r"] - 2.5) < 1e-3 and abs(fp["long_near"] - 10) < 1e-3
# naive full-frame method on the same box would say:
print("naive lateral (photo width = 7 m):", round(xl / (1600 / W), 2), "-", round(xr / (1600 / W), 2))
# a flat pothole 0.6 m x 1.0 m at lateral 4.0-4.6, long 20-21
x1, y2 = to_img(4.0, 20.0); x2, _ = to_img(4.6, 20.0); _, y1 = to_img(4.0, 21.0)
fp = g.footprint((x1, y1, x2, y2), flat=True)
print("pothole", {k: round(v, 3) for k, v in fp.items()})
assert fp["lat_l"] < 4.01 and fp["lat_r"] > 4.59 and abs(fp["long_far"] - 21) < 0.1
# lateral only (no length): lateral still exact
g2 = RoadGeometry({"total_width_m": W, "road_edge_points": img_pts.tolist()}, 1600, 1000, 1.0)
fp2 = g2.footprint((xl, yb - 120, xr, yb), flat=False)
assert abs(fp2["lat_l"] - 1.0) < 1e-3 and fp2["long_near"] is None
# off-road vendor on footpath (lateral -2 to -1)
a, b2 = to_img(-2, 5); bb, _ = to_img(-1, 5)
fp3 = g.footprint((a, b2 - 100, bb, b2), flat=False); print("footpath object lat_r", round(fp3["lat_r"], 2), "(<0 -> ignored)")
# veto keeps the more confident label
dets = [{"cls_name": "street_vendor", "conf": 0.9, "xyxy": (0,0,10,10)}, {"cls_name": "vehicle", "conf": 0.5, "xyxy": (0,0,10,10)}]
kept, v = apply_vehicle_veto(dets); assert [d["cls_name"] for d in kept] == ["street_vendor"] and v == 0
dets[0]["conf"] = 0.4
kept, v = apply_vehicle_veto(dets); assert [d["cls_name"] for d in kept] == ["vehicle"] and v == 1
print("GEOMETRY + VETO TESTS PASSED")
