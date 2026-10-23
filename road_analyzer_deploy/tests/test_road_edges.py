"""
Accuracy test for automatic road-edge detection on synthetic perspective
road photos with known geometry. Reports the lateral error (metres) of the
detected carriageway edges, measured where the true edges are.

Run:  python tests/test_road_edges.py  [--save DIR]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from road_analyzer.road_edges import detect_road_corners  # noqa: E402

PPM = 40  # world texture pixels per metre


def make_scene(seed, W=7.0, markings=True, shadows=True, occlusion=True, camera=(0.0, 0.0),
               footpath=(150, 150, 145), verge=(70, 100, 120), size=(1600, 1000)):
    rng = np.random.default_rng(seed)
    x_min, x_max, L = -4.0, W + 4.0, 60.0
    tw, th = int((x_max - x_min) * PPM), int(L * PPM)
    world = np.zeros((th, tw, 3), np.uint8)
    X = lambda m: int((m - x_min) * PPM)
    Y = lambda m: int(th - m * PPM)
    world[:] = verge
    cv2.rectangle(world, (X(-2.5), 0), (X(0), th), footpath, -1)
    cv2.rectangle(world, (X(W), 0), (X(W + 2.5), th), footpath, -1)
    for k in range(0, th, 18):  # paver joints
        cv2.line(world, (X(-2.5), k), (X(0), k), tuple(int(c * 0.85) for c in footpath), 1)
        cv2.line(world, (X(W), k), (X(W + 2.5), k), tuple(int(c * 0.85) for c in footpath), 1)
    cv2.rectangle(world, (X(-0.15), 0), (X(0), th), (180, 180, 175), -1)          # kerb
    cv2.rectangle(world, (X(W), 0), (X(W + 0.15), th), (180, 180, 175), -1)
    cv2.rectangle(world, (X(0), 0), (X(W), th), (78, 80, 84), -1)                 # asphalt
    if markings:
        cv2.rectangle(world, (X(0.1), 0), (X(0.25), th), (235, 235, 235), -1)     # edge lines
        cv2.rectangle(world, (X(W - 0.25), 0), (X(W - 0.1), th), (235, 235, 235), -1)
    for m in np.arange(0, L, 7.5):                                                # lane dashes
        cv2.rectangle(world, (X(W / 2 - 0.07), Y(m + 3)), (X(W / 2 + 0.07), Y(m)), (230, 230, 230), -1)
    if shadows:
        for _ in range(4):
            y0 = rng.uniform(3, 50)
            sh = world[Y(y0 + 2.5):Y(y0), :].astype(np.float32) * 0.6
            world[Y(y0 + 2.5):Y(y0), :] = sh.astype(np.uint8)
    if occlusion:   # parked car and a vendor cutting into the edges
        cv2.rectangle(world, (X(-0.3), Y(14)), (X(1.7), Y(9.5)), (40, 40, 160), -1)
        cv2.rectangle(world, (X(W - 1.4), Y(24)), (X(W + 0.4), Y(21)), (30, 140, 200), -1)
    noise = rng.normal(0, 6, world.shape)
    world = np.clip(world.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # camera: road-plane metres -> image pixels via the 4 corners of the stretch 0..W x 2..32 m
    w, h = size
    cx, yaw = camera
    near_l, near_r = 120 + cx, 1480 + cx
    far_l, far_r = 690 + cx + yaw, 900 + cx + yaw
    road_pts = np.float32([[0, 2], [W, 2], [W, 32], [0, 32]])
    img_pts = np.float32([[near_l, 980], [near_r, 980], [far_r, 430], [far_l, 430]])
    H_road_img = cv2.getPerspectiveTransform(road_pts, img_pts)
    T = np.array([[1 / PPM, 0, x_min], [0, -1 / PPM, L], [0, 0, 1]])  # world px -> metres
    img = cv2.warpPerspective(world, H_road_img @ T, (w, h), borderValue=(200, 190, 170))
    # sky / buildings above the road plane's visible part
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H_road_img @ T, (w, h))
    sky = np.zeros_like(img); sky[:] = (215, 190, 160)
    img = np.where(mask[..., None] > 0, img, sky)
    return img, H_road_img, W


def lateral_error(det, H_road_img, W):
    """Max lateral error (m) of the detected edges, checked at 3 distances along the true edges."""
    pts = np.float32(det["points"])
    H_img_road = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [W, 0], [W, 10], [0, 10]]))
    errs = []
    for dist in (3, 8, 15):
        for true_lat in (0.0, W):
            v = H_road_img @ np.array([true_lat, dist, 1.0])
            u = H_img_road @ np.array([v[0] / v[2], v[1] / v[2], 1.0])
            errs.append(abs(u[0] / u[2] - true_lat))
    return max(errs)


CASES = [
    ("clean road, edge lines", dict(markings=True, shadows=False, occlusion=False)),
    ("edge lines + shadows + parked car/vendor", dict(markings=True, shadows=True, occlusion=True)),
    ("no markings, shadows, occlusions", dict(markings=False, shadows=True, occlusion=True)),
    ("no markings, camera off-centre", dict(markings=False, shadows=True, occlusion=False, camera=(-150, 60))),
    ("no markings, dusty footpath close to asphalt", dict(markings=False, shadows=False, occlusion=True, footpath=(105, 105, 108))),
    ("edge lines, camera off-centre right", dict(markings=True, shadows=True, occlusion=True, camera=(120, -40))),
]

if __name__ == "__main__":
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    worst = 0.0
    for i, (name, kw) in enumerate(CASES):
        img, Hri, W = make_scene(i, **kw)
        det = detect_road_corners(img)
        if save:
            os.makedirs(save, exist_ok=True)
            cv2.imwrite(os.path.join(save, f"case{i}.jpg"), img)
        if det["points"] is None:
            print(f"[{i}] {name:45s} NOT DETECTED  conf={det['confidence']}  {det['reason']}")
            continue
        e = lateral_error(det, Hri, W)
        worst = max(worst, e)
        print(f"[{i}] {name:45s} max lateral error {e*100:5.1f} cm  conf={det['confidence']}  ({det['method']})")
    print(f"WORST {worst*100:.1f} cm")
