"""
road_analyzer.road_edges
========================
Automatic carriageway-edge detection for a photo taken looking along the road.

Output: the same 4 corners a user would click (near-left, near-right,
far-right, far-left), in the photo's pixel coordinates, plus a confidence
score. core.py turns these into a homography so every detection is measured
in metres on the road plane.

Method
------
1. Road surface: the asphalt colour is sampled from a patch at the bottom
   centre of the photo (where the camera is standing on/near the road). Pixels
   within a colour tolerance of that sample, connected to the patch, form the
   road region. Lightness gets a wider tolerance than colour so shadows and
   sunlit patches stay inside the road.
2. Boundary points: in each image row, the leftmost and rightmost road pixels.
   Rows where the road runs off the photo edge are not used (the true edge
   is outside the frame there).
3. Straight edges: a robust (RANSAC) straight-line fit to the left points and
   to the right points, so parked cars, vendors or garbage cutting into the
   edge in some rows do not bend the result.
4. Painted edge lines (IRC:35 edge marking) sit on the carriageway: where a
   white/yellow strip at the boundary has asphalt beyond it, the boundary is
   moved out across it. A bright strip with no asphalt beyond is the kerb,
   and the boundary stays on its inner side.
5. Corners: the near line is the lowest row of road; the far line is placed
   75 % of the way to the point where the two edges meet (vanishing point),
   limited to where the road region is still visible.

The confidence combines how many boundary points agree with each fitted
line, how many rows had a usable edge, and geometric sanity checks. Below
MIN_CONFIDENCE the caller falls back (and says so in the result).

Only straight roads are modelled. On a curve the near part of the road is
fitted; the far part may deviate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

WORK_WIDTH = 640           # analysis resolution (px)
MIN_CONFIDENCE = 0.55      # below this, do not use the detection
FAR_FRACTION = 0.75        # far line position between near line and vanishing point


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _ransac_line(ys: np.ndarray, xs: np.ndarray, tol: float, iters: int = 300,
                 rng: Optional[np.random.Generator] = None) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """Fit x = a*y + b robustly. Returns ((a, b), inlier_mask)."""
    n = len(ys)
    if n < 8:
        return None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    best_in, best = None, None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(ys[i] - ys[j]) < 5:
            continue
        a = (xs[i] - xs[j]) / (ys[i] - ys[j])
        b = xs[i] - a * ys[i]
        inl = np.abs(xs - (a * ys + b)) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (a, b)
    if best_in is None or best_in.sum() < 6:
        return None, np.zeros(n, bool)
    A = np.vstack([ys[best_in], np.ones(best_in.sum())]).T
    a, b = np.linalg.lstsq(A, xs[best_in], rcond=None)[0]
    inl = np.abs(xs - (a * ys + b)) < tol
    return (float(a), float(b)), inl


def _road_mask(lab: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    h, w = lab.shape[:2]
    y0, y1, x0, x1 = int(0.82 * h), int(0.97 * h), int(0.38 * w), int(0.62 * w)
    seed = lab[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    med = np.median(seed, axis=0)
    mad = np.median(np.abs(seed - med), axis=0) * 1.4826
    # Two separate tests (LAB, 0-255 scale):
    #  - lightness: shadows make asphalt up to ~45 darker, sun/sheen ~25 lighter;
    #    a concrete footpath or kerb is usually 50+ lighter, so it is excluded.
    #  - colour (a, b): asphalt is near-neutral; soil, grass and paint are not.
    labf = lab.astype(np.float32)
    dL = labf[..., 0] - med[0]
    tol_dark = max(45.0, 4.0 * mad[0])
    tol_light = max(25.0, 3.0 * mad[0])
    chroma = np.sqrt((labf[..., 1] - med[1]) ** 2 + (labf[..., 2] - med[2]) ** 2)
    tol_c = max(9.0, 3.5 * float(np.hypot(mad[1], mad[2])))
    mask = ((dL > -tol_dark) & (dL < tol_light) & (chroma < tol_c)).astype(np.uint8)
    colour = mask.astype(bool).copy()
    k = max(3, w // 160) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels = cv2.connectedComponents(mask)
    patch = labels[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return np.zeros_like(mask), (y0, y1, x0, x1), colour
    lbl = np.bincount(vals).argmax()
    road = (labels == lbl).astype(np.uint8)
    # Fill holes (vehicles, potholes, markings inside the road).
    inv = (1 - road).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(inv)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    for k2 in range(1, n2):
        if k2 not in border:
            road[lab2 == k2] = 1
    return road, (y0, y1, x0, x1), colour


def _paint_mask(lab: np.ndarray, road: np.ndarray) -> np.ndarray:
    """White or yellow road paint (IRC:35 markings), relative to the asphalt."""
    L, A, B = (lab[..., i].astype(np.float32) for i in range(3))
    road_L = float(np.median(L[road > 0])) if road.any() else 100.0
    white = (L > road_L + 45) & (np.abs(A - 128) < 18) & (np.abs(B - 128) < 22)
    yellow = (B > 150) & (L > road_L + 15)
    return (white | yellow)


def _extend_over_paint(row: np.ndarray, asphalt: np.ndarray, x0: int, step: int, reach: int, gap: int = 3) -> int:
    """From boundary pixel x0, look outward for a bright strip starting within
    `gap` px (blur leaves a soft transition). If asphalt continues beyond the
    strip, it was paint ON the carriageway: return the outer end of that
    asphalt. If not (footpath, soil), the strip was the kerb: return x0."""
    n = row.shape[0]
    x, k = x0 + step, 0
    while 0 <= x < n and k < gap and not row[x]:
        x += step; k += 1
    if not (0 <= x < n) or not row[x]:
        return x0
    while 0 <= x < n and abs(x - x0) <= reach and row[x]:
        x += step
    k = 0
    while 0 <= x < n and k < gap and not asphalt[x] and not row[x]:
        x += step; k += 1
    if not (0 <= x < n) or not asphalt[x]:
        return x0
    last = x
    while 0 <= x < n and asphalt[x]:
        if abs(x - x0) > reach:
            return x0          # wide road-like surface beyond: a footpath, not a strip
        last = x; x += step
    return last


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------
def detect_road_corners(image_bgr: np.ndarray) -> dict:
    """Return {"points": [[x, y] x4] or None, "confidence": 0-1,
    "method": str, "reason": str, "lines": {...}} in the input image's pixels."""
    H0, W0 = image_bgr.shape[:2]
    s = WORK_WIDTH / W0 if W0 > WORK_WIDTH else 1.0
    img = cv2.resize(image_bgr, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA) if s != 1.0 else image_bgr
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)

    fail = lambda reason, conf=0.0: {"points": None, "confidence": round(conf, 2), "method": "auto",
                                      "reason": reason, "lines": None}
    road, seed_box, colour = _road_mask(lab)
    if road.sum() < 0.08 * h * w:
        return fail("Road surface not found at the bottom centre of the photo.")
    y0s, y1s, x0s, x1s = seed_box
    seed = lab[y0s:y1s, x0s:x1s].reshape(-1, 3).astype(np.float32)
    sL, sa, sb = np.median(seed, axis=0)
    # Asphalt / concrete road: mid lightness, nearly neutral colour.
    if not (45 <= sL <= 200) or np.hypot(sa - 128, sb - 128) > 18:
        return fail("The surface at the bottom centre does not look like a road (colour or brightness).")

    rows = np.where(road.any(axis=1))[0]
    y_top, y_bot = int(rows.min()), int(rows.max())
    if y_bot - y_top < 0.2 * h:
        return fail("Visible road is too short to fit its edges.")
    ys_all = np.arange(y_top + int(0.05 * (y_bot - y_top)), y_bot + 1, 2)
    Ly, Lx, Ry, Rx = [], [], [], []
    paint = _paint_mask(lab, road)
    road_L = float(np.median(lab[..., 0][road > 0]))
    loose = (lab[..., 0].astype(np.float32) < road_L + 50) & ~paint
    # Per row: inner road boundary (xl, xr) and, where resolvable, the outer
    # side of a painted edge line + any asphalt strip beyond it (el, er).
    rows_data = []
    for y in ys_all:
        xs = np.where(road[y])[0]
        if xs.size < 3:
            continue
        xl, xr = int(xs.min()), int(xs.max())
        reach = max(4, int(0.06 * (xr - xl)))
        el = _extend_over_paint(paint[y], loose[y], xl, -1, reach)
        er = _extend_over_paint(paint[y], loose[y], xr, +1, reach)
        rows_data.append((y, xl, xr, el, er))
    # The painted strip has a fixed real width, so in the image it is a fixed
    # fraction of the road width at its row. Measure that fraction where the
    # strip is resolved (near rows) and apply it to every row, so far rows
    # where it blurs away do not tilt the fitted edge.
    def offset_fraction(side):
        fr = [((xl - el) if side == "l" else (er - xr)) / max(1, xr - xl)
              for _, xl, xr, el, er in rows_data if (el < xl if side == "l" else er > xr)]
        return float(np.median(fr)) if len(fr) >= 0.3 * max(1, len(rows_data)) else 0.0
    f_l, f_r = offset_fraction("l"), offset_fraction("r")
    painted_rows = (f_l > 0) + (f_r > 0)
    for y, xl, xr, el, er in rows_data:
        wrow = xr - xl
        if xl > 2:
            Ly.append(y); Lx.append(xl - f_l * wrow)
        if xr < w - 3:
            Ry.append(y); Rx.append(xr + f_r * wrow)
    Ly, Lx, Ry, Rx = map(np.asarray, (Ly, Lx, Ry, Rx))
    tol = 0.015 * w
    left, inl_l = _ransac_line(Ly.astype(float), Lx.astype(float), tol)
    right, inl_r = _ransac_line(Ry.astype(float), Rx.astype(float), tol)
    if left is None or right is None:
        side = "left" if left is None else "right"
        return fail(f"The {side} road edge is not visible (road runs off the photo or is hidden).")

    painted = painted_rows > 0
    method = "road surface + painted edge lines" if painted else "road surface boundary"

    (al, bl), (ar, br) = left, right
    width_at = lambda y: (ar * y + br) - (al * y + bl)
    y_n = float(y_bot)
    if width_at(y_n) < 0.4 * w:
        return fail("Detected road covers less than 40 % of the photo width at the bottom; "
                    "take the photo looking along the road with both edges in view.", 0.2)
    if abs(al - ar) > 1e-6:
        y_v = (br - bl) / (al - ar)
    else:
        y_v = -np.inf
    if y_v >= y_n:
        return fail("Edges do not converge in the expected direction.", 0.2)
    y_f = max(float(y_top) + 0.05 * (y_bot - y_top), y_n - FAR_FRACTION * (y_n - y_v)) if np.isfinite(y_v) \
        else float(y_top) + 0.05 * (y_bot - y_top)
    if y_n - y_f < 0.15 * h or width_at(y_f) <= 2:
        return fail("Not enough road between the near and far lines.", 0.3)

    # confidence: agreement of points with the lines x coverage of rows
    agree = min(inl_l.mean(), inl_r.mean())
    coverage = min(len(Ly), len(Ry)) / max(1, len(ys_all))
    conf = float(np.clip(agree * (0.5 + 0.5 * min(1.0, coverage / 0.6)), 0, 1))

    pts = [[al * y_n + bl, y_n], [ar * y_n + br, y_n], [ar * y_f + br, y_f], [al * y_f + bl, y_f]]
    pts = [[round(x / s, 1), round(y / s, 1)] for x, y in pts]
    return {
        "points": pts if conf >= MIN_CONFIDENCE else None,
        "candidate_points": pts,
        "confidence": round(conf, 2),
        "method": method,
        "reason": "" if conf >= MIN_CONFIDENCE else "Low confidence in the detected edges.",
        "lines": {"left": [al, bl / s], "right": [ar, br / s]},
        "inliers": {"left": round(float(inl_l.mean()), 2), "right": round(float(inl_r.mean()), 2)},
        "row_coverage": round(coverage, 2),
    }


def draw_edges_overlay(image_bgr: np.ndarray, pts) -> np.ndarray:
    """Original photo with the carriageway quad drawn on it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    t = max(2, w // 400)
    poly = np.array(pts, np.int32)
    tint = out.copy()
    cv2.fillPoly(tint, [poly], (0, 200, 255))
    out = cv2.addWeighted(tint, 0.18, out, 0.82, 0)
    cv2.line(out, tuple(poly[0]), tuple(poly[3]), (0, 215, 255), t * 2, cv2.LINE_AA)
    cv2.line(out, tuple(poly[1]), tuple(poly[2]), (0, 215, 255), t * 2, cv2.LINE_AA)
    cv2.line(out, tuple(poly[0]), tuple(poly[1]), (255, 255, 255), t, cv2.LINE_AA)
    cv2.line(out, tuple(poly[3]), tuple(poly[2]), (255, 255, 255), t, cv2.LINE_AA)
    return out


COLORS = {"street_vendor": (45, 98, 194), "cart": (181, 91, 138), "garbage": (47, 125, 107),
          "barricade": (58, 58, 217), "illegal_parking": (181, 111, 47), "pothole": (20, 20, 20),
          "tree_on_road": (40, 140, 40)}


def birdseye_view(image_bgr: np.ndarray, H_img_to_road: np.ndarray, width_m: float, length_m: float,
                  lanes: int, defects: list, px_per_m: float = 50.0) -> np.ndarray:
    """Top-down view of the marked stretch: near edge at the bottom, lanes and
    each detection's footprint drawn to scale."""
    Wp, Lp = int(round(width_m * px_per_m)), int(round(length_m * px_per_m))
    S = np.array([[px_per_m, 0, 0], [0, -px_per_m, Lp], [0, 0, 1]], dtype=np.float64)
    top = cv2.warpPerspective(image_bgr, S @ H_img_to_road, (Wp, Lp), flags=cv2.INTER_LINEAR)
    t = 2
    for i in range(1, max(1, lanes)):
        x = int(round(i * Wp / lanes))
        for y in range(0, Lp, 30):
            cv2.line(top, (x, y), (x, min(Lp, y + 15)), (255, 255, 255), t)
    for d in defects:
        col = COLORS.get(d["cls"], (0, 0, 255))
        x1, x2 = int(d["lat_l"] * px_per_m), int(d["lat_r"] * px_per_m)
        if d.get("long_near") is not None:
            y2 = int(Lp - d["long_near"] * px_per_m)
            y1 = int(Lp - max(d["long_far"], d["long_near"] + 0.3) * px_per_m)
        else:
            y1, y2 = 0, Lp
        cv2.rectangle(top, (x1, max(0, y1)), (x2, min(Lp, y2)), col, 3)
    pad = 28
    canvas = cv2.copyMakeBorder(top, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(canvas, f"{width_m:g} m", (pad + Wp // 2 - 25, Lp + pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas
