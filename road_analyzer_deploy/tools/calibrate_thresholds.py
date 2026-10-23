"""
Calibrate per-class confidence thresholds on a labelled validation set.

For each class, the threshold that maximises F1 (precision/recall balance)
is chosen and written to models/thresholds.json, which core.py loads
automatically. Without this file the built-in default thresholds are used
and the result's "inputs_and_sources" list says so.

Usage (YOLO-format dataset, the same one used for training):
    python tools/calibrate_thresholds.py \
        --model road_analyzer/models/best.pt \
        --images path/to/val/images --labels path/to/val/labels

Each label file: one line per object "cls cx cy w h" (normalised 0-1).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _iou(a, b):
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _read_labels(path: Path, w: int, h: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def best_f1_threshold(scored, n_gt):
    """scored: list of (conf, is_true_positive). Returns (threshold, f1, p, r)."""
    best = (0.45, 0.0, 0.0, 0.0)
    for k in range(5, 96):
        t = k / 100
        tp = sum(1 for c, ok in scored if c >= t and ok)
        fp = sum(1 for c, ok in scored if c >= t and not ok)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best[1]:
            best = (t, f1, p, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    model = YOLO(args.model)
    names = model.names
    scored = defaultdict(list)
    n_gt = defaultdict(int)
    imgs = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = _read_labels(Path(args.labels) / (img_path.stem + ".txt"), w, h)
        for c, _ in gts:
            n_gt[c] += 1
        pred = model.predict(img, conf=0.01, imgsz=args.imgsz, verbose=False, device="cpu")[0]
        used = set()
        dets = sorted(((int(b.cls[0]), float(b.conf[0]), tuple(float(v) for v in b.xyxy[0]))
                       for b in (pred.boxes or [])), key=lambda d: -d[1])
        for c, conf, box in dets:
            best_j, best_iou = None, args.iou
            for j, (gc, gbox) in enumerate(gts):
                if gc != c or j in used:
                    continue
                v = _iou(box, gbox)
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j is not None:
                used.add(best_j)
            scored[c].append((conf, best_j is not None))

    out, report = {}, {}
    for c, name in names.items():
        if n_gt[c] == 0:
            continue
        t, f1, p, r = best_f1_threshold(scored[c], n_gt[c])
        out[name] = t
        report[name] = {"threshold": t, "f1": round(f1, 3), "precision": round(p, 3),
                        "recall": round(r, 3), "ground_truth": n_gt[c]}
    dest = Path(args.model).with_name("thresholds.json")
    dest.write_text(json.dumps({"thresholds": out, "report": report,
                                "images": len(imgs), "iou": args.iou}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
