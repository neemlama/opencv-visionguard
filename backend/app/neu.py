"""Defect localisation for rolled-steel surface inspection (NEU).

Separate from backend/app/vision.py, which handles weld-bead inspection. Same
public idea — classical CV, no learned model — but NEU defects are not dark
pits on a bead, so they need a different feature: NEU's six classes
(crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches) vary
from large low-frequency regions to fine crack networks, while the background is
a comparatively uniform steel surface.

Approach: local anomaly energy. A defect is a region whose intensity departs
from its local surroundings, in EITHER direction (rolled-in scale is brighter,
crazing is darker), so the response is two-sided:

    bg    = box blur (local background)
    anom  = |gray - bg|              two-sided local deviation
    mask  = threshold -> morphology -> connected components

Box blur is used rather than Gaussian because it is O(1) per pixel regardless of
kernel size, which keeps the dominant op cheap (see docs/BENCHMARK-RESULTS.md).

All thresholds are keyword arguments so eval/neu_tune.py can measure them rather
than guess them.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectConfig:
    box: int = 31                # local background window
    anom_thresh: float = 12.0    # minimum |gray-bg| to count as anomalous
    open_k: int = 3              # despeckle
    close_k: int = 7             # fill cracks inside a defect region
    min_area: int = 60           # px, drop specks
    max_frac: float = 0.95       # drop predictions covering nearly everything
    merge_iou: float = 0.30      # merge overlapping components


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    area: int
    score: float
    mean_anom: float


@dataclass
class Detection:
    image_shape: tuple
    regions: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)


def anomaly_map(gray: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    k = cfg.box if cfg.box % 2 == 1 else cfg.box + 1
    bg = cv2.blur(gray, (k, k))
    return cv2.absdiff(gray, bg)


def locate_defects(image: np.ndarray, cfg: DetectConfig = None) -> Detection:
    """Return candidate defect regions for one BGR or gray image."""
    cfg = cfg or DetectConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    anom = anomaly_map(gray, cfg)

    _, mask = cv2.threshold(anom, cfg.anom_thresh, 255, cv2.THRESH_BINARY)
    if cfg.open_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.open_k, cfg.open_k)))
    if cfg.close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_k, cfg.close_k)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = gray.shape[0] * gray.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area:
            continue
        if cfg.max_frac and (w * h) / total > cfg.max_frac:
            continue  # a "defect" filling the frame is a segmentation failure
        sel = labels[y:y + h, x:x + w] == i
        score = float(anom[y:y + h, x:x + w][sel].mean())
        regions.append(Region(int(x), int(y), int(w), int(h), int(area), score, score))

    regions = _merge_overlapping(regions, cfg.merge_iou)

    # Quality signals the agent can act on (same role as blur/glare in the weld path).
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    quality = {
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "coverage": round(sum(r.area for r in regions) / total, 4),
        "region_count": len(regions),
        "max_score": round(max((r.score for r in regions), default=0.0), 3),
        "total_anomaly_energy": round(float(anom.mean()), 4),
    }
    return Detection(gray.shape, regions, quality)


def _merge_overlapping(regions, iou_thresh):
    """Union overlapping boxes; defects often fragment into several components."""
    if not regions:
        return []
    boxes = [(r.x, r.y, r.x + r.w, r.y + r.h, r) for r in regions]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    merged = []
    for x0, y0, x1, y1, r in boxes:
        placed = False
        for m in merged:
            ix0, iy0 = max(x0, m["x0"]), max(y0, m["y0"])
            ix1, iy1 = min(x1, m["x1"]), min(y1, m["y1"])
            iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
            inter = iw * ih
            union = (x1 - x0) * (y1 - y0) + (m["x1"] - m["x0"]) * (m["y1"] - m["y0"]) - inter
            if union > 0 and inter / union >= iou_thresh:
                m["x0"], m["y0"] = min(m["x0"], x0), min(m["y0"], y0)
                m["x1"], m["y1"] = max(m["x1"], x1), max(m["y1"], y1)
                m["score"] = max(m["score"], r.score)
                m["area"] = (m["x1"] - m["x0"]) * (m["y1"] - m["y0"])
                placed = True
                break
        if not placed:
            merged.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                           "score": r.score, "area": r.area})
    return [
        Region(m["x0"], m["y0"], m["x1"] - m["x0"], m["y1"] - m["y0"],
               m["area"], m["score"], m["score"])
        for m in merged
    ]


def iou(a, b):
    """IoU between two Region-like objects."""
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union
