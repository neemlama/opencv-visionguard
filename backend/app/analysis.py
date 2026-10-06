"""Capture-quality analysis and defect classification for rolled-steel surfaces.

This is the "perception" stage. It answers three separate questions, each with
its own evidence, so the agent never has to infer one from another:

  1. Is this capture usable at all?      -> quality_metrics()
  2. What defect is it, and how sure?    -> DefectClassifier
  3. Where is it?                        -> localisation on the anomaly map

Keeping these separate matters. A classifier is confidently wrong on a
defocused or glare-blown capture, so the agent must see capture quality BEFORE
it is allowed to trust a class prediction. Collapsing quality into the score is
exactly the bug documented as C-07 in docs/CHALLENGES.md.

Measured on public data (NEU-CLS 1800 images, stratified 70/30):
  6-class  94.81% held-out accuracy, macro F1 0.948
  7-class  92.43% held-out accuracy, macro F1 0.937 (adds no_defect)
See docs/EVALUATION.md.
"""
import cv2
import numpy as np

from .classifier import DefectClassifier
from .features import (CLASS_NAMES, anomaly_map, describe, quality_metrics)

# Classes that justify stopping the line rather than queueing for review.
SEVERE_CLASSES = ("inclusion", "rolled-in_scale", "pitted_surface")

# Quality gate thresholds. Set from the NEU capture distribution, and recorded
# here rather than buried in the agent so they can be argued with.
# Quality gate thresholds, all derived from measured distributions on 240 real
# NEU captures (eval/quality_gate_calibration.py), not chosen by feel.
SHARPNESS_MIN = 12.0        # Laplacian variance; real captures sit at 1025+
GLARE_MAX = 0.05            # clean p99 = 0.0427
BLOWN_MAX = 0.035           # clean p99 = 0.0267
CRUSHED_MAX = 0.05          # never fires on clean data
# Sensor noise RAISES Laplacian variance, so a noisy frame reads as sharper and
# slips past every other check. Without this the loop auto-passed 196 of 200
# clean captures under noise. Clean p99 = 10.29, noisy p05 = 11.31.
NOISE_MAX = 10.8

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DefectClassifier.load()
    return _MODEL


def localize(gray, anom_thresh=8.0, min_area=400, close_k=9, box=21):
    """Connected components on the anomaly map -> candidate defect regions.

    Defaults come from eval/neu_tune.py (grid over box/anom_thresh/min_area/
    close_k on 240 labelled images): box=21, anom_thresh=8, min_area=400,
    close_k=9 was the best-scoring combination found.

    Honest caveat: even tuned, box-level localisation on NEU is weak (macro F1
    ~0.19 at IoU 0.5) because these defects are diffuse and often cover most of
    the frame. The classifier carries the "which defect" decision; this stage
    supplies a coarse pointer and, more importantly, an independent signal used
    to catch classifier/localiser disagreement.
    """
    anom = anomaly_map(gray, box)
    # Threshold a float32 map and it stays float32, but connectedComponentsWithStats
    # asserts CV_8U/CV_8S. Threshold to 0/1 then scale to a real mask.
    _, binary = cv2.threshold(anom, anom_thresh, 1.0, cv2.THRESH_BINARY)
    mask = (binary * 255).astype(np.uint8)
    if close_k:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = gray.shape[0] * gray.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or (w * h) / total > 0.95:
            continue
        sel = labels[y:y + h, x:x + w] == i
        regions.append({
            "box": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "energy": round(float(anom[y:y + h, x:x + w][sel].mean()), 3),
        })
    regions.sort(key=lambda r: -r["area"])
    return regions, anom


def analyze(image, model=None):
    """Full perception stage. Returns a plain dict (JSON-serialisable)."""
    model = model or get_model()
    gray = _native_gray(image)

    quality = quality_metrics(gray)
    quality["usable"] = bool(
        quality["sharpness"] >= SHARPNESS_MIN
        and quality["glare_ratio"] <= GLARE_MAX
        and quality["blown_highlights"] <= BLOWN_MAX
        and quality["crushed_shadows"] <= CRUSHED_MAX
        and quality["noise_sigma"] <= NOISE_MAX
    )

    feats = np.asarray(describe(gray), dtype=np.float64)
    pred = model.predict_one(feats)

    regions, anom = localize(gray)
    h, w = gray.shape
    return {
        "opencv_version": cv2.__version__,
        "model": {
            "accuracy": model.meta.get("held_out_accuracy"),
            "macro_f1": model.meta.get("macro_f1"),
            "dataset": model.meta.get("dataset"),
            "classes": list(model.class_names),
        },
        "quality": quality,
        "prediction": pred,
        "localization": {
            "region_count": len(regions),
            "regions": regions[:5],
            "coverage": round(sum(r["area"] for r in regions) / (h * w), 4),
            "anomaly_energy": round(float(anom.mean()), 4),
        },
        "features": {"descriptor_len": int(feats.shape[0])},
    }


def _native_gray(image):
    """Grayscale at native resolution (quality must not depend on resizing)."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_for_reanalysis(image):
    """Second-opinion capture: local contrast stretch, so a re-analysis sees more.

    Used by the agent when the first classification is low-confidence. It is a
    genuine new perception of the same scene, not a re-run of the same numbers.
    """
    gray = _native_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
