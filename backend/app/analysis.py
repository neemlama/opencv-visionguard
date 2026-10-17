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
from .features import CLASS_NAMES, describe, quality_metrics
from .localize import ProposalConfig, ProposalScorer, localize_ex

# Classes that justify stopping the line rather than queueing for review.
SEVERE_CLASSES = ("inclusion", "rolled-in_scale", "pitted_surface")

# Quality gate thresholds. Set from the NEU capture distribution, and recorded
# here rather than buried in the agent so they can be argued with.
# Quality gate thresholds, derived from ALL 1800 real NEU captures
# (eval/quality_gate_calibration.py, eval/quality_gate_recalibrate.py).
#
# Two of these are well separated and load-bearing:
#   sharpness  clean p05 = 1025 vs blurred p50 = 1.9   -> enormous margin
#   noise      clean p99 = 10.29 vs noisy p05 = 11.31 -> clean separation
#
# Two are NOT separable and are set permissively so they never refuse a good
# capture:
#   glare      real p99 = 0.561, max = 0.754; our synthetic glare only reaches
#              0.388, so any limit low enough to catch it also rejects ~14% of
#              legitimate captures. Rough, scaled steel has large genuinely
#              saturated regions; a 1xN "largest bright region" feature does not
#              separate them either (clean p99 = 0.324 vs glare p05 = 0.282).
#              Glare detection is therefore NOT claimed - see
#              docs/EVALUATION.md.
#   blown      same reasoning.
SHARPNESS_MIN = 12.0        # Laplacian variance; real captures sit at 1025+
NOISE_MAX = 10.8            # separates cleanly, and prevents a real safety bug
GLARE_MAX = 0.80            # above the real maximum (0.754): effectively off
BLOWN_MAX = 0.70            # above the real p99 (0.409): effectively off
CRUSHED_MAX = 0.05          # never fires on real data

_MODEL = None

# Localisation operating point. Selected on the TUNE validation slice by
# eval/train_scorer.py (which sweeps score_thresh x top_k x nms_iou together) and
# reported once on the disjoint REPORT split. Hard-coded here so the request path
# cannot silently drift from the configuration the report describes.
LOCALIZE_CONFIG = ProposalConfig(variant_factors=(0.85, 2.1), top_k=2,
                                 nms_iou=0.3, score_thresh=0.3)
_SCORER = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DefectClassifier.load()
    return _MODEL


def get_scorer():
    """Cached trained proposal scorer. `ProposalScorer.load` raises if
    backend/app/scorer.npz is missing or was trained on a different feature
    count, which is deliberate - a silently-wrong localiser would be worse than
    a loud one."""
    global _SCORER
    if _SCORER is None:
        _SCORER = ProposalScorer.load()
    return _SCORER


def localize(gray, cfg=None, scorer=None):
    """Multi-scale proposal + learned scorer -> ranked candidate defect regions.

    Replaces the single-threshold connected-component version that lived here
    until 2026-10-17. That version was still the live path while
    `backend/app/localize.py` - the module the report's 0.219 localisation
    number was measured on - sat unimported, so the reported number described
    code that never ran on a request. This wires the measured module in.

    Operating point (`variant_factors`, `top_k`, `nms_iou`, `score_thresh`) is
    the one selected on the TUNE validation slice by eval/train_scorer.py and
    reported once on the disjoint REPORT split: macro F1 0.2200 -> 0.2247, with
    the proposal oracle rising 0.346 -> 0.546.

    Measured cost of wiring it in, stated because it is a real regression:
    localisation 0.64 -> 15.3 ms per image (24x), so `analyze()` goes from 8.4 to
    ~23 ms per image. The cause is structural - 9 (blur x threshold) passes of
    blur+threshold+morphology+connectedComponents instead of one. We took it
    anyway: 23 ms per 200x200 crop is ~43 img/s, well above what a single
    inspection camera needs, and the alternative was a report whose headline
    localisation number did not describe the running system.

    Honest caveat, unchanged by the rewrite: box-level localisation on NEU is
    still weak. The classifier carries the "which defect" decision; this stage
    supplies a coarse pointer and, more importantly, an independent signal used
    to catch classifier/localiser disagreement.
    """
    picked, emap = localize_ex(gray, cfg, scorer=scorer)
    return [
        {
            "box": [c.x, c.y, c.w, c.h],
            "area": int(c.w * c.h),
            "energy": round(float(c.energy), 3),
            "score": round(float(c.score), 4),
            "var_factor": round(float(c.var_factor), 3),
        }
        for c in picked
    ], emap


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

    regions, anom = localize(gray, cfg=LOCALIZE_CONFIG, scorer=get_scorer())
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
