"""Diagnose why weld_defect is scored `pass` under low light / low contrast.

run_eval.py reports a false negative there, but eval/robustness_rules.py shows the
dark-blob detector never loses the blob. So the failure is downstream: either the
Canny edge channel (absolute thresholds 50/150, which are illumination-dependent)
or the score blend that feeds the verdict.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.vision import analyze_image  # noqa: E402
from eval.run_eval import SAMPLES, apply_perturbation  # noqa: E402

CASES = [
    {"name": "none", "kind": "identity"},
    {"name": "dark_-30", "kind": "brightness", "value": 0.70},
    {"name": "contrast_0.6", "kind": "contrast", "value": 0.6},
]


def canny_stats(gray):
    """Compare fixed thresholds against median-proportional thresholds."""
    med = float(np.median(gray))
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    fixed = cv2.Canny(b, 50, 150)
    scaled = cv2.Canny(b, max(1, int(0.66 * med)), max(2, int(1.33 * med)))
    return med, float((fixed > 0).mean()), float((scaled > 0).mean())


def main():
    for name in ("weld_defect.jpg", "weld_clean.jpg"):
        img = cv2.imread(str(SAMPLES / name))
        print(f"\n=== {name} ===")
        for spec in CASES:
            p = apply_perturbation(img, spec)
            data = cv2.imencode(".jpg", p)[1].tobytes()
            v = analyze_image(data)
            gray = cv2.cvtColor(cv2.resize(p, (640, 480)), cv2.COLOR_BGR2GRAY)
            med, fixed_d, scaled_d = canny_stats(gray)
            print(f"  {spec['name']:<14} verdict_score={v['defect_score']:<6} "
                  f"blobs={v['dark_blobs']:<3} edge_density={v['edge_density']:.5f} "
                  f"contours={v['large_contours']}")
            print(f"  {'':<14} gray median={med:6.1f}  "
                  f"canny(50,150) density={fixed_d:.5f}  "
                  f"canny(median-scaled) density={scaled_d:.5f}")


if __name__ == "__main__":
    main()
