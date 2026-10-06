"""Test two candidate robustness fixes before touching the app.

Remaining eval failures and their causes:
  * noise_sigma10/25 -> weld_blurry judged "clean": the Laplacian blur metric
    amplifies high-frequency sensor noise, so a defocused image looks sharp.
  * dark_-30 / contrast_0.6 -> weld_glare judged "clean": the glare test counts
    pixels above an ABSOLUTE V > 240, which darkening removes.

Candidates: pre-blur the Laplacian input, and Otsu-based (illumination-invariant)
glare segmentation instead of a fixed 240 cut.

Measures pristine + perturbed behaviour for each candidate so the change is
evidence-based rather than assumed.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import LABELS, SAMPLES, apply_perturbation  # noqa: E402

SPEC = json.loads(LABELS.read_text())


def blur_var_plain(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_var_preblurred(gray):
    # Laplacian amplifies noise; a 3x3 pre-blur suppresses it before differencing.
    return float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var())


def glare_ratio_fixed(hsv_v):
    return float((hsv_v > 240).mean())


def glare_ratio_otsu(hsv_v):
    # Otsu finds the bright/dark split for THIS image, so it survives dimming.
    t, _ = cv2.threshold(hsv_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((hsv_v >= t).mean())


def measure(label, is_blurry, is_glare):
    """Return (blurry_ok, glare_ok) counters over pristine + perturbed images."""
    stats = {"blur": [0, 0], "glare": [0, 0]}
    for pspec in SPEC["perturbations"]:
        for lab in SPEC["labels"]:
            f = lab["file"]
            img = apply_perturbation(cv2.imread(str(SAMPLES / f)), pspec)
            small = cv2.resize(img, (640, 480))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            v = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hv = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv_v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))[2]

            want_blur = lab["expected_verdict"] == "re-shoot"
            want_glare = lab["expected_verdict"] == "re-shoot-angle"
            got_blur = is_blurry(gray)
            got_glare = is_glare(hsv_v)
            stats["blur"][1] += 1
            stats["glare"][1] += 1
            if got_blur == want_blur:
                stats["blur"][0] += 1
            if got_glare == want_glare:
                stats["glare"][0] += 1
    print(f"{label}")
    for k, (ok, tot) in stats.items():
        print(f"    {k:<6} {ok}/{tot} correct  ({100 * ok / tot:.1f}%)")
    return stats


def main():
    print("=== baseline (current shipped logic) ===")
    measure("blur=Laplacian(gray), glare=V>240",
            lambda g: blur_var_plain(g) < 100.0,
            lambda h: glare_ratio_fixed(h) > 0.05)

    print("\n=== candidate A: pre-blurred Laplacian, Otsu glare ===")
    measure("blur=Laplacian(GaussianBlur(3)), glare=otsu",
            lambda g: blur_var_preblurred(g) < 100.0,
            lambda h: glare_ratio_otsu(h) > 0.05)

    print("\n=== candidate B: pre-blurred Laplacian only ===")
    measure("blur=Laplacian(GaussianBlur(3)), glare=V>240",
            lambda g: blur_var_preblurred(g) < 100.0,
            lambda h: glare_ratio_fixed(h) > 0.05)

    print("\n=== threshold sensitivity for candidate A blur metric ===")
    for thr in (30, 50, 70, 100, 150):
        ok = tot = 0
        for pspec in SPEC["perturbations"]:
            for lab in SPEC["labels"]:
                img = apply_perturbation(cv2.imread(str(SAMPLES / lab["file"])), pspec)
                gray = cv2.cvtColor(cv2.resize(img, (640, 480)), cv2.COLOR_BGR2GRAY)
                want = lab["expected_verdict"] == "re-shoot"
                tot += 1
                ok += (blur_var_preblurred(gray) < thr) == want
        print(f"    threshold {thr:<5} {ok}/{tot} ({100 * ok / tot:.1f}%)")


if __name__ == "__main__":
    main()
