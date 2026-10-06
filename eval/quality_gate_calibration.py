"""Calibrate a capture-quality gate on real data, including sensor noise.

The agent ablation exposed a safety failure: under additive sensor noise the loop
auto-passed 196 of 200 genuinely clean captures. The gate checks sharpness, glare
and clipping, but noise RAISES Laplacian variance, so a noisy frame looks sharper
than a clean one and passes every check.

This measures the discriminating statistics on real NEU captures - pristine, and
degraded by the simulator - so the gate thresholds come from distributions rather
than from intuition.

    python eval/quality_gate_calibration.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.capture import make  # noqa: E402
from eval.splits import dedupe, enumerate_files  # noqa: E402


def noise_sigma(gray):
    """Robust high-frequency noise estimate: MAD of the 3x3 high-pass response.

    Uses the median absolute deviation rather than the standard deviation, so a
    genuine defect (a few percent of pixels) does not inflate the estimate.
    """
    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (3, 3), 0)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F).astype(np.float64).var())


def stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return {
        "noise_sigma": noise_sigma(gray),
        "lap_var": lap_var(gray),
        "contrast": float(gray.std()),
        "mean": float(gray.mean()),
    }


def pct(v, q):
    return float(np.percentile(v, q))


def main():
    items = dedupe(enumerate_files())
    imgs = []
    for item, _n, _g in items[:240]:
        im = cv2.imread(str(item["path"]))
        if im is not None:
            imgs.append(im)
    print(f"measuring {len(imgs)} real NEU captures\n")

    presets = ["clean", "sensor_noise", "soft_focus", "under_exposed",
               "specular_glare", "low_contrast"]
    rows = {}
    for p in presets:
        vals = [stats(make(im, p).image) for im in imgs]
        rows[p] = {k: [v[k] for v in vals] for k in vals[0]}

    print(f"{'preset':<16}{'noise_sigma p50':>18}{'p95':>9}{'p99':>9}"
          f"{'lap_var p50':>14}{'p05':>10}")
    for p in presets:
        n = rows[p]["noise_sigma"]
        lv = rows[p]["lap_var"]
        print(f"{p:<16}{pct(n, 50):>18.3f}{pct(n, 95):>9.3f}{pct(n, 99):>9.3f}"
              f"{pct(lv, 50):>14.1f}{pct(lv, 5):>10.1f}")

    clean_n = rows["clean"]["noise_sigma"]
    noisy_n = rows["sensor_noise"]["noise_sigma"]
    thr = (pct(clean_n, 99) + pct(noisy_n, 5)) / 2
    print(f"\nclean   noise_sigma: p50={pct(clean_n, 50):.3f} p99={pct(clean_n, 99):.3f} max={max(clean_n):.3f}")
    print(f"noisy   noise_sigma: p05={pct(noisy_n, 5):.3f} p50={pct(noisy_n, 50):.3f}")
    print(f"\nsuggested NOISE_MAX threshold (midpoint clean p99 / noisy p05): {thr:.3f}")

    clean_l = rows["clean"]["lap_var"]
    soft_l = rows["soft_focus"]["lap_var"]
    print(f"\nsharpness (Laplacian variance):")
    print(f"  clean   p05={pct(clean_l, 5):.1f} p50={pct(clean_l, 50):.1f}")
    print(f"  blurred p50={pct(soft_l, 50):.1f} p95={pct(soft_l, 95):.1f}")
    print(f"  -> a SHARPNESS_MIN below clean p05 would refuse good captures; "
          f"current gate uses 12.0")

    Path(ROOT / "eval" / "quality_gate.json").write_text(json.dumps({
        "noise_sigma_suggested_max": round(thr, 3),
        "distributions": {p: {k: {"p05": round(pct(v, 5), 4), "p50": round(pct(v, 50), 4),
                                 "p95": round(pct(v, 95), 4), "p99": round(pct(v, 99), 4),
                                 "max": round(max(v), 4)}
                            for k, v in rows[p].items()} for p in presets},
    }, indent=2))
    print("\nwrote eval/quality_gate.json")


if __name__ == "__main__":
    main()
