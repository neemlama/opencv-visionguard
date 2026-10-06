"""Recalibrate the capture gate on the FULL real distribution, not a prefix.

Two calibration errors were found:

1. The glare and blown-highlight limits were set from the first 300 NEU files,
   which are not representative - they are dominated by one or two classes. Across
   all 1800 legitimate captures, glare_ratio reaches 0.75 (p99 = 0.56), while the
   synthetic glare preset only reaches 0.388. So the old limit of 0.05 refused a
   large share of GOOD captures, and the "glare" it detected was really just
   "many bright pixels".
2. Consequently the +0.157 glare result in the ablation was partly an artefact of
   a threshold that rejected both synthetic and natural highlights alike.

The fix needs a feature that separates a specular highlight from naturally bright
steel texture. A glare disc is ONE large connected saturated region; surface
texture is many small scattered ones. This measures that.

    python eval/quality_gate_recalibrate.py
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


def bright_region_stats(img, sat=250):
    """Largest connected saturated region, and its share of the frame."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = (v >= sat).astype(np.uint8) * 255
    if mask.max() == 0:
        return 0.0, 0
    n, _lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = v.shape[0] * v.shape[1]
    return float(areas.max() / total), int(n - 1)


def main():
    items = dedupe(enumerate_files())
    imgs = []
    for item, _n, _g in items:
        im = cv2.imread(str(item["path"]))
        if im is not None:
            imgs.append(im)
    print(f"real captures: {len(imgs)}\n")

    presets = ["clean", "specular_glare", "under_exposed", "soft_focus", "sensor_noise"]
    real_share, real_counts = [], []
    for p in presets:
        shares, counts = [], []
        for im in imgs:
            s, c = bright_region_stats(make(im, p).image)
            shares.append(s)
            counts.append(c)
        shares = np.array(shares)
        print(f"{p:<16} largest-bright-region share: p50={np.percentile(shares,50):.4f} "
              f"p90={np.percentile(shares,90):.4f} p99={np.percentile(shares,99):.4f} "
              f"max={shares.max():.4f}")
        if p == "clean":
            real_share, real_counts = shares, counts

    glare = np.array([bright_region_stats(make(im, "specular_glare").image)[0] for im in imgs])
    print(f"\nclean     p50={np.percentile(real_share,50):.4f} "
          f"p95={np.percentile(real_share,95):.4f} p99={np.percentile(real_share,99):.4f}")
    print(f"glare     p05={np.percentile(glare,5):.4f} "
          f"p50={np.percentile(glare,50):.4f}")
    thr = (np.percentile(real_share, 99) + np.percentile(glare, 5)) / 2
    print(f"\nsuggested BRIGHT_REGION_MAX = {thr:.4f}")
    clean_over = (real_share > thr).mean() * 100
    glare_over = (glare > thr).mean() * 100
    print(f"  would refuse {clean_over:.1f}% of clean captures "
          f"and {glare_over:.1f}% of synthetic-glare captures")

    print("\ncomponent counts (scattered texture vs one blob):")
    print(f"  clean  median components={int(np.median(real_counts))}")
    gc = [bright_region_stats(make(im, 'specular_glare').image)[1] for im in imgs[:120]]
    print(f"  glare  median components={int(np.median(gc))}")

    Path(ROOT / "eval" / "gate_recalibration.json").write_text(json.dumps({
        "bright_region_max_suggested": round(float(thr), 5),
        "clean_refusal_pct": round(float(clean_over), 2),
        "glare_detection_pct": round(float(glare_over), 2),
        "old_limits": {"glare_ratio_max": 0.05, "blown_highlights_max": 0.035,
                       "note": "set from a 300-file prefix; real p99 is 0.561"},
        "real_clean_share_p99": round(float(np.percentile(real_share, 99)), 5),
        "glare_share_p05": round(float(np.percentile(glare, 5)), 5),
    }, indent=2))
    print("\nwrote eval/gate_recalibration.json")


if __name__ == "__main__":
    main()
