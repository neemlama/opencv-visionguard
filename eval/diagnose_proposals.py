"""Why are 65% of ground-truth boxes unreachable by the proposal generator?

`scorer_results.json` reports oracle recall 0.3455 at IoU 0.5: even a perfect
selector tops out there. That ceiling is the reason localisation macro F1 sits
near 0.22, so before building anything we need to know WHICH constraint the
proposal generator is actually hitting. There are two, and they need opposite
fixes:

  * **detection failure** - nothing in the proposal set overlaps the GT box at
    all (best IoU < 0.1). The region is too faint, too small, or too linear for
    an isotropic box-blur anomaly map to fire on it. Fix: more/better proposal
    families (oriented top-hat, more scales, lower thresholds).

  * **refinement failure** - a proposal does overlap the GT box but is the wrong
    size (best IoU 0.1..0.5), usually because neighbouring defects merged into
    one component at this threshold. Fix: refine or re-slice the box, not new
    detectors.

This script splits every GT box into those buckets and prints the distribution
per class, plus the size statistics that explain them. It runs on the TUNE half
ONLY (even indices, per localize_eval.split_indices) so that reading it cannot
contaminate the REPORT split used for published numbers.

    python eval/diagnose_proposals.py --limit 20
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.localize import ProposalConfig, propose_all, _box_iou_vector  # noqa: E402
from eval.neu_eval import CLASSES, IMG_DIR, parse_annotations  # noqa: E402
from eval.localize_eval import build_index, split_indices  # noqa: E402

# A proposal this overlapping but still below the 0.5 threshold is a size or
# merge problem, not a failure to see the defect.
REFINE_LO = 0.1


def bucket(best_iou):
    if best_iou >= 0.5:
        return "hit"
    if best_iou >= REFINE_LO:
        return "refine"
    return "detect_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="images per class (TUNE half)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "eval" / "proposal_diagnosis.json"))
    args = ap.parse_args()

    by_class = build_index()
    pcfg = ProposalConfig(min_area=args.min_area)

    per_class = defaultdict(lambda: defaultdict(int))
    size_rows = []
    ratio_hit, ratio_refine = [], []

    for cls in CLASSES:
        idxs, _ = split_indices(len(by_class[cls]), args.limit, 0)
        for i in idxs:
            xp = by_class[cls][i]
            ip = IMG_DIR / (xp.stem + ".jpg")
            if not ip.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            frame = height * width

            per_scale, _, _ = propose_all(gray, pcfg)
            pool = [b for b in per_scale if len(b)]
            stacked = np.concatenate(pool, axis=0) if pool else np.zeros((0, 4), np.int32)

            for _name, g in parse_annotations(xp):
                per_class[cls]["gt"] += 1
                gb = (g.x, g.y, g.w, g.h)
                area_frac = (g.w * g.h) / frame
                size_rows.append((cls, g.w, g.h, area_frac))

                if not len(stacked):
                    per_class[cls]["detect_fail"] += 1
                    continue
                ious = _box_iou_vector(stacked, gb)
                j = int(ious.argmax())
                best = float(ious[j])
                b = bucket(best)
                per_class[cls][b] += 1

                # Area of the best-overlapping proposal relative to the GT box.
                bx, by, bw, bh = stacked[j]
                r = (bw * bh) / max(1, g.w * g.h)
                if b == "refine":
                    ratio_refine.append(r)
                elif b == "hit":
                    ratio_hit.append(r)

    # ---- report ----------------------------------------------------------
    print(f"{'class':<18}{'gt':>6}{'hit':>7}{'refine':>8}{'detect_fail':>12}"
          f"{'oracle':>9}")
    tot = defaultdict(int)
    for cls in CLASSES:
        d = per_class[cls]
        n = d["gt"] or 1
        for k in ("gt", "hit", "refine", "detect_fail"):
            tot[k] += d[k]
        print(f"{cls:<18}{d['gt']:>6}{d['hit']:>7}{d['refine']:>8}"
              f"{d['detect_fail']:>12}{d['hit'] / n:>9.4f}")

    n = tot["gt"] or 1
    print(f"{'ALL':<18}{tot['gt']:>6}{tot['hit']:>7}{tot['refine']:>8}"
          f"{tot['detect_fail']:>12}{tot['hit'] / n:>9.4f}")

    print("\nWhere the misses come from:")
    print(f"  refinement failures (proposal overlaps, wrong size): "
          f"{tot['refine'] / n:.1%} of all GT")
    print(f"  detection failures   (nothing overlaps at all)      : "
          f"{tot['detect_fail'] / n:.1%} of all GT")

    # ---- size statistics -------------------------------------------------
    areas = np.array([r[3] for r in size_rows])
    wsv = np.array([r[1] for r in size_rows])
    hsv = np.array([r[2] for r in size_rows])
    print("\nGT box size (as fraction of a 200x200 frame):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3} area_frac={np.percentile(areas, q):.4f}  "
              f"w={np.percentile(wsv, q):.0f}  h={np.percentile(hsv, q):.0f}")

    def stats(v, label):
        if not len(v):
            print(f"  {label}: none")
            return
        v = np.array(v)
        print(f"  {label}: n={len(v)}  median={np.median(v):.2f}  "
              f"p25={np.percentile(v, 25):.2f}  p75={np.percentile(v, 75):.2f}  "
              f">3x={np.mean(v > 3):.1%}  <1/3x={np.mean(v < 1 / 3):.1%}")

    print("\nBest-overlapping proposal area / GT area:")
    stats(ratio_refine, "refinement failures")
    stats(ratio_hit, "hits")

    # Per-class median GT area, to explain which classes are size-limited.
    print("\nPer-class median GT area_frac (small classes are hardest to hit):")
    for cls in CLASSES:
        v = [r[3] for r in size_rows if r[0] == cls]
        if v:
            print(f"  {cls:<18}{np.median(v):.4f}")

    Path(args.out).write_text(json.dumps({
        "split": "TUNE half only (even indices)",
        "images_per_class": args.limit,
        "proposal_config": {k: getattr(pcfg, k) for k in
                            ("blur_sizes", "thresholds", "close_k",
                             "min_area", "max_frac")},
        "iou_threshold": args.iou,
        "refine_lo": REFINE_LO,
        "totals": dict(tot),
        "per_class": {c: dict(per_class[c]) for c in CLASSES},
        "gt_area_frac_percentiles": {
            str(q): round(float(np.percentile(areas, q)), 5)
            for q in (5, 25, 50, 75, 95)
        },
        "refine_area_ratio_median": round(float(np.median(ratio_refine)), 4)
        if ratio_refine else None,
        "hit_area_ratio_median": round(float(np.median(ratio_hit)), 4)
        if ratio_hit else None,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()