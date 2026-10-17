"""Pick ONE box size per component with a label-free criterion, not the scorer.

Adding every nested variant to the pool lifted oracle recall 0.355 -> 0.571, but
macro F1 moved only 0.2429 -> 0.2462: the pool tripled, precision fell, and the
learned scorer captured just 53.2% of the enlarged oracle because a linear model
can only apply one global size preference (`var_log` came out negative, i.e. it
mostly learned to shrink).

This probes the alternative: keep the pool near its original size by choosing a
single size per component using a criterion that needs no labels, and let the
scorer keep doing only what it is good at - choosing WHICH components are real.

The criterion is contrast: the right rectangle has high mean anomaly INSIDE it
and low mean anomaly in the ring just outside it. A fragment box fails the first
test (little energy inside); a box merged with its neighbours fails the second
(energy just outside). Maximising `inside_mean - ring_mean` over the factors
needs only integral images, so it costs nothing and cannot leak.

    python eval/probe_variant_pick.py --limit 20
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

from backend.app.localize import (  # noqa: E402
    ProposalConfig, propose_all, _box_iou_vector, _integral_maps, _box_sums,
)
from eval.neu_eval import CLASSES, IMG_DIR, parse_annotations  # noqa: E402
from eval.localize_eval import build_index, split_indices  # noqa: E402

FACTORS = (0.7, 0.85, 1.0, 1.3, 1.7, 2.1)


def box_stats(iiE, boxes, energy_map):
    """(inside_mean, ring_mean) for each box."""
    x, y, w, h = (boxes[:, i].astype(np.float64) for i in range(4))
    inside = _box_sums(iiE, boxes) / np.maximum(1.0, w * h)

    height, width = energy_map.shape
    m = np.maximum(3.0, 0.25 * np.maximum(w, h))
    gx0 = np.clip(x - m, 0, width).astype(np.int64)
    gy0 = np.clip(y - m, 0, height).astype(np.int64)
    gx1 = np.clip(x + w + m, 0, width).astype(np.int64)
    gy1 = np.clip(y + h + m, 0, height).astype(np.int64)
    grown = np.stack([gx0, gy0, np.maximum(1, gx1 - gx0), np.maximum(1, gy1 - gy0)], axis=1)
    ring_sum = np.maximum(0.0, _box_sums(iiE, grown) - _box_sums(iiE, boxes))
    ring_area = np.maximum(1.0, (gx1 - gx0) * (gy1 - gy0) - w * h)
    return inside, ring_sum / ring_area


def expand(boxes, factor, cx, cy, shape):
    height, width = shape
    nw = np.maximum(2, np.rint(boxes[:, 2] * factor)).astype(np.int64)
    nh = np.maximum(2, np.rint(boxes[:, 3] * factor)).astype(np.int64)
    x0 = np.clip(np.rint(cx - nw / 2.0).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.rint(cy - nh / 2.0).astype(np.int64), 0, height - 1)
    w_ = np.maximum(1, np.clip(x0 + nw, 1, width) - x0)
    h_ = np.maximum(1, np.clip(y0 + nh, 1, height) - y0)
    return np.stack([x0, y0, w_, h_], axis=1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "eval" / "variant_pick_probe.json"))
    args = ap.parse_args()

    by_class = build_index()
    pcfg = ProposalConfig(min_area=20)

    images = []
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
            per_scale, energy_map, _ = propose_all(gray, pcfg)
            pool = [b for b in per_scale if len(b)]
            if not pool:
                continue
            base = np.unique(np.concatenate(pool, axis=0), axis=0)
            integral = _integral_maps(energy_map)
            iiE, iiX, iiY = integral
            tot = _box_sums(iiE, base)
            cx = _box_sums(iiX, base) / np.maximum(tot, 1e-6)
            cy = _box_sums(iiY, base) / np.maximum(tot, 1e-6)
            empty = tot <= 1e-6
            cx = np.where(empty, base[:, 0] + base[:, 2] / 2.0, cx)
            cy = np.where(empty, base[:, 1] + base[:, 3] / 2.0, cy)
            images.append((base, cx, cy, gray.shape, energy_map, iiE,
                           [r for _, r in parse_annotations(xp)], cls))
    print(f"cached {len(images)} TUNE images, "
          f"{sum(len(g) for _, _, _, _, _, _, g, _ in images)} GT boxes\n")

    methods = ("contrast", "inside_mean", "mass_over_sqrt_area", "base_plus_contrast")
    hit = defaultdict(int)
    tot_gt = 0
    counts = defaultdict(list)
    per_class = defaultdict(lambda: defaultdict(int))
    chosen_hist = defaultdict(lambda: defaultdict(int))

    for base, cx, cy, shape, emap, iiE, gts, cls in images:
        tot_gt += len(gts)
        per_class[cls]["n"] += len(gts)

        inside0, ring0 = box_stats(iiE, base, emap)
        pools = {"base": base}

        crit = {}
        for f in FACTORS:
            b = expand(base, f, cx, cy, shape)
            ins, ring = box_stats(iiE, b, emap)
            crit[f] = (ins, ring, b)
        pick_contrast = np.argmax(np.stack(
            [crit[f][0] - crit[f][1] for f in FACTORS], axis=1), axis=1)
        pick_inside = np.argmax(np.stack([crit[f][0] for f in FACTORS], axis=1), axis=1)
        pick_mass = np.argmax(np.stack(
            [crit[f][0] * np.sqrt(crit[f][2][:, 2] * crit[f][2][:, 3]) for f in FACTORS],
            axis=1), axis=1)

        def gathered(idx):
            return np.concatenate([crit[FACTORS[i]][2][j:j + 1] for j, i in enumerate(idx)],
                                  axis=0)

        picked = gathered(pick_contrast)
        pools["contrast"] = picked
        pools["inside_mean"] = gathered(pick_inside)
        pools["mass_over_sqrt_area"] = gathered(pick_mass)
        pools["base_plus_contrast"] = np.concatenate([base, picked], axis=0)

        for f_idx, f in enumerate(FACTORS):
            for j, i in enumerate(pick_contrast):
                if i == f_idx:
                    chosen_hist["contrast"][f] += 1

        for key, s in pools.items():
            s = np.unique(s, axis=0)
            counts[key].append(len(s))
            for g in gts:
                ok = bool(len(s)) and \
                    float(_box_iou_vector(s, (g.x, g.y, g.w, g.h)).max()) >= args.iou
                hit[key] += ok
                per_class[cls][key] += ok

    print(f"{'method':<24}{'oracle':>9}{'boxes/img':>12}{'gain':>9}")
    base_r = hit["base"] / max(1, tot_gt)
    for key in ["base"] + list(methods):
        r = hit[key] / max(1, tot_gt)
        print(f"{key:<24}{r:>9.4f}{np.mean(counts[key]):>12.0f}{r - base_r:>+9.4f}")

    print("\nwhich factor the contrast criterion picked:")
    tt = sum(chosen_hist["contrast"].values()) or 1
    for f in FACTORS:
        c = chosen_hist["contrast"][f]
        print(f"  f={f:<5}{c:>6}  ({c / tt:.1%})")

    print(f"\nper-class oracle:")
    print(f"{'class':<18}" + "".join(f"{k:>22}" for k in ["base", "contrast",
                                                           "base_plus_contrast"]))
    for cls in CLASSES:
        n = max(1, per_class[cls]["n"])
        print(f"{cls:<18}" + "".join(
            f"{per_class[cls][k] / n:>22.4f}" for k in
            ["base", "contrast", "base_plus_contrast"]))

    Path(args.out).write_text(json.dumps({
        "split": "TUNE half only (even indices)",
        "images_per_class": args.limit,
        "iou_threshold": args.iou,
        "factors": list(FACTORS),
        "gt_boxes": tot_gt,
        "oracle_recall": {k: round(v / max(1, tot_gt), 4) for k, v in hit.items()},
        "mean_boxes_per_image": {k: round(float(np.mean(v)), 1)
                                 for k, v in counts.items()},
        "contrast_factor_histogram": {str(f): chosen_hist["contrast"][f]
                                      for f in FACTORS},
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()