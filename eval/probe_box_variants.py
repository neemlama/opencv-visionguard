"""Probe: can nested box variants lift the oracle recall above 0.3455?

`diagnose_proposals.py` split the misses into 62.3% refinement failures and only
2.6% detection failures, and showed the refinement failures are bimodal - 16.8%
of proposals are >3x too big (defects merged into one component) and 24.2% are
<1/3x too small (a fragment of a larger defect). That says the fix is a better
RECTANGLE, not a better detector.

This probes three families of nested rectangles grown around each connected
component, at several shrink/grow factors, and reports only the ORACLE recall of
the enlarged pool - i.e. what a perfect selector could reach if the right box
were in the pool. Oracle recall needs no learned scorer and no held-out numbers,
so this is a cheap, honest test of the idea before any retraining.

    python eval/probe_box_variants.py --limit 20
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
    ProposalConfig, propose_all, _box_iou_vector, _components,
)
from eval.neu_eval import CLASSES, IMG_DIR, parse_annotations  # noqa: E402
from eval.localize_eval import build_index, split_indices  # noqa: E402

FAMILIES = ("center", "centroid")


def nested_variants(boxes, factors, energy_map, family):
    """Grow/shrink each box about its centre or its anomaly-weighted centroid."""
    out = []
    for bx, by, bw, bh in boxes:
        x, y, w, h = int(bx), int(by), int(bw), int(bh)
        if family == "centroid":
            patch = energy_map[y:y + h, x:x + w]
            if patch.size == 0:
                continue
            tot = float(patch.sum())
            if tot > 1e-6:
                xs = np.arange(x, x + w, dtype=np.float64)
                ys = np.arange(y, y + h, dtype=np.float64)
                cx = float((patch.sum(axis=0) * xs).sum() / tot)
                cy = float((patch.sum(axis=1) * ys).sum() / tot)
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
        else:
            cx, cy = x + w / 2.0, y + h / 2.0

        for f in factors:
            nw = max(2, int(round(w * f)))
            nh = max(2, int(round(h * f)))
            x0 = int(round(cx - nw / 2.0))
            y0 = int(round(cy - nh / 2.0))
            out.append((x0, y0, nw, nh))
    if not out:
        return np.zeros((0, 4), np.int32)
    return np.unique(np.asarray(out, dtype=np.int32), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "eval" / "box_variant_probe.json"))
    args = ap.parse_args()

    by_class = build_index()
    pcfg = ProposalConfig(min_area=args.min_area)

    factor_sets = {
        "s0.5-1.3": (0.5, 0.7, 1.0, 1.3),
        "s0.4-1.7": (0.4, 0.6, 0.8, 1.0, 1.3, 1.7),
        "s0.3-2.0": (0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.0),
    }

    hit = defaultdict(int)
    tot = 0
    cand_counts = defaultdict(list)
    per_class = defaultdict(lambda: defaultdict(int))

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
                tot += len(parse_annotations(xp))
                continue
            stacked = np.concatenate(pool, axis=0)
            stacked = np.unique(stacked, axis=0)

            pools = {"base": stacked}
            for fam in FAMILIES:
                for fname, factors in factor_sets.items():
                    key = f"{fam}:{fname}"
                    pools[key] = nested_variants(stacked, factors, energy_map, fam)

            for _name, g in parse_annotations(xp):
                tot += 1
                per_class[cls]["__n__"] += 1
                for key, s in pools.items():
                    v = float(_box_iou_vector(s, (g.x, g.y, g.w, g.h)).max()) if len(s) else 0.0
                    ok = v >= args.iou
                    hit[key] += ok
                    if ok:
                        per_class[cls][key] += 1

            for key, s in pools.items():
                cand_counts[key].append(len(s))

    print(f"TUNE images, {tot} GT boxes, IoU {args.iou}\n")
    print(f"{'pool':<22}{'oracle recall':>14}{'mean boxes/img':>16}")
    for key in ["base"] + [f"{f}:{n}" for f in FAMILIES for n in factor_sets]:
        r = hit[key] / max(1, tot)
        c = np.mean(cand_counts[key]) if cand_counts[key] else 0
        mark = ""
        if key != "base" and r > hit["base"] / max(1, tot):
            mark = "  <-- better"
        print(f"{key:<22}{r:>14.4f}{c:>16.0f}{mark}")

    base_r = hit["base"] / max(1, tot)
    best_key = max((k for k in hit if k != "base"),
                   key=lambda k: hit[k])
    print(f"\nbest pool: {best_key}  oracle {hit[best_key] / max(1, tot):.4f} "
          f"(base {base_r:.4f}, +{(hit[best_key] - hit['base']) / max(1, tot):.4f})")

    print(f"\nper-class oracle recall, base -> best:")
    for cls in CLASSES:
        n = per_class[cls]["__n__"]
        b = per_class[cls]["base"]
        print(f"  {cls:<18}{b / max(1, n):.4f} -> {per_class[cls][best_key] / max(1, n):.4f}")
    print(f"\ntotals: {tot} GT boxes; base pool hit {hit['base']}, "
          f"best pool hit {hit[best_key]}")

    Path(args.out).write_text(json.dumps({
        "split": "TUNE half only (even indices)",
        "images_per_class": args.limit,
        "iou_threshold": args.iou,
        "gt_boxes": tot,
        "oracle_recall": {k: round(v / max(1, tot), 4) for k, v in hit.items()},
        "mean_boxes_per_image": {k: round(float(np.mean(v)), 1)
                                 for k, v in cand_counts.items() if v},
        "best_pool": best_key,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()