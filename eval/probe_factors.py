"""Which nested box factors actually earn their place in the proposal pool?

`probe_box_variants.py` showed that growing/shrinking every connected component
can lift oracle recall from 0.35 to 0.54 - but it costs 80 -> 634 boxes per
image, and the scorer has to pick 3 out of 634, so precision and runtime both
suffer. Not every factor is pulling its weight.

This does greedy forward selection over a single factor pool, adding the factor
that buys the most oracle recall per extra box, and prints the whole trace. The
stopping point is a box budget: past it the next factor costs more boxes than
the recall it returns. Oracle recall only - no scorer, no held-out numbers.

    python eval/probe_factors.py --limit 20 --budget 260
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

GRID = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.15, 1.4, 1.7, 2.1)


def variant_pool(boxes, factors, energy_map):
    """Base boxes plus nested variants about each box's anomaly centroid."""
    out = [tuple(int(v) for v in b) for b in boxes]
    for bx, by, bw, bh in boxes:
        x, y, w, h = int(bx), int(by), int(bw), int(bh)
        patch = energy_map[max(0, y):y + h, max(0, x):x + w]
        if patch.size == 0:
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            tot = float(patch.sum())
            if tot > 1e-6:
                xs = np.arange(x, x + w, dtype=np.float64)
                ys = np.arange(y, y + h, dtype=np.float64)
                cx = float((patch.sum(axis=0) * xs).sum() / tot)
                cy = float((patch.sum(axis=1) * ys).sum() / tot)
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
        for f in factors:
            nw = max(2, int(round(w * f)))
            nh = max(2, int(round(h * f)))
            out.append((int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0)), nw, nh))
    if not out:
        return np.zeros((0, 4), np.int32)
    return np.unique(np.asarray(out, dtype=np.int32), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=260,
                    help="stop adding factors past this many boxes/image")
    ap.add_argument("--max-factors", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "eval" / "factor_probe.json"))
    args = ap.parse_args()

    by_class = build_index()
    pcfg = ProposalConfig(min_area=20)

    # Cache base proposals per image so the greedy loop does not re-propose.
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
            base = np.unique(np.concatenate(pool, axis=0), axis=0) if pool \
                else np.zeros((0, 4), np.int32)
            images.append((base, energy_map, [r for _, r in parse_annotations(xp)], cls))
    print(f"cached {len(images)} TUNE images, "
          f"{sum(len(g) for _, _, g, _ in images)} GT boxes\n")

    def score_pool(factors):
        hit = tot = 0
        counts = []
        per_class = defaultdict(lambda: [0, 0])
        for base, emap, gts, cls in images:
            s = variant_pool(base, factors, emap)
            counts.append(len(s))
            for g in gts:
                tot += 1
                per_class[cls][1] += 1
                ok = bool(len(s)) and \
                    float(_box_iou_vector(s, (g.x, g.y, g.w, g.h)).max()) >= args.iou
                hit += ok
                per_class[cls][0] += ok
        return hit / max(1, tot), float(np.mean(counts)), per_class, tot

    base_rec, base_boxes, _, tot = score_pool(())
    print(f"{'factors':<34}{'oracle':>9}{'boxes/img':>11}{'gain':>9}")
    print(f"{'(base only)':<34}{base_rec:>9.4f}{base_boxes:>11.0f}{0.0:>9.4f}")

    chosen = []
    trace = []
    for step in range(args.max_factors):
        best = None
        for f in GRID:
            if f in chosen:
                continue
            rec, boxes, _pc, _t = score_pool(tuple(chosen + [f]))
            if boxes > args.budget:
                continue
            gain = rec - (trace[-1][1] if trace else base_rec)
            per_box = gain / max(1.0, boxes - (trace[-1][2] if trace else base_boxes))
            key = per_box
            if best is None or key > best[0]:
                best = (key, f, rec, boxes)
        if best is None:
            print("  (budget reached - no further factor fits)")
            break
        _key, f, rec, boxes = best
        prev = trace[-1][1] if trace else base_rec
        chosen.append(f)
        trace.append((list(chosen), rec, boxes))
        print(f"{str(chosen):<34}{rec:>9.4f}{boxes:>11.0f}{rec - prev:>9.4f}")

    final_rec, final_boxes, per_class, _t = score_pool(tuple(chosen))
    print(f"\nselected factors: {chosen}")
    print(f"oracle {base_rec:.4f} -> {final_rec:.4f}   "
          f"boxes/img {base_boxes:.0f} -> {final_boxes:.0f}")

    print(f"\nper-class oracle at the selected pool:")
    for cls in CLASSES:
        h, n = per_class[cls]
        print(f"  {cls:<18}{h / max(1, n):.4f}")

    Path(args.out).write_text(json.dumps({
        "split": "TUNE half only (even indices)",
        "images_per_class": args.limit,
        "iou_threshold": args.iou,
        "box_budget": args.budget,
        "grid": list(GRID),
        "selected_factors": chosen,
        "base_oracle": round(base_rec, 4),
        "selected_oracle": round(final_rec, 4),
        "base_boxes_per_image": round(base_boxes, 1),
        "selected_boxes_per_image": round(final_boxes, 1),
        "trace": [{"factors": f, "oracle": round(r, 4), "boxes": round(b, 1)}
                  for f, r, b in trace],
        "per_class_oracle": {c: round(per_class[c][0] / max(1, per_class[c][1]), 4)
                             for c in CLASSES},
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()