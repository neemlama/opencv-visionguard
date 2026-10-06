"""Grid-search the NEU detector config instead of guessing thresholds.

The first hand-picked config scored macro F1 = 0.074 with precision 0.035 - it
was over-segmenting. Rather than nudge numbers by intuition, measure a grid and
let the data pick.

Scores every config on the same subset per class so comparisons are fair, and
prints the top configs plus which parameter actually mattered.

    python eval/neu_tune.py --limit 40
"""
import argparse
import itertools
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np   # noqa: E402

from backend.app.neu import DetectConfig, Region, locate_defects  # noqa: E402
from eval.neu_eval import CLASSES, IMG_DIR, ANN_DIR, parse_annotations, match  # noqa: E402

GRID = {
    "anom_thresh": [8.0, 12.0, 16.0, 20.0, 26.0],
    "min_area": [60, 150, 400, 900],
    "box": [21, 31, 45],
    "close_k": [5, 9, 13],
}


def load_subset(limit):
    by_class = defaultdict(list)
    for x in sorted(ANN_DIR.glob("*.xml")):
        anns = parse_annotations(x)
        if anns:
            by_class[anns[0][0]].append(x)
    subset = []
    for cls in CLASSES:
        for xp in by_class.get(cls, [])[:limit]:
            path = IMG_DIR / (xp.stem + ".jpg")
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            subset.append((path, img, [r for _, r in parse_annotations(xp)]))
    return subset


def score(subset, cfg, iou_thresh=0.5):
    tp = fp = fn = 0
    per_class = defaultdict(lambda: [0, 0, 0])
    for path, img, gts in subset:
        cls = path.stem.rsplit("_", 1)[0]
        preds = locate_defects(img, cfg).regions
        a, b, c, _ = match(preds, gts, iou_thresh)
        tp += a
        fp += b
        fn += c
        s = per_class[cls]
        s[0] += a
        s[1] += b
        s[2] += c
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    per_f1 = []
    for cls in CLASSES:
        s = per_class.get(cls, [0, 0, 0])
        p = s[0] / (s[0] + s[1]) if (s[0] + s[1]) else 0.0
        r = s[0] / (s[0] + s[2]) if (s[0] + s[2]) else 0.0
        per_f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "macro_f1": float(np.mean(per_f1)),
        "per_class_f1": {c: round(v, 3) for c, v in zip(CLASSES, per_f1)},
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval" / "neu_tune.json"))
    args = ap.parse_args()

    subset = load_subset(args.limit)
    print(f"subset: {len(subset)} images ({args.limit}/class)\n")

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"evaluating {len(combos)} configs...")
    t0 = time.time()
    rows = []
    for i, combo in enumerate(combos, 1):
        overrides = dict(zip(keys, combo))
        cfg = DetectConfig(**overrides)
        r = score(subset, cfg)
        rows.append({"config": overrides, **{k: round(v, 4) for k, v in r.items()
                                             if isinstance(v, float)},
                     "per_class_f1": r["per_class_f1"], "tp": r["tp"],
                     "fp": r["fp"], "fn": r["fn"]})
        if i % 20 == 0:
            print(f"  {i}/{len(combos)} best macroF1 so far "
                  f"{max(x['macro_f1'] for x in rows):.4f}")

    rows.sort(key=lambda r: -r["macro_f1"])
    print(f"\ndone in {time.time() - t0:.1f}s. top {args.top} by macro F1:\n")
    for r in rows[: args.top]:
        c = r["config"]
        print(f"  macroF1={r['macro_f1']:.4f} F1={r['f1']:.4f} "
              f"P={r['precision']:.3f} R={r['recall']:.3f} | "
              f"thr={c['anom_thresh']} area={c['min_area']} box={c['box']} close={c['close_k']}")
        print(f"      per-class F1: {r['per_class_f1']}")

    print("\nbaseline (hand-picked, first attempt): macro F1 0.074, P 0.035, R 0.111")

    # Which single parameter moves the needle most?
    print("\nsensitivity (best macro F1 reachable by varying one param):")
    for k in keys:
        vals = defaultdict(float)
        for r in rows:
            vals[r["config"][k]] = max(vals[r["config"][k]], r["macro_f1"])
        span = max(vals.values()) - min(vals.values())
        print(f"  {k:<12} {dict((a, round(b, 3)) for a, b in sorted(vals.items()))}"
              f"  spread={span:.3f}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
