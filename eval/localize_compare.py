"""Apples-to-apples: single-scale (neu.py) vs multi-scale (localize.py).

The honest question is not "is 0.184 better than the old 0.083?" - the old 0.188
was grid-searched on the same images it was reported on (Leak 3). The question
is whether multi-scale beats single-scale when BOTH are tuned on TUNE and
reported on the same disjoint REPORT split.

Writes the comparison, including each arm's oracle ceiling, so a difference can
be attributed to the proposal generator rather than to the evaluation protocol.
"""
import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.app.localize import ProposalConfig  # noqa: E402
from backend.app.neu import DetectConfig, locate_defects  # noqa: E402
from eval.neu_eval import CLASSES, match  # noqa: E402
from eval.localize_eval import (  # noqa: E402
    build_index, split_indices, prepare, score, summarise,
)


def score_single(prepared_single, cfg, iou_thresh):
    per = defaultdict(lambda: dict(tp=0, fp=0, fn=0, gt=0, og=0))
    for cls, gray, gts in prepared_single:
        p = per[cls]
        a, b, c, _ = match(locate_defects(gray, cfg).regions, gts, iou_thresh)
        p["tp"] += a
        p["fp"] += b
        p["fn"] += c
        p["gt"] += c
    return per


def load_single(by_class, idxs_by_class):
    import cv2
    from eval.neu_eval import IMG_DIR, parse_annotations
    out = []
    for cls, idxs in idxs_by_class.items():
        paths = by_class[cls]
        for i in idxs:
            xp = paths[i]
            ip = IMG_DIR / (xp.stem + ".jpg")
            if not ip.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            out.append((cls, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        [r for _, r in parse_annotations(xp)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-limit", type=int, default=60)
    ap.add_argument("--report-limit", type=int, default=150)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "eval" / "localize_compare.json"))
    args = ap.parse_args()

    by_class = build_index()
    tune_idx, rep_idx = {}, {}
    for cls in CLASSES:
        t, r = split_indices(len(by_class[cls]), args.tune_limit, args.report_limit)
        tune_idx[cls], rep_idx[cls] = t, r
    overlap = {(c, i) for c, v in tune_idx.items() for i in v} & \
              {(c, i) for c, v in rep_idx.items() for i in v}
    assert not overlap, f"split overlap: {len(overlap)}"
    print(f"TUNE total={sum(len(v) for v in tune_idx.values())}  "
          f"REPORT total={sum(len(v) for v in rep_idx.values())}  overlap={len(overlap)}\n")

    tune_single = load_single(by_class, tune_idx)
    rep_single = load_single(by_class, rep_idx)
    pcfg = ProposalConfig(min_area=20)
    tune_multi = prepare(by_class, tune_idx, pcfg)
    rep_multi = prepare(by_class, rep_idx, pcfg)

    grid = {
        "anom_thresh": [8.0, 12.0, 16.0, 20.0, 26.0],
        "min_area": [60, 150, 400],
        "box": [21, 31, 45],
        "close_k": [5, 9, 13],
        "open_k": [0, 3],
        "merge_iou": [0.0, 0.3],
    }
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"tuning single-scale over {len(combos)} configs on TUNE...")
    t0 = time.time()
    rows = []
    for combo in combos:
        cfg = DetectConfig(**dict(zip(keys, combo)))
        s = summarise(score_single(tune_single, cfg, args.iou))
        rows.append((s["macro_f1"], dict(zip(keys, combo))))
    rows.sort(key=lambda r: -r[0])
    print(f"done in {time.time()-t0:.1f}s\n")
    for f1, c in rows[:3]:
        print(f"  single-scale tune macroF1={f1:.4f} {c}")

    best_single_f1, best_single_cfg = rows[0]
    rep_s = summarise(score_single(rep_single, DetectConfig(**best_single_cfg), args.iou))

    multi_rows = []
    import eval.localize_eval as LE
    multi_rows = LE.tune_selection(tune_multi, args.iou)
    best_multi_f1, best_multi_cfg = multi_rows[0]
    rep_m = summarise(score(rep_multi, ProposalConfig(min_area=20, **best_multi_cfg), args.iou))
    orc = score(rep_multi, ProposalConfig(min_area=20, **best_multi_cfg), args.iou, oracle=True)
    tot_gt = sum(v["gt"] for v in orc.values())

    print("\n=== HELD-OUT, IDENTICAL IMAGES, BOTH TUNED ON TUNE SPLIT ===")
    print(f"{'arm':<16}{'tune macroF1':>14}{'holdout macroF1':>17}{'P':>8}{'R':>8}{'pooled F1':>11}")
    print(f"{'single-scale':<16}{best_single_f1:>14.4f}{rep_s['macro_f1']:>17.4f}"
          f"{rep_s['precision']:>8.3f}{rep_s['recall']:>8.3f}{rep_s['pooled_f1']:>11.4f}")
    print(f"{'multi-scale':<16}{best_multi_f1:>14.4f}{rep_m['macro_f1']:>17.4f}"
          f"{rep_m['precision']:>8.3f}{rep_m['recall']:>8.3f}{rep_m['pooled_f1']:>11.4f}")
    delta = rep_m["macro_f1"] - rep_s["macro_f1"]
    print(f"{'delta':<16}{'':>14}{delta:>+17.4f}")
    print(f"\nmulti-scale ORACLE recall ceiling @IoU{args.iou}: "
          f"{sum(v['og'] for v in orc.values())/max(1,tot_gt):.4f} over {tot_gt} GT boxes")

    print(f"\n{'class':<18}{'single F1':>12}{'multi F1':>12}")
    for cls in CLASSES:
        a = rep_s["per_class"].get(cls, {}).get("f1", 0.0)
        b = rep_m["per_class"].get(cls, {}).get("f1", 0.0)
        print(f"{cls:<18}{a:>12.3f}{b:>12.3f}")

    Path(args.out).write_text(json.dumps({
        "protocol": "both arms tuned on TUNE split, reported once on disjoint REPORT split",
        "iou_threshold": args.iou,
        "tune_per_class": args.tune_limit,
        "report_per_class": args.report_limit,
        "split_overlap": len(overlap),
        "single_scale": {"tune_macro_f1": best_single_f1,
                         "config": best_single_cfg, "holdout": rep_s},
        "multi_scale": {"tune_macro_f1": best_multi_f1,
                        "config": best_multi_cfg, "holdout": rep_m},
        "delta_holdout_macro_f1": round(delta, 4),
        "multi_scale_oracle_recall": round(sum(v["og"] for v in orc.values()) / max(1, tot_gt), 4),
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()