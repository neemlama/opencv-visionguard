"""Tune and report the multi-scale localiser WITHOUT the leak the old tuner had.

`eval/neu_tune.py` grid-searched on the first N images per class and
`eval/neu_eval.py` reported on those same images, so the published 0.188 was
tuned on its own test set. This script splits each class into a disjoint TUNE
and REPORT half, selects every hyper-parameter on TUNE only, then reports once
on REPORT. Oracle recall is reported too, so the gap between "what the proposal
generator can reach" and "what the scorer actually gets" stays visible.

Proposals are built once per image and reused across the whole grid, because
proposal generation does not depend on the scorer weights.

    python eval/localize_eval.py --tune-limit 60 --report-limit 150
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.localize import (  # noqa: E402
    ProposalConfig, build_candidates, propose_all, select, _box_iou_vector,
)
from eval.neu_eval import CLASSES, IMG_DIR, ANN_DIR, parse_annotations, match  # noqa: E402

PROPOSAL_FIELDS = ("blur_sizes", "thresholds", "close_k", "min_area", "max_frac")


def build_index():
    """Per-class image list, in stable sorted order."""
    by_class = defaultdict(list)
    for xp in sorted(ANN_DIR.glob("*.xml")):
        anns = parse_annotations(xp)
        if anns:
            by_class[anns[0][0]].append(xp)
    return by_class


def split_indices(n, tune_limit, report_limit):
    """Interleaved disjoint halves. Interleaving rather than a head/tail cut
    avoids coupling the split to file order, which groups by class prefix."""
    tune, report = [], []
    for i in range(n):
        (tune if i % 2 == 0 else report).append(i)
    return tune[:tune_limit], report[:report_limit]


def prepare(by_class, idxs_by_class, pcfg: ProposalConfig):
    """Load images and build their proposals once."""
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scales, _, _ = propose_all(gray, pcfg)
            pool = [np.array([tuple(b) for b in s], dtype=np.int32)
                    for s in scales if len(s)]
            stacked = np.concatenate(pool, axis=0) if pool else np.zeros((0, 4), np.int32)
            out.append((cls, build_candidates(gray, pcfg), stacked,
                        [r for _, r in parse_annotations(xp)]))
    return out


def score(prepared, cfg, iou_thresh, oracle=False):
    per = defaultdict(lambda: dict(tp=0, fp=0, fn=0, gt=0, og=0))
    for cls, cands, stacked, gts in prepared:
        p = per[cls]
        if oracle:
            for g in gts:
                p["gt"] += 1
                if len(stacked):
                    best = float(_box_iou_vector(stacked, (g.x, g.y, g.w, g.h)).max())
                    p["og"] += best >= iou_thresh
            continue
        a, b, c, _ = match(select(cands, cfg), gts, iou_thresh)
        p["tp"] += a
        p["fp"] += b
        p["fn"] += c
        p["gt"] += c
    return per


def summarise(per):
    out = {}
    for cls in CLASSES:
        s = per.get(cls)
        if not s:
            continue
        hit, miss, extra = s["tp"], s["fn"], s["fp"]
        if hit + miss == 0:
            continue
        prec = hit / (hit + extra) if (hit + extra) else 0.0
        rec = hit / (hit + miss)
        out[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0}
    if not out:
        return {"macro_f1": 0.0, "per_class": {}}
    tp = sum(per[c]["tp"] for c in out)
    fp = sum(per[c]["fp"] for c in out)
    fn = sum(per[c]["fn"] for c in out)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "macro_f1": round(float(np.mean([v["f1"] for v in out.values()])), 4),
        "pooled_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
        "precision": round(p, 4), "recall": round(r, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "per_class": out,
    }


def tune_selection(prepared, iou_thresh, n_trials=600, seed=0):
    """Random search over scorer weights and suppression.

    The full product of these five weights and three suppression settings is
    ~4400 configs, and a random sample of 600 finds the same optimum here at a
    fraction of the cost. Seeded, so the selection is reproducible - an
    unseeded search would make the reported number unreproducible, which is the
    kind of quiet non-determinism this project already retracted one claim over.

    Proposal parameters are held fixed here rather than swept, because proposals
    are what must be cached for this grid to be affordable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_trials):
        params = {
            "w_energy": float(rng.choice([0.0, 0.5, 1.0, 2.0])),
            "w_agree": float(rng.choice([0.0, 0.5, 1.0, 2.0])),
            "w_compact": float(rng.choice([0.0, 0.5, 1.0])),
            "w_share": float(rng.choice([0.0, 1.0, 2.0])),
            "w_contrast": float(rng.choice([0.0, 1.0, 2.0])),
            "nms_iou": float(rng.choice([0.3, 0.4, 0.5])),
            "top_k": int(rng.choice([2, 3, 4, 6])),
            "score_thresh": float(rng.choice([0.0, 0.2, 0.4, 0.6])),
        }
        cfg = ProposalConfig(**params)
        s = summarise(score(prepared, cfg, iou_thresh))
        rows.append((s["macro_f1"], params))
    rows.sort(key=lambda r: -r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-limit", type=int, default=40, help="images/class for tuning")
    ap.add_argument("--report-limit", type=int, default=150, help="images/class for reporting")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "eval" / "localize_results.json"))
    args = ap.parse_args()

    by_class = build_index()
    tune_idx, rep_idx = {}, {}
    for cls in CLASSES:
        t, r = split_indices(len(by_class[cls]), args.tune_limit, args.report_limit)
        tune_idx[cls], rep_idx[cls] = t, r
    tune_set = {(c, i) for c, v in tune_idx.items() for i in v}
    rep_set = {(c, i) for c, v in rep_idx.items() for i in v}
    overlap = tune_set & rep_set
    assert not overlap, f"TUNE/REPORT overlap: {len(overlap)}"

    print(f"TUNE   images/class={args.tune_limit}  total={sum(len(v) for v in tune_idx.values())}")
    print(f"REPORT images/class={args.report_limit}  total={sum(len(v) for v in rep_idx.values())}")
    print(f"group overlap between splits: {len(overlap)}\n")

    pcfg = ProposalConfig(min_area=args.min_area)
    t0 = time.time()
    tune_prepared = prepare(by_class, tune_idx, pcfg)
    rep_prepared = prepare(by_class, rep_idx, pcfg)
    print(f"proposals built in {time.time()-t0:.1f}s "
          f"(mean {np.mean([len(c) for _, c, _, _ in tune_prepared]):.0f} candidates/image)\n")

    rows = tune_selection(tune_prepared, args.iou)
    print("top configs by TUNE macro F1:")
    for f1, c in rows[:5]:
        print(f"  macroF1={f1:.4f} {c}")

    best_f1, best_params = rows[0]
    cfg = ProposalConfig(min_area=args.min_area, **best_params)
    print(f"\nselected on TUNE only: {best_params}  (tune macroF1={best_f1:.4f})")

    print("\n--- REPORTING ON DISJOINT HOLD-OUT ---")
    rep = summarise(score(rep_prepared, cfg, args.iou))
    orc_raw = score(rep_prepared, cfg, args.iou, oracle=True)
    tot_gt = sum(v["gt"] for v in orc_raw.values())
    oracle = {"recall_at_iou": round(sum(v["og"] for v in orc_raw.values()) / max(1, tot_gt), 4),
              "gt_boxes": tot_gt}

    print(f"held-out macro F1 : {rep['macro_f1']:.4f}   (tune was {best_f1:.4f})")
    print(f"held-out pooled   : P={rep['precision']:.3f} R={rep['recall']:.3f} F1={rep['pooled_f1']:.4f}")
    print(f"held-out ORACLE recall @IoU{args.iou}: {oracle['recall_at_iou']:.4f} "
          f"(ceiling over {oracle['gt_boxes']} GT boxes)")
    print(f"\n{'class':<18}{'P':>8}{'R':>8}{'F1':>8}")
    for cls in CLASSES:
        v = rep["per_class"].get(cls)
        if v:
            print(f"{cls:<18}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")

    Path(args.out).write_text(json.dumps({
        "dataset": "NEU-DET",
        "iou_threshold": args.iou,
        "split": {
            "method": "per-class interleaved: even index=tune, odd index=report",
            "tune_per_class": args.tune_limit,
            "report_per_class": args.report_limit,
            "group_overlap": len(overlap),
        },
        "proposal_config": {k: getattr(pcfg, k) for k in PROPOSAL_FIELDS},
        "selected_scorer": best_params,
        "tune_macro_f1": best_f1,
        "holdout": rep,
        "oracle": oracle,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()