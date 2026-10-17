"""Train the proposal scorer on TUNE, report it on a disjoint REPORT split.

Motivation: the multi-scale proposal set has oracle recall 0.35 @IoU0.5, but the
unsupervised hand-weighted score reached only 0.21 - choosing 1 good box out of
~78 needs supervision. This trains a binary logistic regression on candidate
features, where a candidate is positive when it reaches IoU 0.5 with any ground
truth box in the image.

The discipline matters and is the whole point: training labels come from the TUNE
split and every reported number comes from the REPORT split, which shares no
image with TUNE. That is what separates this from Leak 3, where the old tuner
searched and reported on the same photographs.

    python eval/train_scorer.py --tune-limit 120 --report-limit 200
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.app.localize import (  # noqa: E402
    ProposalConfig, ProposalScorer, SCORER_LEN, SCORER_FEATURES,
    candidate_matrix, fit_scorer, select,
)
from eval.neu_eval import CLASSES, match  # noqa: E402
from eval.localize_eval import build_index, split_indices, prepare  # noqa: E402


def label_candidates(cands, gts, shape, iou_thresh):
    """1 if the candidate reaches IoU>=t with any GT box."""
    y = np.zeros(len(cands), dtype=np.float64)
    if not cands or not gts:
        return y
    boxes = np.array([c.as_box() for c in cands], dtype=np.int32)
    from backend.app.localize import _box_iou_vector
    for g in gts:
        gb = (g.x, g.y, g.w, g.h)
        y[_box_iou_vector(boxes, gb) >= iou_thresh] = 1.0
    return y


def evaluate(prepared, cfg, scorer, iou_thresh, score_thresh=0.0, scores_by_img=None,
             **overrides):
    per = defaultdict(lambda: dict(tp=0, fp=0, fn=0, gt=0))
    for i, (cls, cands, _stacked, gts, shape) in enumerate(prepared):
        c = ProposalConfig(**{**cfg.__dict__, "score_thresh": score_thresh, **overrides})
        sc = None if scores_by_img is None else scores_by_img.get(i)
        a, b, fn, _ = match(
            select(cands, c, scorer=sc if sc is not None else scorer, shape=shape,
                   scores=None if scores_by_img is None else sc),
            gts, iou_thresh)
        p = per[cls]
        p["tp"] += a
        p["fp"] += b
        p["fn"] += fn
        p["gt"] += fn
    return per


def precompute_scores(prepared, scorer):
    """One feature matrix + probability pass per image, reused across a sweep."""
    out = {}
    for i, (_cls, cands, _s, _gts, shape) in enumerate(prepared):
        if cands:
            out[i] = scorer.score(cands, shape)
    return out


def summarise(per):
    out = {}
    for cls in CLASSES:
        s = per.get(cls)
        if not s or (s["tp"] + s["fn"]) == 0:
            continue
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"])
        out[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0}
    if not out:
        return {"macro_f1": 0.0, "per_class": {}}
    tp = sum(per[c]["tp"] for c in out)
    fp = sum(per[c]["fp"] for c in out)
    fn = sum(per[c]["fn"] for c in out)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"macro_f1": round(float(np.mean([v["f1"] for v in out.values()])), 4),
            "pooled_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
            "precision": round(p, 4), "recall": round(r, 4),
            "tp": tp, "fp": fp, "fn": fn, "per_class": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-limit", type=int, default=120)
    ap.add_argument("--report-limit", type=int, default=200)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--variant-factors", default="",
                    help="comma-separated nested box factors, e.g. '0.85,2.1'. "
                         "Empty reproduces the pre-variant proposal pool.")
    ap.add_argument("--skip-report", action="store_true",
                    help="stop after the TUNE validation slice and never read the "
                         "REPORT split. Use this to compare configs; the REPORT "
                         "split is a once-only read.")
    ap.add_argument("--scorer-npz", default="",
                    help="where to save the scorer (default backend/app/scorer.npz)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "scorer_results.json"))
    args = ap.parse_args()

    factors = tuple(float(v) for v in args.variant_factors.split(",") if v.strip())

    by_class = build_index()
    tune_idx, rep_idx = {}, {}
    for cls in CLASSES:
        t, r = split_indices(len(by_class[cls]), args.tune_limit, args.report_limit)
        tune_idx[cls], rep_idx[cls] = t, r
    overlap = {(c, i) for c, v in tune_idx.items() for i in v} & \
              {(c, i) for c, v in rep_idx.items() for i in v}
    assert not overlap, f"split overlap: {len(overlap)}"
    print(f"TUNE total={sum(len(v) for v in tune_idx.values())}  "
          f"REPORT total={sum(len(v) for v in rep_idx.values())}  overlap={len(overlap)}")

    pcfg = ProposalConfig(min_area=args.min_area, variant_factors=factors)
    print(f"proposal pool: variants={list(factors) or 'none (pre-variant pool)'}")
    t0 = time.time()

    import cv2
    from eval.neu_eval import IMG_DIR, parse_annotations

    def prepare_labelled(idxs_by_class):
        rows = defaultdict(list)
        ys = []
        shapes = []
        out = []
        for cls, idxs in idxs_by_class.items():
            for i in idxs:
                xp = by_class[cls][i]
                ip = IMG_DIR / (xp.stem + ".jpg")
                if not ip.exists():
                    continue
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gts = [r for _, r in parse_annotations(xp)]
                from backend.app.localize import build_candidates
                cands = build_candidates(gray, pcfg)
                out.append((cls, cands, None, gts, gray.shape))
                if not cands:
                    continue
                rows["X"].append(candidate_matrix(cands, gray.shape))
                ys.append(label_candidates(cands, gts, gray.shape, args.iou))
        X = np.concatenate(rows["X"])
        y = np.concatenate(ys)
        return X, y, out

    X, y, tune_prepared = prepare_labelled(tune_idx)
    print(f"proposals built in {time.time()-t0:.1f}s")
    print(f"TUNE candidates: {len(y)}  positive: {int(y.sum())} ({y.mean()*100:.2f}%)")

    print("\npos_weight sweep (val = a further disjoint slice of TUNE, NOT REPORT)")
    # Carve a validation slice out of TUNE so pos_weight and score_thresh are
    # chosen without touching REPORT. TUNE images alternate, so take every 4th.
    val_frac = 0.25
    n_tune_imgs = len(tune_prepared)
    val_imgs = set(range(3, n_tune_imgs, 4))  # image-order stride, deterministic
    tr_imgs = [i for i in range(n_tune_imgs) if i not in val_imgs]

    def stack(imgs):
        Xs = [candidate_matrix(tune_prepared[i][1], tune_prepared[i][4])
              for i in imgs if tune_prepared[i][1]]
        ys = [label_candidates(tune_prepared[i][1], tune_prepared[i][3],
                               tune_prepared[i][4], args.iou)
              for i in imgs if tune_prepared[i][1]]
        if not Xs:
            return None, None
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = stack(tr_imgs)
    Xva, yva = stack(sorted(val_imgs))
    print(f"  train {len(ytr)} cands ({ytr.mean()*100:.2f}% pos)  "
          f"val {len(yva)} cands ({yva.mean()*100:.2f}% pos)")

    best = None
    for pw in (1.0, 3.0, 6.0, 10.0):
        W, b, mu, sd = fit_scorer(Xtr, ytr, epochs=args.epochs, pos_weight=pw)
        sc = ProposalScorer(W, b, mu, sd)
        auc = _auc(Xva, yva, sc)
        val_recall_at_rank = _recall_at_budget(tune_prepared, sorted(val_imgs), sc, pcfg,
                                               args.iou, budget=3)
        print(f"  pos_weight={pw:<5} val AUC={auc:.4f}  recall@top3={val_recall_at_rank:.4f}")
        if best is None or val_recall_at_rank > best[0]:
            best = (val_recall_at_rank, pw, W, b, mu, sd)
    _, pw_best, W, b, mu, sd = best
    scorer = ProposalScorer(W, b, mu, sd)
    if args.skip_report:
        print(f"\nchosen pos_weight={pw_best} (on the TUNE validation slice)")
        print("--skip-report: scorer NOT saved and the REPORT split NOT read")
    else:
        scorer.save(args.scorer_npz) if args.scorer_npz else scorer.save()
        print(f"\nchosen pos_weight={pw_best} (on the TUNE validation slice)")
        print(f"saved scorer -> {args.scorer_npz or 'backend/app/scorer.npz'}")

    # Learn per-feature sign for the write-up.
    print(f"\n{'feature':<14}{'weight':>10}")
    for name, w in sorted(zip(SCORER_FEATURES, W), key=lambda t: -abs(t[1])):
        print(f"{name:<14}{w:>+10.3f}")

    # Operating threshold is chosen on the TUNE validation slice too. Selecting
    # it on REPORT and then quoting REPORT would be Leak 3 all over again in a
    # new place, so the REPORT split below is read exactly once.
    print("\nscore_thresh x top_k x nms_iou sweep on the TUNE validation slice:")
    val_prepared = [tune_prepared[i] for i in sorted(val_imgs)]
    val_scores = precompute_scores(val_prepared, scorer)
    best_op, best_th_f1, best_th = (0.0, 4, 0.40), -1.0, 0.0
    for tk in (1, 2, 3, 4, 6):
        for nms in (0.3, 0.4, 0.5):
            for th in (0.0, 0.1, 0.2, 0.3, 0.5):
                s = summarise(evaluate(val_prepared, pcfg, scorer, args.iou, th,
                                      scores_by_img=val_scores,
                                      top_k=tk, nms_iou=nms))
                if s["macro_f1"] > best_th_f1:
                    best_th_f1, best_op = s["macro_f1"], (th, tk, nms)
    th, sel_top_k, sel_nms = best_op
    print(f"  best: score_thresh={th} top_k={sel_top_k} nms_iou={sel_nms} "
          f"-> macroF1={best_th_f1:.4f}")
    print(f"  (top 6 of the grid)")
    grid = []
    for tk in (1, 2, 3, 4, 6):
        for nms in (0.3, 0.4, 0.5):
            for th_ in (0.0, 0.1, 0.2, 0.3, 0.5):
                s = summarise(evaluate(val_prepared, pcfg, scorer, args.iou, th_,
                                      scores_by_img=val_scores,
                                      top_k=tk, nms_iou=nms))
                grid.append((s["macro_f1"], s["precision"], s["recall"], th_, tk, nms))
    grid.sort(reverse=True)
    for f1, pr, rc, th_, tk, nms in grid[:6]:
        print(f"    th={th_:<5} top_k={tk:<2} nms={nms:<4} macroF1={f1:.4f} "
              f"P={pr:.3f} R={rc:.3f}")
    print(f"selected score_thresh={th} top_k={sel_top_k} nms_iou={sel_nms} "
          f"on TUNE-validation (macroF1={best_th_f1:.4f})")

    if args.skip_report:
        from backend.app.localize import _box_iou_vector as _biv
        orc_h = orc_t = 0
        for _cls, cands, _s, gts, _shape in val_prepared:
            boxes = np.array([c.as_box() for c in cands], dtype=np.int32) \
                if cands else np.zeros((0, 4), np.int32)
            for g in gts:
                orc_t += 1
                if len(boxes) and float(_biv(boxes, (g.x, g.y, g.w, g.h)).max()) >= args.iou:
                    orc_h += 1
        orc = orc_h / max(1, orc_t)
        val = summarise(evaluate(val_prepared, pcfg, scorer, args.iou, th,
                                 scores_by_img=val_scores,
                                 top_k=sel_top_k, nms_iou=sel_nms))
        print(f"\n--- TUNE VALIDATION ONLY ({len(val_prepared)} images, "
              f"{orc_t} GT boxes) ---")
        print(f"macro F1={val['macro_f1']:.4f}  P={val['precision']:.3f} "
              f"R={val['recall']:.3f}  pooledF1={val['pooled_f1']:.4f}")
        print(f"oracle recall on this slice = {orc:.4f}")
        print(f"scorer captured {val['recall'] / orc * 100:.1f}% of oracle" if orc else "")
        print(f"\n{'class':<18}{'P':>8}{'R':>8}{'F1':>8}")
        for cls in CLASSES:
            v = val["per_class"].get(cls)
            if v:
                print(f"{cls:<18}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")
        Path(args.out).write_text(json.dumps({
            "mode": "skip-report (TUNE validation only; REPORT split untouched)",
            "variant_factors": list(factors),
            "scorer_len": SCORER_LEN,
            "split": {"tune_per_class": args.tune_limit,
                      "overlap": len(overlap),
                      "note": "TUNE only - REPORT never read in this run"},
            "tune_candidates": int(len(y)),
            "tune_positive_rate": round(float(y.mean()), 4),
            "pos_weight": pw_best,
            "feature_weights": {n: round(float(w), 4)
                                for n, w in zip(SCORER_FEATURES, W)},
            "holdout_score_thresh": th,
            "holdout_top_k": sel_top_k,
            "holdout_nms_iou": sel_nms,
            "tune_validation": val,
            "tune_validation_oracle_recall": round(orc, 4),
        }, indent=2))
        print(f"\nwrote {args.out}")
        return

    print(f"\n--- REPORTING ON DISJOINT HOLD-OUT ({sum(len(v) for v in rep_idx.values())} images) ---")
    Xr, yr, rep_prepared = prepare_labelled(rep_idx)
    print(f"REPORT candidates: {len(yr)} positive: {int(yr.sum())} ({yr.mean()*100:.2f}%)")
    rep = summarise(evaluate(rep_prepared, pcfg, scorer, args.iou, th,
                         top_k=sel_top_k, nms_iou=sel_nms))
    print(f"held-out macro F1={rep['macro_f1']:.4f}  P={rep['precision']:.3f} "
          f"R={rep['recall']:.3f} pooledF1={rep['pooled_f1']:.4f}")
    print(f"\n{'class':<18}{'P':>8}{'R':>8}{'F1':>8}")
    for cls in CLASSES:
        v = rep["per_class"].get(cls)
        if v:
            print(f"{cls:<18}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")

    # Oracle recall on the SAME report split: how much of the proposal space the
    # scorer actually captured. If this is close to 1, a better scorer cannot help
    # and the proposal generator is the remaining bottleneck.
    oracle_rec = 0
    tot_gt = 0
    for cls, cands, _s, gts, shape in rep_prepared:
        boxes = np.array([c.as_box() for c in cands], dtype=np.int32) if cands \
            else np.zeros((0, 4), np.int32)
        from backend.app.localize import _box_iou_vector
        for g in gts:
            tot_gt += 1
            if len(boxes):
                oracle_rec += float(_box_iou_vector(boxes, (g.x, g.y, g.w, g.h)).max()) >= args.iou
    orc = round(oracle_rec / max(1, tot_gt), 4)
    captured = rep["recall"] / orc if orc else 0.0
    print(f"\nORACLE recall on the same report split: {orc:.4f} over {tot_gt} GT boxes")
    print(f"scorer captured {captured*100:.1f}% of what the proposals could reach "
          f"(recall {rep['recall']:.4f} / oracle {orc:.4f})")

    Path(args.out).write_text(json.dumps({
        "dataset": "NEU-DET",
        "iou_threshold": args.iou,
        "scorer_features": list(SCORER_FEATURES),
        "scorer_len": SCORER_LEN,
        "split": {"tune_per_class": args.tune_limit,
                  "report_per_class": args.report_limit,
                  "overlap": len(overlap),
                  "note": "labels from TUNE only; all reported numbers from REPORT"},
        "tune_candidates": int(len(y)),
        "tune_positive_rate": round(float(y.mean()), 4),
        "pos_weight": pw_best,
        "feature_weights": {n: round(float(w), 4) for n, w in zip(SCORER_FEATURES, W)},
        "holdout_score_thresh": th,
        "holdout_top_k": sel_top_k,
        "holdout_nms_iou": sel_nms,
        "variant_factors": list(factors),
        "score_thresh_selected_on": "TUNE validation slice (every 4th TUNE image, stride 3)",
        "holdout": rep,
        "oracle_recall_on_report": orc,
        "scorer_fraction_of_oracle": round(captured, 4),
    }, indent=2))
    print(f"\nwrote {args.out}")


def _auc(X, y, scorer):
    """Rank-based AUC: P(positive scored above negative). Chance is 0.5."""
    if len(X) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    p = _raw_prob(X, scorer)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _raw_prob(X, scorer):
    z = ((X - scorer.mu) / scorer.sd) @ scorer.W + scorer.b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _recall_at_budget(prepared, imgs, scorer, pcfg, iou_thresh, budget=3):
    hit = tot = 0
    for i in imgs:
        cls, cands, _s, gts, shape = prepared[i]
        if not cands or not gts:
            continue
        from backend.app.localize import _box_iou_vector
        boxes = np.array([c.as_box() for c in cands], dtype=np.int32)
        top = select(cands, ProposalConfig(**{**pcfg.__dict__, "score_thresh": 0.0}),
                     top_k=budget, scorer=scorer, shape=shape)
        tb = np.array([c.as_box() for c in top], dtype=np.int32) if top else np.zeros((0, 4), np.int32)
        used = set()
        for g in gts:
            tot += 1
            best, bj = 0.0, None
            for j, tbx in enumerate(tb):
                if j in used:
                    continue
                v = float(_box_iou_vector(np.array([tbx], dtype=np.int32), (g.x, g.y, g.w, g.h))[0])
                if v > best:
                    best, bj = v, j
            if best >= iou_thresh and bj is not None:
                used.add(bj)
                hit += 1
    return hit / max(1, tot)


if __name__ == "__main__":
    main()