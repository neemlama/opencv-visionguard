"""Leakage-free training and honest evaluation of the NEU defect classifier.

This supersedes the previous script. Two leaks were found and fixed:

  1. DUPLICATE LEAKAGE. NEU-CLS and NEU-DET are the same 1800 source photographs
     (eval/check_duplicates.py: 1045 bit-identical cross-set pairs, Cr_1.bmp vs
     crazing_1.jpg). Training on both meant training on each photo twice, and a
     random per-row split could put twins on either side of the boundary. The
     earlier 93.80% was inflated by this.

  2. CALIBRATION LEAKAGE. The temperature was fitted on the same split used to
     report ECE, so the calibration number was optimistic by construction.

Now: dedupe to 1800 unique images (eval/splits.py), split by identity GROUP into
train / val / test, fit weights on train, fit temperature on val, and report on
test only.

    python eval/train_classifier.py --include-background
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.classifier import (META_PATH, MODEL_PATH, DefectClassifier,  # noqa: E402
                                  expected_calibration_error, fit,
                                  fit_temperature, loss)
from backend.app.features import CLASS_NAMES, DESCRIPTOR_LEN  # noqa: E402
from eval.splits import audit, build, grouped_split  # noqa: E402

CLEAN = "no_defect"


def reliability(probs, y, bins=10):
    top = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top > lo) & (top <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "stated": round(float(top[m].mean()), 3),
                         "actual": round(float(correct[m].mean()), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-background", action="store_true")
    ap.add_argument("--test-fraction", type=float, default=0.30)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "eval" / "classifier_metrics.json"))
    args = ap.parse_args()

    print("building leakage-free dataset (dedupe + group split)...")
    t0 = time.time()
    X, y, groups, src, n_unique = build(with_background=args.include_background)
    names = tuple(CLASS_NAMES) + ((CLEAN,) if args.include_background else ())
    n_classes = len(names)
    print(f"  {n_unique} unique source images -> {len(y)} rows "
          f"({int((y < n_classes - 1).sum())} defect, {int((y == n_classes - 1).sum())} clean) "
          f"in {time.time() - t0:.1f}s")
    assert X.shape[1] == DESCRIPTOR_LEN, "descriptor length drifted"

    tr, va, te = grouped_split(y, groups, args.test_fraction, args.val_fraction, args.seed)
    clean_split = audit(y, groups, tr, va, te, "grouped")
    if not clean_split:
        raise SystemExit("group leakage present - refusing to report a number")

    W, b, mu, sd = fit(X[tr], y[tr], n_classes, args.epochs, args.lr, args.l2)
    clf = DefectClassifier(W, b, mu, sd, names)

    Xn = (X - mu) / sd
    print(f"  train loss {loss(Xn[tr], y[tr], W, b):.4f}   "
          f"val loss {loss(Xn[va], y[va], W, b):.4f}   "
          f"test loss {loss(Xn[te], y[te], W, b):.4f}")

    # Temperature fitted on VALIDATION only. Reporting ECE on val would be the
    # same leak as before, just with a different split name.
    ece_test_before = expected_calibration_error(clf.predict_proba(X[te]), y[te])
    temperature, nll_t = fit_temperature(Xn[va], y[va], W, b)
    clf.temperature = temperature
    probs_te = clf.predict_proba(X[te])
    ece_test_after = expected_calibration_error(probs_te, y[te])
    print(f"\n  temperature T={temperature:.2f} fitted on VAL (n={len(va)})")
    print(f"  test ECE  {ece_test_before:.4f} -> {ece_test_after:.4f}  "
          f"(measured on TEST, n={len(te)})")

    pred = probs_te.argmax(axis=1)
    acc = float((pred == y[te]).mean())
    print(f"\nHELD-OUT TEST ACCURACY: {acc * 100:.2f}%   (n={len(te)})")

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t_, p_ in zip(y[te], pred):
        cm[t_, p_] += 1
    print("\nconfusion matrix (rows = true, cols = predicted)")
    print(f"{'':<18}" + "".join(f"{n[:6]:>8}" for n in names))
    per_class = {}
    for i, cname in enumerate(names):
        rec = cm[i, i] / max(1, cm[i].sum())
        prec = cm[i, i] / max(1, cm[:, i].sum())
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cname] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "support": int(cm[i].sum())}
        print(f"{cname:<18}" + "".join(f"{v:>8}" for v in cm[i])
              + f"   R={rec:.3f} P={prec:.3f} F1={f1:.3f}")
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    by_src = {}
    for s in sorted(set(src[te].tolist())):
        m = src[te] == s
        if m.sum():
            by_src[s] = round(float((pred[m] == y[te][m]).mean()), 4)
    print("\ntest accuracy by capture set: "
          + ", ".join(f"{k}={v * 100:.2f}% (n={int((src[te] == k).sum())})"
                      for k, v in by_src.items()))

    errors = sorted(
        ((int(cm[i, j]), names[i], names[j])
         for i in range(n_classes) for j in range(n_classes)
         if i != j and cm[i, j] > 0), reverse=True)
    print("\ntop confusions (true -> predicted, count):")
    for n, a, b in errors[:6]:
        print(f"  {n:>4}  {a} -> {b}")

    rel = reliability(probs_te, y[te])
    print("\nreliability on TEST (stated vs actual):")
    for r in rel:
        if r["n"]:
            print(f"  {r['bin']:<10} n={r['n']:>4}  stated {r['stated']:.3f}  "
                  f"actual {r['actual']:.3f}")

    clf.meta = {
        "dataset": "NEU (deduplicated: CLS and DET are the same survey)",
        "unique_source_images": int(n_unique),
        "rows": int(len(y)),
        "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
        "split": "grouped by source image, stratified by class",
        "split_seed": args.seed,
        "held_out_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "temperature": round(temperature, 4),
        "temperature_fitted_on": "validation split",
        "ece_test_before": round(ece_test_before, 4),
        "ece_test_after": round(ece_test_after, 4),
        "accuracy_by_source": by_src,
        "descriptor_len": int(X.shape[1]),
        "epochs": args.epochs,
    }
    clf.save(MODEL_PATH, META_PATH)
    print(f"\nsaved model -> {MODEL_PATH}   (T={temperature:.2f}, "
          f"test acc {acc * 100:.2f}%)")

    Path(args.out).write_text(json.dumps({
        "per_class": per_class, "confusion": cm.tolist(),
        "held_out_accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "unique_source_images": int(n_unique),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "temperature": round(temperature, 4),
        "ece_test_before": round(ece_test_before, 4),
        "ece_test_after": round(ece_test_after, 4),
        "reliability": rel,
        "top_confusions": [{"count": n, "true": a, "pred": b} for n, a, b in errors[:6]],
        "accuracy_by_source": by_src,
        "config": vars(args),
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
