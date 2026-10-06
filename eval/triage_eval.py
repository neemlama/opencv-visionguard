"""Calibrate the agent's confidence thresholds on the held-out NEU split.

Picking PASS_MIN / REVIEW_MIN / ESCALATE_MIN by feel produces an agent that
sends everything to a human, which is safe but useless - and it hides the real
trade-off. This measures the trade-off instead.

For each candidate ESCALATE_MIN it reports the operating point:

    defect_sensitivity  fraction of real defects NOT silently passed
    auto_escalate       fraction of defects closed without a human re-inspect
    clean_pass_rate     fraction of clean captures released automatically
    human_review_rate   share of traffic that needs a person

Industrial inspection is asymmetric: missing a defect is far worse than an extra
review, so the chosen operating point maximises auto-escalation subject to zero
missed defects on the held-out set.

    python eval/triage_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.classifier import DefectClassifier  # noqa: E402
from backend.app.features import describe, to_gray  # noqa: E402
from eval.train_classifier import (CLS_DIR, DET_ROOT, DET_PREFIXES,  # noqa: E402
                                   PREFIXES, stratified_split)

CLEAN = "no_defect"


def load_holdout(model):
    """Rebuild the same stratified split used for training, then extract features."""
    paths = []
    for ci, p in enumerate(PREFIXES):
        paths += [(p_, ci) for p_ in sorted(CLS_DIR.glob(f"{p}_*.bmp"))]
    for ci, p in enumerate(DET_PREFIXES):
        paths += [(p_, ci) for p_ in sorted((DET_ROOT / "IMAGES").glob(f"{p}_*.jpg"))]
    # Background samples are cheap to regenerate but needed for the clean class.
    from eval.train_classifier import background_samples, N_DEFECT_CLASSES
    X, y = [], []
    for p, ci in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(describe(to_gray(img)))
        y.append(ci)
    bg = background_samples(per_image=2)
    if len(bg):
        X.extend(list(bg))
        y.extend([N_DEFECT_CLASSES] * len(bg))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    tr, te = stratified_split(y, 0.3, seed=0)
    probs = model.predict_proba(X[te])
    return y[te], probs


def operating_point(y, probs, pass_min, review_min, escalate_min, severe):
    n_classes = probs.shape[1]
    is_defect = y != n_classes - 1
    top = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    # Agent logic mirrored from backend/app/agent.decide(). A real defect counts
    # as MISSED only when the classifier actively calls it clean at high
    # confidence. Low-confidence predictions do not silently pass: the agent
    # returns needs-second-opinion, then human review, so they are review traffic,
    # not misses. Counting them as misses (as a first draft here did) overstated
    # the danger enormously - 46 "misses" that were never actually passed.
    escalate = is_defect & (top >= escalate_min) & np.isin(pred, severe)
    review = is_defect & ~escalate & (top >= review_min)
    second_opinion = is_defect & ~escalate & ~review
    missed = is_defect & (pred == n_classes - 1) & (top >= pass_min)
    clean = ~is_defect
    clean_pass = clean & (pred == n_classes - 1) & (top >= pass_min)
    clean_to_review = clean & ~clean_pass

    n_def = max(1, is_defect.sum())
    n_clean = max(1, clean.sum())
    return {
        "escalate_min": escalate_min,
        "pass_min": pass_min,
        "review_min": review_min,
        "auto_escalate_rate": round(float(escalate.sum() / n_def), 4),
        "human_review_rate": round(float(review.sum() / len(y)), 4),
        "second_opinion_rate": round(float(second_opinion.sum() / len(y)), 4),
        "clean_pass_rate": round(float(clean_pass.sum() / n_clean), 4),
        "clean_to_review_rate": round(float(clean_to_review.sum() / n_clean), 4),
        "missed_defects": int(missed.sum()),
        "defect_sensitivity": round(1 - float(missed.sum()) / n_def, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "triage_calibration.json"))
    args = ap.parse_args()

    model = DefectClassifier.load()
    names = list(model.class_names)
    severe = ["inclusion", "rolled-in_scale", "pitted_surface"]
    severe_idx = [names.index(s) for s in severe if s in names]

    y, probs = load_holdout(model)
    print(f"held-out n={len(y)}  classes={names}")
    print(f"defects={int((y != len(names) - 1).sum())}  clean={int((y == len(names) - 1).sum())}\n")

    rows = []
    for escalate_min in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        r = operating_point(y, probs, pass_min=0.80, review_min=0.55,
                            escalate_min=escalate_min, severe=severe_idx)
        rows.append(r)
        print(f"  escalate>={escalate_min:.2f}  auto-escalate={r['auto_escalate_rate'] * 100:5.1f}%  "
              f"review={r['human_review_rate'] * 100:5.1f}%  "
              f"clean-pass={r['clean_pass_rate'] * 100:5.1f}%  "
              f"missed={r['missed_defects']:>3}  sens={r['defect_sensitivity']:.4f}")

    # Choose the most automated operating point that misses nothing held out.
    safe = [r for r in rows if r["missed_defects"] == 0]
    chosen = max(safe, key=lambda r: r["auto_escalate_rate"]) if safe else max(
        rows, key=lambda r: r["defect_sensitivity"])
    print(f"\nchosen: escalate_min={chosen['escalate_min']} "
          f"(auto-escalate {chosen['auto_escalate_rate'] * 100:.1f}%, "
          f"{chosen['missed_defects']} missed defects)")
    if not safe:
        print("WARNING: no threshold missed zero defects on the held-out split. "
              "A real deployment would still route everything to review.")

    # Confidence reliability: is a stated 0.9 actually right 90% of the time?
    print("\nconfidence reliability (predicted vs observed accuracy):")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    top = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y)
    rel = {}
    for lo, hi in bins:
        m = (top >= lo) & (top < hi)
        if m.sum():
            rel[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(m.sum()),
                "stated_mean_confidence": round(float(top[m].mean()), 3),
                "observed_accuracy": round(float(correct[m].mean()), 3),
            }
            print(f"  {lo:.2f}-{hi:.2f}  n={int(m.sum()):>4}  "
                  f"stated {top[m].mean():.3f}  actual {correct[m].mean():.3f}")
    ece = sum(v["n"] / len(y) * abs(v["stated_mean_confidence"] - v["observed_accuracy"])
              for v in rel.values())
    print(f"  expected calibration error: {ece:.4f}")

    Path(args.out).write_text(json.dumps({
        "sweep": rows, "chosen": chosen, "reliability": rel, "ece": round(ece, 4),
        "severe_classes": severe,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
