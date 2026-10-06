"""Calibrate the agent's thresholds on the leakage-free held-out test split.

Uses eval/splits.py (deduplicated, group-aware split) rather than the earlier
per-row split, so these numbers are defensible.

A system that sends everything to a human also misses zero defects, so
sensitivity alone is meaningless. Reported together, as they must be:

  defect_sensitivity     fraction of real defects NOT silently passed
  auto_pass_precision    of images auto-passed, fraction genuinely clean
                         <-- this is what a false accept costs in production
  auto_escalate_rate     defects closed without human re-inspection
  human_review_rate      share of traffic needing a person
  second_opinion_rate    low-confidence cases sent for a second perception

    python eval/triage_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.classifier import DefectClassifier, expected_calibration_error  # noqa: E402
from eval.splits import build, grouped_split  # noqa: E402

CLEAN = "no_defect"
SEVERE = ("inclusion", "rolled-in_scale", "pitted_surface")


def operating_point(y, probs, names, pass_min, review_min, escalate_min):
    n_classes = probs.shape[1]
    clean_idx = n_classes - 1
    is_defect = y != clean_idx
    top = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    severe_idx = [names.index(s) for s in SEVERE if s in names]

    # Mirrors backend/app/agent.py decide()
    escalate = is_defect & (top >= escalate_min) & np.isin(pred, severe_idx)
    review = is_defect & ~escalate & (top >= review_min)
    second = is_defect & ~escalate & ~review
    # A defect is MISSED only when the classifier actively calls it clean with
    # confidence. Low-confidence predictions route to second opinion then review,
    # so they are review traffic, not misses.
    missed = is_defect & (pred == clean_idx) & (top >= pass_min)

    auto_pass = (~is_defect) & (pred == clean_idx) & (top >= pass_min)
    auto_pass_true_clean = auto_pass & (~is_defect)
    clean_reviewed = (~is_defect) & ~auto_pass

    n_def = max(1, int(is_defect.sum()))
    n_clean = max(1, int((~is_defect).sum()))
    return {
        "escalate_min": escalate_min, "review_min": review_min, "pass_min": pass_min,
        "defect_sensitivity": round(1 - float(missed.sum()) / n_def, 4),
        "missed_defects": int(missed.sum()),
        "auto_escalate_rate": round(float(escalate.sum()) / n_def, 4),
        "human_review_rate": round(float(review.sum()) / len(y), 4),
        "second_opinion_rate": round(float(second.sum()) / len(y), 4),
        "auto_pass_rate": round(float(auto_pass.sum()) / len(y), 4),
        "auto_pass_precision": round(float(auto_pass_true_clean.sum())
                                     / max(1, int(auto_pass.sum())), 4),
        "auto_passed_n": int(auto_pass.sum()),
        "clean_reviewed_rate": round(float(clean_reviewed.sum()) / n_clean, 4),
        "false_accepts": int(auto_pass.sum()) - int(auto_pass_true_clean.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "triage_calibration.json"))
    args = ap.parse_args()

    model = DefectClassifier.load()
    names = list(model.class_names)

    X, y, groups, src, n_unique = build(with_background=True)
    tr, va, te = grouped_split(y, groups)
    probs = model.predict_proba(X[te])
    y_te = y[te]

    n_def = int((y_te != len(names) - 1).sum())
    n_clean = int((y_te == len(names) - 1).sum())
    print(f"leakage-free test split: n={len(y_te)}  "
          f"({n_def} defect, {n_clean} clean)  unique source images={n_unique}")
    print(f"model T={model.temperature:.2f}  "
          f"test ECE={expected_calibration_error(probs, y_te):.4f}\n")

    rows = []
    print(f"{'escalate_min':>13} {'sens':>7} {'missed':>7} {'auto-esc':>9} "
          f"{'review':>8} {'2nd-op':>8} {'auto-pass':>10} {'pass-P':>8} {'clean-rev':>10}")
    for em in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        r = operating_point(y_te, probs, names, 0.80, 0.55, em)
        rows.append(r)
        print(f"{em:>13.2f} {r['defect_sensitivity']:>7.4f} {r['missed_defects']:>7} "
              f"{r['auto_escalate_rate'] * 100:>8.1f}% {r['human_review_rate'] * 100:>7.1f}% "
              f"{r['second_opinion_rate'] * 100:>7.1f}% {r['auto_pass_rate'] * 100:>9.1f}% "
              f"{r['auto_pass_precision']:>8.3f} {r['clean_reviewed_rate'] * 100:>9.1f}%")

    safe = [r for r in rows if r["missed_defects"] == 0]
    if safe:
        chosen = max(safe, key=lambda r: r["auto_escalate_rate"])
        print(f"\nchosen: escalate_min={chosen['escalate_min']} "
              f"(auto-escalate {chosen['auto_escalate_rate'] * 100:.1f}%, "
              f"0 missed defects)")
    else:
        best = max(rows, key=lambda r: (r["defect_sensitivity"], r["auto_escalate_rate"]))
        chosen = best
        print(f"\nWARNING: no threshold reaches zero missed defects. "
              f"Best available: escalate_min={chosen['escalate_min']} "
              f"with {chosen['missed_defects']} missed. A real deployment would "
              f"still require human sign-off.")

    print(f"\nAt the shipped setting ESCALATE_MIN=0.85:")
    ship = [r for r in rows if r["escalate_min"] == 0.85][0]
    for k in ("defect_sensitivity", "missed_defects", "auto_escalate_rate",
              "human_review_rate", "second_opinion_rate", "auto_pass_rate",
              "auto_pass_precision", "clean_reviewed_rate"):
        print(f"  {k:<24}{ship[k]}")

    top = probs.max(axis=1)
    print(f"\nconfidence reliability on TEST:")
    for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (top >= lo) & (top < hi)
        if m.sum():
            print(f"  {lo:.2f}-{hi:.2f}  n={int(m.sum()):>4}  "
                  f"stated {top[m].mean():.3f}  actual {(probs[m].argmax(1) == y_te[m]).mean():.3f}")

    Path(args.out).write_text(json.dumps({
        "split": "leakage-free, group-aware, deduplicated",
        "unique_source_images": int(n_unique),
        "test_n": int(len(y_te)), "test_defects": n_def, "test_clean": n_clean,
        "model_temperature": round(model.temperature, 4),
        "test_ece": round(expected_calibration_error(probs, y_te), 4),
        "sweep": rows, "chosen": chosen, "shipped_0.85": ship,
        "severe_classes": list(SEVERE),
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
