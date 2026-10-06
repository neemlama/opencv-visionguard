"""Pick a lighting-robust dark-blob rule before changing the app.

run_eval.py found a false negative: weld_defect is scored `pass` at 30% lower
brightness and at 0.6 contrast, because the black-hat contrast threshold is an
absolute 18 grey levels. A real defect being missed is a safety problem, so the
threshold should adapt to the image's own contrast.

Measures each candidate over the same perturbation sweep the eval harness uses,
counting verdict flips. No AWS calls, no cost.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import LABELS, SAMPLES, apply_perturbation, encode  # noqa: E402

SPEC = json.loads(LABELS.read_text())


def dark_blobs(gray, rule):
    bg = cv2.blur(gray, (25, 25))
    tophat = cv2.subtract(bg, gray)
    if rule == "fixed18":
        thr = 18.0
    elif rule == "adaptive3s":
        thr = max(12.0, 3.0 * float(tophat.std()))
    elif rule == "adaptive4s":
        thr = max(10.0, 4.0 * float(tophat.std()))
    elif rule == "clahe_adaptive":
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        bg = cv2.blur(gray, (25, 25))
        tophat = cv2.subtract(bg, gray)
        thr = max(12.0, 3.0 * float(tophat.std()))
    else:
        raise ValueError(rule)
    _, m = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for x in c if cv2.contourArea(x) > 400), thr


# Ground truth blob presence per sample, from the pristine generator.
TRUTH = {"weld_clean.jpg": 0, "weld_defect.jpg": 1, "weld_damage.jpg": 6}


def main():
    rules = ["fixed18", "adaptive3s", "adaptive4s", "clahe_adaptive"]
    print(f"{'perturbation':<16}" + "".join(f"{r:>16}" for r in rules))
    print("-" * (16 + 16 * len(rules)))
    totals = dict.fromkeys(rules, 0)
    detail = {r: [] for r in rules}

    for pspec in SPEC["perturbations"]:
        row = []
        for rule in rules:
            wrong = []
            for lab in SPEC["labels"]:
                f = lab["file"]
                if f not in TRUTH:
                    continue  # blurry/glare carry no blob ground truth
                img = cv2.imread(str(SAMPLES / f))
                pert = apply_perturbation(img, pspec)
                gray = cv2.cvtColor(cv2.resize(pert, (640, 480)), cv2.COLOR_BGR2GRAY)
                n, _ = dark_blobs(gray, rule)
                if n == 0 and TRUTH[f] > 0:
                    wrong.append(f"{f.replace('weld_','')} MISSED")
                elif n > 0 and TRUTH[f] == 0:
                    wrong.append(f"{f.replace('weld_','')} FALSE+")
            totals[rule] += len(wrong)
            detail[rule].extend(f"{pspec['name']}:{w}" for w in wrong)
            row.append(f"{len(wrong)} bad")
        print(f"{pspec['name']:<16}" + "".join(f"{c:>16}" for c in row))

    print("\ntotal detection errors across sweep (lower is better):")
    for r in sorted(rules, key=lambda k: totals[k]):
        print(f"  {r:<16}{totals[r]:>4}   " + "; ".join(detail[r][:6]))


if __name__ == "__main__":
    main()