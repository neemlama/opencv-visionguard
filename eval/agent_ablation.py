"""Agentic Vision ablation: does the loop actually add value over single-pass?

The award requires an agent whose visual evidence changes a later action, and
asks for "evaluation of task effectiveness and failure handling". A single
impressive trace is not evidence. This measures, on the leakage-free test split:

  single-pass   classify once, then decide (what a non-agentic pipeline does)
  agent loop    quality gate, second opinion on low confidence, classifier/
                localiser disagreement check, bounded retries

Reported, because these are the numbers that matter:

  decision accuracy        single-pass vs loop
  second_opinion_fired     how often the loop escalated to a new perception
  decision_changed         of those, how often the second opinion changed the answer
  errors_corrected         first pass wrong -> loop right  (the value of the loop)
  errors_introduced        first pass right -> loop wrong (the cost of the loop)
  gate_caught              unusable captures the quality gate refused to judge

Run twice: on clean captures, and on captures degraded by the simulator. The
loop's value should show up on the second, and it is important to report both
rather than only the flattering one.

    python eval/agent_ablation.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.analysis import analyze  # noqa: E402
from backend.app.capture import DEGRADATIONS, make  # noqa: E402
from backend.app.classifier import DefectClassifier  # noqa: E402
from eval.splits import build, dedupe, enumerate_files, grouped_split  # noqa: E402

CLEAN_CLASSES = ("crazing", "inclusion", "patches", "pitted_surface",
                 "rolled_in_scale", "scratches", "no_defect")
CLEAN = "no_defect"


def single_pass(model, img):
    """One perception, one decision. No gate, no second opinion, no retry."""
    a = analyze(img, model=model)
    pred = a["prediction"]
    if pred["defect_class"] == CLEAN and pred["confidence"] >= 0.80:
        return "pass", pred, a
    if (pred["defect_class"] != CLEAN
            and pred["defect_class"] in ("inclusion", "rolled_in_scale", "pitted_surface")
            and pred["confidence"] >= 0.85):
        return "escalate-human", pred, a
    if pred["confidence"] < 0.55:
        # A non-agentic system has nowhere to send this, so it guesses.
        return "guess-low-confidence", pred, a
    return "human-review", pred, a


def loop_result(model, img):
    from backend.app.agent import decide, enhance_for_reanalysis

    a = analyze(img, model=model)
    d = decide(a)
    second_fired = d["final"] == "needs-second-opinion"
    if second_fired:
        first = d["first_prediction"]
        a2 = analyze(enhance_for_reanalysis(img), model=model)
        d2 = decide(a2)
        # Must return a2's prediction as the SECOND one. Returning
        # d["first_prediction"] twice made first == second and the
        # "decision changed" counter permanently zero.
        return d2["final"], first, a2["prediction"], a, True
    return d["final"], d.get("first_prediction"), None, a, False


def correct(verdict, true_class):
    """Did this verdict put the right thing in front of a human?

    Anything that DEFERS (escalate, review, second-opinion, recapture) counts as
    correct on a real defect and wrong on a clean surface - deferring is only
    safe when there is something to defer TO. An earlier version did not handle
    'needs-second-opinion', so an undecided loop was scored as an error and the
    ablation reported ~30 introduced errors that were a metric artefact.
    """
    is_defect = true_class != CLEAN
    if verdict in ("escalate-human", "human-review", "guess-low-confidence",
                   "re-capture", "needs-second-opinion"):
        return is_defect
    return not is_defect


DEFERRED = ("escalate-human", "human-review", "guess-low-confidence",
            "re-capture", "needs-second-opinion")


def build_clean_images(n_wanted=200, size=200, seed=3):
    """Real CLEAN captures, cropped from NEU-DET regions outside every GT box.

    Without these the ablation is defects-only, which makes every deferral look
    correct and flatters the agent. Clean captures are drawn from the same real
    photographs (annotated background steel) so nothing synthetic enters.
    """
    import xml.etree.ElementTree as ET
    ann = ROOT / "NEU-DET" / "NEU-DET" / "ANNOTATIONS"
    rng = np.random.default_rng(seed)
    out = []
    for xp in sorted(ann.glob("*.xml")):
        if len(out) >= n_wanted:
            break
        try:
            root = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                          int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax")))))
        img = cv2.imread(str(ROOT / "NEU-DET" / "NEU-DET" / "IMAGES" / (xp.stem + ".jpg")))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < size or w < size:
            continue
        got = False
        for _ in range(24):
            x0 = int(rng.integers(0, w - size + 1))
            y0 = int(rng.integers(0, h - size + 1))
            x1, y1 = x0 + size, y0 + size
            if any(not (x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1)
                   for bx0, by0, bx1, by1 in boxes):
                out.append(img[y0:y1, x0:x1])
                got = True
                break
        if got and len(out) >= n_wanted:
            break
    return out


def run(model, images, labels, preset):
    n = len(images)
    sp_ok = lp_ok = 0
    second_fired = 0
    changed = 0
    corrected = 0
    introduced = 0
    gate_blocked = 0
    # False accept = a genuinely clean surface auto-PASSED with no human. This is
    # the direction that matters in a safety system; extra review is merely costly.
    sp_false_accept = 0
    lp_false_accept = 0
    n_clean = sum(1 for x in labels if x == CLEAN)
    sp_verdicts, lp_verdicts = [], []

    for img, lab in zip(images, labels):
        sv, spred, _ = single_pass(model, img)
        lv, first_pred, second_pred, a, fired = loop_result(model, img)
        sp_verdicts.append(sv)
        lp_verdicts.append(lv)

        s_ok = correct(sv, lab)
        l_ok = correct(lv, lab)
        sp_ok += s_ok
        lp_ok += l_ok
        if lab == CLEAN:
            sp_false_accept += (sv == "pass")
            lp_false_accept += (lv == "pass")

        if not a["quality"]["usable"]:
            gate_blocked += 1
        if fired:
            second_fired += 1
            if second_pred and second_pred["defect_class"] != first_pred["defect_class"]:
                changed += 1
            if (not s_ok) and l_ok:
                corrected += 1
            if s_ok and (not l_ok):
                introduced += 1

    return {
        "preset": preset,
        "n": n,
        "single_pass_accuracy": round(sp_ok / n, 4),
        "loop_accuracy": round(lp_ok / n, 4),
        "delta": round((lp_ok - sp_ok) / n, 4),
        "second_opinion_fired": second_fired,
        "second_opinion_rate": round(second_fired / n, 4),
        "decision_changed": changed,
        "errors_corrected": corrected,
        "errors_introduced": introduced,
        "net_impact": corrected - introduced,
        "gate_refused_unusable": gate_blocked,
        "gate_refusal_rate": round(gate_blocked / n, 4),
        "n_clean": int(n_clean),
        "false_accepts_single_pass": int(sp_false_accept),
        "false_accepts_loop": int(lp_false_accept),
        "false_accept_rate_loop": round(lp_false_accept / max(1, n_clean), 4),
        "single_pass_dist": {v: sp_verdicts.count(v) for v in sorted(set(sp_verdicts))},
        "loop_dist": {v: lp_verdicts.count(v) for v in sorted(set(lp_verdicts))},
    }


def main():
    model = DefectClassifier.load()
    X, y, groups, src, n_unique = build(with_background=True)
    tr, va, te = grouped_split(y, groups)

    # Real images for the held-out test rows, re-read from disk.
    items = dedupe(enumerate_files())
    images, labels = [], []
    for item, _n, g in items:
        idx = np.where(groups == g)[0]
        if len(idx) and idx[0] in set(te.tolist()):
            img = cv2.imread(str(item["path"]))
            if img is not None:
                images.append(img)
                labels.append(CLEAN_CLASSES[int(y[idx[0]])])
    print(f"leakage-free test rows with real images: {len(images)} "
          f"(all defect class)")
    print(f"(classifier test rows incl. background crops: {len(te)})")

    clean_imgs = build_clean_images(200)
    images = images + clean_imgs
    labels = labels + [CLEAN] * len(clean_imgs)
    n_def = len(images) - len(clean_imgs)
    print(f"added {len(clean_imgs)} real CLEAN captures from unannotated regions")
    print(f"ablation set: {len(images)} images "
          f"({n_def} defect / {len(clean_imgs)} clean)\n")

    results = [run(model, images, labels, "clean")]
    for preset in ("soft_focus", "under_exposed", "specular_glare",
                   "low_contrast", "sensor_noise"):
        deg = []
        for img in images:
            deg.append(make(img, preset).image)
        results.append(run(model, deg, labels, preset))

    print(f"{'condition':<18}{'n':>5}{'single':>9}{'loop':>8}{'delta':>8}"
          f"{'2nd-op':>8}{'changed':>9}{'fixed':>7}{'broke':>7}{'gate':>7}"
          f"{'false-acc':>11}")
    for r in results:
        print(f"{r['preset']:<18}{r['n']:>5}{r['single_pass_accuracy']:>9.4f}"
              f"{r['loop_accuracy']:>8.4f}{r['delta']:>+8.4f}"
              f"{r['second_opinion_rate']:>8.3f}{r['decision_changed']:>9}"
              f"{r['errors_corrected']:>7}{r['errors_introduced']:>7}"
              f"{r['gate_refusal_rate']:>7.3f}"
              f"{str(r['false_accepts_loop']) + '/' + str(r['n_clean']):>11}")

    agg_fired = sum(r["second_opinion_fired"] for r in results)
    agg_changed = sum(r["decision_changed"] for r in results)
    print(f"\nacross all conditions: second opinion fired {agg_fired} times, "
          f"changed the answer {agg_changed} times")
    for r in results:
        if r["second_opinion_fired"]:
            print(f"  {r['preset']:<18} fired {r['second_opinion_fired']:>4}, "
                  f"changed {r['decision_changed']:>3}, "
                  f"corrected {r['errors_corrected']:>3}, introduced {r['errors_introduced']:>2}")

    Path(ROOT / "eval" / "agent_ablation.json").write_text(json.dumps(
        {"conditions": results,
         "note": "single-pass = one perception then decide; loop = quality gate + "
                 "second opinion on low confidence + disagreement check"},
        indent=2))
    print("\nwrote eval/agent_ablation.json")


if __name__ == "__main__":
    main()
