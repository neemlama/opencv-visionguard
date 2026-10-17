"""Verify every number asserted in docs/REPORT.md against the artifacts on disk.

Reports are only useful if they cannot drift from the code. This reads the JSON
that each claim cites and fails loudly on any mismatch.

    python eval/verify_report_claims.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def load(rel, *keys):
    p = ROOT / rel
    if not p.exists():
        FAILURES.append(f"MISSING FILE {rel}")
        return None
    d = json.loads(p.read_text())
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            FAILURES.append(f"{rel}: key '{k}' not found")
            return None
        d = d[k]
    return d


def check(label, actual, expected, tol=0.0):
    ok = actual is not None and abs(float(actual) - expected) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {label:<52} {actual}")
    if not ok:
        FAILURES.append(f"{label}: report says {expected}, artifact says {actual}")


def check_str(label, actual, expected):
    """Equality for non-numeric claims (config values read back as strings)."""
    ok = actual == expected
    print(f"  {'OK ' if ok else 'FAIL'}  {label:<52} {actual}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected!r}, artifact says {actual!r}")


def main():
    print("C1-C2  classifier")
    acc = load("eval/classifier_metrics.json", "held_out_accuracy")
    check("accuracy", acc, 0.9275, 5e-4)
    check("macro F1", load("eval/classifier_metrics.json", "macro_f1"), 0.9265, 5e-4)
    check("ECE test after scaling", load("eval/classifier_metrics.json", "ece_test_after"), 0.0361, 5e-4)
    src = load("eval/classifier_metrics.json", "accuracy_by_source")
    if isinstance(src, dict):
        det = src.get("neu-det", src.get("DET", src.get("det")))
        check("DET accuracy", det, 0.9481, 5e-4)

    print("C3-C6  triage / calibration")
    tr = load("eval/triage_calibration.json")
    if tr:
        ch = tr.get("chosen", {})
        for key in ("missed_defects", "auto_pass_precision", "auto_pass_rate",
                    "human_review_rate", "auto_escalate_rate", "defect_sensitivity"):
            if key in ch:
                print(f"  --   {key:<52} {ch[key]}")
    ch = tr.get("chosen", {}) if tr else {}
    check("missed defects", ch.get("missed_defects"), 0)
    check("defect sensitivity", ch.get("defect_sensitivity"), 1.0, 1e-9)
    check("human review rate", ch.get("human_review_rate"), 0.4106, 5e-4)
    check("auto-pass rate", ch.get("auto_pass_rate"), 0.0709, 5e-4)
    check("auto-pass precision", ch.get("auto_pass_precision"), 1.0, 1e-9)
    check("test ECE", tr and tr.get("test_ece"), 0.0361, 5e-4)

    print("C7-C8  ablation")
    ab = json.loads((ROOT / "eval" / "agent_ablation.json").read_text())
    cond = {r["preset"]: r for r in ab["conditions"]}
    for preset, single, loop in (("clean", 0.7000, 0.7257),
                                 ("soft_focus", 0.7243, 0.7297),
                                 ("under_exposed", 0.6770, 0.7243),
                                 ("specular_glare", 0.5730, 0.7338),
                                 ("low_contrast", 0.7027, 0.7230),
                                 ("sensor_noise", 0.3108, 0.4838)):
        r = cond.get(preset, {})
        check(f"{preset} single-pass", r.get("single_pass_accuracy"), single, 5e-4)
        check(f"{preset} loop", r.get("loop_accuracy"), loop, 5e-4)
    sn = cond.get("sensor_noise", {})
    check("noise false accepts single-pass", sn.get("false_accepts_single_pass"), 200)
    check("noise false accepts loop", sn.get("false_accepts_loop"), 0)
    sg = cond.get("specular_glare", {})
    check("glare false accepts loop", sg.get("false_accepts_loop"), 8)
    check("glare gate refusal rate", sg.get("gate_refusal_rate"), 0.0081, 5e-4)
    check("soft_focus gate refusal rate", cond["soft_focus"].get("gate_refusal_rate"), 1.0, 1e-9)
    # The release gate for wiring the localiser in: loop must not lose to
    # single-pass in any condition, and false accepts must not rise.
    check("loop wins in all 6 conditions",
          sum(1 for p in cond
              if cond[p]["loop_accuracy"] > cond[p]["single_pass_accuracy"]), 6)

    print("C11  localisation (was unchecked until 2026-10-17)")
    # Localisation was the one headline metric no assertion covered, which is how
    # a number measured on a module that was not wired into the request path sat
    # in the README unchallenged. Both arms of the A/B are asserted here so the
    # comparison in the report cannot drift from the artifacts.
    loc = load("eval/scorer_results.json")
    if loc:
        check("localisation macro F1 (variant arm)", loc["holdout"]["macro_f1"], 0.2247, 5e-4)
        check("localisation precision", loc["holdout"]["precision"], 0.2527, 5e-4)
        check("localisation recall", loc["holdout"]["recall"], 0.1966, 5e-4)
        check("proposal oracle recall", loc["oracle_recall_on_report"], 0.5455, 5e-4)
        check_str("variant factors",
                  ",".join(str(v) for v in loc["variant_factors"]), "0.85,2.1")
        check("operating top_k", loc["holdout_top_k"], 2)
        check("operating nms_iou", loc["holdout_nms_iou"], 0.3, 1e-9)
        check("scorer feature count", loc["scorer_len"], 18)
    base = load("eval/scorer_results_BASELINE_swept.json")
    if base:
        check("localisation macro F1 (baseline arm)", base["holdout"]["macro_f1"], 0.2200, 5e-4)
        check("baseline oracle recall", base["oracle_recall_on_report"], 0.3455, 5e-4)
    diag = load("eval/proposal_diagnosis.json", "totals")
    if isinstance(diag, dict):
        n = max(1, diag.get("gt", 1))
        check("misses that were refinement (not detection)",
              round(diag.get("refine", 0) / n, 4), 0.6229, 5e-4)

    print("C12  wiring: is the measured localiser the one that runs?")
    # Guards the exact regression found on 2026-10-17: a module measured and
    # reported but never imported by the request path.
    try:
        import inspect

        from backend.app import analysis as _an
        src = inspect.getsource(_an)
        check("analysis.py imports the measured localiser",
              1 if "from .localize import" in src else 0, 1)
        check("analysis.py still defines the old single-threshold body",
              1 if "connectedComponentsWithStats" in src else 0, 0)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"wiring check could not run: {exc}")

    print("C9  COOL A/B")
    bench = load("bench/results/comparison-us-east-1.json")
    if bench:
        arms = bench  # list of arms
        def fps(tag):
            for a in arms:
                lbl = str(a.get("label", a.get("arm", "")))
                if lbl.strip().upper().startswith(tag):
                    return a.get("fps")
            return None
        check("COOL fps (A)", fps("A"), 302.60, 0.01)
        check("DIY  fps (B)", fps("B"), 321.79, 0.01)
        check("A/B ratio", round(fps("A") / fps("B"), 4), 0.9406, 1e-3)

    print("C10  pricing / vCPU")
    pr = load("bench/instance_pricing.json")
    if isinstance(pr, dict):
        for k, want in (("c8g.medium", 0.03988), ("c8g.large", 0.07976),
                        ("c7i.large", 0.08925)):
            check(f"{k} usd/hr", pr["prices"].get(k), want, 5e-5)
        check("c8g.medium vcpu", pr["specs"]["c8g.medium"].get("vcpu"), 1)
        check("c8g.large vcpu", pr["specs"]["c8g.large"].get("vcpu"), 2)
        check("c7i.large vcpu", pr["specs"]["c7i.large"].get("vcpu"), 2)

    print("recapture check: does the loop actually converge on soft focus?")
    from eval.demo_loop import run_loop  # noqa: E402
    from pathlib import Path as P
    img_path = ROOT / "data" / "NEU-DET" / "NEU-DET" / "IMAGES" / "inclusion_1.jpg"
    if not img_path.exists():
        cands = list(ROOT.glob("**/IMAGES/inclusion_1.jpg"))
        img_path = cands[0] if cands else None
    if img_path:
        import cv2
        res = run_loop(scene=cv2.imread(str(img_path)), preset="soft_focus")
        check("soft_focus recapture iterations > 1", 1 if res["iterations"] > 1 else 0, 1)
        check("soft_focus final is not 'capture-unusable'",
              0 if res["final"] == "capture-unusable" else 1, 1)
    else:
        print("  --   inclusion_1.jpg not found; skipping runtime check")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} MISMATCH(ES):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all report claims match the artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())