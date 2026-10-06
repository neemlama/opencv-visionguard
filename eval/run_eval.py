"""ForgeSight evaluation harness.

Produces the evidence the submission rules ask for: "evaluation evidence
appropriate to the project, including failure cases or limitations".

Three evaluations:

1. Decision accuracy on a labelled set - exact agent verdict plus a 3-way
   triage confusion matrix (defect / clean / unusable-image).
2. Robustness sweep - every image under 9 perturbations. This is where failure
   cases actually show up; a detector that only works on pristine inputs is not
   evidence of anything.
3. Latency distribution - p50/p95, not just a mean.

Runs offline against backend/app/vision.py + agent.py (same code the HTTP API
serves), so no server is required.

    python eval/run_eval.py
    python eval/run_eval.py --out eval/results.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agent import run_agent_loop  # noqa: E402
from backend.app.vision import analyze_image  # noqa: E402

SAMPLES = ROOT / "samples"
LABELS = ROOT / "eval" / "labels.json"

# Agent verdicts mapped to the triage class that actually matters operationally.
# "escalate-human" and "human-review" are both "a human must look at this".
VERDICT_CLASS = {
    "pass": "clean",
    "human-review": "defect",
    "escalate-human": "defect",
    "re-shoot": "unusable",
    "re-shoot-angle": "unusable",
}
CLASSES = ("clean", "defect", "unusable")


def expected_class(label):
    if not label["usable_image"]:
        return "unusable"
    return "defect" if label["expected_defect"] else "clean"


def encode(img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()


def classify(data):
    vision = analyze_image(data)
    trace = run_agent_loop(vision)
    return vision, trace


def apply_perturbation(img, spec):
    kind = spec["kind"]
    v = spec.get("value")
    if kind == "identity":
        return img
    if kind == "brightness":
        return np.clip(img.astype(np.float32) * v, 0, 255).astype(np.uint8)
    if kind == "noise":
        rng = np.random.default_rng(1234)  # fixed: sweep stays reproducible
        return np.clip(img.astype(np.int16) + rng.normal(0, v, img.shape), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        data = encode(img)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if kind == "resize":
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * v), int(h * v)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "rotate":
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), v, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "contrast":
        f = v
        return np.clip((img.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown perturbation {kind}")


def confusion(observed, truth):
    matrix = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for o, t in zip(observed, truth):
        matrix[t][o] += 1
    return matrix


def metrics(matrix):
    out = {}
    for c in CLASSES:
        tp = matrix[c][c]
        fn = sum(matrix[c][k] for k in CLASSES) - tp
        fp = sum(matrix[a][c] for a in CLASSES) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "support": sum(matrix[c].values()),
        }
    total = sum(sum(r.values()) for r in matrix.values())
    correct = sum(matrix[c][c] for c in CLASSES)
    out["accuracy"] = round(correct / total, 4) if total else 0.0
    out["n"] = total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    ap.add_argument("--latency-iters", type=int, default=200)
    args = ap.parse_args()

    spec = json.loads(LABELS.read_text())
    labels = spec["labels"]
    perturbations = spec["perturbations"]

    # ---------- 1. baseline accuracy ----------
    rows = []
    obs_cls, truth_cls = [], []
    for lab in labels:
        path = SAMPLES / lab["file"]
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"missing sample {path} - run python -m backend.app.make_samples")
        vision, trace = classify(encode(img))
        got = trace["final"]
        cls = VERDICT_CLASS.get(got, "unknown")
        exp = expected_class(lab)
        obs_cls.append(cls)
        truth_cls.append(exp)
        rows.append({
            "file": lab["file"],
            "expected_verdict": lab["expected_verdict"],
            "got_verdict": got,
            "verdict_match": got == lab["expected_verdict"],
            "expected_class": exp,
            "got_class": cls,
            "class_match": cls == exp,
            "defect_score": vision["defect_score"],
            "dark_blobs": vision["dark_blobs"],
            "blur_variance": vision["blur_variance"],
            "glare_ratio": vision["glare_ratio"],
            "trace_steps": len(trace["trace"]),
        })

    matrix = confusion(obs_cls, truth_cls)
    base_metrics = metrics(matrix)

    # ---------- 2. perturbation sweep ----------
    sweep = []
    for pspec in perturbations:
        kept, flips, per = 0, [], []
        for lab in labels:
            img = cv2.imread(str(SAMPLES / lab["file"]))
            pert = apply_perturbation(img, pspec)
            try:
                vision, trace = classify(encode(pert))
            except Exception as e:  # a crash is itself a result
                per.append({"file": lab["file"], "error": f"{type(e).__name__}: {e}"})
                continue
            exp_cls = expected_class(lab)
            got_cls = VERDICT_CLASS.get(trace["final"], "unknown")
            ok = got_cls == exp_cls
            kept += 1
            per.append({
                "file": lab["file"],
                "expected_class": exp_cls,
                "got_class": got_cls,
                "ok": ok,
                "verdict": trace["final"],
                "defect_score": vision["defect_score"],
            })
            if not ok:
                flips.append(f"{lab['file']}: {exp_cls}->{got_cls} ({trace['final']})")
        sweep.append({
            "perturbation": pspec["name"],
            "correct": kept - len(flips),
            "total": kept,
            "accuracy": round((kept - len(flips)) / kept, 4) if kept else 0.0,
            "flips": flips,
            "per_image": per,
        })

    # ---------- 3. latency ----------
    img = cv2.imread(str(SAMPLES / "weld_clean.jpg"))
    data = encode(img)
    for _ in range(20):
        classify(data)
    samples = []
    for _ in range(args.latency_iters):
        t0 = time.perf_counter()
        classify(data)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    latency = {
        "n": len(samples),
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[int(len(samples) * 0.95)], 3),
        "p99_ms": round(samples[int(len(samples) * 0.99)], 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "fps_equivalent": round(1000 / statistics.mean(samples), 1),
        "platform": f"{cv2.__version__}",
    }

    results = {
        "opencv_version": cv2.__version__,
        "baseline": {"rows": rows, "confusion": matrix, "metrics": base_metrics},
        "perturbation_sweep": sweep,
        "latency_local": latency,
        "known_limitations": [
            "Thresholds are hand-tuned on 5 synthetic images; no labelled real-weld "
            "dataset, so scores are a triage signal rather than calibrated probability.",
            "Pits below the 400px area floor are invisible to the dark-blob detector.",
            "No geometric tolerance model: the system reports 'a dark blob exists', not "
            "'this weld violates ISO 17663 dimension X'.",
            "The sample set is single-scenario (flat gray plate, one lighting model). "
            "Real weld beads vary in colour, texture and spatter.",
            "Latency figures are single-process, warm-cache, local x86; AWS figures live "
            "in docs/BENCHMARK-RESULTS.md.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # ---------- console report ----------
    print(f"OpenCV {cv2.__version__}\n")
    print("=== 1. baseline decision accuracy ===")
    print(f"{'sample':<18}{'expected':<18}{'got':<18}{'score':>7}  match")
    for r in rows:
        print(f"{r['file']:<18}{r['expected_verdict']:<18}{r['got_verdict']:<18}"
              f"{r['defect_score']:>7}  {'ok' if r['verdict_match'] else 'DIFFERS'}")
    print(f"\nexact verdict match : {sum(r['verdict_match'] for r in rows)}/{len(rows)}")
    print(f"triage accuracy    : {base_metrics['accuracy']}")
    for c in CLASSES:
        m = base_metrics[c]
        print(f"  {c:<9} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")
    print("\nconfusion (rows = truth, cols = observed)")
    print(f"{'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for t in CLASSES:
        print(f"{t:<10}" + "".join(f"{matrix[t][o]:>10}" for o in CLASSES))

    print("\n=== 2. perturbation sweep (failure hunting) ===")
    print(f"{'perturbation':<18}{'acc':>7}  flips")
    for s in sweep:
        flag = "ok" if not s["flips"] else f"{len(s['flips'])} flip(s)"
        print(f"{s['perturbation']:<18}{s['accuracy']:>7}  {flag}")
        for f in s["flips"]:
            print(f"    - {f}")

    print("\n=== 3. latency (local, warm) ===")
    print(f"  p50={latency['p50_ms']}ms  p95={latency['p95_ms']}ms  "
          f"p99={latency['p99_ms']}ms  fps~{latency['fps_equivalent']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()