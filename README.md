# ForgeSight — Agentic Visual Inspection of Rolled-Steel Surface Defects

OpenCV AI Competition 2026, powered by AWS (deadline Oct 26, 2026 11:45pm PDT).
Targets Overall ($5k/$3k/$2k) + Best Use of COOL ($1k) + Agentic Vision ($1k).
Rules confirm one entry may take one Overall Award **and** both Special Awards.

**Domain: one.** Rolled-steel surface defect inspection. Not welds, PCBs, solar
or pipes — an earlier draft claimed all four and had only built one. Scope was cut
to the single domain we could actually evaluate.

---

## What it does

A capture of a steel surface goes in. The system decides what should happen next,
and sometimes acts on the world and looks again.

1. **Capture quality gate** (OpenCV 5) — sharpness, glare, blown highlights,
   crushed shadows. An unusable frame is *never* classified; it is sent back with
   a specific physical fix.
2. **Defect classification** (OpenCV 5 features + NumPy logistic regression) —
   which of the six NEU defect classes, or clean.
3. **Localisation** (OpenCV 5) — connected components on a local-anomaly map,
   used as a coarse pointer and as an *independent* check on the classifier.
4. **Agentic loop** — the evidence above changes the next action: recapture with a
   named fix, second-opinion re-analysis, escalate to a stop-the-line work order,
   queue human review, or pass.

## Measured results (public data, held-out split)

| metric | value |
|---|---|
| defect classification, 7 classes | **93.80% accuracy**, macro F1 0.929 |
| accuracy on the capture set used at inference | **95.40%** |
| confidence calibration (ECE) | **0.0218** (was 0.0750 before temperature scaling) |
| defect sensitivity at the shipped operating point | **1.0000** — 0 of 1080 defects missed |
| auto-escalation rate | 42.8% of defects closed without human re-inspection |
| clean captures released automatically | 48.1% |
| COOL vs stock OpenCV, production pipeline | **0.97x — parity** (see below) |
| COOL vs stock OpenCV, earlier Canny-heavy pipeline | **1.27x** |

Trained and evaluated on **NEU-CLS + NEU-DET** (3600 real photographs, 6 defect
classes) plus a `no_defect` class built from annotated-background crops. No
synthetic data is used anywhere in the evaluation.

## The COOL finding, stated plainly

On our production pipeline COOL is **not** faster (0.97x). One op —
`gabor_filter2D_x16` — is 82% of runtime and COOL leaves it at exactly 1.00x.
COOL does accelerate `Canny` (1.49x), `resize` (1.33x) and `cvtColor` (1.23x), but
those are under 6% of our workload.

On the earlier Canny-dominated pipeline the same hardware and libraries gave
**1.27x**.

We report both. We did not rewrite the feature bank to make COOL look good — that
would be benchmark-fitting. Full per-op tables in `docs/BENCHMARK-RESULTS.md`.

## Structure

| path | purpose |
|---|---|
| `backend/app/features.py` | OpenCV 5 descriptor: 16 Gabor, Sobel, blur/anomaly, Laplacian |
| `backend/app/classifier.py` | NumPy logistic regression + temperature scaling |
| `backend/app/analysis.py` | perception: quality gate, classification, localisation |
| `backend/app/agent.py` | decision loop, confidence bands, bounded retries |
| `backend/app/capture.py` | capture-settings simulator (fixes change the render) |
| `backend/app/model.npz` | trained weights + fitted temperature |
| `bench/bench.py`, `aws/run_bench.py` | 3-arm COOL benchmark on AWS Graviton4 |
| `eval/` | training, calibration, localisation eval, loop demo |
| `docs/` | evaluation, benchmark, challenges, cost guard |

## Run it

```powershell
cd C:\Users\wecal\Documents\Codes\opencv-visionguard
pip install opencv-python fastapi "uvicorn[standard]" python-multipart python-pillow
python eval\train_classifier.py --include-background   # writes backend/app/model.npz
uvicorn backend.app.main:app --port 8000
# http://localhost:8000  ·  /docs for the API
```

Evidence, no AWS needed:

```powershell
python eval\train_classifier.py --include-background   # accuracy + confusion matrix
python eval\triage_eval.py                            # operating-point calibration
python eval\neu_eval.py --limit 100                   # localisation, IoU per class
python eval\demo_loop.py                              # closed-loop traces
```

AWS benchmark (~`$0.015`, auto-terminating):

```powershell
aws login
python aws\run_bench.py --region us-east-1 `
    --arm "B=ami-0bec8cef5313300ad:c8g.medium" `
    --arm "A=ami-01db31139bc5615d8:c8g.medium:cool"
python bench\show.py
```

## OpenCV 5 compliance

The **application runs OpenCV 5.0.0** on every request (`/health` reports it).

The benchmark arms run **4.14** because COOL's images bundle 4.12/4.14 — the
Marketplace listing calls the product "based on OpenCV 5", which does not match
either image. Pinning both arms to the same version is what makes the A/B delta
attributable to the library instead of a version change. Building OpenCV 5 from
source on Graviton for the control would remove the caveat at the cost of hours of
compile time; it is a documented gap, not an oversight.

## Known limitations

Stated up front, quantified in `docs/EVALUATION.md`:

- **Localisation is weak** — box-level IoU 0.5 macro F1 ~0.19 even after grid
  search. NEU defects are diffuse and often span the frame. The classifier carries
  the decision; localisation is a pointer and a cross-check.
- **`no_defect` recall is 0.728** and rests on only 269 background crops, because
  most NEU frames have annotations covering nearly the whole image. This is the
  weakest part of the model.
- `scratches` ↔ `inclusion` and `inclusion` ↔ `pitted_surface` are the dominant
  confusions — thin linear features.
- No real production line: single flat 200x200 steel crops, no video, no camera
  geometry, no mill tolerance model.
- Human approval is a stub endpoint, not a workflow system.

## Submission checklist

- [x] OpenCV 5 substantive image analysis, live-checked via `/health`
- [x] agent trace where vision evidence changes the next action
- [x] real public dataset, held-out metrics, confusion matrix, calibration
- [x] 3-arm reproducible COOL benchmark on Graviton4
- [x] failure cases and limitations documented (`docs/CHALLENGES.md`)
- [x] closed capture loop (fix changes settings, scene re-rendered)
- [ ] AWS-hosted demo endpoint
- [ ] architecture diagram
- [ ] technical report + <=5 min video
