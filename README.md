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

## Measured results (public data, leakage-free split)

**1800 unique source images** — NEU-CLS and NEU-DET turned out to be the *same*
survey, so both are deduplicated and the split is grouped by source image
(verified zero group overlap). See `docs/EVALUATION.md` §1a.

| metric | value |
|---|---|
| defect classification, 7 classes | **92.75%** accuracy, macro F1 0.927 |
| accuracy on the capture set used at inference | **94.81%** |
| **missed defects** (test split) | **0 of 540** |
| **auto-pass precision** | **1.000** — nothing auto-passed was defective |
| auto-escalate rate | 38.0% of defects closed without a human |
| clean captures released automatically | 54.3% |
| confidence calibration, ECE on test | **0.0338** (temperature fitted on validation) |
| throughput after the feature study | **2.20x faster** than the 16-filter bank, −0.33 accuracy pts |
| COOL vs stock OpenCV, library-only | **0.93x — COOL slower** |
| COOL (Graviton) vs stock OpenCV (x86) | 1.11x — but that is the *hardware* |
| cost per 1000 frames | Graviton $0.000067 vs x86 $0.000043 — **x86 is cheaper** |

Two methodology leaks were found and fixed before any of these numbers were
quoted: the dataset duplication above, and a temperature that had been fitted on
the same split used to report ECE. Both are documented rather than quietly
corrected.

## The COOL finding, stated plainly

**COOL is not faster than stock OpenCV on our workload (0.93x), and it is not
cheaper either.** The operation that dominates our pipeline is a Gabor
`filter2D`, and COOL leaves it at 1.00x. COOL does accelerate `Canny` (1.57x) and
`cvtColor` (1.24x), but those are not in our hot path.

Two claims we tested rather than assumed, and which did **not** hold:

- *"Graviton is cheaper per frame."* It is not, at these instance sizes.
  `t3.medium` costs $0.000043 per 1000 frames vs `c8g.medium` at $0.000067,
  because Graviton's hourly rate is ~42% higher at this size. COOL's licence fee
  is additional and excluded from that figure.
- *"COOL-accelerated ops dominate a vision pipeline."* They dominate the earlier
  Canny-heavy pipeline (1.27x), not this one.

The one favourable comparison is COOL-on-Graviton vs stock-on-x86 at **1.11x** —
but the DIY-on-Graviton arm is 1.19x, so that gap is the **hardware**, not COOL.

We did not rewrite the feature bank to flatter COOL. When a cheaper bank turned
out to be equally accurate we shipped it *because it is better product*, and
published the trade-off. Full tables in `docs/BENCHMARK-RESULTS.md`.

**Open question we have asked the organiser:** our COOL image is a `-v2-` build
bundling OpenCV **4.14**, while the Marketplace listing describes v3.1 as built on
OpenCV **5.0**. Until that is resolved we are treating the main podium and the
Agentic Vision Award as the realistic targets and COOL as a possible bonus. Draft
email: `docs/EMAIL-opencv-org.md`.

## Structure

| path | purpose |
|---|---|
| `backend/app/features.py` | OpenCV 5 descriptor: 8 Gabor, Sobel, blur/anomaly, Laplacian |
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
