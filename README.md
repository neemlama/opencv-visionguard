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
| missed defects | **0 of 540** → sensitivity **≥ 99.44%** (95%, rule of three) |
| **auto-pass precision** | **1.000** — nothing auto-passed was defective |
| auto-escalate rate | 38.0% of defects closed without a human |
| clean surfaces released automatically | 54.3% |
| confidence calibration, ECE on test | **0.0338** (temperature fitted on validation) |
| **agent loop vs single-pass** | better in **all 6** capture conditions; **+0.161** on glare, **+0.173** on sensor noise |
| false accepts under sensor noise | **0 of 200** (was 196 before the noise gate) |
| throughput, Gabor bank 16→8 | **1.83x** end-to-end on AWS, −0.33 accuracy pts |
| COOL vs stock OpenCV 5, library-only | **0.94x — COOL slower** |
| price per vCPU, Graviton vs Intel | Graviton $0.03988 vs Intel $0.04462 — **Graviton 11% cheaper** |
| per-core throughput | 322 vs 330 fps/vCPU — within 2.4% |

Three methodology leaks were found and fixed before any of these numbers were
quoted: the dataset duplication, a temperature fitted on the split used to
report ECE, and a localisation grid search run on the same images it was
reported on. One safety failure was found by the ablation (196/200 false accepts
under noise) and fixed. All are documented rather than quietly corrected.

## The COOL finding, stated plainly

**COOL is 6% slower than stock OpenCV 5 on our workload (0.94x), and it does not
help where we need it.** The operation that dominates our pipeline is a Gabor
`filter2D`, and COOL leaves it at exactly 1.00x. COOL *does* accelerate `Canny`
(1.57x) and `cvtColor` (1.24x), but neither is in our hot path.

This is now **version-independent**. We measured two different COOL images:

| COOL image | bundled OpenCV | A/B library-only |
|---|---|---|
| `COOL-Graviton4-v2-AMI` | 4.14.0-pre | 0.93x |
| `Graviton5-COOL-v3` | **5.1.0-dev** | **0.94x** |

Both images point the same way, so this is a property of our workload rather
than of a build.

Three claims we previously made and have now **retracted**:

- *"COOL on Graviton beats stock on x86 (1.11x)."* The DIY-on-Graviton arm was
  1.19x, so that gap was the hardware, not COOL.
- *"Graviton is cheaper per frame."* Not at matched vCPU. Correct rates are
  $0.03988/hr for `c8g.medium` (we had hardcoded $0.072) and `c8g.medium` is
  **1 vCPU**, not 2. What *is* true: Graviton is ~11% cheaper per vCPU.
- *"The feature change is 2.20x."* It did not reproduce. It is **1.83x** on AWS
  and 1.42x locally.

What we did **not** do: rewrite the feature bank to flatter COOL. When a cheaper
bank turned out to be equally accurate we shipped it *because it is better
product*, and published the trade-off either way. Full tables in
`docs/BENCHMARK-RESULTS.md`.

**Open question we have asked the organiser:** whether a COOL benchmark is
eligible when the live API runs pip OpenCV 5 rather than COOL. Draft email:
`docs/EMAIL-opencv-org.md`. Note we have since found a COOL image carrying
**OpenCV 5.1**, which may make the question moot — awaiting their reply.

## Structure

| path | purpose |
|---|---|
| `backend/app/features.py` | OpenCV 5 descriptor: 8 Gabor, Sobel, blur/anomaly, Laplacian |
| `backend/app/classifier.py` | NumPy logistic regression + temperature scaling |
| `backend/app/analysis.py` | perception: quality gate, classification, localisation |
| `backend/app/localize.py` | multi-scale proposal generator, nested box variants, learned proposal scorer |
| `backend/app/agent.py` | decision loop, confidence bands, bounded retries |
| `backend/app/capture.py` | capture-settings simulator (fixes change the render) |
| `backend/app/model.npz` | trained weights + fitted temperature |
| `bench/bench.py`, `aws/run_bench.py` | 3-arm COOL benchmark on AWS Graviton4 |
| `eval/diagnose_proposals.py` | buckets every missed box: wrong size vs never seen |
| `eval/probe_factors.py` | greedy factor selection for the nested box pool |
| `eval/probe_variant_pick.py` | the label-free box-size criteria that failed |
| `eval/localize_wiring_check.py` | old vs new localiser, end-to-end through the agent |
| `eval/verify_report_claims.py` | asserts every number in the report against the artifacts |
| `eval/` | training, calibration, localisation eval, loop demo |
| `docs/` | evaluation, benchmark, challenges, cost guard |

## What makes this agentic rather than a decision tree

The award requires that visual evidence changes a later action. Three mechanisms,
each measured on 740 real captures (`eval/agent_ablation.py`):

| mechanism | evidence |
|---|---|
| **quality gate** refuses to classify an unusable capture, and names the physical fix | loop beats single-pass in all 6 conditions; **+0.161** on specular glare, **+0.173** on sensor noise |
| **second opinion** applies a genuinely different perception (CLAHE local-contrast stretch) when confidence < 0.55 | measured firing 359 times and changing the class 165; **net-negative, so disabled by default** (fires 0 times in the current ablation) |
| **disagreement check** — classifier says clean but localisation finds structure — refuses to auto-pass | independent cross-check on the classifier; now load-bearing — wiring in the rebuilt localiser took its mean `region_count` on defect images from 0.608 to 1.833, and glare false accepts fell **34 → 8** |

**Reported honestly:** the second opinion changed the class 165 times but
**corrected 0 errors and introduced 17**, so it is off unless enabled. The loop's
measured value comes from the quality gate and the disagreement check. We have
**not** isolated the veto's contribution with a dedicated ablation — it is the
mechanism consistent with the measurements, not a proven cause.

### Closed-loop demonstration

`backend/app/capture.py` models **capture settings**, not pixel patches: a fix
changes the camera setting (focus, exposure, polariser, denoise) and the scene is
re-rendered, so the next perception is of a genuinely different frame. An earlier
pixel-patch version could not converge — unsharp masking cannot undo blur, so
the loop requested refocus forever.

```
iter 0: class=scratches confidence=0.404 margin=0.138
   -> classifier not trusted on first look
   -> second opinion: CLAHE local-contrast stretch, then re-classify
iter 1: class=crazing confidence=0.874 margin=0.826
   -> queue human review with region (workorder WO-8cb28f)
```

The UI labels this explicitly as **simulator mode**, and the plain HTTP path
returns `re-capture` with `reason: no_capture_device_attached` rather than
faking a fix.

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

- **Localisation is weak** — box-level IoU 0.5 macro F1 **0.2247** on a disjoint
  hold-out, against an honest same-protocol baseline of 0.2200. We raised the
  proposal generator's oracle recall from 0.346 to **0.546** by adding nested box
  variants, and macro F1 moved **+0.0047**; the scorer captures only 36% of the
  ceiling, so it is the bottleneck again. `crazing` (0.061) and `patches` (0.053)
  are barely above zero and `pitted_surface` regressed 0.278 → 0.250. The
  classifier carries the decision; localisation is a pointer and an independent
  cross-check. It costs 3x latency (`analyze()` 20.5 → 62.3 ms per crop).
- **A negative result worth reading**: every label-free rule we tried for
  choosing a box's size made localisation *worse* (`docs/EVALUATION.md` §4d).
  NEU annotations mark a defect's semantic extent, which follows no energy
  iso-contour, so box size has to be learned from labels.
- **`no_defect` recall is 0.790** and rests on only 269 background crops, because
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
- [x] AWS-hosted demo endpoint — deployed and verified on the COOL Graviton4 AMI,
      `/health` reporting OpenCV 5.0.0 (`docs/DEPLOYMENT.md`). Held on-demand for
      cost; `up` brings it back in ~3 min. **Must be up on submission day.**
- [x] architecture diagram (`docs/architecture.png`, source `docs/architecture-eraser.json`)
- [x] technical report (`docs/REPORT.md`)
- [ ] <=5 min video — script, shot list and pre-flight in `docs/VIDEO-SCRIPT.md`
