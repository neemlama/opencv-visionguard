# Evaluation — ForgeSight

Reproduce: `python eval/run_eval.py` (offline, no AWS, no cost).
Raw output: `eval/results.json`. Ground truth: `eval/labels.json`.

The submission rules ask for "evaluation evidence appropriate to the project,
including failure cases or limitations". This is that evidence — including the
part where the harness found a bug in our own detector.

---

## 1. What is being evaluated

Three separate things, because they fail differently:

| evaluation | question |
|---|---|
| **decision accuracy** | does the agent reach the right action on a labelled image? |
| **robustness sweep** | does that hold when the image degrades? |
| **latency** | what does it cost to run, at the tail not just the mean? |

The agent's five possible verdicts collapse to three operational classes,
because "escalate-human" and "human-review" both mean *a person must look at
this*:

| verdict | class |
|---|---|
| `pass` | clean |
| `human-review`, `escalate-human` | defect |
| `re-shoot`, `re-shoot-angle` | unusable |

Evaluation runs against `backend/app/vision.py` + `agent.py` directly — the same
code the HTTP API serves.

---

## 2. Baseline decision accuracy

5 labelled synthetic weld images, 9-condition ground truth.

| sample | expected | observed | defect_score | match |
|---|---|---|---|---|
| `weld_clean` | pass | pass | 0.015 | yes |
| `weld_defect` | human-review | human-review | 0.311 | yes |
| `weld_damage` | escalate-human | escalate-human | 1.000 | yes |
| `weld_blurry` | re-shoot | re-shoot | 0.000 | yes |
| `weld_glare` | re-shoot-angle | re-shoot-angle | 0.000 | yes |

**Exact verdict match 5/5. Triage accuracy 1.00.**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 1 |
| defect | 1.000 | 1.000 | 1.000 | 2 |
| unusable | 1.000 | 1.000 | 1.000 | 2 |

Confusion matrix (rows = truth, cols = observed):

| | clean | defect | unusable |
|---|---|---|---|
| **clean** | 1 | 0 | 0 |
| **defect** | 0 | 2 | 0 |
| **unusable** | 0 | 0 | 2 |

**This table is too small to be impressive and we say so up front.** 5 images
cannot support a precision claim. It demonstrates the decision logic is wired
correctly, nothing more.

### A bug the harness caught in our labels

The first run scored 0.60 accuracy with two "failures" that were actually
*correct* behaviour: `weld_blurry` and `weld_glare` were labelled
`usable_image: true`, so the harness expected a defect judgement while the agent
correctly refused to make one. The labels were wrong, not the detector. Fixed in
`eval/labels.json`.

Worth stating plainly: the first version of our own evaluation set was wrong.

---

## 3. Robustness sweep — where it actually breaks

Every image under 9 perturbations (identity, brightness ±30%, Gaussian noise
σ=10/25, JPEG q20, 2x downscale, 5° rotation, 0.6 contrast). 45 evaluations.

| perturbation | accuracy | failure |
|---|---|---|
| none | 1.00 | — |
| bright +30% | 1.00 | — |
| JPEG q20 | 1.00 | — |
| rotate 5° | 1.00 | — |
| dark −30% | 0.80 | glare image judged `pass` |
| contrast 0.6 | 0.80 | glare image judged `pass` |
| noise σ=10 | 0.80 | blurry image judged `pass` |
| noise σ=25 | 0.80 | blurry image judged `pass` |
| **downscale 0.5x** | **0.40** | **all 5 images rejected as `re-shoot`** |

### The serious one we fixed: a missed defect under low light

Before the fix, `dark −30%` and `contrast 0.6` produced **`weld_defect:
defect → clean (pass)`** — a **false negative on a real defect**. For a safety
inspection that is the worst possible failure, so it was chased down rather than
listed as a caveat.

Diagnosis (`eval/diagnose_fn.py`) showed the dark-blob detector was *not* at
fault — it kept finding the pit in every condition (`dark_blobs=1`
throughout). The failure was upstream in the score blend:

| condition | blobs | edge_density | defect_score | verdict |
|---|---|---|---|---|
| pristine | 1 | 0.00790 | 0.317 | human-review |
| dark −30% | 1 | 0.00250 | **0.296** | **pass** |
| contrast 0.6 | 1 | 0.00250 | **0.296** | **pass** |

`Canny` used **absolute thresholds (50, 150)**, so dimming the image starved it
of edges; `edge_density` fell 3.2x, dragging the blended score from 0.317 to
**0.296 — 0.004 below the 0.3 review cut**. A real defect auto-passed by four
thousandths.

Two fixes, both applied:

1. **Median-proportional Canny thresholds** (`0.66·median`, `1.33·median`, the
   ratio OpenCV recommends when thresholds are unspecified). Makes the edge
   channel illumination-invariant.
2. **A safety floor in the agent**: if `dark_blobs > 0`, the verdict can never
   be `pass` — the floor is `human-review`. Blob count is the stable signal, so
   it is the right thing to gate on. Auto-clearing an anomaly the system itself
   detected is the one outcome a safety inspection must never produce.

After the fix both cases are correct and baseline stays 5/5.

### Rejected alternatives — measured, not assumed

Two plausible fixes were tested and **rejected on evidence**
(`eval/robustness_candidates.py`, 45 conditions each):

| variant | blur correct | glare correct |
|---|---|---|
| **shipped: Laplacian(gray), V > 240** | **84.4%** | **95.6%** |
| pre-blurred Laplacian + Otsu glare | 26.7% | 51.1% |
| pre-blurred Laplacian only | 26.7% | 95.6% |

A 3x3 pre-blur before the Laplacian collapses the blur metric (it lowers the
variance of *sharp* images too, so everything reads as defocused), and Otsu
splits mid-tone images roughly in half, halving glare accuracy. Both would have
been committed on intuition; both were worse.

### Remaining limitations, each with a cause

| limitation | cause | severity |
|---|---|---|
| **2x downscale rejects everything** (0.40) | the blur threshold (Laplacian variance < 100) is **resolution-dependent**; downscaling then upsampling costs high-frequency detail, so every image looks defocused. In a real deployment the threshold must be calibrated per capture resolution. | high — it is an availability failure, not a safety one |
| noise makes a blurry image read as sharp | the Laplacian amplifies high-frequency noise, inflating the variance that the blur test depends on. The obvious fix (pre-blur) was measured and is worse — see above. | medium — degrades to "no action" rather than a wrong pass on the defect sample |
| glare missed at low light | the glare test counts pixels above an **absolute** `V > 240`; dimming removes them. Otsu was tested and was worse. | low — a missed glare prompt, not a missed defect; the defect path is unaffected |

---

## 4. Latency

Local x86, warm cache, 200 iterations after 20 warmup:

| metric | value |
|---|---|
| p50 | ~9 ms |
| p95 | ~13 ms |
| p99 | ~29 ms |
| fps equivalent | ~104 |

The p99 being 3x the p50 is normal Python/GC jitter and is exactly why the
benchmark reports medians rather than means. AWS figures (median of 30, after 5
discarded warmups, on Graviton4) are in `docs/BENCHMARK-RESULTS.md`.

Note the local latency (~9 ms) exceeds the AWS Graviton4 figure (~1.1 ms): this
dev box is a Windows laptop and the AWS number is the measurement that matters.

---

## 5. What this evaluation does not establish

- **No labelled real-weld dataset.** 5 synthetic images on one flat gray plate
  under one lighting model. Real beads vary in colour, texture, spatter and heat
  tint. Nothing here transfers automatically to a real inspection line.
- **Precision/recall are computed on 5 samples.** Reported for completeness;
  treat them as "the logic is wired correctly", not as a performance claim.
- **The defect score is a triage signal, not a probability.** It is a hand-tuned
  blend of edge and blob evidence on synthetic data. It has not been calibrated
  against any ground-truth defect population.
- **No geometric tolerance model.** The system reports "a dark anomaly exists",
  not "this bead violates specification X at dimension Y". Real weld acceptance
  needs width/undercut/porosity measurements against a standard.
- **Sub-400px anomalies are invisible** to the dark-blob detector by design.
- **Single run per configuration** on AWS; no cross-instance variance estimate.
- **Not tested adversarially.** No adversarial images, no tampering, no attempt
  to make the system approve a bad weld deliberately.

---

## 6. Files

| file | purpose |
|---|---|
| `eval/labels.json` | ground truth + perturbation definitions |
| `eval/run_eval.py` | main harness: accuracy, sweep, latency |
| `eval/robustness_rules.py` | compares dark-blob threshold strategies for speed **and** recall |
| `eval/robustness_candidates.py` | compares blur/glare fixes (the rejected ones) |
| `eval/diagnose_fn.py` | isolates why a real defect was auto-passed |
| `eval/results.json` | raw output |
| `bench/detector_variants.py` | speed/recall table behind the detector replacement |
