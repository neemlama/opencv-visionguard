# ForgeSight — Agentic Visual Inspection of Rolled-Steel Surface Defects

**OpenCV AI Competition 2026, powered by AWS** · deadline Oct 26 2026
Team ForgeSight · domain: rolled-steel surface defect inspection

---

## 0. Claim → evidence map

Every number in this report traces to a file. Nothing is asserted without a
command that reproduces it.

| # | Claim | Value | Evidence |
|---|---|---|---|
| C1 | Defect classification, 7 classes | **92.75%**, macro F1 0.927 | `eval/classifier_metrics.json` |
| C2 | Accuracy on the capture set used at inference | **94.81%** | same, `accuracy_by_source` |
| C3 | Missed defects on test | **0 of 540** | `eval/triage_calibration.json` |
| C4 | Defect sensitivity (95%, rule of three) | **≥ 99.44%** | derived from C3 |
| C5 | Auto-pass precision | **1.000** | `eval/triage_calibration.json` |
| C6 | Confidence calibration, ECE on test | **0.0361** | same, `test_ece` |
| C7 | Agent loop beats single-pass | 5 of 6 conditions, +0.005 to +0.068 | `eval/agent_ablation.json` |
| C8 | False accepts under sensor noise | **0 of 200** (was 200/200) | `eval/agent_ablation.json` |
| C9 | COOL vs stock OpenCV 5, library-only | **0.94x — COOL slower** | `bench/results/comparison-us-east-1.json` |
| C10 | Graviton price per vCPU vs Intel | **11% cheaper** | `bench/instance_pricing.json` |
| C11 | Gabor bank 16→8 speedup | **1.83x** end-to-end on AWS, −0.33 acc pts | `docs/BENCHMARK-RESULTS.md` §2 |
| C12 | Localisation macro F1 at IoU 0.5 | **~0.19 — weak** | `eval/neu_results.json` |
| C13 | Live endpoint | verified on COOL AMI | `docs/DEPLOYMENT.md` |

---

## 1. The limitation we want you to see first

**Every robustness number in this report comes from simulated capture
degradations applied to NEU photographs. None of it comes from a real camera on
a real production line.**

We have no physical inspection rig. The `CaptureSimulator`
(`backend/app/capture.py`) holds a scene plus capture settings — focus, exposure,
polariser, denoise, contrast — and applies them to real NEU images. That is a
faithful model of *what a camera setting does*, and it lets the agent genuinely
act and re-perceive. It is **not** evidence that our thresholds work on your line.

Three concrete consequences:

1. The simulator is **unfaithful in at least one measurable way**. Our synthetic
   specular-glare preset produces a highlight *less* extreme than natural NEU
   captures contain (`glare_ratio` 0.388 synthetic versus 0.561 at the real p99
   and 0.754 at the maximum). Real specular glare on scaled steel is therefore
   **not** represented, which is why we do not claim glare detection (§5.3).
2. Sensor noise is additive Gaussian. Real sensor noise is shot-noise-like and
   pipeline-dependent, so our `NOISE_MAX = 10.8` threshold is unverified in the
   field.
3. NEU images are 200x200 flat crops. There is no camera geometry, no motion, no
   depth, no illumination gradient across a real coil surface.

The classifier results (C1–C6) rest on real labelled photographs and are
unaffected by this. The *agentic* results (C7–C8) are simulator-derived and should
be read as evidence of logic, not of field performance.

---

## 2. Problem and scope

Rolled-steel surface inspection is a high-volume, low-margin, safety-relevant
task. Operators inspect thousands of square metres of strip per shift for six
defect classes; misses carry rework cost and, for inclusion-type defects,
integrity risk. Human inspection is slow, inconsistent and tiring, and the
defects are visually subtle at 200x200.

**One domain, deliberately.** An earlier draft of this project claimed welds,
PCBs, solar panels and pipes. We cut that to a single domain because we had
evidence for exactly one. Claims we cannot evaluate are not claims we make.

We chose to keep the system **classical CV plus a small learned classifier**
rather than a deep detector, for three reasons: the dataset is small (1800
unique images), the failure mode we care about most is a *false accept*, and an
end-to-end pipeline whose every stage is inspectable is easier to argue about.

### What we deliberately did not build

- No video or streaming. Static captures only.
- No geometric tolerance model. The system says "inclusion, here is the region" —
  it does not check the defect against a mill specification.
- No LLM planner (§5.4).
- No mill integration, no historian, no MES write-back.

---

## 3. Architecture

```
                      ┌──────────────────────────────────────────────┐
   capture (JPEG/BMP) │  CAPTURE QUALITY GATE            [OpenCV 5]   │
   ──────────────────▶│  sharpness · noise_sigma · saturation        │
                      │  unusable ⇒ REFUSE to classify               │
                      └───────────────┬──────────────────────────────┘
                                      │ usable
                      ┌───────────────▼──────────────────────────────┐
                      │  FEATURE EXTRACTION             [OpenCV 5]   │
                      │  8× Gabor filter2D (6σ+1 kernels)            │
                      │  2× Sobel · magnitude · orientation hist      │
                      │  3× box blur + absdiff (multi-scale anomaly)  │
                      │  Laplacian roughness · 4×4 intensity profile │
                      └───────────────┬──────────────────────────────┘
                                      │ 69-dim descriptor
                      ┌───────────────▼──────────────────────────────┐
                      │  CLASSIFIER  [NumPy logistic regression]     │
                      │  temperature T=0.75 (fitted on validation)   │
                      └───────────────┬──────────────────────────────┘
                                      │ class + confidence + margin
                      ┌───────────────▼──────────────────────────────┐
                      │  LOCALISATION                [OpenCV 5]      │
                      │  anomaly map → threshold → morphology →        │
                      │  connected components → boxes                │
                      └───────────────┬──────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────┐
   │  AGENT                                                     │
   │  quality failed  → name physical fix → (simulator) re-render │
   │                   → perceive again                         │
   │  disagree        → classifier says clean but localiser found │
   │                     structure ⇒ refuse to auto-pass          │
   │  conf ≥ 0.85 & severe class → STOP-THE-LINE work order      │
   │  else            → human review                              │
   │  clean & high conf & no regions → pass                       │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
                     /approve/{id}  —  human signs off
```

Deployment: single `c8g.medium` (AWS Graviton4) launched from the COOL
Marketplace AMI, FastAPI + nginx under systemd, driven over AWS Systems Manager
(no inbound SSH), IMDSv2 required. See `docs/DEPLOYMENT.md`.

---

## 4. Data and vision pipeline

### 4.1 The dataset, and the duplicate that invalidated our first result

We trained initially on "NEU-CLS + NEU-DET", believing they were separate capture
sets — 3600 images. A perceptual-hash pass (`eval/check_duplicates.py`) showed
they are **the same 1800 source photographs**:

| evidence | count |
|---|---|
| cross-set pairs, pHash == 0 **and** dHash == 0 (bit-identical) | **1045** |
| same-label cross-set pairs within perceptual threshold 6 | 7701 |

The file numbering is explicit: `Cr_1.bmp` ↔ `crazing_1.jpg`,
`Pa_101.bmp` ↔ `patches_101.jpg`. A random per-row split could place twins on
either side of the boundary, so the model was partly scored on memorised images.

Fix (`eval/splits.py`): deduplicate to **1800 unique images**, keeping the DET
copy because inference runs on DET captures, then split **by identity group**.

```
[grouped] train=1138  val=310  test=621
group overlap: {'train&val': 0, 'train&test': 0, 'val&test': 0}  → CLEAN
```

**Accuracy moved 93.80% → 92.75%.** The leak was worth 0.3–0.7 points, not
catastrophically — but 92.75% is a number we can defend.

A methodological note: 13389 "cross-label" pairs also fell inside the hash
threshold. Those are pHash **collisions**, not contradictory labels — a 64-bit
8x8 DCT hash is not discriminative at threshold 6. We therefore key identity on
(class, source index) and use hashing only to *verify*, which is the robust
order.

### 4.2 The `no_defect` class

NEU contains only defective images, so a triage system that must sometimes say
"pass" has no clean class to learn from. We build one from regions **outside**
every annotation box — annotated background steel. That yields 269 crops.

Known weakness: annotation boxes are imperfect, so some "clean" crops contain
unannotated defect. `no_defect` has the **lowest recall of any class (0.790)**
and the fewest training samples; we treat it as the weakest part of the model.

We checked for a shortcut (crop geometry or flat texture giving away "clean")
and found no evidence — recall would be high if one existed. We have **not**
proven its absence, because the dataset contains no genuinely defect-free steel.

### 4.3 Pipeline and a feature bug that cost us accuracy

The descriptor is 8 Gabor convolutions plus gradients, multi-scale anomaly
energy and roughness (69 dims).

Early on, `cv2.getGaborKernel((0,0), ...)` — asking OpenCV to auto-size the
kernel — returned a **1x1 kernel for 12 of our 16 orientations**. Twelve
features were duplicates of mean intensity and carried no orientation
information. Computing the size explicitly as `6·σ+1` (verified bit-identical to
the auto-sized kernel on 5.0.0) improved accuracy from 93.02% to 93.80% on the
then-current split. The same `(0,0)` also tries to allocate 1.8e19 bytes on the
4.14 control build.

### 4.4 Localisation — the weak component, stated plainly

Box-level detection against ground truth, IoU 0.5, after a 180-config grid
search, reaches **macro F1 ≈ 0.19**.

NEU defects are diffuse: `crazing_1` is annotated 2,2–193,194, i.e. almost the
whole frame. A merged connected-component box cannot reach IoU 0.5 against such
an annotation, and `rolled-in_scale` has no box annotation in the files we
parsed.

We therefore **do not use localisation to decide**. It supplies a coarse pointer
and — more usefully — an independent signal: when the classifier says clean but
the localiser finds structure, the agent refuses to auto-pass. A weak second
opinion is still useful as a veto.

---

## 5. The agent loop, and the ablation

### 5.1 Why it qualifies as agentic

The award requires image results to influence a later plan, tool call, action or
human-approval request, and warns that "a chatbot that only explains a fixed
vision result is not enough". Ours:

1. vision metrics decide whether the frame is judgeable at all;
2. if not, the agent names the **physical fix** and, in simulator mode, applies
   it by changing capture settings and **re-rendering the scene**;
3. perception runs again on a genuinely different frame;
4. confidence and severity choose pass / review / stop-the-line, always with
   human approval.

An earlier simulator applied pixel patches (an unsharp mask to undo blur). That
**could not converge** — sharpening cannot recreate detail the blur destroyed, so
the loop requested refocus indefinitely. Rebuilding it as a settings model fixed
convergence in one step.

### 5.2 Ablation: does the loop earn its place?

740 real captures — 540 defect from the leakage-free test split plus 200 genuinely
clean crops. Single-pass = one perception then decide.

| condition | single-pass | loop | delta | gate refusals | false accepts (of 200 clean) |
|---|---|---|---|---|---|
| clean | 0.7000 | 0.7176 | **+0.018** | 0.013 | 2 |
| soft focus | 0.7243 | 0.7297 | **+0.005** | 1.000 | 0 |
| under-exposed | 0.6770 | 0.7068 | **+0.030** | 0.054 | 1 |
| specular glare | 0.5730 | 0.6405 | **+0.068** | 0.008 | **34** |
| low contrast | 0.7027 | 0.7162 | **+0.014** | 0.081 | 3 |
| sensor noise | 0.3108 | 0.2892 | **−0.022** | 0.543 | 0 |

The loop helps in five of six conditions, most where a capture is not fit to
judge. It is **slightly harmful under sensor noise** (−0.022) because the gate
refuses 54% of frames including good ones; false accepts stay at 0/200, so the
error is over-caution, not unsafety.

### 5.2a Which degradations the loop can actually fix

Gate refusal rate per condition tells us which failures the agent detects at all:

| condition | gate refusal rate | recaptures? |
|---|---|---|
| soft focus | **1.000** | **yes — converges in one step** |
| sensor noise | 0.543 | yes |
| low contrast | 0.081 | no |
| under-exposed | **0.054** | **no** |
| specular glare | 0.008 | no |

**Soft focus is the only condition where the loop genuinely detects a bad
capture, acts, and converges** (`sharpness` collapses from ~1250 to 2.0; the
agent requests `steady_and_refocus`; the simulator re-renders at `focus_blur=0`;
the classifier then returns the correct class). It leads the demo for that
reason, and it is the whole agentic claim in one trace.

**Under-exposure is a gap we did not expect.** The gate measures sharpness,
noise and saturation but has **no brightness or exposure check**, so it refuses
only 5.4% of under-exposed frames. In the demo, an under-exposed `patches`
image is read as `no_defect` at 62% confidence and queued for review rather than
recaptured. The +0.030 under exposure comes from confidence deferral, not from
the agent acting. Fixing this means adding an exposure metric (mean luminance
against a calibrated target) — we did not have time to add and validate one, so
we list it as an open defect rather than quietly omitting it.

We deliberately did not add a brightness threshold on the fly: a hand-picked
luminance cut-off would have looked like progress in the demo while adding
another threshold fitted on the same data that already misled us once (§5.3).

### 5.3 What we retracted from our own results

An earlier run of this table showed **+0.157 on specular glare**. That was an
artefact of a **miscalibrated gate**: the glare limit had been derived from the
**first 300 NEU files**, which are dominated by one or two classes. Across all
1800 legitimate captures `glare_ratio` reaches **0.75** (p99 = 0.561), while our
synthetic glare preset only reaches **0.388**. Any limit low enough to catch the
synthetic glare also refuses ~14% of good captures.

Recalibrated permissively, the result falls to **+0.068**. A
`largest-bright-region` feature was tried as a better discriminator and failed:
clean p99 = 0.324 versus glare p05 = 0.282. Rough, scaled steel has genuinely
large saturated regions, so specular glare is not separable from legitimate
bright surface using intensity statistics alone.

Consequences we accept:

- **Glare detection is not claimed.** It is disabled by default.
- Under the glare preset the gate does not refuse, and **34 of 200 clean
  captures are auto-passed**. That is the safety cost, and the clearest argument
  for building a real glare feature.
- Under noise the loop over-refuses.

### 5.4 No LLM planner, and a disabled second opinion

We use **no LLM reasoning chain**. "Agentic" here means a gated loop in which
vision metrics select the next action. We think that is more defensible than an
unmeasured chain, and we would rather be judged on what we measured.

We also **disabled the second-opinion step** (CLAHE local-contrast stretch and
re-classify) on evidence: across all conditions it fired **359 times**, changed
the predicted class **165 times**, and **corrected 0 errors while introducing
17**. It is net-negative. It remains behind `ENABLE_SECOND_OPINION = False`
because the negative result is part of this submission.

### 5.5 A safety bug the ablation found

The first ablation run showed the loop **auto-passed 200 of 200 clean captures
under sensor noise**. Additive noise *raises* Laplacian variance, so noisy frames
measured as *sharper* than clean ones and passed every check; the classifier then
confidently answered "no defect" on noise texture.

Fixed with a robust noise estimate — MAD of the 3x3 high-pass response,
median-based so a real defect cannot inflate it — and a threshold from 240 real
captures: clean p99 = 10.286, noisy p05 = 11.305, so `NOISE_MAX = 10.8`.
**False accepts: 200/200 → 0/200.**

This is the strongest argument for having built the ablation: a metric written for
the award found a defect that would otherwise have shipped.

---

## 6. COOL on AWS Graviton

### 6.1 Method

Three questions need three different comparisons, and conflating them is the
usual way these benchmarks go wrong.

| arm | instance | vCPU | burstable | library | OpenCV | fps |
|---|---|---|---|---|---|---|
| A | `c8g.medium` | 1 | no | **COOL** | 5.1.0-dev | 302.60 |
| B | `c8g.medium` | 1 | no | DIY pip | 5.0.0 | 321.79 |
| D | `c8g.large` | 2 | no | DIY pip | 5.0.0 | 330.43 |
| C | `c7i.large` | 2 | no | DIY pip | 5.0.0 | 659.02 |

All arms run **OpenCV 5**. COOL comes from Marketplace AMI
`ami-033e481a24f94c8cb`, verified on-instance as `libopencv_*.so.5.1.0`. The
control is `opencv-python==5.0.0.93` — same major, closest stable release.
Timing is the median of 30 iterations after 5 discarded warmups.

### 6.2 The library question

**A/B = 0.94x. COOL is 6% slower than stock OpenCV on our workload.**

This is version-independent. Two different COOL images agree:

| COOL image | bundled OpenCV | A/B |
|---|---|---|
| `COOL-Graviton4-v2-AMI` | 4.14.0-pre | 0.93x |
| `Graviton5-COOL-v3` | **5.1.0-dev** | **0.94x** |

Per-op, COOL's wins are real but not in our hot path: `canny_legacy` 1.57x,
`cvtColor` 1.24x, `connected_components` ≈ 1.0x. The dominant operation — Gabor
`filter2D` — is **1.00x**. `adaptiveThreshold` remains a COOL regression
(0.57x), which we found by measurement and reported rather than worked around.

**Conclusion:** COOL's benefit is determined entirely by which operations a
workload uses, and the operation a Gabor-texture classifier depends on is not
among them.

### 6.3 The hardware question, with corrections

Two claims we made earlier were **wrong**:

- **Our price was wrong.** We hardcoded `c8g.medium` at $0.072/hr. The AWS Price
  List API says **$0.03988/hr**. Every cost figure we had published was ~1.8x
  too high.
- **Our vCPU matching was wrong.** `c8g.medium` is **1 vCPU**; we had compared
  it against `t3.medium`, which is **2 vCPU and burstable**. A short benchmark can
  run on burst credits that production cannot sustain.

Corrected, vCPU-matched, both non-burstable:

| | fps | fps/vCPU | USD/hr | USD/hr/vCPU |
|---|---|---|---|---|
| `c8g.large` (2 vCPU) | 330.43 | 165 | 0.07976 | 0.0399 |
| `c7i.large` (2 vCPU) | 659.02 | 330 | 0.08925 | 0.0446 |

At matched vCPU, Intel delivers ~2x the absolute throughput. **Per core the two
are within 2.4%** (322 vs 330 fps/vCPU) and **Graviton is ~11% cheaper per
vCPU**. So Graviton wins narrowly on price-per-core, not on speed-per-core. We
report both framings because either alone would mislead.

### 6.4 Cost

`USD / 1000 frames = (usd_per_hour / 3600) × (1000 / fps)`

| arm | USD/1000 frames |
|---|---|
| B, DIY Graviton 1 vCPU | 3.4e-05 |
| C, DIY Intel 2 vCPU | 3.8e-05 |
| A, COOL Graviton 1 vCPU | 3.7e-05 |

COOL's Marketplace licence fee is **not** included in arm A, so A's true cost is
higher. Region us-east-1, on-demand, no Savings Plans or Spot.

### 6.5 An accuracy-versus-latency note

Halving the Gabor bank (16 → 8) is **1.83x** end-to-end on AWS for **−0.33
accuracy points** on the 7-class task, with triage behaviour unchanged. We adopted
it because it is better product throughput — **not** to flatter COOL, which
remains 0.94x on it.

We earlier reported 2.20x; that did not reproduce (1.42x locally) and is
retracted.

---

## 7. Failure cases and limitations

The full log is `docs/CHALLENGES.md` (C-01 … C-17). The ones a judge should know:

| id | failure | how we caught it | status |
|---|---|---|---|
| C-01 | detector blind to dark pits (false negatives) | traced to edge-only features | fixed |
| C-05 | COOL 0.76x *slower* on the first valid run | per-op profiling | root-caused |
| C-08 | **dataset duplicated across two releases** | perceptual hash + filenames | fixed, −0.72 pts |
| C-09 | calibration temperature fitted on the reported split | review | fixed |
| C-10 | `getGaborKernel((0,0))` silently returned 1x1 kernels | inspecting kernel shapes | fixed, +0.78 pts |
| C-14 | **auto-passed 200/200 clean captures under noise** | writing the ablation | fixed, → 0/200 |
| C-15 | wrong AWS price, wrong vCPU matching | Price List API | corrected |
| C-16 | 2.20x speedup did not reproduce | re-measurement | retracted |

### Open limitations

1. **Simulator-derived robustness** (§1). The largest caveat in the project.
2. **Localisation macro F1 ≈ 0.19.** Not used for decisions.
3. **`no_defect` recall 0.790** from 269 crops; crops inherit annotation noise.
4. **Glare detection does not work** and is disabled; 34/200 false accepts
   under the glare preset.
5. **The gate has no exposure check**, so only 5.4% of under-exposed captures are
   refused and exposure never triggers a recapture (§5.2a).
6. **Confusions:** `scratches ↔ inclusion` and `inclusion ↔ pitted_surface` —
   thin, low-contrast linear features.
7. **Splits are by image.** NEU-CLS and NEU-DET are the same survey, so
   near-duplicate frames cannot straddle our split — but we have not verified
   that *within* one set there are no capture-session duplicates. A
   session-aware split would be the honest next step.
8. **Single training run.** No seed variance, no cross-validation.
9. **No video, no camera geometry, no mill tolerance model.**
10. **Human approval is a stub endpoint**, not a workflow system.

---

## 8. Reproducibility

```powershell
aws login

# 1. Datasets are NOT redistributed. See docs/DATASETS.md.
#    NEU-CLS/NEU-CLS/*.bmp  and  NEU-DET/NEU-DET/{IMAGES,ANNOTATIONS}

# 2. Confirm the leakage-free dataset and train (writes backend/app/model.npz)
python eval\check_duplicates.py
python eval\splits.py
python eval\train_classifier.py --include-background

# 3. Evidence, all offline and free
python eval\verify_report_claims.py      # checks every number in THIS report
python eval\triage_eval.py               # operating point + calibration
python eval\neu_eval.py --limit 100      # localisation IoU
python eval\agent_ablation.py            # loop vs single-pass
python eval\quality_gate_calibration.py  # gate thresholds
python eval\feature_variants.py          # accuracy vs latency
python eval\demo_loop.py --image patches_1.jpg --preset specular_glare

# 4. AWS benchmark (~$0.02, auto-terminating)
python aws\instance_pricing.py
python aws\run_bench.py --region us-east-1 --opencv-version 5.0.0.93 `
    --arm "A=ami-033e481a24f94c8cb:c8g.medium:cool@0.03988" `
    --arm "B=ami-0bec8cef5313300ad:c8g.medium@0.03988" `
    --arm "D=ami-0bec8cef5313300ad:c8g.large@0.07976" `
    --arm "C=ami-0045d7fc2ad003464:c7i.large@0.08925"

# 5. Live endpoint (COOL AMI, Graviton4)
python aws\deploy_endpoint.py up --ttl-hours 4
python aws\deploy_endpoint.py status
python aws\deploy_endpoint.py down
```

### Expected cost

| activity | cost |
|---|---|
| all evaluation (offline) | $0 |
| one A+B+D+C benchmark run | ~$0.05 |
| endpoint up for 4 hours | $0.16 |
| endpoint across the full judging window (14 d) | ~$13 |

A `ForgeSight Hackathon Cap` budget of $25/month with alarms at 50% / 80% / 100%
actual and 100% forecast, plus a launch-time pre-flight that refuses to start an
instance whose projected cost would breach the cap. See `docs/COST-GUARD.md`.

### AMI and instance references

| purpose | identifier |
|---|---|
| COOL, OpenCV 5.1.0-dev | `ami-033e481a24f94c8cb` |
| COOL, OpenCV 4.14 (earlier runs) | `ami-01db31139bc5615d8` |
| Ubuntu 24.04 arm64 control | `ami-0bec8cef5313300ad` |
| Ubuntu 24.04 amd64 control | `ami-0045d7fc2ad003464` |
| instances | `c8g.medium`, `c8g.large`, `c7i.large` (us-east-1) |

---

## 9. Responsible use

- No personal data; no faces, no identifiers. Industrial surface imagery only.
- Datasets are public research data and are **not redistributed** with this
  submission; see `docs/DATASETS.md` for provenance and attribution.
- **No escalation can close without a human.** `POST /approve/{id}` is the only
  path to a closed work order, and it records `decided_by: human`.
- The system errs toward review, not toward passing: across the test split it
  auto-passes 7.1% of traffic at 100% precision and sends 41% to a person.
- Captures are assumed to be of material under inspection. Deployment on a line
  handling identifiable product would need that addressed first.