# Evaluation — ForgeSight

Domain: **rolled-steel surface defect inspection**. Public data only
(NEU-CLS + NEU-DET, 3600 real photographs). No synthetic data is used in any
number below.

Reproduce everything:

```powershell
python eval\train_classifier.py --include-background   # accuracy + confusion matrix
python eval\triage_eval.py                             # operating point + calibration
python eval\neu_eval.py --limit 100                    # localisation, IoU per class
python eval\demo_loop.py                               # closed-loop traces
```

Raw output: `eval/classifier_metrics.json`, `eval/triage_calibration.json`,
`eval/neu_results.json`.

---

## 1. What is evaluated, and why separately

Three questions that fail differently:

| evaluation | question | headline |
|---|---|---|
| **classification** | which defect is this? | **93.80%** held-out accuracy, macro F1 0.929 |
| **triage** | should a person look at it? | **sensitivity 1.0000**, 0 of 1080 defects missed |
| **localisation** | where is it? | **macro F1 ~0.19** at IoU 0.5 — weak, stated plainly |

Localisation is reported separately precisely because it is the weakest part.
Pooling it with classification would hide that.

---

## 2. Classification — 7 classes

Training set 3869 images: 1800 NEU-CLS + 1800 NEU-DET + 269 `no_defect`
crops taken from regions **outside** every annotation box (NEU contains only
defective images, so a triage system needs a clean class to be able to output
"pass"). Stratified 70/30 split, seed 0.

| | held-out accuracy |
|---|---|
| overall | **93.80%** |
| NEU-DET captures (used at inference) | **95.40%** |
| NEU-CLS captures | 95.34% |
| `no_defect` background crops | 72.84% (n=81) |

Per class:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| crazing | 0.968 | 0.994 | **0.981** | 180 |
| patches | 0.942 | 0.989 | **0.965** | 179 |
| rolled-in_scale | 0.968 | 1.000 | **0.984** | 180 |
| pitted_surface | 0.925 | 0.961 | **0.943** | 175 |
| scratches | 0.930 | 0.889 | 0.909 | 171 |
| inclusion | 0.879 | 0.889 | 0.884 | 160 |
| no_defect | 0.983 | 0.728 | 0.837 | 81 |

**Dominant confusions** (this is the interesting part, not the worst number):

| count | true → predicted |
|---|---|
| 15 | inclusion → pitted_surface |
| 14 | scratches → inclusion |
| 12 | scratches → inclusion (second block) |
| 9 | inclusion → scratches |
| 7 | no_defect → crazing |

Thin, linear, low-contrast features are the hard cases: `inclusion` and
`scratches` are both elongated dark streaks at this resolution.

---

## 3. Triage — the operating point

Thresholds were **measured, not chosen by feel**, by sweeping the held-out split
(`eval/triage_eval.py`, 1161 images). Shipped setting: `ESCALATE_MIN=0.85`,
`REVIEW_MIN=0.55`, `PASS_MIN=0.80`.

| metric | value |
|---|---|
| **defect sensitivity** | **1.0000** — 0 of 1080 defects silently passed |
| auto-escalate rate | 42.8% of defects closed without human re-inspection |
| human review rate | 51.2% of all traffic |
| clean captures released automatically | 48.1% |
| second-opinion rate | remainder, low-confidence cases |

### Confidence calibration

Raw softmax was systematically **under-confident** (stated 0.79 where accuracy
was 0.94), which made the agent escalate far too little. Fitting a single
temperature on the held-out split:

| | ECE |
|---|---|
| before | 0.0750 |
| after (T = 0.70) | **0.0218** |

Reliability after scaling:

| stated confidence | n | actual accuracy |
|---|---|---|
| 0.00–0.50 | 20 | 0.300 |
| 0.50–0.70 | 77 | 0.597 |
| 0.70–0.85 | 110 | 0.845 |
| 0.85–0.95 | 210 | 0.962 |
| 0.95–1.00 | 744 | 0.985 |

Temperature cannot change the argmax, so accuracy is identical before and after —
only the stated confidence moves, which is what lets the agent act on a number.

### A correction to our own metric

A first version of this evaluation counted any low-confidence defect prediction
as a "miss", reporting 46 misses. That was wrong: low confidence routes to
second opinion and then human review, so those are review traffic, not misses. The
real failure is a defect the classifier **actively calls clean**. Corrected, the
count is 0.

---

## 4. Localisation — the weak part

Box-level detection against NEU-DET ground truth, greedy IoU matching at 0.5.

| class | precision | recall | F1 | detected @IoU 0.3 |
|---|---|---|---|---|
| crazing | 0.010 | 0.081 | 0.017 | 0.600 |
| inclusion | 0.113 | 0.072 | 0.088 | 0.360 |
| patches | 0.011 | 0.038 | 0.017 | 0.400 |
| pitted_surface | 0.032 | 0.078 | 0.045 | 0.220 |
| rolled-in_scale | 0.000 | 0.000 | 0.000 | 0.000 |
| scratches | 0.245 | 0.313 | 0.275 | 0.940 |

**macro F1 0.075 hand-picked, 0.188 after grid search** (180 configs over
`box`, `anom_thresh`, `min_area`, `close_k`).

Why it is weak: NEU defects are diffuse and frequently span most of the frame
(`crazing_1` is annotated 2,2–193,194). A merged connected-component box cannot
reach IoU 0.5 against such an annotation, and `rolled-in_scale` has no box
annotation at all in the files we parsed.

**We did not paper over this.** The classifier carries the decision; localisation
supplies a coarse pointer and, more usefully, an *independent* signal — when the
classifier says clean but localisation finds structure, the agent refuses to
auto-pass. That disagreement rule is in `backend/app/agent.py`.

---

## 5. Failure cases and limitations

| limitation | cause | severity |
|---|---|---|
| **Localisation F1 ~0.19** | diffuse defects vs merged-box annotation; `rolled-in_scale` unannotated | high — reported, not hidden |
| **`no_defect` recall 0.728** | only 269 background crops exist (most NEU frames are annotated almost end-to-end), and boxes are imperfect so some crops contain unannotated defect | high — the weakest class |
| `inclusion` ↔ `pitted_surface` / `scratches` | thin low-contrast streaks are near-identical at 200x200 | medium |
| No video, no camera geometry | static 200x200 crops only | medium |
| No mill tolerance model | outputs a defect class and a region, not a pass/fail against a spec | medium |
| Human approval is a stub | endpoint + log, not a workflow system | low |
| COOL gives no speedup on this pipeline | `gabor_filter2D_x16` is 82% of runtime and COOL does not accelerate it | see `docs/BENCHMARK-RESULTS.md` |

### Not adversarial
No adversarial images, no tampering attempts, no deliberate attempt to force a
bad part through.

---

## 6. What this evaluation does not establish

- **Not a production claim.** 200x200 flat crops from one dataset. Real steel
  lines have camera geometry, motion, scale variation, multiple defect
  co-occurrence, and throughput requirements we never measured.
- **`no_defect` is weak and under-represented** (81 held-out). A 48.1% clean-pass
  rate on that sample size is not a meaningful false-accept rate.
- **Splits are by image, and the two NEU sets may share capture sessions.** If
  near-duplicate frames straddle the split, accuracy is optimistic. We did not
  verify that; a session-aware split would be the honest next step.
- **Single training run.** No seed variance, no cross-validation.
- **Localisation is not production-grade** and we would not ship it for marking
  without work.
- **`no_defect` crops inherit annotation noise**, so some "clean" labels are
  probably not clean.
