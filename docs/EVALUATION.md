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
| **classification** | which defect is this? | **92.75%** held-out accuracy, macro F1 0.927 |
| **triage** | should a person look at it? | **0 missed defects**, auto-pass precision **1.000** |
| **localisation** | where is it? | **macro F1 ~0.19** at IoU 0.5 — weak, stated plainly |

Localisation is reported separately precisely because it is the weakest part.
Pooling it with classification would hide that.

---

## 1a. Two leaks found and fixed before quoting any number

Both were found by review, not by the tooling, and both inflated our result.

### Leak 1 — NEU-CLS and NEU-DET are the same photographs

We trained on "CLS + DET" believing they were different capture sets.
`eval/check_duplicates.py` shows they are **the same 1800 source images**:

| evidence | count |
|---|---|
| cross-set pairs with pHash == 0 AND dHash == 0 (bit-identical content) | **1045** |
| same-label cross-set pairs within perceptual threshold 6 | 7701 |
| exact pixel-duplicate groups within a set | 2 |

The file numbering makes the mapping explicit: `Cr_1.bmp` ↔ `crazing_1.jpg`,
`Pa_101.bmp` ↔ `patches_101.jpg`. (The 13389 "cross-label" pairs are pHash
collisions, not contradictory labels — a 64-bit 8x8 DCT hash is not discriminative
at threshold 6, which is why identity is keyed on class + source index and the hash
is used only to verify.)

Consequence: a random per-row split could put `Cr_1` in train and `crazing_1` in
test, and the model scored a memorised image.

**Fix** (`eval/splits.py`): deduplicate to **1800 unique images**, keeping the DET
copy since inference runs on DET captures, then split **by identity group** so no
group can straddle a boundary.

```
raw files enumerated : 3600 (1800 CLS + 1800 DET)
unique source images : 1800   <- CLS and DET are the same survey
rows after dedupe    : 2069  (defect 1800 + background 269)
[grouped] train=1138 val=310 test=621
         group overlap: {'train&val': 0, 'train&test': 0, 'val&test': 0}  -> CLEAN
```

**Effect on the headline: 93.80% → 92.75%** (the 16-Gabor model; 93.08% → 92.75%
for the shipped 8-Gabor one). The leak was worth ~0.3–0.7 points, not
catastrophically — but the number is now defensible, which matters more.

### Leak 2 — temperature fitted on the split used to report ECE

The calibration temperature was fitted on the held-out split and ECE was then
reported **on that same split**, so the calibration figure was optimistic by
construction.

**Fix:** three-way split. Weights fitted on train, temperature fitted on
**validation**, everything reported on **test**.

| | ECE on TEST |
|---|---|
| before scaling | 0.0886 |
| after scaling (T = 0.75, fitted on val) | **0.0338** |

---

## 2. Classification — 7 classes

Deduplicated to 1800 unique images plus 269 `no_defect` crops taken from regions
**outside** every annotation box (NEU contains only defective images, so a triage
system needs a clean class to be able to output "pass"). Grouped split, seed 0.

| | held-out accuracy |
|---|---|
| overall | **92.75%** (n=621) |
| NEU-DET captures (used at inference) | 94.81% |
| `no_defect` background crops | 79.01% (n=81) |

| class | precision | recall | F1 |
|---|---|---|---|
| rolled-in_scale | 0.957 | 1.000 | **0.978** |
| patches | 0.977 | 0.956 | **0.966** |
| crazing | 0.937 | 0.989 | 0.962 |
| pitted_surface | 0.904 | 0.944 | 0.924 |
| scratches | 0.976 | 0.889 | 0.930 |
| no_defect | 0.970 | 0.790 | 0.871 |
| inclusion | 0.804 | 0.911 | 0.854 |

**Dominant confusions:** `scratches → inclusion` (10), `no_defect → inclusion` (6),
`inclusion → pitted_surface` (5), `no_defect → rolled-in_scale` (4),
`no_defect → crazing` (4).

Thin, linear, low-contrast streaks are the hard cases; `inclusion` is the weakest
class and `no_defect` recall is limited by having only 269 crops to learn from.

---

## 3. Triage — the operating point, with an interval

A system that sends everything to a human also misses zero defects. **Sensitivity
alone is meaningless**, so it is reported with an interval, the automation rate,
and the precision of automatic passes.

Leakage-free test split, 621 rows (540 defect, 81 clean), thresholds swept by
`eval/triage_eval.py`. Shipped setting `ESCALATE_MIN=0.85`:

| metric | value |
|---|---|
| **missed defects** | **0 of 540** |
| sensitivity, point estimate | 1.0000 |
| **sensitivity, 95% lower bound (rule of three)** | **≥ 0.9944** |
| **auto-pass precision** | **1.000** — nothing auto-passed was defective |
| auto-pass rate | 7.1% of all traffic |
| clean surfaces released automatically | 54.3% |
| auto-escalate rate | 38.0% of defects closed without human re-inspection |
| human review rate | 51.0% of all traffic |
| second-opinion rate | 2.9% of all traffic |

**On the interval.** With 0 misses in 540, the rule of three gives a 95% upper
bound on the miss rate of 3/540 = 0.56%, so sensitivity is **≥ 99.44%** at 95%
confidence. We quote that rather than 1.0000, because a point estimate of
"perfect" from 540 samples is not a measurement of perfection.

**On the 51% review rate.** It is high, and we say so. Sensitivity is 1.0000 at
*every* threshold from 0.50 to 0.95, so the threshold is not buying safety — it is
buying automation. The sweep in `eval/triage_calibration.json` shows
auto-escalate falling from 50.7% to 27.4% as the threshold rises, with misses
staying at 0 throughout. A deployment would trade along that curve; we picked 0.85
rather than the most aggressive 0.50.

### Confidence calibration (fitted on validation, reported on test)

| stated confidence | n | actual accuracy |
|---|---|---|
| 0.00–0.50 | 18 | 0.444 |
| 0.50–0.70 | 51 | 0.725 |
| 0.70–0.85 | 72 | 0.889 |
| 0.85–0.95 | 120 | 0.933 |
| 0.95–1.00 | 360 | 0.992 |

Mildly **under**-confident in the mid bands, so `ESCALATE_MIN=0.85` is
conservative rather than aggressive. ECE on test: 0.0886 → **0.0338** after
temperature scaling (T = 0.75, fitted on validation).

---

## 4. Agentic ablation — does the loop earn its place?

`eval/agent_ablation.py`, 740 real captures (540 defect from the leakage-free
test split + 200 genuinely clean crops from unannotated regions). Single-pass =
one perception then decide. Loop = quality gate + second opinion on low
confidence + classifier/localiser disagreement check.

| condition | single-pass | loop | delta | 2nd opinion fired | class changed | errors fixed | errors introduced | false accepts (of 200 clean) |
|---|---|---|---|---|---|---|---|---|
| clean | 0.7000 | 0.7162 | +0.0162 | 0.169 | 56 | 0 | 2 | 2 |
| soft focus | 0.7243 | 0.7297 | +0.0054 | 0.000 | 0 | 0 | 0 | 0 |
| under-exposed | 0.6770 | 0.6946 | +0.0176 | 0.137 | 53 | 0 | 9 | 1 |
| **specular glare** | 0.5730 | 0.7297 | **+0.1568** | 0.000 | 0 | 0 | 0 | 0 |
| low contrast | 0.7027 | 0.7081 | +0.0054 | 0.180 | 56 | 0 | 6 | 3 |
| **sensor noise** | 0.3108 | 0.3473 | **+0.0365** | 0.000 | 0 | 0 | 0 | **0** |

**The loop beats single-pass in every condition**, and the gain concentrates
exactly where it should: on captures that are not fit to judge.

### The honest decomposition — the gate works, the second opinion does not

Across all conditions the second opinion **fired 359 times and changed the class
165 times, yet corrected 0 errors and introduced 17.** Essentially all of the
loop's measured value comes from the **capture-quality gate**, which refuses to
classify unusable frames, not from the second-opinion reasoning.

We report that rather than quoting the +0.157 glare figure alone. The second
opinion currently costs a little accuracy and buys caution; it is not earning its
place and we would not claim it does.

### A safety failure this ablation found, and we fixed

The first run of this ablation showed the loop **auto-passed 196 of 200 clean
captures under sensor noise**. Root cause: the gate checked sharpness, glare and
clipping, but additive noise *raises* Laplacian variance, so a noisy frame reads as
*sharper* than a clean one and passed every check. The classifier then confidently
answered "no defect" on noise.

Fixed by adding a robust noise estimate to the gate — MAD of the 3x3 high-pass
response, median-based so a real defect cannot inflate it. Threshold derived from
240 real captures (`eval/quality_gate_calibration.py`):

| | noise_sigma p50 | p99 / p05 |
|---|---|---|
| clean NEU captures | 8.803 | p99 = 10.286 |
| + sensor noise | 13.065 | p05 = 11.305 |

Clean max (10.286) sits below noisy p05 (11.305), so **NOISE_MAX = 10.8**
separates them. False accepts under noise: **196/200 → 0/200.** The same
calibration pass relaxed the glare and blown-highlight limits to clean p99
(0.05 / 0.035), which removed spurious refusals on good captures.

This is the clearest argument for running the ablation at all: a metric we added
for the award found a defect that would have shipped.

---

## 5. Accuracy versus latency

`docs/BENCHMARK-RESULTS.md` §2. Halving the Gabor bank is **1.83x** end-to-end on
AWS (1.42x on this laptop) for **−0.33 accuracy points** on the 7-class task,
with triage behaviour unchanged. An earlier **2.20x** figure did not reproduce and
is retracted.

---

## 6. A shortcut we checked for, and did not find

`no_defect` crops are taken from regions *outside* annotation boxes in the same
photographs as the defect crops, so the classifier could in principle learn
"crop boundary / flat texture = clean" rather than "no defect = clean".

**Test:** background crops and defect images come from the same 200x200 captures
with the same sensor and lighting, so a shortcut would have to be geometric rather
than photometric. The measured evidence against a shortcut:

- `no_defect` is the **hardest** class to learn from — only 269 crops exist,
  because most NEU frames are annotated almost end-to-end — and it has the
  **lowest recall (0.790)** of any class. A shortcut would give high recall.
- Its precision is 0.970: when it says clean it is almost always right, but it
  misses clean surfaces. That is the signature of an under-powered signal, not a
  leaked one.
- Its dominant confusions are `no_defect → inclusion` (6) and
  `no_defect → rolled-in_scale` (4) — texture inside the crop being read as
  defect, which is a genuine ambiguity, not a label artefact.

**Not proven clean, though.** The crop regions inherit annotation noise, so some
"clean" crops almost certainly contain unannotated defect. A cleaner test would
need genuinely defect-free steel, which NEU does not provide. We flag this as an
open methodological weakness rather than claiming it resolved.

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
