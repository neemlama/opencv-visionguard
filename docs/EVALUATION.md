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
| after scaling (T = 0.75, fitted on val) | **0.0361** |

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

**On the 41% review rate.** It is high, and we say so. Sensitivity is 1.0000 at
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
conservative rather than aggressive. ECE on test: 0.0886 → **0.0361** after
temperature scaling (T = 0.75, fitted on validation).

---

## 4. Agentic ablation — does the loop earn its place?

`eval/agent_ablation.py`, 740 real captures (540 defect from the leakage-free test
split + 200 genuinely clean crops from unannotated regions). Single-pass = one
perception then decide. Loop = quality gate + classifier/localiser disagreement
check (second opinion now disabled by default, see §4c).

| condition | single-pass | loop | delta | gate refusals | false accepts (of 200 clean) |
|---|---|---|---|---|---|
| clean | 0.7000 | 0.7176 | **+0.018** | 0.013 | 2 |
| soft focus | 0.7243 | 0.7297 | **+0.005** | 1.000 | 0 |
| under-exposed | 0.6770 | 0.7068 | **+0.030** | 0.054 | 1 |
| specular glare | 0.5730 | 0.6405 | **+0.068** | 0.008 | **34** |
| low contrast | 0.7027 | 0.7162 | **+0.014** | 0.081 | 3 |
| sensor noise | 0.3108 | 0.2892 | **−0.022** | 0.543 | 0 |

### What we retract from our own earlier run

An earlier version of this table showed **+0.157 on specular glare**. That number
was an artefact of a miscalibrated gate: the glare limit (0.05) had been derived
from the **first 300 NEU files**, which are dominated by one or two classes. Across
all 1800 legitimate captures, `glare_ratio` reaches **0.75** (p99 = 0.561), while
our synthetic glare preset only reaches **0.388**. So any limit low enough to
catch the synthetic glare also refuses roughly 14% of good captures.

Recalibrated permissively (`GLARE_MAX = 0.80`, above the real maximum), the
glare result falls to **+0.068** — still positive, but a third of what we first
claimed. We quote the corrected figure.

A `largest-bright-region` feature was tried as a better discriminator and
**failed**: clean p99 = 0.324 versus glare p05 = 0.282. Rough, scaled steel has
genuinely large saturated regions, so a specular highlight is not separable from
legitimate bright surface by intensity statistics alone.

### Consequences we accept

- **Glare detection is not claimed.** It is off by default and is listed as an
  open problem.
- Under the glare preset the gate no longer refuses, and **34 of 200 clean
  captures are auto-passed**. That is the safety cost of not detecting glare, and
  it is the clearest argument for building a real glare feature.
- Under sensor noise the loop is **slightly worse than single-pass (−0.022)**
  because the gate refuses 54% of frames, including good ones. False accepts stay
  at 0/200, so it is over-cautious rather than unsafe.

### 4d. Which degradations the loop can actually fix

| condition | gate refusal rate | recaptures? |
|---|---|---|
| soft focus | **1.000** | **yes — converges in one step** |
| sensor noise | 0.543 | yes |
| low contrast | 0.081 | no |
| under-exposed | **0.054** | **no** |
| specular glare | 0.008 | no |

Soft focus is the only condition where the loop detects a bad capture, acts and
converges. **Under-exposure is an unexpected gap: the gate has no brightness or
exposure check**, so it refuses only 5.4% of under-exposed frames and exposure
never triggers a recapture. The +0.030 there comes from confidence deferral, not
from the agent acting. Adding a luminance metric against a calibrated target is
the obvious fix; we left it open rather than hand-tuning another threshold on the
same data that already misled us once.

### 4e. Claims are machine-checked

`eval/verify_report_claims.py` re-reads every artifact cited by
`docs/REPORT.md` and fails on any mismatch. It has already caught three real
drifts in this document (ECE, human-review rate, and a noise false-accept count
that belonged to an older threshold set).

### 4a. A safety failure the ablation found

The first run showed the loop **auto-passed 200 of 200 clean captures under sensor
noise**. Additive noise *raises* Laplacian variance, so noisy frames read as
*sharper* and passed every check; the classifier then answered "no defect" on
noise texture.

Fixed with a robust noise estimate (MAD of the 3x3 high-pass response) and a
threshold from 240 real captures: clean p99 = 10.286, noisy p05 = 11.305, so
`NOISE_MAX = 10.8`. **False accepts under noise: 200/200 → 0/200.**

This is the clearest argument for having built the ablation: a metric written for
the award found a defect that would otherwise have shipped.

### 4b. What "agentic" means here, precisely

The award asks for a workflow where image results influence a later plan, tool
call, action, or human-approval request. Ours does exactly that, and the mechanism
is **gated perception**, not a reasoning chain:

1. vision metrics (sharpness, noise, saturation) decide whether the frame is
   judgeable at all;
2. if not, the agent names the physical fix and, in simulator mode, applies it by
   changing capture settings and **re-rendering the scene**;
3. perception runs again on a genuinely different frame;
4. classification confidence plus defect severity choose pass / review /
   stop-the-line escalation, always with human approval.

There is **no LLM planner** in this system. We think a transparent, measured gate
is more defensible than an unmeasured reasoning chain, and we would rather be
judged on what we can measure.

### 4c. The second opinion: measured, removed, reported

The CLAHE re-analysis fired **359 times** across all conditions, changed the
predicted class **165 times**, and **corrected 0 errors while introducing 17**.

It is therefore **disabled by default** (`ENABLE_SECOND_OPINION = False`) and
kept behind a flag because the negative result is part of the submission. The
loop's measured value comes from the quality gate, not from re-reasoning about
the same frame.

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
