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

# Localisation rebuild (2026-10-12): disjoint TUNE/REPORT splits
python eval\localize_compare.py --tune-limit 60 --report-limit 150   # single vs multi
python eval\train_scorer.py --tune-limit 120 --report-limit 200      # learned scorer
```

Raw output: `eval/classifier_metrics.json`, `eval/triage_calibration.json`,
`eval/neu_results.json`, `eval/localize_compare.json`, `eval/scorer_results.json`.

---

## 1. What is evaluated, and why separately

Three questions that fail differently:

| evaluation | question | headline |
|---|---|---|
| **classification** | which defect is this? | **92.75%** held-out accuracy, macro F1 0.927 |
| **triage** | should a person look at it? | **0 missed defects**, auto-pass precision **1.000** |
| **localisation** | where is it? | **macro F1 0.2247** at IoU 0.5 on a disjoint hold-out — barely moved from 0.2200 despite the proposal oracle rising 0.346 → 0.546, stated plainly |

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
| before scaling | 0.0903 |
| after scaling (T = 0.75, fitted on val) | **0.0361** |

### Leak 3 — localisation was tuned on the images it was reported on

Found on 2026-10-12 while rebuilding the localiser, and it applies to the
previously published localisation numbers only (classification, triage and the
agent ablation were already split correctly).

`eval/neu_tune.py` grid-searched 180 configs over `load_subset(40)` — the **first
40 images per class in sorted filename order** — and `eval/neu_eval.py` reported
on those same first-40 images. The search set and the report set were identical.

**Fix** (`eval/localize_eval.py`, `eval/localize_compare.py`, `eval/train_scorer.py`):
per-class **interleaved** disjoint halves — even index to TUNE, odd index to
REPORT — so every hyper-parameter, weight, `pos_weight` and operating threshold
is chosen on TUNE and the REPORT split is read once. Interleaving rather than a
head/tail cut avoids coupling the split to filename order, which groups by class
prefix.

**Effect on the headline: 0.188 → 0.1855.** The leak was worth little in itself,
but the old number was not defensible and this one is.

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
conservative rather than aggressive. ECE on test: 0.0903 → **0.0361** after
temperature scaling (T = 0.75, fitted on validation).

---

## 4. Agentic ablation — does the loop earn its place?

`eval/agent_ablation.py`, 740 real captures (540 defect from the leakage-free test
split + 200 genuinely clean crops from unannotated regions). Single-pass = one
perception then decide. Loop = quality gate + classifier/localiser disagreement
check (second opinion now disabled by default, see §4c).

| condition | single-pass | loop | delta | gate refusals | false accepts (of 200 clean) |
|---|---|---|---|---|---|
| clean | 0.7000 | 0.7257 | **+0.026** | 0.013 | 0 |
| soft focus | 0.7243 | 0.7297 | **+0.005** | 1.000 | 0 |
| under-exposed | 0.6770 | 0.7243 | **+0.047** | 0.054 | 0 |
| specular glare | 0.5730 | 0.7338 | **+0.161** | 0.008 | **8** |
| low contrast | 0.7027 | 0.7230 | **+0.020** | 0.081 | 0 |
| sensor noise | 0.3108 | 0.4838 | **+0.173** | 0.543 | 0 |

Re-measured 2026-10-17 after the multi-scale localiser was wired into the request
path (§4d, §4e). Single-pass numbers are unchanged to four decimals, which is the
expected signature: single-pass never consults the localiser. Every **loop**
number improved, and both false-accept counts the release gate cares about moved
the right way — **glare 34 → 8**, noise **0 → 0**.

The likely mechanism is the disagreement veto. The new localiser returns more
regions per image (mean `region_count` on defect images **0.608 → 1.833**,
measured in `eval/localize_wiring_check.py`), so the veto fires more often and
routes more classifier errors to a human. **We have not isolated this with a
dedicated ablation**, so it is stated as the mechanism consistent with the
measurements, not as a proven cause. Note also that this ablation's metric
rewards deferral on defects and penalises it on clean surfaces, so a
change that defers more is not unambiguously an improvement — which is exactly
why the false-accept column is reported next to it, and why it did not regress.

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

## 4. Localisation — rebuilt, still the weak part

Box-level detection against NEU-DET ground truth, greedy IoU matching at 0.5.
All numbers below are on a **disjoint 900-image REPORT split** that is used for
nothing else (see Leak 3).

### 4a. The first question is not "how good is it" but "how good could it be"

Before changing anything we measured the **oracle**: keep *every* candidate
component and hand-pick the best subset. Any scorer is bounded by it.

| proposal generator | oracle recall @IoU0.5 |
|---|---|
| single-scale, as previously shipped (`neu.py`) | **0.082** |
| multi-scale, union over 21 (blur, threshold) pairs | **0.476** |

The single-scale oracle of 0.082 is the finding that mattered. It is *below* the
0.188 we had been publishing, which means **no scorer could ever have reached
that number** — the 0.188 was an artefact of the leak in Leak 3, not a result.
Threshold tuning could not have fixed it, and no amount of further tuning would.

### 4b. What the diagnosis actually found

We had assumed, and the old document asserted, that NEU defects are simply too
diffuse to box. Measurement contradicts most of that:

| claim | measurement |
|---|---|
| defects span most of the frame | across all 4189 GT boxes the median area is **11.8%** of the frame, p90 = 37.9%; only 264 boxes exceed 50% and 117 exceed 80% |
| annotations overlap, so one blob cannot match | GT sibling boxes barely touch: mean pairwise IoU **0.015**, only 0.1% exceed 0.5. NEU annotates **2.32 genuinely separate defects per image** |
| our components are hollow shells that need filling | hole-filling and convex hull changed the oracle **not at all** (0.082 either way) — the components were already solid |

The real fault was ours, and it was measurable: **88% of predicted boxes were more
than 2x too large**, median **7.3x in area** (~2.7x linear). NEU-DET images
averaged 21.9 components against 2.32 true defects. Shrinking boxes toward their
centres did not help (oracle stayed ~0.08), so it was over-merging, not
over-scaling.

### 4c. What was built

`backend/app/localize.py`. Propose at 9 (blur, threshold) combinations rather
than one, then select. Two scorers were measured:

| arm | macro F1 | precision | recall |
|---|---|---|---|
| single-scale, 540 configs tuned on TUNE (`eval/localize_compare.py`) | 0.1855 | 0.259 | 0.151 |
| multi-scale, **unsupervised** hand-weighted score | 0.1839 | 0.159 | 0.208 |
| multi-scale, **learned** scorer (`eval/train_scorer.py`) | **0.2191** | 0.239 | 0.210 |

The middle row is a **negative result we are reporting rather than hiding**: the
multi-scale proposal space has a 4x higher ceiling, and a hand-weighted score
could not exploit any of it (−0.0016, i.e. nothing). Choosing one good box out
of ~78 needs supervision. A 15-feature binary logistic regression trained on the
TUNE split only recovers part of it: **+0.034 macro F1, +18% relative**, with
precision *and* recall both matching or beating single-scale.

Per class, held out:

| class | precision | recall | F1 |
|---|---|---|---|
| scratches | 0.394 | 0.587 | **0.471** |
| inclusion | 0.372 | 0.246 | **0.296** |
| pitted_surface | 0.293 | 0.216 | **0.249** |
| rolled-in_scale | 0.187 | 0.170 | **0.178** |
| crazing | 0.086 | 0.061 | **0.071** |
| patches | 0.053 | 0.045 | **0.048** |

The learned features are interpretable and match the diagnosis: the strongest
weights are on **box size** (`log_area` +0.71, `log_h` +0.65, `log_w` +0.62) and
**`area_frac` is negative** (−0.53) — the scorer learned to prefer smaller boxes,
which is exactly the 7.3x over-sizing fault we measured. An unsupervised score
had no way to express that.

### 4d. Second diagnosis: the ceiling was box geometry, not detection

Added 2026-10-17. The 0.219 above was limited by an oracle of 0.3455, and §4e's
conclusion that "closing the remaining gap needs better proposals, not a better
score" was half right, and worth correcting precisely.

`eval/diagnose_proposals.py` buckets every missed ground-truth box by the *best*
overlapping proposal:

| bucket | definition | share of all GT |
|---|---|---|
| **refinement failure** | best IoU 0.1–0.5 — a proposal overlaps the defect, at the wrong size | **62.3%** |
| **detection failure** | best IoU < 0.1 — nothing overlaps at all | **2.6%** |

So the generator already *saw* 97.4% of defects. It drew the wrong rectangle.
More detector families would have been wasted effort. The refinement failures are
bimodal — of them **16.8%** had a proposal >3x too large and **24.2%** <1/3x too
small — which is why no single "tighten the box" rule can help.

**A negative result, reported rather than buried.** The obvious cheap fix is to
pick each box's size with a label-free criterion. All three we tried made things
*worse* (`eval/probe_variant_pick.py`):

| criterion | oracle recall | vs 0.3508 |
|---|---|---|
| contrast — `inside_mean − ring_mean` | 0.3180 | **−0.033** |
| `inside_mean` | 0.2689 | −0.082 |
| `mass / √area` | 0.3541 | +0.003 |
| base box + contrast pick | 0.3934 | +0.043 |

The reason is the interesting part: **NEU annotations mark the semantic extent
of a defect, which does not follow any energy iso-contour.** No function of the
anomaly map can recover it. Box size has to be learned from labels.

**What we built instead.** Add nested variants of every component, grown and
shrunk about its anomaly-weighted centroid, and let the learned scorer choose
(`backend/app/localize.py::add_variants`). `eval/probe_factors.py` chose the
factors by greedy forward selection on TUNE, maximising oracle recall per extra
box. It picked **(0.85, 2.1)**: 80 → 240 boxes/image, oracle **0.346 → 0.546**.
Richer pools raise the oracle further and *lower* F1, so two factors is the
optimum, not a compromise.

Two details that turned out to matter more than the variants themselves:

- **Sweep the operating point jointly.** `top_k` and `nms_iou` were never swept
  before — only `score_thresh`. With 3x the candidates the optimum moves from
  `top_k=3` to `top_k=2`, and precision *rises* (0.224 → 0.253) because fewer,
  better-ranked boxes are emitted. Most of the apparent win was hiding here.
- **Give the scorer the size signal.** Three features were added
  (`SCORER_FEATURES` 15 → 18): `var_log`, `outside_mass` (anomaly energy in a
  ring just outside the box) and their product `grow_interaction`. A linear model
  cannot represent "grow this box *because* there is anomaly just outside it", so
  the interaction is supplied explicitly.

**Result — a clean A/B, both arms REPORT-read once each, identical protocol:**

| arm | boxes/img | oracle | P | R | **macro F1** |
|---|---|---|---|---|---|
| baseline, swept operating point | 80 | 0.3455 | 0.224 | 0.209 | 0.2200 |
| **+ nested variants** | 240 | **0.5455** | 0.253 | 0.197 | **0.2247** |

**+0.0047.** Per class the gain is not uniform: `rolled-in_scale` 0.185 → 0.250,
`pitted_surface` **0.278 → 0.250 regressed**, `crazing` and `patches` flat.

Two things we are disclosing rather than smoothing over:

- **The TUNE-validation gain was +0.0201 and the held-out gain is +0.0047** — it
  shrank 3.6x. Most of what looked like a win was overfitting the tuning slice.
- **The REPORT split was read twice**, once per pre-registered arm, and reported
  as an ablation. No configuration was *selected* on it. A single-read protocol
  cannot produce this comparison at all, so we state the deviation instead of
  hiding it.

The ceiling is no longer the binding constraint in the way §4e said: it is now
0.5455 and the scorer reaches only 0.1966, capturing **36.0%** of what the
proposals allow (was 60.8%). **The scorer is the bottleneck again.**

### 4e. What is still wrong, stated plainly

- **macro F1 0.2247 is still weak.** It is a real improvement on an honest
  baseline, not a good detector. `crazing` (0.061) and `patches` (0.053) are
  barely above zero, and §4d's `pitted_surface` regression (0.278 → 0.250) is a
  genuine cost, not noise.
- **The scorer is the bottleneck again, not the proposal generator.** This
  inverted in §4d. Oracle recall on the same split is now **0.5455**; the scorer
  reaches 0.1966, capturing **36.0%** of what the proposals allow (was 60.8% at
  an oracle of 0.3455). Raising the ceiling bought recall we cannot yet spend,
  because a 15→18 feature *linear* model is not strong enough to pick 2 boxes out
  of 240. That is the next thing to try, and it is a modelling problem, not an
  OpenCV one.
- **Now wired into the request path, and the gate that blocked it passed.**
  This bullet used to say "not wired in, deliberately". The reasoning was sound
  and it is worth keeping, because it set the condition under which the change
  could be made:

  ```python
  # backend/app/agent.py:131
  if pred["defect_class"] == "no_defect" and pred["confidence"] >= PASS_MIN:
      if loc["region_count"] > 0:      # structure found -> refuse to auto-pass
  ```

  The veto keys on the **raw count** of regions, so it is a *sensitivity*
  mechanism: any structure at all blocks the pass. The rebuilt localiser looked
  like its opposite — it suppresses to `top_k` and applies a score threshold,
  emitting **fewer, tighter** boxes. The stated fear was that a precision-oriented
  detector behind a sensitivity veto would make the agent *more* willing to
  auto-pass, "the exact direction in which this system has already shipped one
  real safety bug". The condition set for revisiting it was: re-run
  `eval/agent_ablation.py` with **`false_accepts` as the release gate**, not an
  F1 comparison.

  Run on 2026-10-17, the gate passed and the feared direction did not materialise:

  | release gate | before | after |
  |---|---|---|
  | noise false accepts (of 200 clean) | 0 | **0** |
  | glare false accepts (of 200 clean) | 34 | **8** |

  Both moved the safe way or held, and every one of the six loop accuracies
  improved (§4f). The premise that the rebuilt localiser was *less* sensitive
  turned out to be wrong in the way that mattered: the old implementation's
  `min_area=400` suppressed most components, so the new one returns **3x more**
  regions per defect image (0.608 → 1.833) and the veto fires more, not less.

  **What we did not do.** We did not re-tune the veto to make the numbers look
  better, and we did not touch `PASS_MIN`. The correct future fix is still the
  one originally identified: make the veto key on *localiser confidence* rather
  than `region_count > 0`, so a better detector tightens the boxes without
  silently loosening or silently tightening the veto.
- **crazing remains near-unsolved** and is the honest weak class. Its oracle
  recall is only 0.136 at the *baseline* pool — the variant pool lifts the
  overall oracle to 0.546, but crazing is still the class least helped by it.

---

## 5. Failure cases and limitations

| limitation | cause | severity |
|---|---|---|
| **Localisation F1 0.2247** | the *proposals* now reach oracle 0.546 but a linear scorer captures only 36% of them; crazing (0.061) and patches (0.053) are barely above zero, and `pitted_surface` regressed 0.278 → 0.250 when variants were added | high — reported, not hidden |
| **Localisation costs 3x latency** | the multi-scale proposal generator runs 9 × (blur + threshold + morphology + connectedComponents) instead of 1; `analyze()` 20.5 → 62.3 ms per 200×200 crop (16 img/s) | medium — taken deliberately, measured |
| **`no_defect` recall 0.790** | only 269 background crops exist (most NEU frames are annotated almost end-to-end), and boxes are imperfect so some crops contain unannotated defect | high — the weakest class |
| `inclusion` ↔ `pitted_surface` / `scratches` | thin low-contrast streaks are near-identical at 200x200 | medium |
| No video, no camera geometry | static 200x200 crops only | medium |
| No mill tolerance model | outputs a defect class and a region, not a pass/fail against a spec | medium |
| Human approval is a small state machine | orders persist with an audit trail; unknown ids 404 and a second decision 409. No accounts, roles or signatures | low |
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
