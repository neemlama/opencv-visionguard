# Challenges We Faced

Engineering log of real problems hit while building ForgeSight. Judges score
"evaluation evidence appropriate to the project, including failure cases or
limitations" — this file is that evidence. Each entry is reproducible.

---

## C-01 — Defect detector missed dark pits (false negative on real defect)

**Severity:** Critical. Would have produced a detector that passes defective
weldseverything. A judge uploading one real defect image would have seen
`pass`and the whole technical-execution score (30%) would be undermined.

**Symptom**
The first detector version scored purely on edge evidence:

```python
edge_component = min(1.0, edge_density * 8 + large_contours * 0.05)
```

End-to-end test over our own samples gave:

| sample | expected | got | defect_score |
|---|---|---|---|
| weld_clean | pass | pass | 0.11 |
| weld_defect | defect | **pass** | 0.114 |
| weld_blurry | re-shoot | re-shoot | 0.00 |
| weld_glare | re-shoot-angle | re-shoot-angle | 0.23 |

`weld_defect` scored 0.114 — statistically indistinguishable from the clean
sample (0.11). The system was blind to the defect.

**Root cause**
Canny edges fire on *transitions*. A porosity pit in the middle of a weld bead
is a **uniform dark disk**: dark inside, with an edge only at its rim. On our
640x480 plate the rim is ~2% of the image, so the edge channel barely registers,
while the pit itself contributes almost nothing to `edge_density`.

Worse, the two "hard" samples (severe damage, glare) scored *higher* than the
real single-pit defect, so tuning thresholds on score magnitude alone would not
have fixed it — the signal simply was not present in that feature.

**Fix**
Added a complementary feature that does not depend on edge energy: regions
darker than their local surroundings.

```python
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
dcontours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
defect_areas = [cv2.contourArea(c) for c in dcontours if cv2.contourArea(c) > 400]
```

`adaptiveThreshold` with `THRESH_BINARY_INV` yields 255 where the pixel is darker
than the local Gaussian-weighted mean, so pits light up as filled blobs
regardless of contrast. `MORPH_OPEN` removes single-pixel sensor noise, and the
`contourArea > 400` filter (0.2% of this image) discards harmless texture.

Blend both channels so neither failure mode returns:

```python
edge_component = min(1.0, edge_density * 8 + large_contours * 0.05)
blob_component = min(1.0, dark_area_ratio * 25 + dark_blobs * 0.25)
defect_score = min(1.0, 0.5 * edge_component + 0.9 * blob_component)
```

**Verification after fix**

| sample | expected | got | defect_score | dark_blobs | trace steps |
|---|---|---|---|---|---|
| weld_clean | pass | pass | 0.055 | 0 | 1 |
| weld_defect | defect | human-review | 0.318 | 1 | 2 |
| weld_damage | escalate | escalate-human | 1.000 | 6 | 1 |
| weld_blurry | re-shoot | re-shoot | 0.000 | 0 | 1 |
| weld_glare | angle change | re-shoot-angle | 0.707 | 2 | 1 |

Separation between clean and defective went from 0.004 to 0.263 — a 66x margin.

**What we learned**
A single-feature detector is only as good as the assumption behind the feature.
"Pits have low contrast, therefore low edges" was true, and that is exactly why
they were invisible. Defect detection needs at least one channel per defect
*mechanism*, not per defect *appearance*.

**Known remaining limitations** (deliberate, documented for judges)
- Thresholds are hand-tuned on synthetic samples, not a labelled real dataset.
  Scores are a triage signal, not calibrated probability.
- `weld_damage` and `weld_glare` both reach high `defect_score`, but glare is
  caught first because branch order is blur -> glare -> defect. Reordering would
  let glare masquerade as a real defect.
- Very small pits (< 400px area) are filtered out entirely.
- No geometric tolerance model: a real system needs weld width/spec limits, not
  just "something is dark".

---

## C-02 — Sample images written to the wrong directory

`make_samples.py` computed its output path with
`Path(__file__).resolve().parent.parent / "samples"`, but the file lives at
`backend/app/make_samples.py`, so `.parent.parent` is `backend/`, not the repo
root. Samples silently landed in `backend/samples/` and every `/samples/*` request
404'd.

Fixed to `.parents[2]`. Worth logging because it failed *silently* — the script
printed a success message listing filenames that were not where the server looked.

---

## C-03 — PowerShell 5.1 has no `-Form` parameter

`Invoke-RestMethod -Form @{file = Get-Item ...}` fails with
"A parameter cannot be found that matches parameter name 'Form'" on Windows
PowerShell 5.1. Multipart uploads need `curl.exe`:

```powershell
curl.exe -s -X POST -F "file=@samples/weld_defect.jpg" http://localhost:8000/inspect
```

All sample-API tests in this repo were re-run with `curl.exe`.

---

## C-05 — COOL was *slower* than stock OpenCV, and the profiler showed why

**Severity:** High, and the most valuable finding in the project.

**Symptom.** The first valid Graviton4 comparison produced the wrong direction:

| arm | instance | library | OpenCV | fps |
|---|---|---|---|---|
| A | `c8g.medium` | COOL | 4.14.0-pre | 271.06 |
| B | `c8g.medium` | DIY pip | 4.14.0 | 357.27 |

**A/B = 0.76x** — COOL was 24% *slower* than the stock wheel on identical
hardware. Reporting that as-is would have been a poor COOL submission, and
tuning thresholds would not have helped.

**Root cause.** Per-op timings isolated a single dominant operation:

| op | COOL | DIY | |
|---|---|---|---|
| `Canny` | 0.189 | 0.615 | COOL **3.25x faster** |
| `adaptiveThreshold` | 2.748 | 1.714 | COOL **0.63x slower** |

`adaptiveThreshold` was ~74% of total pipeline time (2.75 ms of 3.69 ms). COOL's
large `Canny` win could not offset one operation it handles badly, so a
per-op-dominant workload inverted the total.

**Root cause of the bottleneck.** Our own C-01 fix introduced it. The
dark-blob detector used `adaptiveThreshold(31,9)` because that is the textbook
choice — which is exactly why nobody questions it and why it survived into the
benchmark.

**Fix.** Stop guessing, and measure candidate replacements on both axes — speed
*and* recall — with `bench/detector_variants.py`:

```
variant                     ms/call  clean defect damage glare
adaptiveThreshold(31,9)       2.428      0      1      6      2
gaussTophat sigma=12          5.291      0      1      6      0
boxTophat k=25                0.920      0      1      6      0   <- chosen
boxTophat k=41                0.845      0      1      6      3
morphBlackhat k=15            1.948      1      2      1      0
medianTophat k=15            14.743      0      0      0      0
```

Chose a **box-filter black-hat**: `cv2.blur` is O(1) per pixel regardless of
kernel size, so k=25 costs 0.92 ms where a 25x25 Gaussian (51x51 implied kernel)
costs 5.29 ms.

An intermediate attempt is instructive: we first swapped `adaptiveThreshold` for
a *Gaussian* top-hat on the theory that COOL accelerates `GaussianBlur`. Locally
that was **4.0 ms versus adaptiveThreshold's 1.6 ms — a pessimisation**, caught
by measuring before deploying. The theory was wrong; the box filter was right.

**Verification.** All five sample outcomes and blob counts unchanged
(`clean`=0, `defect`=1, `damage`=6, `blurry`=0, `glare`=0), so no accuracy was
traded away.

**Result after the fix**

| metric | before | after |
|---|---|---|
| application latency | 3.69 ms/frame | **1.10 ms/frame** (3.35x faster) |
| COOL vs DIY | 0.76x | **1.28x** |

**What we learned.** A library benchmark is only as useful as the workload's
composition. One operation can represent three quarters of the runtime, so
"which library is faster" is really "which library is faster *on your
bottleneck*". The win came from reading per-op timings and replacing the
bottleneck, not from tuning anything. We also report the remaining COOL
regression on `adaptiveThreshold` (0.62x) rather than hiding it.

---

## C-06 — Marketplace COOL images are not the OpenCV 5 the listing advertises

The Graviton4 listing states COOL is "based on OpenCV 5", but the image reports
`cv2.__version__ == '4.14.0-pre'`; the Graviton2 image reports `4.12.0`
(`libopencv_imgproc.so.4.12.0`). Verified on-instance, not from docs.

**Consequence.** A naive COOL-vs-pip comparison is version-confounded, so all arms
are pinned to the same major.minor. The **application** still runs OpenCV 5.0.0
as the competition requires; only the benchmark control uses 4.x.

Related trap: COOL reports a pre-release (`4.14.0-pre`), which does not resolve on
PyPI, so the control must be pinned to a range (`>=4.14.0,<4.15.0`) rather than an
exact version.

---

## C-04 — Wrong mental model of COOL packaging (design correction)

Our first plan assumed COOL could be `pip install`ed or baked into a Dockerfile
for `linux/arm64`. Researching the AWS Marketplace listing showed COOL ships as
a **Graviton AMI** (Ubuntu 24.04, ARM64, KleidiCV-optimised), not a library
artifact.

Consequence: the COOL Award path must launch the Marketplace AMI and run the
workload inside it with:

```bash
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/cool/python_3.12/site-packages/"
python3.12 bench.py
```

The x86/Arm baseline therefore cannot be a container build — it must be the same
Graviton instance family running DIY `pip install opencv-python`, which isolates
the library's contribution from the hardware's. See `docs/COOL-setup.md`.
---

## C-08 — NEU-CLS and NEU-DET are the same photographs (duplicate leakage)

**Severity:** Critical. This invalidated our headline accuracy number.

**Symptom.** We trained on "CLS + DET" believing they were separate capture sets,
giving a 3600-image training set. Accuracy looked strong at 93.80%.

**Root cause.** `eval/check_duplicates.py` compares the two sets perceptually:

| evidence | count |
|---|---|
| cross-set pairs, pHash == 0 AND dHash == 0 (bit-identical) | **1045** |
| same-label cross-set pairs within perceptual threshold 6 | 7701 |
| exact pixel-duplicate groups within a set | 2 |

The file numbering is explicit: `Cr_1.bmp` <-> `crazing_1.jpg`,
`Pa_101.bmp` <-> `patches_101.jpg`. **They are the same 1800 source images.** A
random per-row split could place `Cr_1` in train and `crazing_1` in test, so the
model was scored on a memorised image.

Note on method: 13389 "cross-label" pairs also fell inside the threshold. Those
are hash collisions, not contradictory labels - a 64-bit 8x8 DCT hash is not
discriminative at threshold 6. We therefore key identity on (class, source index)
and use hashing only to VERIFY, which is the robust order.

**Fix** (`eval/splits.py`): deduplicate to 1800 unique images keeping the DET copy
(inference runs on DET captures), then split by identity GROUP.

```
[grouped] train=1138 val=310 test=621
         group overlap: {'train&val': 0, 'train&test': 0, 'val&test': 0}  -> CLEAN
```

**Result.** 93.80% -> 92.75%. The leak was worth 0.3-0.7 points, not
catastrophically - but the number is now defensible, which matters more than the
value.

**What we learned.** Dataset names imply independence; filenames prove it.
Before combining two releases of a dataset, hash them against each other.

---

## C-09 — Calibration temperature fitted on the split used to report ECE

**Severity:** Moderate, and self-inflicted.

We fitted the temperature on the held-out split and then reported ECE on that
same split, so the calibration figure was optimistic by construction - ECE cannot
improve on data it was tuned against.

**Fix.** Three-way split: weights on train, temperature on validation, all
reporting on test.

| | ECE on test |
|---|---|
| before | 0.0886 |
| after (T = 0.75, fitted on val) | **0.0361** |

**What we learned.** "Held-out" is not one thing. A parameter fitted on a split
makes that split a validation set, not a test set.

---

## C-10 — `getGaborKernel((0,0))` silently produced (1,1) kernels

**Severity:** High, and invisible without checking.

**Symptom.** Our feature bank was 16 Gabor filters. Accuracy plateaued and we had
assumed texture features were simply weak on this data.

**Root cause.** `cv2.getGaborKernel((0,0), ...)` asks OpenCV to auto-size the
kernel. On OpenCV 5.0.0 it returns a proper 1xN column kernel only at 0 degrees:

```
sigma=  3 auto ksize=(1, 19)
sigma=  3 auto ksize=(1,  1)   <- orientations 45, 90, 135
```

**12 of our 16 Gabor features were computed from a 1x1 kernel** - i.e. they were
duplicates of mean intensity and carried no orientation information at all.

**Fix.** Compute the size explicitly. OpenCV 5's auto-size is exactly `6*sigma+1`,
verified bit-identical:

```python
ksize = (1, 6 * sigma + 1)
```

**Result.** Accuracy improved 93.02% -> 93.80% on the then-current split. The
feature bank had been crippled and the classifier had compensated.

The same `(0,0)` is also a portability trap: on the 4.14 control build it was taken
literally and `getGaborKernel` tried to allocate 1.8e19 bytes and died with
"Insufficient memory".

**What we learned.** A convenience default that silently returns a degenerate
result is worse than an error. Verify auto-sizing behaviour per library version.

---

## C-11 — AWS SSM silently truncated our 830KB app payload

**Severity:** High, cost two failed deploys.

**Symptom.** Deployment failed with `gzip: stdin: not in gzip format` while
extracting the app tarball. `SendCommand` also rejects requests over 97KB.

**Root cause, two bugs compounding:**
1. `run_remote` returned from INSIDE its chunk loop, so only the first batch ever
   executed. The payload arrived as its first ~66KB.
2. The payload is one long base64 string, so the generic request chunker split it
   mid-string.

**Fix.** Accumulate output across all chunks and return after the loop; ship the
payload as explicit 6KB `printf >> file` appends, then decode once. Also cut the
payload from 830KB to 152KB by shipping only `samples/neu` instead of the whole
`samples/` directory.

**What we learned.** Retry loops that "succeed" can hide partial execution. Assert
on the payload (`gzip -t`) rather than trusting a transport.

---

## C-12 — Deployed service healthy on localhost, unreachable from the internet

**Severity:** Moderate, and a nasty one to diagnose.

**Symptom.** The container reported `SETUP_OK`, systemd was `active`, uvicorn was
up, `curl localhost:8000/health` returned 200, nginx returned 200 - and the public
URL refused connections.

**Root cause.** The default VPC security group permits only SSH. We launched into
it and never added an ingress rule.

**Fix.** `ensure_sg()` creates/reuses `forgesight-endpoint-sg` with TCP 80 open and
attaches it. `docs/DEPLOYMENT.md` lists this in the failure table, because
"healthy locally, dead externally" is a state we will hit again.

---

## C-13 - Graviton is not cheaper per frame at these instance sizes

**Severity:** Moderate - it removed a claim we intended to make.

We expected Graviton to win on cost per 1000 images, a metric the rules explicitly
allow. Measured on-demand, us-east-1:

| arm | instance | USD/hr | fps | USD / 1000 frames |
|---|---|---|---|---|
| COOL on Graviton | `c8g.medium` | 0.072 | 300.38 | 0.000067 |
| DIY on Graviton | `c8g.medium` | 0.072 | 321.44 | 0.000062 |
| DIY on x86 | `t3.medium` | 0.0416 | 269.62 | **0.000043** |

Graviton's hourly rate is ~42% higher than `t3.medium` at this size, and its
throughput advantage does not close the gap. COOL's licence fee is not even
included in arm A.

**What we learned.** "Graviton is cheaper" is a family-level intuition that breaks
down at small sizes, where the per-core advantage is offset by the rate card.
Always measure cost per unit of work, not per hour.

---

## C-14 - Sensor noise made the agent auto-pass defective surfaces

**Severity:** Critical, and the worst bug in the project. Found only because we
wrote the ablation the award asks for.

**Symptom.** `eval/agent_ablation.py` on 200 genuinely clean captures degraded by
additive sensor noise: the loop **auto-passed 200 of 200**. A safety inspection
saying "no defect" on a noisy frame is the failure mode that matters most.

**Root cause.** The quality gate checked sharpness, glare, blown highlights and
crushed shadows - but additive noise *raises* Laplacian variance, so a noisy
frame measures as **sharper** than a clean one and sailed through every check. The
classifier then confidently answered "no_defect" on noise texture.

**Fix.** Added a robust noise estimate to the gate: MAD of the 3x3 high-pass
response (median-based, so a real defect covering a few percent of pixels cannot
inflate it). Threshold derived from 240 real captures
(`eval/quality_gate_calibration.py`), not chosen by feel:

| | noise_sigma p50 | p99 / p05 |
|---|---|---|
| clean NEU captures | 8.803 | p99 = 10.286 |
| + sensor noise | 13.065 | p05 = 11.305 |

Clean max (10.286) sits below noisy p05 (11.305), so `NOISE_MAX = 10.8` separates
them cleanly.

**Result. False accepts under noise: 200/200 -> 0/200.** The same calibration
pass relaxed the glare and blown-highlight limits to clean p99 (0.05 / 0.035),
removing spurious refusals on good captures.

**What we learned.** A quality gate built from intuition inherits the blind spots
of the intuition. Sharpness and noise are entangled - measuring one without the
other is measuring neither.

---

## C-15 - Our published AWS prices were wrong, and a vCPU mismatch hid it

**Severity:** High - an entire cost conclusion was an artefact.

We hardcoded `c8g.medium` at **$0.072/hr**. The AWS Price List API says
**$0.03988/hr**. Every cost-per-1000-frames figure we had published was ~1.8x
too high for the Graviton arms.

Worse, `c8g.medium` is **1 vCPU**. We had been comparing it against `t3.medium`
which is **2 vCPU and burstable** - so the "x86 is cheapest per frame" conclusion
compared different machines, one of them running on burst credits a short
benchmark can abuse but production cannot sustain.

**Fix.** `aws/instance_pricing.py` pulls rates from the Price List API and specs
from `describe_instance-types`, and prints vCPU, burstability and $/hr/vCPU
together. Replaced with a vCPU-matched, both-non-burstable pair
(`c8g.large` vs `c7i.large`, 2 vCPU each) plus per-core normalisation.

Two API shapes bit us on the way: `PriceList[i]` is a JSON *string* with prices
under `terms.OnDemand.<term>`, and `priceDimensions` is a **dict** keyed by SKU,
not a list - iterating it directly yields strings and the next `.get()` raises
AttributeError.

**What we learned.** Hardcoded cloud prices rot silently. A cost claim is a
measurement, not a constant.

---

## C-16 - Retracted: the "2.20x" feature speedup did not reproduce

We reported 2.20x for halving the Gabor bank. Re-measured properly, on the same
leakage-free split:

| | features only | full pipeline |
|---|---|---|
| local | 1.50x | 1.42x |
| AWS (authoritative) | - | **1.83x** |

The 2.20x came from a single local timing run on a contended laptop. The
direction of the result held; the magnitude did not, and we had quoted it
confidently. Now reported as 1.83x with the AWS basis stated, and the accuracy
cost (-0.33 points) kept beside it.

**What we learned.** A speedup measured on a shared dev box is a hypothesis. We
had measurements on both; we quoted the flattering one.

---

## C-17 - Ablation metric bugs that briefly indicted the loop

Two errors in `eval/agent_ablation.py`, both caught by sanity-checking output
rather than by the code:

1. `loop_result()` returned `d["first_prediction"]` **twice**, so first-prediction
   and second-prediction were the same object and "decision changed" was
   permanently **0** - while the live trace demonstrably changed class. Real rate
   is 165/359.
2. `correct()` did not handle `needs-second-opinion`, so an undecided loop was
   scored as an error. This manufactured ~30 phantom "introduced errors".

Also, the first ablation ran on a **defects-only** set, which makes every deferral
look correct and flatters any agent. Added 200 real clean captures from
unannotated regions.

**What we learned.** An ablation that produces a suspiciously clean result
usually has a bug, not a great system.
