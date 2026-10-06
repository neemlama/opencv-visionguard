# Benchmark Results — us-east-1, 2026-10-06

Real AWS measurements. Raw JSON: `bench/results/comparison-us-east-1.json`.
Format with `python bench\show.py`.

Two pipelines were benchmarked on identical hardware with identical library
versions. **They disagree, and that is the most useful thing we measured.**

---

## 1. Production pipeline — rolled-steel surface inspection (NEU)

The pipeline the product actually runs: **8** Gabor `filter2D` responses, 2 Sobel
gradients, magnitude, orientation histogram, 3 box blurs + `absdiff`, Laplacian,
and the localiser (blur + absdiff + threshold + morphology + connected
components).

| arm | instance | library | OpenCV | fps | ms/frame | USD/hr | **USD / 1000 frames** |
|---|---|---|---|---|---|---|---|
| A | `c8g.medium` Graviton4 | **COOL** | 4.14.0-pre | 300.38 | 3.33 | 0.072 | **0.000067** |
| B | `c8g.medium` Graviton4 | DIY pip | 4.14.0 | 321.44 | 3.11 | 0.072 | 0.000062 |
| C | `t3.medium` x86_64 | DIY pip | 4.14.0 | 269.62 | 3.71 | 0.0416 | **0.000043** |

| comparison | meaning | result |
|---|---|---|
| **A/B** | **library only** (same instance, same version) | **0.93x — COOL slower** |
| A/C | COOL on Graviton vs DIY on x86 | **1.11x — Graviton wins throughput** |
| B/C | DIY Graviton vs DIY x86 | 1.19x — the *hardware* wins, not COOL |

### The three honest conclusions

1. **COOL is not faster than stock OpenCV on our workload** (0.93x). The dominant
   op is `gabor_filter2D`, which COOL leaves untouched.
2. **Graviton is genuinely faster than x86 here** (1.19x DIY-on-DIY), and COOL
   inherits that: A/C = 1.11x. Any A-vs-C headline is mostly a hardware result.
3. **x86 is the cheapest per frame.** `t3.medium` ($0.0416/hr) is ~42% cheaper per
   hour than `c8g.medium` ($0.072/hr), and its throughput deficit does not close
   the gap: $0.000043 vs $0.000067 per 1000 frames. We tested this specifically
   because "Graviton is cheaper" is a common claim and **it does not hold at these
   instance sizes.** COOL's Marketplace licence fee is additional and is *not*
   included in arm A's figure, so A's true cost is higher still.

### Per-op, COOL vs stock on identical hardware (A vs B)

| op | COOL faster by |
|---|---|
| `canny_legacy` (not in pipeline) | **1.57x** |
| `cvtColor` | **1.24x** |
| `connected_components` | ~1.0x |
| `blur_box*`, `absdiff`, `morph_close` | ~1.0x |
| `gabor_filter2D` (dominant op) | **1.00x** |
| `adaptiveThreshold_legacy` (not in pipeline) | **0.57x** |

COOL's genuine wins (`Canny`, `cvtColor`) are not in our hot path, and the op that
*is* in our hot path is the one COOL does not accelerate.

### Caveats on these figures

- **Single run per arm.** The B/C per-op ratios (up to 3.5x on trivial ops) are
  larger than the end-to-end ratio and are probably inflated by run-to-run and
  burstable-instance variance. `t3.medium` is a burstable family. Treat the
  per-op table as indicative and the end-to-end numbers as the reliable ones.
- COOL's licence fee is excluded from arm A.
- Region: us-east-1. On-demand prices; no Savings Plans or Spot.

---

## 2. Feature-variant study — is a cheaper descriptor just as good?

Because COOL's benefit is op-dependent, the descriptor's composition determines
the answer. `eval/feature_variants.py` trained each variant on the same
leakage-free split:

| variant | Gabor filters | ms/frame | fps | accuracy |
|---|---|---|---|---|
| gabor16 (was production) | 16 | 12.589 | 79.4 | 95.37% |
| **gabor8 (now production)** | **8** | **5.723** | **174.7** | **95.37%** |
| gabor4 | 2 | 6.518 | 153.4 | 93.89% |
| gabor0 (gradients only) | 0 | 5.841 | 171.2 | 90.19% |

**Halving the Gabor bank is 2.20x faster at identical accuracy.** Dropping Gabor
entirely costs 5.18 points, so orientation-selective texture is doing real work —
it just does not need 16 convolutions.

We adopted `gabor8`. Two honesty notes:

- On the 6-class task above the accuracy was *exactly* equal. On the full 7-class
  task including the `no_defect` class, gabor8 scores **92.75%** vs **93.08%** for
  gabor16 — a real 0.33-point cost, not zero. Triage behaviour was unchanged
  (0 missed defects, auto-pass precision 1.000) and the clean auto-pass rate
  actually improved (42.0% → 45.7%).
- This was **not** chosen to flatter COOL. It was adopted because 2.2x throughput
  for 0.33 accuracy points is a good trade for a real-time inspection line, and
  the measurement is published either way. COOL's position on the cheaper pipeline
  is still parity (0.93x).

---

## 3. Earlier weld pipeline — retained as a contrast, not a product

The pre-rewrite pipeline was Canny/blur/contour dominated. Same instances, same
libraries, **before the Gabor kernel-size fix**, so it is a different pipeline
measuring different work:

| arm | library | OpenCV | fps | ms/frame |
|---|---|---|---|---|
| A | COOL | 4.14.0-pre | 893.42 | 1.12 |
| B | DIY pip | 4.14.0 | 702.60 | 1.42 |

**A/B = 1.27x**, with `Canny` at 3.15x — the op COOL accelerates.

Labelled explicitly: this is **not** the shipped pipeline and its numbers must not
be compared against section 1. It exists to show that COOL's benefit is entirely a
function of which operations a workload uses.

---

## 3. Why we did not tune the pipeline to flatter COOL

It would have been easy to drop the Gabor bank for cheaper features until COOL
looked good. That is benchmark-fitting, not engineering: it would optimise for
the measurement rather than the product. The steel classifier needs orientation
selective texture features, and Gabor is the standard tool for that, so it stays.

## 4. A lead, not a result

`gabor_filter2D_x16` uses 16 separate calls with 1xN column kernels. `filter2D`
appears not to use a separable fast path for these, which is why 16 small
convolutions cost 4.68 ms on a 128x128 image. Restructuring that is a genuine
optimisation worth doing on its own merits — for COOL *and* for stock OpenCV. We
have not done it, and it would help both arms equally, so it cannot change the
A/B conclusion.

---

## 5. Method

- Workload mirrors `backend/app/features.py` + `analysis.py` exactly.
- Inputs are **real NEU photographs**, shipped inline as base64, byte-identical
  on every arm.
- Median of 30 iterations after 5 discarded warmups.
- Both arms pinned to OpenCV `>=4.14.0,<4.15.0` so the delta is the library, not a
  major-version change. Pre-release strings like `4.14.0-pre` do not resolve on
  PyPI, hence a range.
- Library self-identifies from `/opt/cool` or `/opt/opencv_install`, so the JSON
  is itself proof of what executed.

## 6. Reproduction

```powershell
aws login
python aws\run_bench.py --region us-east-1 `
    --arm "B=ami-0bec8cef5313300ad:c8g.medium" `
    --arm "A=ami-01db31139bc5615d8:c8g.medium:cool"
python bench\show.py
```

AMI IDs (us-east-1):
| arm | AMI | note |
|---|---|---|
| A | `ami-01db31139bc5615d8` | `COOL-Graviton4-v2-AMI-prod-k6u24vxijyzvg`, OpenCV 4.14.0-pre |
| B | `ami-0bec8cef5313300ad` | Ubuntu 24.04 arm64 `20260923`, `opencv-python>=4.14.0` |

Helpers: `aws/find_ami_types.py` (DryRun instance-type probe, costs nothing),
`aws/probe_cool_ami.py` (discovers a COOL install layout).

## 7. Cost

`c8g.medium` at $0.072/h; each A+B pair ~$0.015 plus the COOL licence fee. Every
instance is terminated in a `finally` block.

## 8. Issues hit

`UnsupportedOperation` for `t4g.*` on COOL images · 40 GB snapshot vs a hardcoded
20 GB volume · `/opt/cool` absent on Graviton2 (real path `/opt/opencv_install`) ·
`detect_build()` mislabelling COOL as DIY · env exports not persisting between
SSM command entries · `libGL.so.1` missing · `exec >` swallowing later output ·
two JSON blobs in stdout · SSM reporting Success on a failed benchmark ·
**unquoted pip spec treated `>` as shell redirection** · **SendCommand 97 KB
parameter cap** · **`getGaborKernel((0,0))` returning `(1,1)` kernels on 12 of 16
orientations**, which silently crippled the feature bank until the kernel size was
made explicit.
