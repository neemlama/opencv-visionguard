# Benchmark Results — us-east-1, 2026-10-06

Real AWS measurements. Raw JSON: `bench/results/comparison-us-east-1.json`.
Format with `python bench\show.py`.

Two pipelines were benchmarked on identical hardware with identical library
versions. **They disagree, and that is the most useful thing we measured.**

---

## 1. Production pipeline — rolled-steel surface inspection (NEU)

The pipeline the product actually runs: 16 Gabor `filter2D` responses, 2 Sobel
gradients, magnitude, orientation histogram, 3 box blurs + `absdiff`, Laplacian,
and the localiser (blur + absdiff + threshold + morphology + connected
components).

| arm | instance | library | OpenCV | fps | ms/frame |
|---|---|---|---|---|---|
| A | `c8g.medium` (Graviton4) | **COOL** | 4.14.0-pre | 170.23 | 5.87 |
| B | `c8g.medium` (Graviton4) | DIY pip | 4.14.0 | 175.90 | 5.68 |

**A/B = 0.97x — parity. COOL is not faster on our production workload.**

### Why: one op is 82% of the pipeline and COOL does not touch it

| op | COOL (A) | DIY (B) | COOL faster by |
|---|---|---|---|
| **`gabor_filter2D_x16`** | **4.681** | **4.681** | **1.00x** |
| `sobel_x2` | 0.0282 | 0.0284 | 0.99x |
| `magnitude` | 0.0126 | 0.0126 | 1.00x |
| `blur_box9` | 0.0264 | 0.0239 | 0.95x |
| `blur_box21` | 0.0293 | 0.0277 | 0.95x |
| `blur_box41` | 0.0364 | 0.0339 | 0.93x |
| `absdiff` | 0.0033 | 0.0032 | 0.97x |
| `threshold` | 0.0054 | 0.0032 | **0.59x** |
| `morph_close` | 0.0143 | 0.0146 | 1.02x |
| `connected_components` | 0.0466 | 0.0493 | 1.06x |
| `laplacian` | 0.0131 | 0.0106 | 0.81x |
| `resize_128` | 0.0021 | 0.0028 | **1.33x** |
| `cvtColor` | 0.0129 | 0.0162 | **1.23x** |
| `canny_legacy` (unused) | 0.1886 | 0.2813 | **1.49x** |
| `adaptiveThreshold_legacy` (unused) | 0.1599 | 0.0892 | **0.56x** |

(ms/call. The two `legacy` ops are not in the production path; they are kept for
continuity and because `adaptiveThreshold` is COOL's clearest regression.)

COOL's wins are real — `Canny` 1.49x, `resize` 1.33x, `cvtColor` 1.23x — but on
this pipeline those operations together account for under 6% of runtime. The
dominant op is a 1xN convolution, and COOL leaves it untouched at exactly 1.00x.

**This is the honest conclusion: COOL's benefit is entirely determined by which
operations a workload happens to use, and the operation our classifier depends on
is not among them.**

---

## 2. Earlier weld pipeline — retained as a contrast, not a second product

The pre-rewrite pipeline was Canny/blur/contour dominated. Same instances, same
versions:

| arm | library | OpenCV | fps | ms/frame |
|---|---|---|---|---|
| A | COOL | 4.14.0-pre | 893.42 | 1.12 |
| B | DIY pip | 4.14.0 | 702.60 | 1.42 |

**A/B = 1.27x**, with `Canny` at 3.15x — exactly the op COOL accelerates.

Same hardware, same libraries, same measurement harness. **1.27x on one
pipeline, 0.97x on the other.** We report both; we do not ship only the flattering
one.

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
