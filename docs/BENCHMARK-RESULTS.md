# Benchmark Results — us-east-1

Real AWS measurements. Raw JSON: `bench/results/comparison-us-east-1.json`.
Format with `python bench\show.py`. Prices from the AWS Price List API via
`python aws/instance_pricing.py`.

---

## 1. Definitive benchmark — COOL on OpenCV 5, correct prices, vCPU disclosed

All arms run **OpenCV 5**. COOL comes from the Marketplace AMI
`ami-033e481a24f94c8cb` ("Graviton5-COOL-v3"), verified on-instance as
`libopencv_*.so.5.1.0` / `cv2.__version__ == 5.1.0-dev`. The DIY control is
`opencv-python==5.0.0.93` — same major, closest stable release.

| arm | instance | vCPU | burstable | library | OpenCV | fps | ms/frame | USD/hr | **USD/1000 frames** |
|---|---|---|---|---|---|---|---|---|---|
| A | `c8g.medium` | 1 | no | **COOL** | 5.1.0-dev | 302.60 | 3.31 | 0.03988 | 3.7e-05 |
| B | `c8g.medium` | 1 | no | DIY pip | 5.0.0 | 321.79 | 3.11 | 0.03988 | **3.4e-05** |
| D | `c8g.large` | 2 | no | DIY pip | 5.0.0 | 330.43 | 3.03 | 0.07976 | 6.7e-05 |
| C | `c7i.large` | 2 | no | DIY pip | 5.0.0 | 659.02 | 1.52 | 0.08925 | 3.8e-05 |

### The library question (COOL Award): A vs B

**A/B = 0.94x — COOL is 6% *slower* than stock OpenCV** on identical hardware,
identical OS, same OpenCV major version.

This is now a **version-independent** result. We measured COOL on two different
images and both agree:

| COOL image | bundled OpenCV | A/B library-only |
|---|---|---|
| `COOL-Graviton4-v2-AMI` | 4.14.0-pre | 0.93x |
| `Graviton5-COOL-v3` | **5.1.0-dev** | **0.94x** |

Per-op, COOL's wins are real but not in our hot path: `canny_legacy` 1.57x,
`cvtColor` 1.24x, `connected_components` ~1.0x — while the dominant op,
Gabor `filter2D`, is **1.00x**. `adaptiveThreshold` remains a COOL regression
(0.57x).

### The hardware/cost question

Two earlier claims of ours were **wrong** and are corrected here.

**Wrong price.** We hardcoded `c8g.medium` at $0.072/hr. The Price List API says
**$0.03988/hr**. Every cost figure we had published was ~1.8x too high for the
Graviton arms.

**Wrong vCPU matching.** `c8g.medium` is **1 vCPU**; the `t3.medium` we compared
against is **2 vCPU and burstable**, so the earlier "x86 is cheapest" claim
compared different machines, one of them on burst credits.

Corrected readings:

| comparison | vCPU-matched? | result |
|---|---|---|
| D vs C (`c8g.large` vs `c7i.large`, 2 vCPU, both non-burstable) | yes | Intel **2.0x faster** (659 vs 330 fps) |
| per-core: B (321.79 fps / 1 vCPU) vs C (659.02 / 2 vCPU) | yes | 322 vs 330 fps/vCPU — **within 2.4%** |
| price per vCPU | yes | Graviton $0.03988 vs Intel $0.04462 — **Graviton 11% cheaper** |
| cheapest USD/1000 frames | not matched | B (Graviton 1 vCPU) $3.4e-05 |

**Honest summary:** per core the two are equivalent and Graviton is ~11% cheaper
per vCPU, so Graviton wins narrowly on price-performance. In absolute throughput
at matched vCPU, Intel wins decisively. Graviton's advantage is that it is
*cheaper per unit of core*, not that it is faster per core.

### Caveats

- **Single run per arm.** Treat per-op ratios on cheap ops as indicative.
- COOL's Marketplace licence fee is **not** in arm A's cost.
- COOL is `5.1.0-dev`, a pre-release. The control is `5.0.0` stable. Same major
  version, different minor.
- Region us-east-1, on-demand, no Savings Plans or Spot.

---

## 2. Feature-variant study — what the speedup actually measures

Earlier we quoted **2.20x**. That did **not reproduce**; it was a single local
timing run on a contended laptop. Measured properly:

| variant | Gabor | features only | full pipeline | accuracy (6-class) |
|---|---|---|---|---|
| gabor16 | 16 | 8.651 ms (115.6 fps) | 10.349 ms (96.6 fps) | 95.37% |
| **gabor8 (shipped)** | **8** | **5.769 ms (173.4 fps)** | **7.309 ms (136.8 fps)** | **95.37%** |
| gabor4 | 2 | 4.411 ms (226.7 fps) | 5.556 ms (180.0 fps) | 93.89% |
| gabor0 | 0 | 4.092 ms (244.4 fps) | 5.091 ms (196.4 fps) | 90.19% |

**Local: 1.50x features-only, 1.42x full-pipeline. AWS: ~1.83x full-pipeline**
(5.68 ms → 3.11 ms on `c8g.large`/`c8g.medium`, medians of 30 after discarded
warmup). The AWS figure is the one to quote.

Accuracy cost on the full 7-class task including `no_defect`:
**93.08% → 92.75% (−0.33 points)**. Triage behaviour unchanged (0 missed defects,
auto-pass precision 1.000). Dropping Gabor entirely saves little more and costs
5.18 points, so orientation-selective texture earns its place — it just does not
need 16 convolutions.

Adopted because it is better product, not to flatter COOL. COOL is still 0.94x on
it.

---

## 3. Earlier weld pipeline — contrast only, not the product

Pre-Gabor-fix, Canny/blur/contour dominated. Different pipeline measuring
different work; **its numbers must not be compared to section 1**:

| arm | library | fps | ms/frame |
|---|---|---|---|
| A | COOL 4.14.0-pre | 893.42 | 1.12 |
| B | DIY 4.14.0 | 702.60 | 1.42 |

**A/B = 1.27x**, `Canny` at 3.15x — the op COOL accelerates. Retained only to
show that COOL's benefit is entirely a function of which operations a workload
uses.

---

## 4. Reproduction

```powershell
aws login
python aws\instance_pricing.py     # verify prices and vCPU counts
python aws\run_bench.py --region us-east-1 --opencv-version 5.0.0.93 `
    --arm "A=ami-033e481a24f94c8cb:c8g.medium:cool@0.03988" `
    --arm "B=ami-0bec8cef5313300ad:c8g.medium@0.03988" `
    --arm "D=ami-0bec8cef5313300ad:c8g.large@0.07976" `
    --arm "C=ami-0045d7fc2ad003464:c7i.large@0.08925"
python bench\show.py
```

COOL AMI versions available to this account:

| AMI | name | OpenCV |
|---|---|---|
| `ami-033e481a24f94c8cb` | Graviton5-COOL-v3 | **5.1.0-dev** |
| `ami-01db31139bc5615d8` | COOL-Graviton4-v2-AMI | 4.14.0-pre |
| `ami-0b775ed8c83396c73` | COOL-V3 (G2 family) | 4.12.0 |

## 5. Cost of all benchmarking

Roughly 20 short-lived instances across the project, all terminated in `finally`
blocks. **Under $1** against a $25 cap with no alarm triggered.
