# Benchmark Results — us-east-1, 2026-10-06

Real measurements on AWS. Raw JSON: `bench/results/comparison-us-east-1.json`.
Reproduce: §6. Format with `python bench\show.py`.

## Method

Workload = the ForgeSight inspection path on the 5 seeded sample images:
`resize -> cvtColor -> GaussianBlur -> Canny -> adaptiveThreshold -> morphologyEx
OPEN -> findContours x2`.

- **Median** of 30 iterations after 5 discarded warmup iterations.
- Inputs generated **on each instance** by a seeded generator (`SEED=20261006`),
  verified byte-identical. Unseeded noise would have made each arm measure
  different images and voided the comparison.
- **All three arms pinned to OpenCV 4.12.0** so the A-vs-B delta is the library,
  not a major-version change. See §5.

## 1. Results (c6g.large / c6i.large, 4 vCPU each)

| arm | instance | arch | library | OpenCV | fps | ms/frame |
|---|---|---|---|---|---|---|
| A | `c6g.large` | aarch64 | **COOL** | 4.12.0 | 226.73 | 4.41 |
| B | `c6g.large` | aarch64 | DIY pip | 4.12.0 | 202.02 | 4.95 |
| C | `c6i.large` | x86_64 (Xeon 8375C) | DIY pip | 4.12.0 | 257.36 | 3.89 |

| comparison | meaning | result |
|---|---|---|
| **A/B** | **library only** (same instance, same version) | **1.12x** |
| A/C | library + hardware | 0.88x |
| B/C | hardware only | 0.78x |

## 2. Per-op, COOL vs DIY on identical hardware (A vs B)

| op | COOL (A) | DIY (B) | COOL faster by |
|---|---|---|---|
| `GaussianBlur` | 0.1001 | 0.2910 | **2.91x** |
| `dilate` | 0.0792 | 0.1005 | 1.27x |
| `findContours` | 0.1002 | 0.1238 | 1.24x |
| `Canny` | 0.3357 | 0.4126 | 1.23x |
| `resize` | 0.0484 | 0.0493 | 1.02x |
| `adaptiveThreshold` | 2.9358 | 2.9077 | **0.99x** |

## 3. The honest read

**COOL wins 1.12x end-to-end, and the reason is instructive.**

COOL's per-op gains are large and consistent (1.2x–2.9x) on five of six
operations — `GaussianBlur` at 2.91x is a genuinely large win.

But **end-to-end is only 1.12x because `adaptiveThreshold` is ~67% of our
pipeline (2.94 ms of 4.41 ms), and COOL does not accelerate it at all (0.99x).**
One unaccelerated op dominating the profile caps the total. This is a real
property of our workload, not noise: the dark-blob detector added to fix the
false-negative in `docs/CHALLENGES.md` C-01 is what makes `adaptiveThreshold`
the hot spot.

Two further results reported as-is, without spin:

- **B/C = 0.78x**: our DIY Graviton run is *slower* than DIY Intel on this
  workload. Graviton is not automatically faster here.
- **A/C = 0.88x**: COOL on Graviton does not beat DIY on x86 in raw throughput.

Per-op hardware note (B vs C): Graviton wins `resize` (1.13x) and loses
`findContours` (0.51x) and `dilate` (0.40x).

**Where this argues for COOL:** workloads dominated by blur, edges and contour
work — the classic video pre/post-processing path the AWS blog measured, where
they reported ~30%. A pipeline dominated by one unoptimised operation sees much
less. Stating that plainly is more credible than quoting only the 2.91x.

## 4. Cost per 1000 frames

`cost = (usd_per_hour / 3600) x (1000 / fps)`

| arm | usd/hr | s/frame | usd/1000 frames |
|---|---|---|---|
| A COOL `c6g.large` | 0.072 | 4.410 | 0.0000882 |
| C DIY `c6i.large` | 0.085 | 3.885 | 0.0000917 |

A is marginally cheaper per 1000 frames despite lower fps, because Graviton is
cheaper per hour. The COOL AMI also carries a separate Marketplace licence fee,
so A's true cost is **higher** than shown — this table understates COOL's cost.

## 5. Compliance note: OpenCV version

The competition requires **OpenCV 5** for substantive analysis. Two facts:

1. **The application runs OpenCV 5.0.0** (`backend/app/vision.py`), verified via
   `/health` on every run.
2. **The COOL Graviton 2 AMI bundles OpenCV 4.12** — confirmed on the instance:
   `/opt/opencv_install/lib/libopencv_imgproc.so.4.12.0`, `cv2.__version__`
   reports `4.12.0`. Its libs are KleidiCV-accelerated.

Because of that, the benchmark pins all arms to 4.12.0. Comparing COOL 4.12
against DIY 5.x would have confounded the optimisation with a major version
change and produced a meaningless number.

**Next step to remove this caveat:** subscribe to **Cloud Optimized OpenCV for
AWS Graviton 4** (Marketplace, latest version 3.1, listed as OpenCV 5-based) and
re-run on `c8g`/`c8gn`. That yields an all-OpenCV-5 comparison.

## 6. Reproduction

```powershell
aws login
python aws\run_bench.py --region us-east-1 --opencv-version 4.12.0.88 `
    --arm "A=ami-03175684e1b021dda:c6g.large:cool" `
    --arm "B=ami-0bec8cef5313300ad:c6g.large" `
    --arm "C=ami-0045d7fc2ad003464:c6i.large"
python bench\show.py
```

AMI IDs (us-east-1):
- A `ami-03175684e1b021dda` — COOL Graviton2 (`OpenCV-graviton-optimized-AMI-prod-2jrbxn57qotcw`)
- B `ami-0bec8cef5313300ad` — Ubuntu 24.04 arm64 `20260923`
- C `ami-0045d7fc2ad003464` — Ubuntu 24.04 amd64 `20260923`

Helper tools: `aws/find_ami_types.py` (DryRun probe for supported instance
types) and `aws/probe_cool_ami.py` (discovers a COOL install layout).

## 7. Issues hit (also in docs/CHALLENGES.md)

| # | Problem | Fix |
|---|---|---|
| 1 | COOL AMI rejects `t4g.*`: `UnsupportedOperation` | COOL Graviton2 supports only m6g/c6g and minimum `.large`. Found with `aws/find_ami_types.py` (DryRun, no cost) |
| 2 | AMI snapshot is 40 GB, hardcoded 20 GB volume → `InvalidBlockDeviceMapping` | stop overriding the root volume; `--volume-size` opt-in |
| 3 | `Unsupported`: instance type not offered in that AZ | launcher walks candidate AZs (`us-east-1e` rejects `c6g.large`) |
| 4 | **`/opt/cool` does not exist on the Graviton2 image** | probed it: real layout is `/opt/opencv_install/{lib,lib/python3.12/site-packages}`. Detection now probes both paths |
| 5 | COOL reported as `build=DIY` | `detect_build()` checked only `/opt/cool`; would have mislabelled the COOL arm in the results file |
| 6 | `python3.12` ran but `cv2` missing | env exports do not reliably carry between separate RunShellScript command entries — inline them per command |
| 7 | `ImportError: libGL.so.1` | install `libgl1 libglib2.0-0t64` |
| 8 | samples missing on instance | generate on-instance from the seeded generator (SSM command size limit) |
| 9 | **unseeded RNG in the sample generator** | fixed seed; arms would otherwise benchmark different images |
| 10 | `exec > log` swallowed later output | scope redirect to a subshell |
| 11 | two JSON blobs in stdout → "Extra data" | `__BENCH_JSON_BEGIN__` / `__BENCH_JSON_END__` sentinels |
| 12 | SSM reported Success on a failed benchmark | capture `BENCH_EXIT`, surface remote logs, print diagnostics |
| 13 | version confound: COOL 4.12 vs DIY 5.x | `--opencv-version` pin |

## 8. Cost of these runs

8 short-lived instances (4 benchmark attempts + probes + a 2-arm run), all
auto-terminated in `finally` blocks; `$0.04`–`$0.20` at `c6g.large`/`c6i.large`
rates. Within the `$25` cap with no action triggered.