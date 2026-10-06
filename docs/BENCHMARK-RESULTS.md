# Benchmark Results — us-east-1, 2026-10-06

Real measurements on AWS. Reproduce with the commands in §4. Raw JSON in
`bench/results/comparison-us-east-1.json`.

## Method

Workload = the ForgeSight inspection path, on the 5 seeded sample images:
`resize -> cvtColor -> GaussianBlur -> Canny -> adaptiveThreshold -> morphologyEx
OPEN -> findContours x2`.

Timing uses the **median** of 30 iterations after 5 discarded warmup iterations,
so one scheduling hiccup on a shared instance cannot deflate a number.
`adaptiveThreshold` dominates at ~62% of pipeline cost, which matters for §3.

Inputs are **generated on each instance** by a fixed-seed generator
(`SEED = 20261006`), verified byte-identical across regenerations. This was a
real bug: the generator originally used unseeded `np.random`, so each arm would
have benchmarked *different images* and the comparison would have been
meaningless.

## 1. Measured results

| arm | instance | arch | build | OpenCV | fps | ms/frame |
|---|---|---|---|---|---|---|
| B | `t4g.small` | aarch64 | DIY pip | 5.0.0 | **212.70** | 4.70 |
| C | `t3.small` | x86_64 (Xeon Platinum 8259CL) | DIY pip | 5.0.0 | **194.39** | 5.14 |

**B/C = 1.09x** hardware-only (Graviton vs x86, identical library).

### Per-op, Graviton vs x86 (ms/call, DIY build on both)

| op | Graviton (B) | x86 (C) | B faster by |
|---|---|---|---|
| `resize` | 0.0500 | 0.0998 | **2.00x** |
| `GaussianBlur` | 0.1960 | 0.3888 | **1.98x** |
| `Canny` | 0.3179 | 0.3775 | 1.19x |
| `findContours` | 0.0995 | 0.0930 | 0.93x |
| `dilate` | 0.0827 | 0.0604 | 0.73x |
| `adaptiveThreshold` | 2.9270 | 2.1458 | **0.73x** |

## 2. Reading these numbers honestly

Graviton wins ~2x on `resize` and `GaussianBlur` but **loses ~0.73x on
`adaptiveThreshold` and `dilate`**. End-to-end that nets to only 1.09x, because
`adaptiveThreshold` is 62% of our pipeline and Graviton is slower at it.

This is the interesting finding, not a disappointment: `adaptiveThreshold` is
precisely one of the operations AWS says COOL optimises (resize, adaptive
Gaussian threshold, contour detection). Our DIY-Graviton arm is the **worst case
for COOL** — the hardware is weak exactly where the library is meant to help.
Arm A should therefore show a much larger per-op gain on `adaptiveThreshold`
than the 1.09x end-to-end figure, which makes it the arm that matters most.

## 3. COOL arm (A) — pending Marketplace subscription

Blocked on one console action: subscribe to the **Cloud Optimized OpenCV for AWS
Graviton** listing (seller: OpenCV, Ubuntu 24.04, ARM64, latest 3.1) so the AMI
becomes visible to this account. `aws ec2 describe-images --owners
aws-marketplace` currently returns nothing, which confirms no subscription yet.

Once subscribed:

```powershell
python aws\run_bench.py --region us-east-1 `
    --arm "A=<COOL_AMI_ID>:t4g.small:cool" `
    --arm "B=ami-0bec8cef5313300ad:t4g.small" `
    --arm "C=ami-0045d7fc2ad003464:t3.small"
python bench\show.py
```

Note arm B must stay `t4g.small` — same family as arm A — or the library delta is
confounded with the hardware delta.

## 4. Full reproduction

```powershell
aws login
# AMI ids used here (us-east-1, re-derive per docs/COOL-setup.md section 2):
#   arm64 Ubuntu 24.04 : ami-0bec8cef5313300ad  (ubuntu-noble-24.04-arm64-server-20260923)
#   amd64 Ubuntu 24.04 : ami-0045d7fc2ad003464  (ubuntu-noble-24.04-amd64-server-20260923)
python aws\run_bench.py --region us-east-1 `
    --arm "B=ami-0bec8cef5313300ad:t4g.small" `
    --arm "C=ami-0045d7fc2ad003464:t3.small"
python bench\show.py
```

The orchestrator drives instances over Systems Manager (no SSH keypair), walks
candidate AZs until one offers the instance type (`us-east-1e` rejects `t4g.*`),
and terminates every instance in a `finally` block.

## 5. Issues hit getting these numbers (see also docs/CHALLENGES.md)

| # | Problem | Fix |
|---|---|---|
| 1 | `ImportError: libGL.so.1` on headless host | install `libgl1 libglib2.0-0t64`; kept full `opencv-python` (not `-headless`) so the control matches AWS's published DIY baseline |
| 2 | `Unsupported`: `t4g.small` not offered in `us-east-1e` | launcher tries each candidate AZ in turn |
| 3 | samples not on the instance (`/samples`) | generate on-instance from the seeded generator instead of uploading (SSM commands are size-limited) |
| 4 | **unseeded RNG in sample generator** | fixed `SEED`; arms would otherwise benchmark different images |
| 5 | `exec > log` in the install step swallowed later output | scope the redirect to a subshell |
| 6 | two JSON blobs in stdout, `json.loads` "Extra data" | sentinels `__BENCH_JSON_BEGIN__` / `__BENCH_JSON_END__` |
| 7 | SSM `get_command_invocation` rejects `Filters`; `list_command_invocations` filter needs lowercase `key`/`value` scalar and rejects `*` | poll `list_command_invocations` unfiltered |

## 6. Cost of this run

2 instances x ~4 minutes + 20 GB gp3 each, auto-terminated. Under **$0.02**,
consistent with the $0.01 estimate in `docs/COST-GUARD.md`.