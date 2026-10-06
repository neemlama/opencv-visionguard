# COOL on AWS Graviton — Setup & Benchmark Guide

Everything here is verified against the AWS Marketplace listing and the AWS
Spatial/Physical AI blog "Accelerating OpenCV on Graviton – the COOL framework"
(April 2026). No invented AMI IDs: steps tell you how to look them up.

---

## 1. What COOL actually is (read this first)

| Property | Value |
|---|---|
| Delivery method | **Amazon Machine Image (AMI)** — *not* pip, *not* Docker |
| Architecture | 64-bit Arm (Graviton) |
| OS | Ubuntu 24.04 LTS |
| Based on | OpenCV 5, with **KleidiCV** optimizations |
| Optimized for | resize, adaptive Gaussian threshold, contour detection |
| Claimed gains | up to 70% faster on some ops, ~20% average; most modules >1.2x, many imgproc ops ~1.8x |

**Implication:** the COOL Award requires launching the Marketplace AMI on
Graviton and executing the workload *inside* it. There is no `pip install cool`.

### Marketplace listings
| Listing | Notes |
|---|---|
| Cloud Optimized OpenCV For AWS Graviton 2 | version 1.0 |
| Cloud Optimized OpenCV For Graviton3 | — |
| Cloud Optimized OpenCV For AWS Graviton4 | latest version 3.1 |

All under seller **OpenCV** on AWS Marketplace.

---

## 2. Finding the AMI ID for your region

```powershell
# 1) Subscribe / "Try for free" on the Marketplace listing page in the AWS console,
#    choosing your region. Do this BEFORE looking up the AMI ID.

# 2) Find the AMI the subscription granted you:
aws ec2 describe-images --owners aws-marketplace `
    --filters "Name=name,Values=*OpenCV*Graviton*" `
    --query "Images[].{Id:ImageId,Name:Name,Arch:Architecture}" --output table
```

Pick the one matching the Graviton generation of the instance family you will
launch (Graviton2 → `t4g`/`m6g`, Graviton3 → `c7g`/`m7g`, Graviton4 → `c8g`).
Copy the AMI ID into `.env` (see §6).

---

## 3. Inside the instance: activating COOL

COOL lives at `/opt/cool`. Two environment variables are required:

```bash
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/cool/python_3.12/site-packages/"
python3.12 bench.py
```

Confirm COOL is actually loaded (not falling back to a pip build):

```bash
python3.12 -c "import cv2, os; print(cv2.__version__); print(os.environ.get('PYTHONPATH'))"
python3.12 -c "import cv2; print(cv2.getBuildInformation())"
```

`bench.py` prints `"build": "COOL"` only when `/opt/cool` is present on disk, so
the JSON output is itself proof of which library ran.

> Note: COOL is based on OpenCV 5. On our local Windows dev box we also run
> OpenCV 5.0.0, so the algorithm under test is the same major version on both
> sides of the comparison.

---

## 4. The three-way benchmark (this is the COOL Award evidence)

A two-way COOL-vs-x86 comparison confounds *library* and *hardware*. We run
three arms on the same workload and the same images:

| Arm | Instance | Library | Answers |
|---|---|---|---|
| A | Graviton (`t4g`/`c7g`/`c8g`) | **COOL** (Marketplace AMI) | Award eligibility: COOL on Arm |
| B | Graviton, *same family* | DIY `pip install opencv-python` | Isolates the library's contribution |
| C | x86 (`t3`/`c7i`) | DIY `pip install opencv-python` | Hardware baseline + Arm/x86 coexistence |

Arm B is the honest control and matches AWS's own published methodology (they
compared COOL against DIY pip OpenCV *on the same Graviton4 instance type*).

Because COOL only exists as an AMI, arm B must **not** be a container build —
build arm B from the same Ubuntu 24.04 base so the only difference is the
library.

Metrics recorded per arm:
- wall-clock for the workload, ops/sec, equivalent fps
- **per-operation** timings for `resize`, `GaussianBlur`, `adaptiveThreshold`,
  `Canny`, `findContours`, `dilate` (the ops COOL actually optimizes)
- CPU model + instance type + library build path
- estimated USD cost per 1,000 frames from on-demand rates in that region

---

## 5. Cost model

`t4g.*` is typically ~20% cheaper per hour than the equivalent `t3.*`, and
Graviton offers better price/performance. So the win compounds:

```
cost_per_1000_frames = (usd_per_hour / 3600) * (1000 / fps)
```

Substituting the measured `fps` per arm is the whole argument. Fill the
`usd_per_hour` from the pricing page for **your** region rather than trusting a
number written here.

---

## 6. Automating it

```powershell
# One-time: AWS CLI login for this shell
aws login

# Launch both arms, run the workload, collect JSON, terminate.
# Uses AWS Systems Manager (no SSH keypair needed).
python aws/run_bench.py --region ap-south-1 --ami-arm64 $env:COOL_AMI `
    --instance-arm64 t4g.small --instance-x86 t3.small
```

`aws/run_bench.py` launches each instance, waits for an SSM-registered managed
instance, sends the workload, downloads `bench_result.json`, then terminates the
instance. Both instances exist only for the duration of the run, so the
comparison is cost-trivial (< $0.10 typical).

---

## 7. Reproducibility checklist for the judges

- [ ] COOL version string and `/opt/cool` path printed in arm A output
- [ ] Exact AMI ID, instance type, and region recorded
- [ ] Same workload script + same input images on all three arms
- [ ] Warmup iterations discarded, N iterations measured, variance reported
- [ ] `pip freeze` from each arm
- [ ] Raw JSON committed alongside the report

---

## 8. Reference implementation

The OpenCV organization publishes its own harness, which we cite rather than
reinvent: `github.com/opencv/COOL-Benchmark` — spins up `t4g`/`m7g`/`c7g`/`m6i`,
runs a configurable pipeline from an in-memory asset pool (no S3/disk noise), and
reports throughput, latency and cost side by side. It takes the AMI via
`MARKETPLACE_AMI_ID` and deploys `opencv-mcp-server.py` to each instance at
runtime, which is also a path to an MCP-based agentic demonstration.