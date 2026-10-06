# Email to competition@opencv.org — draft, not sent

I cannot send email from this environment. Review, fill in the bracketed fields,
and send. It is deliberately short and asks three answerable questions rather
than arguing a case.

**To:** competition@opencv.org
**Subject:** OpenCV AI Competition 2026 — COOL / OpenCV 5 eligibility question (team ForgeSight)

---

Hello,

Team ForgeSight is preparing an entry for the OpenCV AI Competition 2026 and has
three eligibility questions about the Best Use of COOL Award. We would rather ask
now than submit something that turns out to be ineligible.

**1. Our COOL image is OpenCV 4.14, while the listing describes OpenCV 5.**

We subscribed to *Cloud Optimized OpenCV For AWS Graviton 4* and launched the AMI
we were granted: `ami-03175684e1b021dda`-family AMI named
`COOL-Graviton4-v2-AMI-prod-k6u24vxijyzvg`. On that instance:

```
LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib PYTHONPATH=/opt/cool/python_3.12/site-packages \
  python3.12 -c "import cv2; print(cv2.__version__)"
# -> 4.14.0-pre
# /opt/cool/cpp_sdk/lib/libopencv_imgproc.so.4.14.0
```

However, the Marketplace page for the Graviton 4 product lists *Latest version:
Cloud Optimized OpenCV 3.1*, and the v3.1 release notes say:

> "Version 3.1 of OpenCV Graviton Optimized, now built on OpenCV 5.0 ... upgrading
> the core library from OpenCV 4.0 to 5.0"

Is there a v3.1 AMI we should be launching instead of the v2 build our
subscription granted? We want to be benchmarking the current release.

**2. Does a COOL benchmark on OpenCV 4.14 satisfy the OpenCV 5 requirement?**

The competition states every entry must use OpenCV 5 for substantive image or
video analysis. Our application runs OpenCV 5.0.0 on every request (verifiable via
a `/health` endpoint on our deployed instance, which reports `5.0.0`). Our COOL
benchmark, however, necessarily runs on 4.14 because that is what the COOL image
provides, and we deliberately pin the DIY control to the same version so the
measured delta is attributable to the library rather than to a version change.

Is that acceptable — OpenCV 5 in the application, with the COOL measurement
isolated to a version-matched comparison? Or must the COOL arm itself be OpenCV 5?

**3. May the COOL claim rest on the benchmark workload?**

The award requires that "COOL must execute the claimed core workload on AWS
Graviton". In our system the production API runs pip OpenCV 5, so COOL executes
the *benchmark* workload rather than the live request path. Would you consider
that sufficient, or do you expect COOL to be running the deployed application
itself?

For context on what we are submitting, so you can judge the answers in context:
our entry is agentic visual inspection of rolled-steel surface defects, evaluated
on the public NEU surface-defect dataset (1800 unique photographs, deduplicated
and split by source image so no duplicate straddles train and test). Our measured
COOL result on that workload is **parity (0.97x)**, not a speedup — the operation
that dominates our pipeline, Gabor `filter2D`, is not accelerated by COOL, while
COOL does accelerate `Canny`, `resize` and `cvtColor`. We report that honestly
rather than selecting a workload that flatters it. We would still like to
understand whether that evidence is eligible.

Happy to provide the AMI ID, instance types, raw benchmark JSON and the exact
activation commands if that would help you check.

Thank you,
[Name]
Team ForgeSight
[contact / GitHub or repo link]

---

## Notes for us

- **Question 1 is the one to watch.** If a v3.1 AMI exists and is obtainable, the
  cleanest outcome is a COOL run on OpenCV 5, which dissolves questions 2 and 3.
  We checked `aws ec2 describe-images --owners aws-marketplace` after subscribing
  and only the v2 AMI was visible to our account.
- There is also a **Docker** listing, *Cloud-Optimized OpenCV for AWS Graviton 4
  [Docker]*, described as "Hardware-optimized OpenCV 5.x build ... with integrated
  KleidiCV, ARMPL, and Python 3.10, 3.11, and 3.12". Worth subscribing to as a
  fallback if an AMI route is unavailable.
- **Do not overstate the COOL result** in the submission. Our own measurement is
  parity on the production workload; the 1.27x figure came from an earlier,
  Canny-dominated pipeline and must be labelled as a different pipeline if used at
  all (see `docs/BENCHMARK-RESULTS.md`).
