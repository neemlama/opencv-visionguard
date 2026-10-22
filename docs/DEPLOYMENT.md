# Deployment — verified on AWS

The demo endpoint and the COOL benchmark run on the **same machine**: a COOL
Graviton4 AMI (`c8g.medium`), so the hardware a judge clicks is the hardware the
A/B was measured on. They do not run the same *library*, and the next section is
explicit about that.

## Why the API runs OpenCV 5 while COOL sits on the same box

The competition requires OpenCV 5 for substantive analysis. The COOL Award
requires COOL to execute the claimed core workload. Both can hold in one process
only if the image carries COOL *and* OpenCV 5. Our COOL images bundle 4.12 / 4.14,
so the deployed API installs `opencv-python==5.0.0.93` from pip and **deliberately
does not export COOL's `LD_LIBRARY_PATH` / `PYTHONPATH`** for the service. COOL
stays installed at `/opt/cool` and is used only by `bench/bench.py`.

Stated plainly: **the deployed API does not execute COOL**, so the A/B in
`docs/BENCHMARK-RESULTS.md` is a library-only measurement and the COOL Award's
"COOL executes the claimed core workload" criterion is not met by this deployment.
`Graviton5-COOL-v3` (`ami-033e481a24f94c8cb`) does carry
`libopencv_*.so.5.1.0`, so a single deployment could satisfy both; that is not
what is running, and this document does not claim otherwise.

Verified on the running instance:

```
/health  ->  {"ok":true,"opencv":"5.0.0", ...}
ls -d /opt/cool  ->  /opt/cool
```

Both are true at once, and `/health` is the check a judge can repeat.

## Commands

```powershell
aws login
python aws\deploy_endpoint.py up --ttl-hours 3     # launch + install + verify
python aws\deploy_endpoint.py status               # URL, uptime, accrued cost
python aws\deploy_endpoint.py down                 # terminate
```

Bring it back up on demand any time; the instance is disposable.

## What `up` does

1. **Budget pre-flight** — refuses to launch if the forecast plus this run would
   exceed the `$25` cap.
2. Creates/reuses an SSM instance profile (no SSH keys, no inbound SSH).
3. Creates/reuses security group `forgesight-endpoint-sg` with **port 80 open**.
   The default VPC security group allows only SSH, which is why an earlier
   deployment was healthy on `localhost` but refused external connections.
4. Launches `c8g.medium` from the COOL Graviton4 AMI, walking AZs until one
   offers the type (`us-east-1e` does not).
5. Ships the app as a base64 tarball over SSM, **in explicit 6 KB appends** — a
   single ~150 KB command gets split by the request chunker and arrives corrupt.
6. Installs nginx + a systemd unit, then verifies `/health`, the OpenCV version,
   and COOL's presence.
7. Prints the URL and a hard deadline. It does **not** stop itself.

## Verified behaviour on AWS

`c8g.medium`, COOL Graviton4 AMI, us-east-1:

| check | result |
|---|---|
| `/health` external | `{"ok":true,"opencv":"5.0.0", ...}` |
| `/samples` | 6 labelled NEU images served |
| `/inspect` on a real NEU photograph | classified, localised, decided |
| `/inspect/loop?preset=soft_focus` | recaptures, re-renders, settles (below) |
| `/approve/{id}` | 404 on an unknown id, 409 on a second decision |
| COOL present on disk | `/opt/cool` (benchmark only, not the API runtime) |
| nginx | 200 on port 80 |

Closed-loop trace from the **current build**, 2026-10-22. Reproduce it with
`python eval/demo_loop.py --image inclusion_1.jpg --preset soft_focus`; the HTTP
route `/inspect/loop?preset=soft_focus` produces the same trace, and
`eval/smoke_test.py` asserts that it does. Work-order ids differ per run.

```
1. iter 0: sharpness=1.16 glare=0.0 blown=0.0
   -> capture is not fit to judge - refuse to classify
   -> recapture with: steady_and_refocus, boost_contrast
2. iter 0: applied 'steady_and_refocus' -> focus_blur 4.0 -> 0.0
   -> capture settings corrected, scene re-rendered
   -> run perception again on the recaptured frame
3. iter 1: class=inclusion confidence=0.416 margin=0.104 regions=2
   -> capture is usable; trust the classifier if confident enough
   -> predict inclusion at 41.6%
4. iter 1: confidence 0.416 below 0.55
   -> classifier not trusted; second opinion disabled by default (net-negative)
   -> queue human review (workorder WO-xxxxxx)
5. iter 1: settled on 'human-review'
=> FINAL: human-review  iterations=2  human=true  final focus_blur=0.0
```

The capture is refused, the *camera setting* changes, the scene is rendered
again from the corrected setting, and the second perception is a different
frame. That is the Agentic Vision criterion being satisfied, and it is
reproducible from the trace and from the test.

An earlier version of this block showed a trace whose second step was a CLAHE
second opinion. That path is disabled by default (`ENABLE_SECOND_OPINION =
False`, measured net-negative), so the trace could no longer be produced by the
shipped code. It has been replaced rather than left standing.

## Operating schedule and cost

`c8g.medium` is **$0.03988/hr**, not the $0.072/hr we used to assume. The Price
List API figure is $0.03988 and `docs/BENCHMARK-RESULTS.md` §2 retracted the
0.072 number; this file had drifted and is corrected here. At the corrected rate:

| window | hours | cost |
|---|---|---|
| now (Oct 12) → submission (Oct 26), continuous | 336 | **$13.40** |
| judging (Oct 27) → Nov 9, continuous | 312 | **$12.44** |
| now → Nov 9, continuous (the whole period) | 672 | **$26.80 — over the $25 cap** |

So a continuously-alive endpoint is affordable up to submission but **not**
through the full judging window. Recommended schedule:

| phase | when | mode | cost |
|---|---|---|---|
| build | now – Oct 26 | on demand, short TTL | ~$0.10 total so far |
| submission | Oct 26 | **up for the video / screen-share** | < $1 |
| judging | Oct 27 – Nov 9 | up during judging hours, down nightly | ~$5–10 |

One 10-hour judging day at the corrected rate is **$0.40**.

```powershell
aws login                                                    # session expires ~every few hours
python aws\deploy_endpoint.py up --ttl-hours 10              # covers a judging day
python aws\deploy_endpoint.py down                           # stop nightly
```

Note that an **expired AWS session is the most likely cause of a failed
submission-time deploy**, and it is silent until a call is made. Re-run
`aws sts get-caller-identity` before relying on the endpoint.

Cost so far for all AWS work (benchmark runs, probes, three endpoint deploys):
**under $1**, against a `$25` cap with no alarm triggered.

## Security notes

- No inbound SSH. SSM (SSM-managed instances) is the only control channel.
- IMDSv2 required (`HttpTokens: required`) on every instance we launch.
- No credentials, keys or secrets in the image or the repo.
- Security group exposes only TCP 80. Tighten to a judge IP for the live window
  if preferred.
- Human approval is a required step for any escalation — the model cannot close
  a work order.

## If the deploy fails

| symptom | cause | fix |
|---|---|---|
| `InvalidBlockDeviceMapping` | volume smaller than the AMI snapshot (COOL G4 is 50 GB) | do not override `BlockDeviceMappings` |
| `gzip: stdin: not in gzip format` | app payload split mid-string | ship as appends, not one command |
| `UnsupportedOperation` | instance type not offered in that AZ | launcher walks AZs |
| healthy on `localhost`, refused externally | default SG allows only SSH | attach `forgesight-endpoint-sg` |
| `/health` empty right after setup | systemd still starting | re-check; service is `Restart=always` |
| UnicodeEncodeError printing remote output | Windows cp1252 console | `sys.stdout.reconfigure(errors="replace")` |
