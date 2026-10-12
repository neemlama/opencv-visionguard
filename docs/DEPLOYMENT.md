# Deployment — verified on AWS

The demo endpoint and the COOL Award evidence run on the **same instance**: a
COOL Graviton4 AMI (`c8g.medium`), so the machine judges benchmark is also the
machine they click.

## Why the API runs OpenCV 5 while COOL sits on the same box

The competition requires OpenCV 5 for substantive analysis. COOL's images bundle
OpenCV 4.12 / 4.14. So the deploy installs `opencv-python==5.0.0.93` from pip and
**deliberately does not export COOL's `LD_LIBRARY_PATH` / `PYTHONPATH`** for the
service. COOL stays installed at `/opt/cool` and is used by `bench/bench.py`.

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
| `/inspect/loop` | full multi-step trace (below) |
| COOL present | `/opt/cool` |
| nginx | 200 on port 80 |

Closed-loop trace captured from the deployed instance:

```
1. iter 0: class=scratches confidence=0.404 margin=0.138 regions=3
   -> capture is usable; trust the classifier if confident enough
2. iter 0: confidence 0.404 below 0.55
   -> classifier not trusted on first look
   -> second opinion: CLAHE local-contrast stretch, then re-classify
3. iter 0: first look scratches @ 0.404
   -> apply CLAHE local-contrast stretch and re-analyse the new image
4. iter 1: class=crazing confidence=0.874 margin=0.826 regions=0
   -> capture is usable; trust the classifier if confident enough
5. iter 1: class=crazing conf=0.874
   -> queue human review with region (workorder WO-8cb28f)
=> FINAL: human-review  iterations=2  human=True
```

The first perception was low-confidence, so the agent took a *different*
perception and reached a different, much more confident answer. That is the
Agentic Vision criterion being satisfied, and it is reproducible from the trace.

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
