# Cost Guard — ForgeSight on AWS

Account `557723775608`, region `us-east-1`. Set up 2026-10-06.
Standing policy agreed with owner: **on-demand now, always-on during judging.**

---

## 1. Hard cap: $25/month

Budget **"ForgeSight Hackathon Cap"** — COST, MONTHLY, $25.00, scope: **all
services (account-wide)**.

Account-wide scope is deliberate. A tag-scoped budget is cheaper but *blind to
untagged spend*, so an accidental `run_instances` without our tag would be
invisible. A hard cap has to see everything.

Four alarms, all confirmed `OK`:

| Type | Threshold | Value | Purpose |
|---|---|---|---|
| ACTUAL | 50% | $12.50 | early warning, time to react |
| ACTUAL | 80% | $20.00 | act now |
| ACTUAL | 100% | $25.00 | cap breached |
| FORECASTED | 100% | $25.00 | **the one that prevents an overrun** — fires before the money is gone |

Alarms publish to SNS topic `forgesight-budget-alerts`.

> **Action needed:** subscribe your email so you actually receive them.
> `python aws/setup_budget.py --limit 25 --email you@example.com`
> SNS then sends a confirmation email that must be clicked.

---

## 2. The alarm is not the guard — because alarms lag

AWS Budget notifications are email-based and **can lag by hours**. A runaway
instance keeps billing in that window. Two layers close that gap.

### 2a. Budget action (armed, but semantics unverified)

```
actions : APPLY_IAM_POLICY / ApprovalModel=AUTOMATIC / STANDBY / 100%
```

An action **is** armed and fires without human approval.

**Correction to an earlier note in this file:** we first concluded that no
auto-stop action could be armed on this account, based on the shape of
`CreateBudgetAction.Definition`:

```
IamActionDefinition  ->  {PolicyArn, Roles, Groups, Users}
SsmActionDefinition  ->  {ActionSubType, Region, InstanceIds}
ScpActionDefinition  ->  {PolicyId, TargetIds}
```

The documented tag-scoped EC2 stop (`IAMAction` with `InstanceTypes` + `Tags`) is
genuinely **absent from this API model**. That part was right. But the
`IamActionDefinition` shape *is* accepted, so the action was created
successfully — our earlier "cannot be armed" claim was wrong.

What remains genuinely unverified: whether that action actually **stops EC2
instances** on this account, since the only role reference available is
`Roles=["forgesight-budget-stopper"]` with no instance-type or tag selector. We
have not observed it fire. Treat it as a best-effort net, not the primary guard.

`create_budget_action` is **not idempotent** — re-running stacks duplicates, so
the script deletes existing actions before creating exactly one.

### 2b. Code-level enforcement (primary)

`aws/endpoint.py` enforces the cap *at launch time*:

| Mechanism | What it does |
|---|---|
| pre-flight | refuses to launch when `forecast + this run > cap`; override requires explicit `--force` |
| TTL | every launch is time-boxed; the hard deadline is printed at launch |
| cost tag | every instance carries `CostBudgetTag=forgesight` so stop/status can find it |
| `stop` | terminates every tagged instance |

`aws/run_bench.py` terminates its instances in a `finally` block, so a crashed
benchmark cannot leave instances running. Verified after the first real run:

```
$ aws ec2 describe-instances --filters instance-state-name=running,pending,...
[]
```

### Verified behaviour

```
$ python -c "... preflight(0.0776, 480)"   # 20 days of c8g.medium
[budget] cap=$25.00  spent=$0.00  forecast=$0.00
[budget] this run +$37.2480 (480h @ $0.0776/h) -> projected $37.25
[budget] REFUSED: projected $37.25 exceeds cap $25.00      -> False

$ ... preflight(0.0168, 4)                 # 4h t4g.small
[budget] OK: $24.93 headroom                                  -> True
```

The guard refuses the $37 case and allows the $0.07 case.

---

## 3. Cost model

us-east-1 on-demand list prices.

| Item | Rate | Cost |
|---|---|---|
| 3-arm COOL benchmark | 3 instances × ~10 min | **~$0.01** |
| COOL AMI licence | hourly, on top of EC2 | small; per Marketplace listing |
| Demo endpoint, 4 h on-demand | t4g.small $0.0168/h | $0.07 |
| Demo endpoint, always-on 20 days | t4g.small × 480 h | $8.06 |
| Always-on on `c8g.medium` instead | $0.0776/h × 480 h | $37.25 — **refused by guard** |
| EBS 20 GB gp3 | ~$0.16/month | $0.16 |
| Elastic IP | $0.005/h while attached | $0.02 per 4 h |

The benchmark is effectively free. **Only a long-lived endpoint has real cost**,
which is why on-demand is the default and the TTL is short.

Note `t4g.*` is ~20% cheaper per hour than the equivalent `t3.*`, so Graviton
compounds the win — relevant to the COOL benchmark numbers too.

---

## 4. Why on-demand is safe for this competition

The rules require *"a working web endpoint **or** an arranged live screen-share
demonstration."* A permanently running server is **not** mandatory.

So the plan costs ~$0 while building, and only switches to always-on when judging
opens:

| Phase | Dates | Mode | Cost |
|---|---|---|---|
| Build | Oct 6 – 26 | on-demand, 4 h TTL | ~$0.10 |
| Judging | Oct 27 – Nov 9 | always-on for judges | ~$7–8 |

Judging-period switch-over (do it once, on Oct 27):

```powershell
python aws\endpoint.py start --ami-ubuntu-arm64 ami-... `
    --instance-type t4g.small --ttl-hours 400 --eip
python aws\endpoint.py status      # note the URL for the report
```

Then to be sure nothing runs past the deadline:

```powershell
python aws\endpoint.py stop
```

---

## 5. Routine commands

```powershell
python aws\endpoint.py status   # cap, month-to-date, uptime, accrued cost, URL
python aws\endpoint.py stop     # terminate every CostBudgetTag=forgesight instance
python aws\setup_budget.py --limit 25 --email you@example.com   # re-arm / verify
aws ce get-cost-and-usage --time-period Start=2026-10-01,End=2026-10-07 `
    --granularity MONTHLY --metrics BlendedCost --group-by Type=DIMENSION,Key=SERVICE
```

Cost Explorer data lags by a few hours to a day, so `status` showing
`month-to-date=$0.00` right after a launch is expected, not a bug.

---

## 6. Known gaps

- **No hard kill switch.** If an instance runs past its TTL, nothing stops it
  automatically — the TTL prints a deadline, it does not enforce one. A nightly
  EventBridge stop rule would close this gap and needs
  `scheduler.amazonaws.com` + Lambda permissions.
- **Alarms need an email subscriber** to be useful; pending.
- **Forecast is inert** on this account: `get_cost_forecast` requires Start =
  today (its earliest supported date is derived from available billing data) and
  returns `ForecastResults=None` because there is almost no usage history. The
  guard therefore relies on actual month-to-date spend, which is exact.
- **Prices are hardcoded** in `DEFAULT_USD_PER_HOUR` for us-east-1. Override with
  `--usd-per-hour` if you change region or instance family.