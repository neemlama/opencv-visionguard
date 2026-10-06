"""ForgeSight demo endpoint: start / stop / status with a hard budget pre-flight.

Why this exists instead of relying on the AWS Budget alarm
---------------------------------------------------------
AWS Budget notifications are email-based and can lag by hours, and the API model
available to this account does not expose the tag-scoped instance-stop budget
action (its IAMActionDefinition takes only PolicyArn/Roles/Groups/Users, with no
InstanceTypes or Tags). So an unattended instance could keep billing after the
alarm email arrives.

This script therefore enforces the cap *before* spending:

  * pre-flight -- refuses to launch when the budget forecast already projects a
    breach of the cap, so an over-budget launch needs an explicit --force;
  * TTL        -- every launch is time-boxed, with the deadline printed;
  * tag        -- every instance carries CostBudgetTag=forgesight so stop/status
                  and the budget action (where available) can find it.

Usage
-----
  python aws/endpoint.py start --ami-ubuntu-arm64 ami-... --ttl-hours 4 --eip
  python aws/endpoint.py status
  python aws/endpoint.py stop
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import boto3

COST_TAG_KEY = "CostBudgetTag"
COST_TAG_VALUE = "forgesight"
NAME_TAG = "forgesight-endpoint"
BUDGET_NAME = "ForgeSight Hackathon Cap"
DEFAULT_REGION = "us-east-1"

# us-east-1 on-demand list price, USD/hour. Override with --usd-per-hour.
DEFAULT_USD_PER_HOUR = {
    "t4g.small": 0.0168,
    "t4g.medium": 0.0336,
    "t3.small": 0.0208,
    "c7g.small": 0.0365,
    "m7g.small": 0.0584,
    # Elastic IP, charged while associated to a running instance
    "elastic-ip": 0.005,
}


def ec2(region):
    return boto3.client("ec2", region_name=region)


def budget_clients():
    b = boto3.client("budgets", region_name="us-east-1")
    ce = boto3.client("ce", region_name="us-east-1")
    return b, ce


def account_id():
    return boto3.client("sts", region_name="us-east-1").get_caller_identity()["Account"]


def find_budget(budgets):
    for b in budgets.describe_budgets(AccountId=account_id())["Budgets"]:
        if b["BudgetName"] == BUDGET_NAME:
            return b
    return None


def _month_bounds():
    today = datetime.now(timezone.utc).date()
    first = today.replace(day=1)
    nxt = first.replace(day=28) + timedelta(days=4)
    return first, nxt.replace(day=1)


def month_to_date(ce):
    first, _ = _month_bounds()
    today = datetime.now(timezone.utc).date()
    res = ce.get_cost_and_usage(
        TimePeriod={"Start": first.isoformat(), "End": today.isoformat()},
        Granularity="MONTHLY",
        Metrics=["BlendedCost"],
    )
    return max(
        0.0, float(res["ResultsByTime"][0]["Total"]["BlendedCost"]["Amount"])
    )


def forecast_rest_of_month(ce):
    """Cost Explorer forecast. Best-effort refinement.

    GetCostForecast derives its "Earliest supported Start" from this account's
    available billing data, which is currently today. So the window is
    today -> today+30d. It also returns ForecastResults=None when the account
    has too little usage history to project, which is normal for a quiet
    account. Either way the guard falls back to actual month-to-date spend,
    which is always available.
    """
    today = datetime.now(timezone.utc).date()
    try:
        res = ce.get_cost_forecast(
            TimePeriod={
                "Start": today.isoformat(),
                "End": (today + timedelta(days=30)).isoformat(),
            },
            Metric="BLENDED_COST",
            Granularity="MONTHLY",
        )
        fc = res.get("ForecastResults") or {}
        val = fc.get("ForecastedValue", {}).get("Amount")
        if val is not None:
            return float(val)
    except Exception as e:  # forecast is best-effort; fall back to actuals
        print(f"[budget] forecast unavailable ({type(e).__name__}), using actuals")
    return 0.0


def preflight(ce, budgets, hourly, hours):
    """Refuse a launch that would project past the cap."""
    b = find_budget(budgets)
    if not b:
        print("[budget] no budget found - proceeding without a cap check")
        return True
    cap = float(b["BudgetLimit"]["Amount"])
    actual = max(
        0.0, float(b.get("CalculatedSpend", {}).get("ActualSpend", {}).get("Amount", 0.0))
    )
    forecast = forecast_rest_of_month(ce)
    baseline = forecast if forecast > 0 else actual
    added = hourly * hours
    projected = baseline + added

    print(f"[budget] cap=${cap:.2f}  spent=${actual:.2f}  forecast=${forecast:.2f}")
    print(f"[budget] this run +${added:.4f} ({hours}h @ ${hourly}/h) -> projected ${projected:.2f}")
    if projected > cap:
        print(f"[budget] REFUSED: projected ${projected:.2f} exceeds cap ${cap:.2f}")
        return False
    print(f"[budget] OK: ${cap - projected:.2f} headroom")
    return True


USER_DATA = r"""#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/forgesight-setup.log) 2>&1
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv nginx unzip
python3.12 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
/opt/venv/bin/pip install -q opencv-python fastapi "uvicorn[standard]" python-multipart boto3

mkdir -p /opt/app
if [ -n "${SOURCE_URL}" ]; then
  echo "fetching app from ${SOURCE_URL}"
  curl -fsSL -o /tmp/app.tgz "${SOURCE_URL}" || echo "source fetch failed; continuing"
  tar -xzf /tmp/app.tgz -C /opt/app || true
fi

cat > /etc/systemd/system/forgesight.service <<'UNIT'
[Unit]
Description=ForgeSight inspection API
After=network.target
[Service]
WorkingDirectory=/opt/app
Environment=PYTHONPATH=/opt/app
ExecStart=/opt/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now forgesight

cat > /etc/nginx/sites-available/default <<'NGX'
server {
    listen 80;
    server_name _;
    client_max_body_size 12M;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
NGX
nginx -t
systemctl reload nginx
echo "forgesight ready"
"""


def cmd_start(args):
    region = args.region
    e = ec2(region)
    hourly = args.usd_per_hour or DEFAULT_USD_PER_HOUR.get(args.instance_type, 0.02)
    if args.eip:
        hourly += DEFAULT_USD_PER_HOUR["elastic-ip"]

    budgets, ce = budget_clients()
    if not args.force and not preflight(ce, budgets, hourly, args.ttl_hours):
        sys.exit("Launch refused by budget pre-flight. Fix the cap, shorten --ttl-hours, or pass --force.")

    existing = find_endpoint(e)
    if existing:
        print(f"[endpoint] already running: {existing}")
        cmd_status(args)
        return

    resp = e.run_instances(
        ImageId=args.ami_ubuntu_arm64,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        AssociatePublicIpAddress=True,
        UserData=USER_DATA.replace("${SOURCE_URL}", args.source_url or ""),
        IamInstanceProfile={"Name": "forgesight-bench-ssm"},
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": NAME_TAG},
                    {"Key": COST_TAG_KEY, "Value": COST_TAG_VALUE},
                ],
            }
        ],
        BlockDeviceMappings=[
            {"DeviceName": "/dev/sda1",
             "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True}}
        ],
        MetadataOptions={"HttpTokens": "required"},
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"[endpoint] launched {iid} ({args.instance_type})")
    deadline = datetime.now(timezone.utc).timestamp() + args.ttl_hours * 3600
    print(f"[endpoint] cost ceiling for this run: ${hourly * args.ttl_hours:.4f}")
    print(f"[endpoint] HARD DEADLINE: {datetime.fromtimestamp(deadline, timezone.utc):%Y-%m-%d %H:%M UTC}")
    print(f"[endpoint] it will NOT stop itself. Run:  python aws/endpoint.py stop")

    if args.eip:
        eip = e.allocate_address(Domain="vpc")
        assoc = e.associate_address(
            InstanceId=iid, AllocationId=eip["AllocationId"],
            AllowReassociation=True,
        )
        print(f"[endpoint] url: http://{assoc['PublicIp']}")
    e.get_waiter("instance_running").wait(InstanceIds=[iid])
    print("[endpoint] running. bootstrap takes ~90s (pip install + nginx).")


def find_endpoint(e):
    resp = e.describe_instances(
        Filters=[{"Name": "tag:Name", "Values": [NAME_TAG]},
                 {"Name": "instance-state-name",
                  "Values": ["running", "pending", "stopping", "stopped"]}]
    )
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            return i
    return None


def cmd_stop(args):
    e = ec2(args.region)
    found = []
    resp = e.describe_instances(
        Filters=[{"Name": f"tag:{COST_TAG_KEY}", "Values": [COST_TAG_VALUE]}]
    )
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            if i["State"]["Name"] in ("running", "pending", "stopping", "stopped"):
                found.append(i)
    if not found:
        print("[endpoint] nothing tagged", COST_TAG_KEY, "- nothing to stop")
        return
    ids = [i["InstanceId"] for i in found]
    e.terminate_instances(InstanceIds=ids)
    print(f"[endpoint] terminated: {', '.join(ids)}")


def cmd_status(args):
    e = ec2(args.region)
    inst = find_endpoint(e)
    budgets, ce = budget_clients()
    b = find_budget(budgets)
    actual = (
        max(0.0, float(b.get("CalculatedSpend", {}).get("ActualSpend", {}).get("Amount", 0)))
        if b else 0.0
    )
    mtd = month_to_date(ce)
    print(f"[budget] cap=${b['BudgetLimit']['Amount'] if b else 'n/a'}  "
          f"month-to-date=${mtd:.2f}  health={b['HealthStatus']['Status'] if b else 'n/a'}")
    if not inst:
        print("[endpoint] not running  (cost accruing: $0.00/hr)")
        return
    launch = inst.get("LaunchTime")
    hrs = (datetime.now(timezone.utc) - launch.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    rate = (args.usd_per_hour or DEFAULT_USD_PER_HOUR.get(inst["InstanceType"], 0.02))
    print(f"[endpoint] {inst['InstanceId']} {inst['InstanceType']} {inst['State']['Name']}")
    print(f"[endpoint] uptime {hrs:.2f}h -> accrued ~${rate * hrs:.4f} (+EIP if attached)")
    print(f"[endpoint] public: http://{inst.get('PublicIpAddress')}")


def main():
    ap = argparse.ArgumentParser(description="ForgeSight demo endpoint with budget guard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--ami-ubuntu-arm64", required=True)
    p.add_argument("--instance-type", default="t4g.small")
    p.add_argument("--ttl-hours", type=float, default=4.0,
                   help="time-box the run; printed as a hard deadline")
    p.add_argument("--usd-per-hour", type=float, default=0.0)
    p.add_argument("--eip", action="store_true", help="stable URL (+$0.005/hr while attached)")
    p.add_argument("--source-url", default="", help="tarball URL of the app to deploy")
    p.add_argument("--force", action="store_true", help="bypass the budget pre-flight")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("status")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--usd-per-hour", type=float, default=0.0)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()