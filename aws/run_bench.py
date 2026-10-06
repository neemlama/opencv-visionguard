"""Multi-arm COOL benchmark orchestrator.

An "arm" is one benchmark run: an AMI, an instance type, and whether COOL is
preinstalled in that image. Specified as ARM=AMI:TYPE[:cool].

  python aws/run_bench.py --region us-east-1 \
      --arm "B=ami-0bec8cef5313300ad:t4g.small" \
      --arm "C=ami-0045d7fc2ad003464:t3.small"

  # add the COOL arm once the Marketplace AMI is subscribed:
  python aws/run_bench.py --region us-east-1 \
      --arm "A=ami-COOL:t4g.small:cool" \
      --arm "B=ami-0bec8cef5313300ad:t4g.small" \
      --arm "C=ami-0045d7fc2ad003464:t3.small"

Why B exists: comparing COOL-on-Graviton against x86 conflates the library with
the hardware. Arm B runs the same instance family with a plain pip OpenCV, so the
A-vs-B delta is attributable to COOL alone. That is also the methodology AWS
itself used in the COOL launch blog.

Instances are driven over Systems Manager (no SSH keypair) and are always
terminated in a finally block.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BENCH_REMOTE = "/tmp/bench.py"
REMOTE_SAMPLES = "/tmp/samples"
OUT_DIR = Path(__file__).resolve().parents[1] / "bench" / "results"
BENCH_LOCAL = Path(__file__).resolve().parents[1] / "bench" / "bench.py"
SAMPLES_LOCAL = Path(__file__).resolve().parents[1] / "backend" / "app" / "make_samples.py"

# Ubuntu 24.04 already ships Python 3.12, so the DIY arm needs no deadsnakes.
# libgl1 + libglib are required by opencv-python on a headless host; without
# them `import cv2` dies with "libGL.so.1: cannot open shared object file".
# This deliberately installs the full opencv-python (not -headless) so the
# control matches AWS's own published DIY baseline.
INSTALL_DIY = (
    "export DEBIAN_FRONTEND=noninteractive; "
    # Subshell-scoped redirect: a bare `exec >` would keep redirecting for the
    # whole SSM script and swallow the result markers printed later.
    "( apt-get update -qq && "
    "  apt-get install -y -qq python3-venv python3-pip libgl1 libglib2.0-0t64 && "
    "  python3 -m venv /opt/benchenv && "
    "  /opt/benchenv/bin/pip install -q --upgrade pip && "
    "  /opt/benchenv/bin/pip install -q opencv-python numpy "
    ") > /tmp/install.log 2>&1 || { echo INSTALL_FAILED; tail -20 /tmp/install.log; }; "
    "/opt/benchenv/bin/python -c \"import cv2; print('opencv', cv2.__version__)\""
)


def parse_arm(spec):
    """ARM=AMI:TYPE[:cool] -> dict. Tolerant of the "cool" suffix being absent."""
    name, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"bad --arm '{spec}', expected ARM=AMI:INSTANCE_TYPE[:cool]")
    parts = rest.split(":")
    if len(parts) < 2:
        raise SystemExit(f"bad --arm '{spec}', expected ARM=AMI:INSTANCE_TYPE[:cool]")
    ami, itype = parts[0], parts[1]
    use_cool = any(p.lower() == "cool" for p in parts[2:])
    return {"name": name.strip(), "ami": ami, "instance_type": itype, "cool": use_cool}


def subnet_candidates(ec2):
    """Default-VPC public subnets; SSM needs egress to reach the endpoint.

    Returns every candidate rather than one, because Graviton is not offered in
    all AZs (us-east-1e rejects t4g.*), so the launcher walks the list.
    """
    resp = ec2.describe_subnets(
        Filters=[
            {"Name": "default-for-az", "Values": ["true"]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]},
        ]
    )
    subs = [{"SubnetId": s["SubnetId"], "az": s["AvailabilityZone"]} for s in resp["Subnets"]]
    if not subs:
        raise SystemExit(
            "No default public subnet with auto-assign public IP. "
            "Create one or pass --subnet-id."
        )
    return subs


def ensure_instance_profile(iam, name):
    """Minimal SSM profile. Reuses an existing profile to stay idempotent."""
    try:
        iam.get_instance_profile(InstanceProfileName=name)
        print(f"[iam] reusing instance profile {name}")
    except ClientError:
        try:
            iam.create_instance_profile(InstanceProfileName=name)
            iam.add_role_to_instance_profile(
                InstanceProfileName=name, RoleName=name
            )
            print(f"[iam] created instance profile {name}")
        except ClientError as e:
            print(f"[iam] SKIP: {e.response['Error']['Message']}")
            return None
        time.sleep(10)
    return iam.get_instance_profile(InstanceProfileName=name)["InstanceProfile"]["Arn"]


def ensure_ssm_role(iam, name):
    """Role granting the AmazonSSMManagedInstanceCore managed policy."""
    trust = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    try:
        iam.get_role(RoleName=name)
    except ClientError:
        try:
            iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust)
            iam.attach_role_policy(
                RoleName=name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            )
            iam.get_waiter("role_exists").wait(RoleName=name)
            print(f"[iam] created role {name} + SSM core policy")
            time.sleep(10)
        except ClientError as e:
            print(f"[iam] SKIP SSM role: {e.response['Error']['Message']}")
            return None
    return name


def launch(ec2, arm, profile_arn, subnet=None):
    kwargs = dict(
        ImageId=arm["ami"],
        InstanceType=arm["instance_type"],
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Arn": profile_arn},
        UserData="#!/bin/bash\n# COOL is preinstalled in the Marketplace AMI\n"
        if arm["cool"]
        else "#!/bin/bash\n# DIY arm: dependencies installed at run time\n",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"forgesight-bench-{arm['name']}"},
                    {"Key": "CostBudgetTag", "Value": "forgesight"},
                ],
            }
        ],
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
        MetadataOptions={"HttpTokens": "required"},
    )
    if subnet:
        kwargs["SubnetId"] = subnet["SubnetId"]
    resp = ec2.run_instances(**kwargs)
    iid = resp["Instances"][0]["InstanceId"]
    az = resp["Instances"][0].get("Placement", {}).get("AvailabilityZone", "?")
    print(f"[{arm['name']}] launched {iid} ({arm['instance_type']}, az {az}, ami {arm['ami']})")
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    return iid


def launch_with_fallback(ec2, arm, profile_arn, candidates):
    """Walk candidate AZs until one actually offers the instance type."""
    last = None
    for cand in candidates:
        try:
            return launch(ec2, arm, profile_arn, cand)
        except ClientError as e:
            last = e
            if e.response["Error"]["Code"] in ("Unsupported", "InvalidParameterValue"):
                print(f"[{arm['name']}] {cand['az']} unavailable for "
                      f"{arm['instance_type']}, trying next AZ")
                continue
            raise
    msg = last.response["Error"]["Message"] if last else "unknown"
    raise SystemExit(f"[{arm['name']}] no AZ offers {arm['instance_type']}: {msg}")


def wait_for_ssm(ssm, iid, arm_name, attempts=60):
    print(f"[{arm_name}] waiting for SSM agent...")
    for _ in range(attempts):
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [iid]}]
        )["InstanceInformationList"]
        if info:
            print(f"[{arm_name}] SSM online")
            return True
        time.sleep(10)
    return False


def send_and_wait(ssm, iid, commands, timeout=1200):
    cid = ssm.send_command(
        InstanceIds=[iid],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
    )["Command"]["CommandId"]
    print(f"    command {cid} dispatched")
    terminal = {"Success", "Cancelled", "Failed", "TimedOut", "Delivery Timed Out",
                "Execution Timed Out", "Undeliverable"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        # get_command_invocation takes no Filters in this API version, so poll
        # list_command_invocations first to learn whether the invocation exists.
        # No Status filter: this API rejects "*" as a filter value, and we only
        # ever target a single instance so an unfiltered list is unambiguous.
        invocations = ssm.list_command_invocations(
            CommandId=cid, InstanceId=iid
        )["CommandInvocations"]
        if not invocations:
            time.sleep(10)
            continue
        status = invocations[0]["Status"]
        if status in terminal:
            out = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
            return out["Status"], out.get("StandardOutputContent", ""), out.get(
                "StandardErrorContent", ""
            )
        time.sleep(10)
    raise SystemExit(f"command timed out on {iid}")


def run_arm(region, arm, profile_arn, candidates, usd_per_hour):
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    iid = None
    try:
        iid = launch_with_fallback(ec2, arm, profile_arn, candidates)
        if not wait_for_ssm(ssm, iid, arm["name"]):
            print(f"[{arm['name']}] SSM never registered (check subnet egress)", file=sys.stderr)
            return None

        src = BENCH_LOCAL.read_text()
        gen = SAMPLES_LOCAL.read_text()
        # Samples are regenerated on the instance rather than uploaded: SSM
        # commands are size-limited, and the generator is seeded, so every arm
        # gets byte-identical inputs while still measuring real decode work.
        cmds = [
            f"cat > {BENCH_REMOTE} <<'EOFBENCH'\n{src}\nEOFBENCH",
            f"cat > /tmp/make_samples.py <<'EOFGEN'\n{gen}\nEOFGEN",
        ]
        if arm["cool"]:
            cmds.append(
                'export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"; '
                'export PYTHONPATH="/opt/cool/python_3.12/site-packages/"'
            )
            runner = f"python3.12 {BENCH_REMOTE}"
            gen_runner = f"python3.12 /tmp/make_samples.py {REMOTE_SAMPLES}"
        else:
            cmds.append(INSTALL_DIY)
            runner = f"/opt/benchenv/bin/python {BENCH_REMOTE}"
            gen_runner = f"/opt/benchenv/bin/python /tmp/make_samples.py {REMOTE_SAMPLES}"
        cmds.append(f"mkdir -p {REMOTE_SAMPLES}")
        cmds.append(f"{gen_runner}")
        cmds.append(f"{runner} --label {arm['name']} --instance-type {arm['instance_type']} "
                    f"--samples {REMOTE_SAMPLES} --usd-per-hour {usd_per_hour} "
                    f"--out /tmp/bench_result.json")
        cmds.append("echo __BENCH_JSON_BEGIN__")
        cmds.append("cat /tmp/bench_result.json")
        cmds.append("echo __BENCH_JSON_END__")

        st, out, err = send_and_wait(ssm, iid, cmds)
        if st != "Success":
            print(f"[{arm['name']}] FAILED ({st})\n{err[-1500:]}", file=sys.stderr)
            return None
        # The workload also prints its own JSON to stdout, so isolate the result
        # with sentinels rather than grabbing the first "{" and last "}" (which
        # spans both blobs and fails to parse).
        if "__BENCH_JSON_BEGIN__" not in out or "__BENCH_JSON_END__" not in out:
            print(f"[{arm['name']}] result markers missing:\n{out[-1500:]}", file=sys.stderr)
            return None
        blob = out.split("__BENCH_JSON_BEGIN__")[1].split("__BENCH_JSON_END__")[0]
        try:
            return json.loads(blob.strip())
        except json.JSONDecodeError as e:
            print(f"[{arm['name']}] bad result JSON ({e}):\n{blob[:800]}", file=sys.stderr)
            return None
    finally:
        if iid:
            print(f"[{arm['name']}] terminating {iid}")
            try:
                ec2.terminate_instances(InstanceIds=[iid])
            except ClientError as e:
                print(f"[{arm['name']}] terminate failed: {e}", file=sys.stderr)


def print_summary(results):
    print("\n=== summary ===")
    print(f"{'arm':<4}{'build':<6}{'opencv':<9}{'arch':<8}{'fps':>9}{'ms/frame':>11}")
    for r in results:
        ms = r["mean_frame_seconds"] * 1000
        print(f"{r['label']:<4}{r['build']:<6}{r['opencv_version']:<9}"
              f"{r['arch']:<8}{r['fps']:>9}{ms:>11.2f}")
    by = {r["label"]: r for r in results}
    if "A" in by and "B" in by:
        print(f"\nA/B library-only speedup : {by['A']['fps'] / by['B']['fps']:.2f}x")
    if "A" in by and "C" in by:
        print(f"A/C library+hardware     : {by['A']['fps'] / by['C']['fps']:.2f}x")
    if "B" in by and "C" in by:
        print(f"B/C hardware-only        : {by['B']['fps'] / by['C']['fps']:.2f}x")
    costs = {r["label"]: r.get("usd_per_1000_frames") for r in results}
    if any(v is not None for v in costs.values()):
        print("\ncost per 1000 frames:", {k: v for k, v in costs.items() if v is not None})


def main():
    ap = argparse.ArgumentParser(description="COOL vs DIY OpenCV benchmark")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--arm", action="append", required=True,
                    help="ARM=AMI:INSTANCE_TYPE[:cool]  (repeatable)")
    ap.add_argument("--usd-per-hour", type=float, default=0.0)
    ap.add_argument("--subnet-id", default="")
    ap.add_argument("--profile", default="forgesight-bench-ssm")
    args = ap.parse_args()

    arms = [parse_arm(s) for s in args.arm]
    region = args.region
    iam = boto3.client("iam", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)

    ensure_ssm_role(iam, args.profile)
    profile_arn = ensure_instance_profile(iam, args.profile)
    if not profile_arn:
        raise SystemExit("cannot create instance profile; need iam:CreateInstanceProfile")

    if args.subnet_id:
        candidates = [{"SubnetId": args.subnet_id, "az": args.subnet_id}]
    else:
        candidates = subnet_candidates(ec2)
    print(f"[net] candidate AZs: {', '.join(c['az'] for c in candidates)}")

    results = []
    for arm in arms:
        results.append(run_arm(region, arm, profile_arn, candidates, args.usd_per_hour))
    results = [r for r in results if r]

    if not results:
        raise SystemExit("all arms failed; see stderr above")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"comparison-{region}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_file}")
    print_summary(results)


if __name__ == "__main__":
    main()