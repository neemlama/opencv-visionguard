"""Three-arm COOL benchmark orchestrator.

Arms (same workload script, same sample images):
  A  Graviton + COOL (Marketplace AMI)      -> COOL Award eligibility
  B  Graviton, same family + DIY pip OpenCV -> isolates the library's contribution
  C  x86 + DIY pip OpenCV                   -> hardware baseline / Arm-x86 coexistence

Arm B exists because a COOL-vs-x86 comparison confounds library and hardware.
Arm B holds hardware constant so the delta is attributable to COOL alone.

Drives instances over AWS Systems Manager (no SSH keypair needed) and always
terminates them in a finally block.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BENCH_REMOTE = "/tmp/bench.py"
OUT_DIR = Path(__file__).resolve().parents[1] / "bench" / "results"

INSTALL_DIY = (
    "sudo apt-get update -qq && "
    "sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip >/dev/null && "
    "python3.12 -m venv /opt/benchenv && "
    "/opt/benchenv/bin/pip install -q --upgrade pip && "
    "/opt/benchenv/bin/pip install -q opencv-python numpy"
)

RUN_BENCH = f"/opt/benchenv/bin/python {BENCH_REMOTE}"  # arm B / C
RUN_BENCH_COOL = (
    'export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"; '
    'export PYTHONPATH="/opt/cool/python_3.12/site-packages/"; '
    f"python3.12 {BENCH_REMOTE}"
)


def client(region, service):
    return boto3.client(service, region_name=region)


def ensure_instance_profile(iam, region):
    """Reuse or create a minimal SSM instance profile (no S3/other permissions)."""
    name = "forgesight-bench-ssm"
    try:
        role = iam.get_role(RoleName=name)["Role"]
    except ClientError:
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
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust)["Role"]
        iam.attach_role_policy(
            RoleName=name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        )
        iam.get_waiter("role_exists").wait(RoleName=name)
        time.sleep(10)  # eventual consistency before attaching a profile
    try:
        iam.get_instance_profile(InstanceProfileName=name)
    except ClientError:
        iam.create_instance_profile(InstanceProfileName=name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=name, RoleName=role["RoleName"]
        )
        time.sleep(10)
    return name


def subnet_for(ec2, region, arch):
    """Pick a default-VPC subnet that can reach the internet (SSM needs egress)."""
    resp = ec2.describe_subnets(
        Filters=[
            {"Name": "default-for-az", "Values": ["true"]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]},
        ]
    )
    if not resp["Subnets"]:
        raise SystemExit(
            "No default public subnet with auto-assign public IP found. "
            "Create one, or pass --subnet-id explicitly."
        )
    return resp["Subnets"][0]["SubnetId"]


def launch(ec2, ssm, args, arm, ami, instance_type, label, use_cool):
    block = {
        "ImageId": ami,
        "InstanceType": instance_type,
        "InstanceProfileName": {"Arn": args.profile_arn},
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"forgesight-{label}"},
                    {"Key": "forgesight", "Value": "bench"},
                ],
            }
        ],
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
        "MetadataOptions": {"HttpTokens": "required"},
    }
    resp = ec2.run_instances(
        MinCount=1,
        MaxCount=1,
        UserData=INSTALL_DIY if not use_cool else "#!/bin/bash\n# COOL preinstalled in AMI\n",
        **block,
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"[{label}] launched {iid} ({instance_type}, ami {ami})")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[iid])
    print(f"[{label}] running, waiting for SSM agent...")
    for _ in range(60):
        ping = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [iid]}]
        )["InstanceInformationList"]
        if ping:
            break
        time.sleep(10)
    else:
        raise SystemExit(f"[{label}] SSM agent never registered - check subnet egress")
    return iid


def send_and_wait(ssm, iid, commands, timeout=900):
    doc = ssm.send_command(
        InstanceIds=[iid], DocumentName="AWS-RunShellScript", Parameters={"commands": commands}
    )
    cid = doc["Command"]["CommandId"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = ssm.get_command_invocation(
            CommandId=cid, InstanceId=iid, Filters=[{"Key": "Status", "Values": ["*"]}]
        )
        st = out["Status"]
        if st["Status"] in ("Success", "Cancelled", "Failed", "TimedOut"):
            return st, out.get("StandardOutputContent", ""), out.get("StandardErrorContent", "")
        time.sleep(10)
    raise SystemExit(f"command timed out on {iid}")


def run_arm(args, arm, ami, instance_type, label, use_cool, region):
    ec2, ssm = client(region, "ec2"), client(region, "ssm")
    iid = None
    try:
        iid = launch(ec2, ssm, args, arm, ami, instance_type, label, use_cool)
        bench_src = (Path(__file__).resolve().parents[1] / "bench" / "bench.py").read_text()
        s3cmd = f"cat > {BENCH_REMOTE} <<'EOFBENCH'\n{bench_src}\nEOFBENCH"

        cmds = [s3cmd]
        if not use_cool:
            cmds.append(INSTALL_DIY)
        runner = RUN_BENCH_COOL if use_cool else RUN_BENCH
        cmds.append(
            f"{runner} --label {label} --instance-type {instance_type} "
            f"--usd-per-hour {args.usd_per_hour} --out /tmp/bench_result.json"
        )
        cmds.append("cat /tmp/bench_result.json")

        st, out, err = send_and_wait(ssm, iid, cmds)
        if st["Status"] != "Success":
            print(f"[{label}] FAILED\n{err[-2000:]}", file=sys.stderr)
            return None
        return json.loads(out[out.index("{") : out.rindex("}") + 1])
    finally:
        if iid:
            print(f"[{label}] terminating {iid}")
            ec2.terminate_instances(InstanceIds=[iid])


def main():
    ap = argparse.ArgumentParser(description="COOL (Graviton) vs DIY OpenCV benchmark")
    ap.add_argument("--region", required=True)
    ap.add_argument("--ami-arm64", required=True, help="COOL Marketplace AMI id (Graviton)")
    ap.add_argument("--ami-ubuntu-arm64", default="", help="plain Ubuntu 24.04 arm64 AMI (arm B)")
    ap.add_argument("--ami-ubuntu-amd64", default="", help="plain Ubuntu 24.04 amd64 AMI (arm C)")
    ap.add_argument("--instance-arm64", default="t4g.small")
    ap.add_argument("--instance-x86", default="t3.small")
    ap.add_argument("--usd-per-hour", type=float, default=0.0)
    ap.add_argument("--subnet-id", default="")
    ap.add_argument("--profile", default="forgesight-bench-ssm")
    args = ap.parse_args()

    region = args.region
    iam, ec2 = client(region, "iam"), client(region, "ec2")
    profile = ensure_instance_profile(iam, region)
    args.profile_arn = iam.get_instance_profile(InstanceProfileName=profile)["InstanceProfile"]["Arn"]

    if args.subnet_id:
        ec2.describe_subnets(SubnetIds=[args.subnet_id])
    else:
        subnet_for(ec2, region, "arm")

    results = []
    results.append(
        run_arm(args, "A", args.ami_arm64, args.instance_arm64, "cool-arm", True, region)
    )
    if args.ami_ubuntu_arm64:
        results.append(
            run_arm(args, "B", args.ami_ubuntu_arm64, args.instance_arm64, "diy-arm", False, region)
        )
    if args.ami_ubuntu_amd64:
        results.append(
            run_arm(args, "C", args.ami_ubuntu_amd64, args.instance_x86, "x86-arm", False, region)
        )

    results = [r for r in results if r]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"comparison-{region}.json").write_text(json.dumps(results, indent=2))

    print("\n=== summary ===")
    for r in results:
        print(f"{r['label']:<10} {r['build']:<4} {r['opencv_version']:<8} {r['fps']:>8} fps")
    if len(results) == 2:
        a, b = results
        print(f"\nspeedup A/B (library only): {a['fps'] / b['fps']:.2f}x")
    if len(results) == 3:
        print(f"speedup A/C (library + arch): {results[0]['fps'] / results[2]['fps']:.2f}x")


if __name__ == "__main__":
    main()