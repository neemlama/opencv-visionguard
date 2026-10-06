"""Probe a COOL AMI to find how the optimised OpenCV is actually installed.

The AWS listing documents /opt/cool, but that path is version-specific and the
Graviton2 AMI may lay out differently. This walks the instance and reports what
is really there, so the benchmark activates the right paths instead of guessing.

Dry-run-free by design: it launches, probes, prints, then terminates.
"""
import argparse
import json
import time

import boto3
from botocore.exceptions import ClientError

PROBE = r"""
echo "=== /opt ==="; ls -la /opt 2>&1 | head -20
echo "=== search for cool ==="
sudo find / -maxdepth 5 -iname '*cool*' -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -30
echo "=== python interpreters ==="
ls /usr/bin/python3* 2>&1
echo "=== opencv anywhere on disk ==="
sudo find / -maxdepth 7 -name 'cv2' -type d -not -path '/proc/*' 2>/dev/null | head -20
echo "=== shared libs ==="
sudo find / -maxdepth 6 -name 'libopencv*' 2>/dev/null | head -20
echo "=== pip opencv (system python) ==="
python3.12 -m pip list 2>/dev/null | grep -i -E 'opencv|numpy' || echo "none in python3.12"
for v in /opt/*/bin/python* /usr/local/bin/python3*; do
  [ -x "$v" ] && echo "--- $v ---" && "$v" -c "import cv2,sys; print('cv2', cv2.__version__, sys.executable)" 2>&1 | head -3
done
echo "=== env hints ==="
env | grep -i -E 'cool|opencv|LD_LIBRARY|PYTHONPATH' || echo "no cool env set"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--ami", required=True)
    ap.add_argument("--instance-type", default="c6g.large")
    ap.add_argument("--profile", default="forgesight-bench-ssm")
    args = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    ssm = boto3.client("ssm", region_name=args.region)
    iam = boto3.client("iam", region_name=args.region)
    arn = iam.get_instance_profile(InstanceProfileName=args.profile)[
        "InstanceProfile"
    ]["Arn"]

    iid = None
    try:
        subnets = ec2.describe_subnets(
            Filters=[
                {"Name": "default-for-az", "Values": ["true"]},
                {"Name": "map-public-ip-on-launch", "Values": ["true"]},
            ]
        )["Subnets"]
        for sub in subnets:
            try:
                iid = ec2.run_instances(
                    ImageId=args.ami,
                    InstanceType=args.instance_type,
                    MinCount=1,
                    MaxCount=1,
                    SubnetId=sub["SubnetId"],
                    IamInstanceProfile={"Arn": arn},
                    MetadataOptions={"HttpTokens": "required"},
                )["Instances"][0]["InstanceId"]
                break
            except ClientError as e:
                if e.response["Error"]["Code"] in ("Unsupported", "UnsupportedOperation"):
                    continue
                raise
        if not iid:
            raise SystemExit("could not launch")
        print(f"probing {iid}")
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])

        for _ in range(60):
            if ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [iid]}]
            )["InstanceInformationList"]:
                break
            time.sleep(10)
        else:
            raise SystemExit("SSM never came online")

        cid = ssm.send_command(
            InstanceIds=[iid],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [PROBE]},
        )["Command"]["CommandId"]
        terminal = {"Success", "Failed", "Cancelled", "TimedOut", "Delivery Timed Out"}
        for _ in range(90):
            inv = ssm.list_command_invocations(
                CommandId=cid, InstanceId=iid
            )["CommandInvocations"]
            if inv and inv[0]["Status"] in terminal:
                out = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
                print(out.get("StandardOutputContent", ""))
                if out.get("StandardErrorContent"):
                    print("--- stderr ---")
                    print(out["StandardErrorContent"][-2000:])
                break
            time.sleep(10)
    finally:
        if iid:
            print(f"terminating {iid}")
            ec2.terminate_instances(InstanceIds=[iid])


if __name__ == "__main__":
    main()