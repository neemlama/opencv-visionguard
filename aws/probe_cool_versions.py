"""Probe a COOL Graviton AMI for every usable OpenCV it contains.

The Marketplace listing for the Graviton4 product says version 3.1 is "now built
on OpenCV 5.0" and documents preconfigured environments at
/opt/cool/venvs/python_3.12/bin/activate. Our first COOL runs used the v2 AMI
and reported cv2 4.14.0-pre via a hand-rolled PYTHONPATH, so we may have simply
been activating the wrong OpenCV.

This enumerates every interpreter and environment on the box and reports the
version each one loads, so the COOL/OpenCV-5 question is answered by measurement
rather than by documentation.

    python aws/probe_cool_versions.py --ami ami-...
"""
import argparse
import json
import time

import boto3
from botocore.exceptions import ClientError

PROBE = r"""
echo "=== /opt/cool top level ==="
ls -la /opt/cool 2>&1 | head -25
echo "=== venvs ==="
ls -la /opt/cool/venvs 2>&1 | head -20
echo "=== every python under /opt/cool ==="
find /opt/cool -maxdepth 4 -name 'python3*' -type f 2>/dev/null | head -20
echo "=== version per interpreter/env ==="
for py in $(find /opt/cool -maxdepth 4 -name 'python3.1*' -type f 2>/dev/null | head -10); do
  echo "--- $py"
  "$py" -c "import cv2,sys; print('  cv2', cv2.__version__, '|', cv2.__file__)" 2>&1 | head -2
done
echo "=== manual env approach (what our benchmark used) ==="
LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib PYTHONPATH=/opt/cool/python_3.12/site-packages \
  python3.12 -c "import cv2; print('  manual cv2', cv2.__version__, '|', cv2.__file__)" 2>&1 | head -3
echo "=== system python, no env ==="
python3.12 -c "import cv2; print('  system cv2', cv2.__version__)" 2>&1 | head -2
echo "=== shared libs ==="
ls /opt/cool/cpp_sdk/lib 2>/dev/null | head -8
find /opt/cool -name 'libopencv_imgproc.so*' 2>/dev/null | head -5
echo "=== build info grep ==="
grep -a -o -E 'OpenCV 5[^,]*|OpenCV 4[^,]*' /opt/cool/python_3.12/site-packages/cv2/config.py 2>/dev/null | head -3
echo "PROBE_DONE"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--ami", required=True)
    ap.add_argument("--instance-type", default="c8g.medium")
    ap.add_argument("--profile", default="forgesight-bench-ssm")
    args = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    ssm = boto3.client("ssm", region_name=args.region)
    iam = boto3.client("iam", region_name=args.region)
    arn = iam.get_instance_profile(InstanceProfileName=args.profile)["InstanceProfile"]["Arn"]

    iid = None
    try:
        for s in ec2.describe_subnets(
                Filters=[{"Name": "default-for-az", "Values": ["true"]},
                         {"Name": "map-public-ip-on-launch", "Values": ["true"]}]
        )["Subnets"]:
            try:
                iid = ec2.run_instances(
                    ImageId=args.ami, InstanceType=args.instance_type,
                    MinCount=1, MaxCount=1, SubnetId=s["SubnetId"],
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
                    Filters=[{"Key": "InstanceIds", "Values": [iid]}])["InstanceInformationList"]:
                break
            time.sleep(10)

        cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                               Parameters={"commands": [PROBE]})["Command"]["CommandId"]
        terminal = {"Success", "Failed", "Cancelled", "TimedOut", "Delivery Timed Out"}
        for _ in range(120):
            inv = ssm.list_command_invocations(CommandId=cid, InstanceId=iid)["CommandInvocations"]
            if inv and inv[0]["Status"] in terminal:
                print(ssm.get_command_invocation(
                    CommandId=cid, InstanceId=iid).get("StandardOutputContent", ""))
                break
            time.sleep(10)
    finally:
        if iid:
            print(f"terminating {iid}")
            ec2.terminate_instances(InstanceIds=[iid])


if __name__ == "__main__":
    main()
