"""Find which instance types a Marketplace AMI actually supports.

Uses DryRun so nothing is launched and nothing is charged. A Marketplace AMI can
reject a perfectly valid Graviton family (e.g. t4g.small on the COOL Graviton2
AMI), and the error only names the product, not the supported list.
"""
import argparse
import sys

import boto3
from botocore.exceptions import ClientError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--ami", required=True)
    ap.add_argument(
        "--types",
        default=",".join(
            [
                "t4g.nano", "t4g.small", "t4g.medium", "t4g.large",
                "m6g.large", "m6g.xlarge", "m6g.2xlarge",
                "c6g.large", "c6g.xlarge", "c6g.2xlarge",
                "r6g.large", "r6g.xlarge",
                "m6gd.large", "c6gn.large", "x2gd.medium", "x2gd.large",
            ]
        ),
    )
    args = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    ok, bad = [], []
    for itype in args.types.split(","):
        try:
            ec2.run_instances(
                ImageId=args.ami,
                InstanceType=itype,
                MinCount=1,
                MaxCount=1,
                DryRun=True,
            )
            ok.append(itype)
            print(f"  OK       {itype}")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"][:70]
            # DryRun itself returns DryRunOperation when the config is valid but
            # we asked for a dry run, so a DryRunOperation error means supported.
            if code == "DryRunOperation":
                ok.append(itype)
                print(f"  SUPPORTED{itype}")
            else:
                bad.append((itype, code))
                print(f"  no       {itype:<14} {code}")

    print(f"\nsupported ({len(ok)}): {', '.join(ok) or 'none'}")
    if not ok:
        sys.exit("no supported instance type found for this AMI")


if __name__ == "__main__":
    main()