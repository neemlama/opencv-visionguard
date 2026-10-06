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
import base64
import json
import re
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
NEU_SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "neu"

# COOL install layout differs by AMI generation. The Graviton2 image
# (ami-03175684e1b021dda) uses /opt/opencv_install/{lib,lib/python3.12/site-packages}
# and has NO /opt/cool at all; the Graviton4 listing documents /opt/cool. Detect at
# runtime instead of hardcoding either path.
#
# The env exports MUST be in the same command as the python invocation: relying on
# exports carrying between separate AWS-RunShellScript command entries failed in
# practice - python3.12 ran fine but cv2 was missing.
COOL_ENV = (
    "if [ -d /opt/opencv_install ]; then "
    "  export LD_LIBRARY_PATH=/opt/opencv_install/lib:$LD_LIBRARY_PATH; "
    "  export PYTHONPATH=/opt/opencv_install/lib/python3.12/site-packages; "
    "elif [ -d /opt/cool ]; then "
    "  export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH; "
    "  export PYTHONPATH=/opt/cool/python_3.12/site-packages/; "
    "else echo COOL_LAYOUT_NOT_FOUND; fi; "
)


VER_RE = re.compile(r"(\d+)\.(\d+)")


def pin_for_cool_version(cool_version):
    """Map a COOL-reported version to a pip range that actually resolves.

    COOL images report pre-release builds (e.g. "4.14.0-pre") which do not exist
    on PyPI, so pinning the exact string makes the DIY control fail to install.
    Instead install the newest stable release of the same major.minor, which is
    the closest fair control.
    """
    m = VER_RE.search(cool_version or "")
    if not m:
        return ""
    major, minor = int(m.group(1)), int(m.group(2))
    return f">={major}.{minor}.0,<{major}.{minor + 1}.0"


def diy_install_cmd(opencv_version=""):
    """DIY control install.

    `opencv_version` pins the pip wheel so the A-vs-B delta is attributable to
    COOL alone. The COOL AMIs bundle different majors (Graviton2 ships 4.12,
    Graviton4 ships 4.14.0-pre), and comparing across majors would confound the
    optimisation with a version change. Accepts a range like ">=4.14.0,<4.15.0".
    """
    spec = f"opencv-python{opencv_version}" if opencv_version else "opencv-python"
    # Single-quote the spec: an unquoted "opencv-python>=4.14.0,<4.15.0" makes
    # bash treat ">" as output redirection, so pip never runs and cv2 is missing.
    spec_q = "'" + spec + "'"
    return (
        "export DEBIAN_FRONTEND=noninteractive; "
        # Subshell-scoped redirect: a bare `exec >` keeps redirecting for the
        # whole SSM script and swallows the result markers printed later.
        "( apt-get update -qq && "
        "  apt-get install -y -qq python3-venv python3-pip libgl1 libglib2.0-0t64 && "
        "  python3 -m venv /opt/benchenv && "
        "  /opt/benchenv/bin/pip install -q --upgrade pip && "
        f"  /opt/benchenv/bin/pip install -q {spec_q} numpy "
        ") > /tmp/install.log 2>&1 || { echo INSTALL_FAILED; tail -20 /tmp/install.log; }; "
        "/opt/benchenv/bin/python -c \"import cv2; print('opencv', cv2.__version__)\""
    )

# libgl1 + libglib are required by opencv-python on a headless host; without
# them `import cv2` dies with "libGL.so.1: cannot open shared object file".
# The full opencv-python (not -headless) is installed deliberately so the control
# matches AWS's own published DIY baseline.


# us-east-1 on-demand list prices, used only for the pre-run estimate.
USD_PER_HOUR = {
    "c8g.medium": 0.072, "c8g.large": 0.144, "c8g.xlarge": 0.288,
    "m8g.medium": 0.077, "m8g.large": 0.154,
    "r8g.medium": 0.084, "r8g.large": 0.168,
    "c6g.medium": 0.036, "c6g.large": 0.072,
    "c6i.large": 0.085, "c6i.medium": 0.0425,
    "t4g.small": 0.0168, "t3.small": 0.0208,
}
# Typical wall-clock per arm: boot + SSM + (DIY only) pip install of OpenCV.
MINUTES_PER_ARM = {"cool": 4, "diy": 8}


def estimate_cost(arms, detected_version=""):
    """Rough pre-run cost so the operator can see spend BEFORE committing.

    The COOL Marketplace product also bills a licence fee per running hour that
    is not modelled here, so treat this as a lower bound.
    """
    total = 0.0
    print("\n=== cost estimate (us-east-1 on-demand, lower bound) ===")
    for a in arms:
        rate = USD_PER_HOUR.get(a["instance_type"], 0.05)
        mins = MINUTES_PER_ARM["cool" if a["cool"] else "diy"]
        cost = rate * mins / 60
        total += cost
        print(f"  arm {a['name']:<2} {a['instance_type']:<12} ~{mins} min  ~${cost:.4f}")
    print(f"  total ~${total:.4f}  (+ COOL licence fee on COOL arms)")
    return total


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


def launch(ec2, arm, profile_arn, subnet=None, volume_size=None):
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
        MetadataOptions={"HttpTokens": "required"},
    )
    # Only override the root volume when asked. A hardcoded 20GB broke the COOL
    # arm: its AMI snapshot is 40GB and EC2 rejects a smaller volume with
    # InvalidBlockDeviceMapping. Omitting the mapping uses the AMI's own sizing,
    # and the root volume is DeleteOnTermination by default.
    if volume_size:
        kwargs["BlockDeviceMappings"] = [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_size, "VolumeType": "gp3",
                        "DeleteOnTermination": True},
            }
        ]
    if subnet:
        kwargs["SubnetId"] = subnet["SubnetId"]
    resp = ec2.run_instances(**kwargs)
    iid = resp["Instances"][0]["InstanceId"]
    az = resp["Instances"][0].get("Placement", {}).get("AvailabilityZone", "?")
    print(f"[{arm['name']}] launched {iid} ({arm['instance_type']}, az {az}, ami {arm['ami']})")
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    return iid


def launch_with_fallback(ec2, arm, profile_arn, candidates, volume_size=None):
    """Walk candidate AZs until one actually offers the instance type."""
    last = None
    for cand in candidates:
        try:
            return launch(ec2, arm, profile_arn, cand, volume_size)
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


TERMINAL = {"Success", "Cancelled", "Failed", "TimedOut", "Delivery Timed Out",
            "Execution Timed Out", "Undeliverable"}
# SendCommand rejects a request whose parameters exceed 97KB. bench.py plus the
# inline base64 NEU samples overflow that, so commands are dispatched in chunks.
MAX_COMMAND_BYTES = 80 * 1024


def chunk_commands(commands, limit=MAX_COMMAND_BYTES):
    chunks, cur, size = [], [], 0
    for c in commands:
        n = len(c) + 1
        if cur and size + n > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def _await_command(ssm, cid, iid, timeout):
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
            time.sleep(5)
            continue
        status = invocations[0]["Status"]
        if status in TERMINAL:
            out = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
            return out["Status"], out.get("StandardOutputContent", ""), out.get(
                "StandardErrorContent", ""
            )
        time.sleep(5)
    raise SystemExit(f"command {cid} timed out on {iid}")


def send_and_wait(ssm, iid, commands, timeout=1200):
    """Dispatch in chunks, concatenating stdout so markers survive."""
    stdout, stderr, status = "", "", "Success"
    for i, batch in enumerate(chunk_commands(commands), 1):
        cid = ssm.send_command(
            InstanceIds=[iid],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": batch},
        )["Command"]["CommandId"]
        print(f"    chunk {i}/{len(chunk_commands(commands))} ({sum(len(c) for c in batch) // 1024} KB)")
        status, out, err = _await_command(ssm, cid, iid, timeout)
        stdout += out
        stderr += err
        if status != "Success":
            break
    return status, stdout, stderr


def run_arm(region, arm, profile_arn, candidates, usd_per_hour, volume_size=None,
            opencv_version=""):
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    iid = None
    try:
        iid = launch_with_fallback(ec2, arm, profile_arn, candidates, volume_size)
        if not wait_for_ssm(ssm, iid, arm["name"]):
            print(f"[{arm['name']}] SSM never registered (check subnet egress)", file=sys.stderr)
            return None

        src = BENCH_LOCAL.read_text()
        # Samples: real NEU photographs shipped inline as base64, one SSM command
        # per image. The pipeline measures image analysis, so synthetic images
        # would misreport it, and AWS-RunShellScript caps a single command's size,
        # hence one command per file. They are byte-identical across every arm, so
        # the comparison stays controlled.
        cmds = [f"cat > {BENCH_REMOTE} <<'EOFBENCH'\n{src}\nEOFBENCH",
                f"mkdir -p {REMOTE_SAMPLES}"]
        for img_path in sorted(NEU_SAMPLES.glob("*.jpg")):
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            cmds.append(
                f"echo {b64} | base64 -d > {REMOTE_SAMPLES}/{img_path.name}"
            )
        if arm["cool"]:
            cmds.append(
                "echo '--- COOL layout ---'; "
                "ls -d /opt/opencv_install /opt/cool /opt/kleidicv 2>&1; "
                "which python3.12; python3.12 -V"
            )
            cmds.append(
                COOL_ENV + "python3.12 -c \"import cv2; "
                "print('COOL cv2', cv2.__version__)\" > /tmp/coolcheck.log 2>&1; "
                "echo COOLCHECK_EXIT=$?; cat /tmp/coolcheck.log"
            )
            runner = f"python3.12 {BENCH_REMOTE}"
            prefix = COOL_ENV
        else:
            cmds.append(diy_install_cmd(opencv_version))
            runner = f"/opt/benchenv/bin/python {BENCH_REMOTE}"
            prefix = ""
        # Capture the workload exit code explicitly: a bare `cat` at the end would
        # return 0 and make SSM report Success even when the benchmark failed.
        cmds.append(
            f"{prefix}{runner} --label {arm['name']} --instance-type {arm['instance_type']} "
            f"--samples {REMOTE_SAMPLES} --usd-per-hour {usd_per_hour} "
            f"--out /tmp/bench_result.json > /tmp/bench_stdout.log 2>&1; echo BENCH_EXIT=$?"
        )
        cmds.append("echo __BENCH_JSON_BEGIN__")
        cmds.append("cat /tmp/bench_result.json 2>/dev/null || echo NO_RESULT_FILE")
        cmds.append("echo __BENCH_JSON_END__")
        cmds.append("echo '--- bench stdout ---'; tail -25 /tmp/bench_stdout.log")

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
        if "NO_RESULT_FILE" in blob or not blob.strip():
            print(f"[{arm['name']}] benchmark produced no result file.\n"
                  f"--- remote output ---\n{out[-2500:]}\n--- stderr ---\n{err[-1500:]}",
                  file=sys.stderr)
            return None
        try:
            return json.loads(blob.strip())
        except json.JSONDecodeError as e:
            print(f"[{arm['name']}] bad result JSON ({e}):\n{blob[:800]}\n"
                  f"--- remote output ---\n{out[-2000:]}", file=sys.stderr)
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
    ap.add_argument(
        "--volume-size",
        type=int,
        default=0,
        help="override root volume GB (0 = use the AMI's own sizing)",
    )
    ap.add_argument(
        "--opencv-version",
        default="",
        help="pin the DIY arms' opencv-python version, e.g. 4.12.0.88. Match the "
             "COOL arm's version so the A-vs-B delta is the library, not a "
             "major-version change.",
    )
    args = ap.parse_args()

    arms = [parse_arm(s) for s in args.arm]
    region = args.region

    # COOL arms run first so the DIY control can be pinned to the *same* OpenCV
    # version automatically. The COOL AMIs bundle different majors (Graviton2
    # ships 4.12), and comparing across majors would confound the optimisation
    # with a version change.
    arms.sort(key=lambda a: not a["cool"])

    estimate_cost(arms, args.opencv_version)

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
    pinned = args.opencv_version
    for arm in arms:
        res = run_arm(
            region, arm, profile_arn, candidates, args.usd_per_hour,
            args.volume_size, pinned,
        )
        results.append(res)
        if res and arm["cool"] and not pinned:
            # Match the control to COOL's bundled major.minor from now on.
            v = pin_for_cool_version(res["opencv_version"])
            pinned = v
            print(f"[{arm['name']}] COOL bundles OpenCV {res['opencv_version']}; "
                  f"pinning DIY control to opencv-python{v}")

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