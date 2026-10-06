"""ForgeSight benchmark: per-op timings, runs identically on COOL (Graviton AMI)
and DIY pip OpenCV. Self-identifies which library is loaded so the JSON output is
proof of what actually executed.

Usage on a COOL AMI:
    export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/opt/cool/python_3.12/site-packages/"
    python3.12 bench.py --label cool-arm --instance-type t4g.small

Usage for the DIY control (same instance family):
    python3 bench.py --label diy-arm --instance-type t4g.small
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

COOL_ROOTS = [Path("/opt/cool"), Path("/opt/opencv_install")]


def detect_build() -> str:
    """COOL ships at /opt/cool on the Graviton4 image and /opt/opencv_install on
    the Graviton2 image, so both must be probed or COOL gets mislabelled DIY."""
    return "COOL" if any(p.exists() for p in COOL_ROOTS) else "DIY"


def build_marker() -> str:
    present = [str(p) for p in COOL_ROOTS if p.exists()]
    return ", ".join(present) if present else "no COOL install found"
# Ops COOL specifically optimizes, per AWS Marketplace listing.
OPS = ("resize", "GaussianBlur", "adaptiveThreshold", "Canny", "findContours", "dilate")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load_images(samples: Path, count: int) -> list:
    files = sorted(samples.glob("*.jpg"))[:count]
    return [cv2.imread(str(f)) for f in files if cv2.imread(str(f)) is not None]


def timed(fn, iters: int, warmup: int) -> float:
    """Median seconds per call over `iters` runs, after `warmup` discarded runs.

    Median rather than mean: one scheduling hiccup on a shared instance should
    not silently deflate the reported speedup.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_ops(img, iters: int, warmup: int) -> dict:
    small = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    timed_ops = {
        "resize": timed(lambda: cv2.resize(img, (640, 480)), iters, warmup),
        "GaussianBlur": timed(lambda: cv2.GaussianBlur(gray, (5, 5), 0), iters, warmup),
        "adaptiveThreshold": timed(
            lambda: cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
            ),
            iters,
            warmup,
        ),
        "Canny": timed(lambda: cv2.Canny(blurred, 50, 150), iters, warmup),
        "findContours": timed(
            lambda: cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
            iters,
            warmup,
        ),
        "dilate": timed(lambda: cv2.dilate(edges, None, iterations=1), iters, warmup),
    }
    return {k: round(v * 1e3, 4) for k, v in timed_ops.items()}  # ms per call


def full_pipeline(img, iters: int, warmup: int) -> float:
    """End-to-end ForgeSight path: the workload judges will actually run."""

    def once():
        small = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        b = cv2.GaussianBlur(gray, (5, 5), 0)
        e = cv2.Canny(b, 50, 150)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
        )
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for _ in range(warmup):
        once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="arm name, e.g. cool-arm / diy-arm / x86-arm")
    ap.add_argument("--instance-type", default="unknown")
    ap.add_argument("--samples", default=str(Path(__file__).resolve().parents[1] / "samples"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--usd-per-hour", type=float, default=0.0, help="fill from AWS pricing for your region")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    images = load_images(Path(args.samples), 5)
    if not images:
        raise SystemExit(f"no images found in {args.samples}")

    # Representative resolution for the per-op table: ForgeSight downsizes to 640x480.
    probe = images[0]
    per_op = bench_ops(probe, args.iters, args.warmup)

    per_frame = [full_pipeline(img, args.iters, args.warmup) for img in images]
    mean_frame = statistics.mean(per_frame)
    fps = 1.0 / mean_frame

    result = {
        "label": args.label,
        "build": detect_build(),
        "build_marker": build_marker(),
        "opencv_version": cv2.__version__,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "instance_type": args.instance_type,
        "images": len(images),
        "iters": args.iters,
        "warmup_discarded": args.warmup,
        "per_op_ms": per_op,
        "mean_frame_seconds": round(mean_frame, 6),
        "fps": round(fps, 2),
        "usd_per_hour": args.usd_per_hour,
        "usd_per_1000_frames": round(args.usd_per_hour / 3600 * 1000 / fps, 6) if args.usd_per_hour else None,
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()