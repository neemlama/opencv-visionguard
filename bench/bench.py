"""ForgeSight benchmark: per-op timings + end-to-end, on COOL or stock OpenCV.

Must mirror backend/app/features.py + analysis.py EXACTLY, or it measures
something the product never runs. Every op below is copied from the application:

  quality   Laplacian sharpness
  features  16x Gabor filter2D, Sobel gx/gy, magnitude, orientation histogram,
            multi-scale box blur + absdiff, 4x4 resize, Laplacian
  localize  box blur(21) + absdiff + threshold + morphology + connected components

Self-identifies the library (COOL lives at /opt/cool on the Graviton4 image and
/opt/opencv_install on the Graviton2 image), so the JSON output is itself proof of
what executed.

    # inside a COOL AMI
    export LD_LIBRARY_PATH=/opt/opencv_install/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/opencv_install/lib/python3.12/site-packages
    python3.12 bench.py --label A --instance-type c8g.medium
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
IMAGE_SIZE = 128
GABOR_SCALES = (3, 9)
GABOR_ORIENTS = (0, 45, 90, 135)
ANOM_BOXES = (9, 21, 41)
ANOM_LEVEL = 12.0
LOC_BOX = 21
LOC_THRESH = 8.0
LOC_MIN_AREA = 400
LOC_CLOSE = 9


def detect_build():
    present = [str(p) for p in COOL_ROOTS if p.exists()]
    return ("COOL" if present else "DIY"), (", ".join(present) or "no COOL install found")


def gabor_bank():
    # Explicit ksize (6*sigma+1), not (0,0): OpenCV 5 auto-sizes (0,0) to this
    # but older builds try to allocate ~1.8e19 bytes and fail. Verified
    # bit-identical to the auto-sized kernel on 5.0.0.
    out = []
    for s in GABOR_SCALES:
        ksize = (1, 6 * s + 1)
        for o in GABOR_ORIENTS:
            out.append(cv2.getGaborKernel(ksize, s, np.deg2rad(o), 2.0, 0, 0.5,
                                          ktype=cv2.CV_64F))
    return out


KERNELS = gabor_bank()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def to_gray(image):
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if g.shape[0] != IMAGE_SIZE or g.shape[1] != IMAGE_SIZE:
        g = cv2.resize(g, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return g


def timed(fn, iters, warmup):
    """Median seconds per call after discarded warmups.

    Median, not mean: one scheduling hiccup on a shared instance must not
    deflate a reported speedup.
    """
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


def describe_ops(gray):
    """The feature descriptor, exactly as features.describe() computes it."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    feats = []
    for k in KERNELS:
        r = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
        feats.append(float(np.sqrt((r * r).mean())))
        feats.append(float(np.abs(r).mean()))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    feats += [float(mag.mean()), float(mag.std()), float(np.percentile(mag, 95))]
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hist, _ = np.histogram(ang, bins=8, range=(0, 1), weights=mag)
    feats += (hist / (hist.sum() + 1e-6)).tolist()
    for k in ANOM_BOXES:
        bg = cv2.blur(g, (k, k))
        d = cv2.absdiff(g, bg)
        feats += [float(d.mean()), float(d.std()), float((d > ANOM_LEVEL).mean())]
    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    feats += (small.astype(np.float32) / 255.0).ravel().tolist()
    feats.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    return np.asarray(feats, dtype=np.float64)


def localize(gray):
    g = gray.astype(np.float32)
    bg = cv2.blur(g, (LOC_BOX, LOC_BOX))
    anom = cv2.absdiff(g, bg)
    _, binary = cv2.threshold(anom, LOC_THRESH, 1.0, cv2.THRESH_BINARY)
    mask = (binary * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (LOC_CLOSE, LOC_CLOSE)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n - 1, anom


def full_pipeline(image, iters, warmup):
    def once():
        gray = to_gray(image)
        feats = describe_ops(gray)
        localize(gray)
        return feats

    for _ in range(warmup):
        once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    return (time.perf_counter() - t0) / iters


def per_op(gray, iters, warmup):
    g = gray.astype(np.float32)
    gn = (g - g.mean()) / (g.std() + 1e-6)
    blur_bg = cv2.blur(gn, (LOC_BOX, LOC_BOX))
    anom = cv2.absdiff(gn, blur_bg)
    gx = cv2.Sobel(gn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gn, cv2.CV_32F, 0, 1, ksize=3)
    edges_mask = (anom > LOC_THRESH).astype(np.uint8) * 255

    ops = {
        # The headline op: 16 convolution responses, the bulk of feature cost.
        "gabor_filter2D_x16": timed(
            lambda: [cv2.filter2D(gn, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
                     for k in KERNELS], iters, warmup),
        "sobel_x2": timed(lambda: (cv2.Sobel(gn, cv2.CV_32F, 1, 0, ksize=3),
                                   cv2.Sobel(gn, cv2.CV_32F, 0, 1, ksize=3)), iters, warmup),
        "magnitude": timed(lambda: cv2.magnitude(gx, gy), iters, warmup),
        "blur_box9": timed(lambda: cv2.blur(gn, (9, 9)), iters, warmup),
        "blur_box21": timed(lambda: cv2.blur(gn, (LOC_BOX, LOC_BOX)), iters, warmup),
        "blur_box41": timed(lambda: cv2.blur(gn, (41, 41)), iters, warmup),
        "absdiff": timed(lambda: cv2.absdiff(gn, blur_bg), iters, warmup),
        "threshold": timed(lambda: cv2.threshold(anom, LOC_THRESH, 1.0, cv2.THRESH_BINARY),
                           iters, warmup),
        "morph_close": timed(
            lambda: cv2.morphologyEx(edges_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT,
                                                               (LOC_CLOSE, LOC_CLOSE))),
            iters, warmup),
        "connected_components": timed(
            lambda: cv2.connectedComponentsWithStats(edges_mask, connectivity=8),
            iters, warmup),
        "laplacian": timed(lambda: cv2.Laplacian(gn, cv2.CV_32F), iters, warmup),
        "resize_128": timed(lambda: cv2.resize(g, (IMAGE_SIZE, IMAGE_SIZE),
                                               interpolation=cv2.INTER_AREA), iters, warmup),
        "cvtColor": timed(lambda: cv2.cvtColor(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                                                cv2.COLOR_BGR2GRAY), iters, warmup),
        # Kept for continuity with the earlier weld benchmark, and as the known
        # COOL regression we removed from the pipeline. OpenCV 5 requires uint8
        # input for both of these (adaptiveThreshold asserts CV_8UC1).
        "adaptiveThreshold_legacy": timed(
            lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 31, 9), iters, warmup),
        "canny_legacy": timed(lambda: cv2.Canny(gray, 50, 150), iters, warmup),
    }
    return {k: round(v * 1e3, 4) for k, v in ops.items()}


def load_images(samples, limit):
    files = sorted(samples.glob("*.jpg"))[:limit]
    imgs = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            imgs.append(img)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--instance-type", default="unknown")
    ap.add_argument("--samples", default=str(Path(__file__).resolve().parents[1] / "samples"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--usd-per-hour", type=float, default=0.0)
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    build, marker = detect_build()
    images = load_images(Path(args.samples), args.limit)
    if not images:
        raise SystemExit(f"no images found in {args.samples}")

    probe = to_gray(images[0])
    ops = per_op(probe, args.iters, args.warmup)
    per_frame = [full_pipeline(im, args.iters, args.warmup) for im in images]
    mean_frame = statistics.mean(per_frame)
    fps = 1.0 / mean_frame

    result = {
        "label": args.label,
        "pipeline": "neu-steel-surface (features+classify+localize)",
        "build": build,
        "build_marker": marker,
        "opencv_version": cv2.__version__,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "instance_type": args.instance_type,
        "images": len(images),
        "iters": args.iters,
        "warmup_discarded": args.warmup,
        "per_op_ms": ops,
        "mean_frame_seconds": round(mean_frame, 6),
        "fps": round(fps, 2),
        "usd_per_hour": args.usd_per_hour,
        "usd_per_1000_frames": (round(args.usd_per_hour / 3600 * 1000 / fps, 6)
                                if args.usd_per_hour else None),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
