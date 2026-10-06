"""Accuracy-versus-latency study across feature variants.

The production descriptor runs 16 Gabor convolutions, which is 82% of pipeline
time. Two questions this answers, both legitimate product questions rather than
benchmark-tuning:

  1. Does a cheaper descriptor classify NEU just as well? If yes, shipping it is
     better engineering - the product gets faster at no accuracy cost.
  2. What does COOL do to each variant? If COOL's advantage is concentrated in
     the ops a cheaper variant avoids, the parity result on the full pipeline is
     explained rather than merely reported.

Each variant is trained on the SAME leakage-free split and evaluated on the SAME
held-out test set, so the comparison is fair.

    python eval/feature_variants.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.classifier import fit, fit_temperature  # noqa: E402
from eval.splits import build, grouped_split  # noqa: E402

CLEAN_IDX = 6
ANOM_BOXES = (9, 21, 41)
ANOM_LEVEL = 12.0
CLEAN_CLASSES = ("crazing", "inclusion", "patches", "pitted_surface",
                 "rolled-in_scale", "scratches", "no_defect")


def gabor(scale, orient):
    return cv2.getGaborKernel((1, 6 * scale + 1), scale, np.deg2rad(orient),
                              2.0, 0, 0.5, ktype=cv2.CV_64F)


def common(gray, out):
    """Ops present in every variant: gradients, orientation histogram,
    multi-scale anomaly energy, coarse profile, roughness."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out += [float(mag.mean()), float(mag.std()), float(np.percentile(mag, 95))]
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hist, _ = np.histogram(ang, bins=8, range=(0, 1), weights=mag)
    out += (hist / (hist.sum() + 1e-6)).tolist()
    for k in ANOM_BOXES:
        d = cv2.absdiff(g, cv2.blur(g, (k, k)))
        out += [float(d.mean()), float(d.std()), float((d > ANOM_LEVEL).mean())]
    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    out += (small.astype(np.float32) / 255.0).ravel().tolist()
    out.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    return out


def gabor_block(gray, scales, orients):
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    out = []
    for s in scales:
        for o in orients:
            r = cv2.filter2D(g, cv2.CV_32F, gabor(s, o), borderType=cv2.BORDER_REPLICATE)
            out += [float(np.sqrt((r * r).mean())), float(np.abs(r).mean())]
    return out


VARIANTS = {
    # name: (scales, orientations)
    "gabor16 (production)": ((3, 5, 9, 15), (0, 45, 90, 135)),
    "gabor8": ((3, 9), (0, 45, 90, 135)),
    "gabor4": ((5,), (0, 90)),
    "gabor0 (gradients only)": ((), ()),
}


def descriptor(gray, scales, orients):
    out = []
    if scales:
        out += gabor_block(gray, scales, orients)
    out = common(gray, out)
    return np.asarray(out, dtype=np.float64)


def to_gray(img, size=128):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def median_time(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return float(np.median(s) * 1e3)


def main():
    print("loading leakage-free split ...")
    images = []
    from eval.splits import dedupe, enumerate_files
    for item, _n, _g in dedupe(enumerate_files()):
        img = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((to_gray(img), CLEAN_CLASSES.index(item["label"])))
    y = np.array([lab for _, lab in images])
    print(f"  {len(images)} unique source images")

    # Identical group structure to eval/splits.py: group = (class, index)
    from eval.splits import identity_group
    groups = np.array([identity_group(CLEAN_CLASSES[l], i)
                       for i, (_, l) in enumerate(images)])
    tr, va, te = grouped_split(y, groups)

    rows = []
    for name, (scales, orients) in VARIANTS.items():
        X = np.stack([descriptor(g, scales, orients) for g, _ in images])
        W, b, mu, sd = fit(X[tr], y[tr], 7, epochs=400, lr=0.5, l2=1e-3)
        Xn = (X - mu) / sd
        T, _ = fit_temperature(Xn[va], y[va], W, b)

        Z = (Xn[te] @ W + b) / T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        acc = float((P.argmax(1) == y[te]).mean())

        probe = images[0][0]
        ms = median_time(lambda: descriptor(probe, scales, orients))
        n_gabor = len(scales) * len(orients)
        rows.append({"variant": name, "gabor_filters": n_gabor,
                     "descriptor_len": int(X.shape[1]), "ms_per_frame": round(ms, 3),
                     "fps": round(1000 / ms, 1), "accuracy": round(acc, 4),
                     "temperature": round(T, 3)})
        print(f"  {name:<26} gabor={n_gabor:<3} len={X.shape[1]:<4} "
              f"{ms:6.3f} ms  {1000 / ms:7.1f} fps  acc={acc * 100:6.2f}%")

    base = rows[0]
    print(f"\nrelative to {base['variant']}:")
    for r in rows:
        print(f"  {r['variant']:<26} speed {base['ms_per_frame'] / r['ms_per_frame']:.2f}x  "
              f"accuracy {r['accuracy'] * 100 - base['accuracy'] * 100:+.2f} pts")

    Path(ROOT / "eval" / "feature_variants.json").write_text(json.dumps({
        "note": "all variants trained and tested on the same leakage-free split",
        "variants": rows}, indent=2))
    print("\nwrote eval/feature_variants.json")


if __name__ == "__main__":
    main()
