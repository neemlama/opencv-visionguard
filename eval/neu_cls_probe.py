"""Feasibility check: can OpenCV features + a NumPy classifier beat 90% on NEU-CLS?

Run BEFORE committing to a design. NEU-CLS is 1800 images / 6 classes, so a
meaningful held-out split is possible - unlike our 5 synthetic samples.

No sklearn or torch in this environment (and no cv2.ml in the OpenCV 5 wheel), so
the classifier head is implemented directly in NumPy. If accuracy is poor here,
the whole learned-classifier direction is wrong and we should stay classical and
report the low F1 honestly instead.

    python eval/neu_cls_probe.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLS_DIR = ROOT / "NEU-CLS" / "NEU-CLS"
CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")  # crazing, inclusion, patches,
# pitted_surface, rolled-in_scale, scratches  (NEU-CLS filename prefixes)
CLASS_NAMES = {
    "Cr": "crazing", "In": "inclusion", "Pa": "patches",
    "PS": "pitted_surface", "RS": "rolled-in_scale", "Sc": "scratches",
}

GABOR_SCALES = (3, 5, 9, 15)
GABOR_ORIENTS = (0, 45, 90, 135)
GABOR_SIGMA = 2.0
GABOR_GAMMA = 0.5


def gabor_bank():
    kernels = []
    for s in GABOR_SCALES:
        for o in GABOR_ORIENTS:
            k = cv2.getGaborKernel((0, 0), s, np.deg2rad(o), GABOR_SIGMA, 0, GABOR_GAMMA, ktype=cv2.CV_64F)
            kernels.append(k)
    return kernels


KERNELS = gabor_bank()


def describe(gray, out):
    """Append a fixed-length descriptor of `gray` to list `out`."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)  # illumination-invariant

    # Gabor energy: local orientation/selectivity signature of the texture
    for k in KERNELS:
        r = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
        e = float(np.sqrt((r * r).mean()))
        out.append(e)
        out.append(float(np.abs(r).mean()))

    # Gradient statistics
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out.append(float(mag.mean()))
    out.append(float(mag.std()))
    out.append(float(np.percentile(mag, 95)))
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hist, _ = np.histogram(ang, bins=8, range=(0, 1), weights=mag)
    hist = hist / (hist.sum() + 1e-6)
    out.extend(hist.tolist())

    # Multi-scale local anomaly energy (same idea as the localiser)
    for k in (9, 21, 41):
        bg = cv2.blur(g, (k, k))
        d = cv2.absdiff(g, bg)
        out.append(float(d.mean()))
        out.append(float(d.std()))
        out.append(float((d > 12).mean()))

    # Coarse intensity profile
    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    out.extend((small.astype(np.float32) / 255.0).ravel().tolist())

    # Laplacian energy (surface roughness / pit texture)
    out.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))


def load_image(p, size=128):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------- NumPy classifier heads ----------

def nearest_centroid(Xtr, ytr, Xte, n_classes):
    cents = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(n_classes)])
    var = Xtr.var(axis=0) + 1e-6
    d = ((Xte[:, None, :] - cents[None, :, :]) ** 2 / var).sum(axis=2)
    return d.argmin(axis=1)


def logistic_regression(Xtr, ytr, Xte, n_classes, epochs=400, lr=0.5, l2=1e-3):
    """Multinomial logistic regression, full-batch gradient descent."""
    n, d = Xtr.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), ytr] = 1.0
    for _ in range(epochs):
        Z = Xtr @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (Xtr.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
    Z = Xte @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(axis=1, keepdims=True)
    return P.argmax(axis=1), P


def main():
    paths = []
    for c in CLASSES:
        ps = sorted(CLS_DIR.glob(f"{c}_*.bmp"))
        paths += [(p, CLASSES.index(c)) for p in ps]
    print(f"loaded {len(paths)} images, {len(CLASSES)} classes, "
          f"{len(KERNELS)} gabor filters")

    t0 = time.time()
    X, y = [], []
    for p, c in paths:
        g = load_image(p)
        if g is None:
            continue
        f = []
        describe(g, f)
        X.append(f)
        y.append(c)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    print(f"features {X.shape} in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / len(X) * 1000:.2f} ms/image)")

    # Stratified 70/30 split, fixed seed for reproducibility
    rng = np.random.default_rng(0)
    tr, te = [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(len(idx) * 0.7)
        tr += list(idx[:k])
        te += list(idx[k:])
    tr, te = np.array(tr), np.array(te)

    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    for name, fn in (("nearest_centroid", nearest_centroid), ("logreg", logistic_regression)):
        t = time.time()
        out = fn(Xn[tr], y[tr], Xn[te], len(CLASSES))
        pred, probs = (out if isinstance(out, tuple) else (out, None))
        acc = (pred == y[te]).mean()
        print(f"\n{name}: held-out accuracy = {acc * 100:.2f}%  ({time.time() - t:.1f}s)")
        cm = np.zeros((len(CLASSES), len(CLASSES)), int)
        for t_, p_ in zip(y[te], pred):
            cm[t_, p_] += 1
        print("      " + " ".join(f"{c:>5}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            rec = cm[i, i] / max(1, cm[i].sum())
            prec = cm[i, i] / max(1, cm[:, i].sum())
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            print(f"  {CLASS_NAMES[c]:<18}" + " ".join(f"{v:>5}" for v in cm[i])
                  + f"   R={rec:.2f} P={prec:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main()
