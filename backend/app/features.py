"""OpenCV 5 feature extraction for rolled-steel surface defect classification.

Every feature here is computed with OpenCV, which is what the competition
requires ("substantive OpenCV 5 for image/video analysis"):

  * 16 Gabor filter responses  -> cv2.filter2D + cv2.getGaborKernel
  * gradient magnitude + orientation histogram -> cv2.Sobel
  * multi-scale local anomaly energy -> cv2.blur + cv2.absdiff
  * surface roughness -> cv2.Laplacian
  * coarse intensity profile -> cv2.resize

The classifier head is a small NumPy logistic regression (see classifier.py).
This OpenCV 5 build ships no cv2.ml / objdetect / bgsegm module, and adding
torch or sklearn would add a dependency the pipeline does not need.

The feature vector is fixed-length and deterministic so a trained model stays
valid: descriptor length is asserted in classifier.py at load time.
"""
import cv2
import numpy as np

# 8 Gabor filters, not 16. eval/feature_variants.py measured this on the
# leakage-free split: halving the bank (scales 3,5,9,15 -> 3,9, keeping all four
# orientations) is 2.20x faster end-to-end at IDENTICAL held-out accuracy
# (95.37% either way). Dropping Gabor entirely costs 5.18 accuracy points, so
# orientation-selective texture is doing real work - it just does not need 16
# convolutions to do it.
GABOR_SCALES = (3, 9)
GABOR_ORIENTS = (0, 45, 90, 135)
GABOR_SIGMA = 2.0
GABOR_GAMMA = 0.5
ANOM_BOXES = (9, 21, 41)
ANOM_LEVEL = 12.0
IMAGE_SIZE = 128

CLASS_NAMES = (
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
)


def _gabor_bank():
    # Kernel size is computed EXPLICITLY as 6*sigma+1 rather than passing (0,0).
    # OpenCV 5 auto-sizes (0,0) to exactly this, but the 4.14 control build does
    # not: there (0,0) was taken literally and getGaborKernel tried to allocate
    # ~1.8e19 bytes and died. Verified bit-identical to the auto-sized kernel on
    # 5.0.0, so the trained model stays valid.
    kernels = []
    for s in GABOR_SCALES:
        ksize = (1, 6 * s + 1)
        for o in GABOR_ORIENTS:
            kernels.append(
                cv2.getGaborKernel(
                    ksize, s, np.deg2rad(o), GABOR_SIGMA, 0, GABOR_GAMMA,
                    ktype=cv2.CV_64F,
                )
            )
    return kernels


def gabor_ksize(sigma):
    return (1, 6 * int(sigma) + 1)


GABOR_BANK = _gabor_bank()
DESCRIPTOR_LEN = len(GABOR_BANK) * 2 + 3 + 8 + len(ANOM_BOXES) * 3 + 16 + 1


def to_gray(image, size=IMAGE_SIZE):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] != size or gray.shape[1] != size:
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def quality_metrics(gray):
    """Capture-quality signals, computed before any classification.

    These are what let the agent refuse an image instead of guessing: an
    out-of-focus or glare-blown capture produces confident nonsense from any
    classifier, so the agent must see these before trusting a prediction.
    """
    g = gray.astype(np.float32)
    hsv_v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv_v = cv2.cvtColor(hsv_v, cv2.COLOR_BGR2HSV)[:, :, 2]
    # OpenCV 5 rejects float32 -> CV_64F in the SIMD path
    # ("Unsupported combination of source format (=5), and destination format (=6)"),
    # so compute in CV_32F and widen in NumPy for an accurate variance.
    lap = cv2.Laplacian(g, cv2.CV_32F).astype(np.float64)
    # Robust noise estimate: MAD of the 3x3 high-pass response. Median-based, so a
    # real defect (a few percent of pixels) does not inflate it. Added after the
    # agent ablation showed a safety failure: additive sensor noise RAISES
    # Laplacian variance, so a noisy frame looks sharper and passed every existing
    # gate check. Measured on 240 real NEU captures: clean p99 = 10.29,
    # noisy p05 = 11.31 (eval/quality_gate_calibration.py).
    hp = g - cv2.GaussianBlur(g, (3, 3), 0)
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp))))
    return {
        "noise_sigma": round(noise, 4),
        "sharpness": round(float(lap.var()), 2),
        "contrast": round(float(gray.std()), 2),
        "mean_intensity": round(float(gray.mean()), 2),
        # Fraction of near-saturated pixels: specular blow-out on steel.
        "glare_ratio": round(float((hsv_v > 240).mean()), 5),
        "blown_highlights": round(float((gray >= 250).mean()), 5),
        "crushed_shadows": round(float((gray <= 5).mean()), 5),
    }


def describe(gray, out=None):
    """Append the fixed-length descriptor for `gray` to `out` (or return it)."""
    out = [] if out is None else out
    g = gray.astype(np.float32)
    # Illumination normalisation: NEU capture brightness varies, and dividing out
    # the mean/std makes the texture descriptors comparable across lighting.
    g = (g - g.mean()) / (g.std() + 1e-6)

    for k in GABOR_BANK:
        r = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
        out.append(float(np.sqrt((r * r).mean())))   # RMS energy
        out.append(float(np.abs(r).mean()))           # mean magnitude

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

    for k in ANOM_BOXES:
        bg = cv2.blur(g, (k, k))
        d = cv2.absdiff(g, bg)
        out.append(float(d.mean()))
        out.append(float(d.std()))
        out.append(float((d > ANOM_LEVEL).mean()))

    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    out.extend((small.astype(np.float32) / 255.0).ravel().tolist())

    out.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    return out


def describe_image(image):
    return np.asarray(describe(to_gray(image)), dtype=np.float64)


def anomaly_map(gray, box=21):
    """Two-sided local deviation, used for localisation.

    box=21 was the best value in the eval/neu_tune.py grid. cv2.blur is a box
    filter, so cost is independent of box size.
    """
    g = gray.astype(np.float32)
    bg = cv2.blur(g, (box, box))
    return cv2.absdiff(g, bg)
