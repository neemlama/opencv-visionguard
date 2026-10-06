"""Compare dark-blob detector variants: speed AND recall on the samples.

The C-01 fix needs to keep finding dark pits (recall), and the COOL story needs
the op to be one COOL actually accelerates (speed). These can conflict, so both
are measured together rather than assuming.

Recall proxy: for each variant, count blobs >400px on the defect-bearing samples
and confirm clean stays at 0.
"""
import time
from pathlib import Path

import cv2
import numpy as np

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
ITER = 20
WARMUP = 3


def t(fn, iter=ITER, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    best = []
    for _ in range(iter):
        s = time.perf_counter()
        fn()
        best.append(time.perf_counter() - s)
    return float(np.median(best)) * 1e3


def blobs(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for x in c if cv2.contourArea(x) > 400)


def v_adaptive(gray):
    m = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )
    return blobs(m)


def gauss_tophat(sigma, contrast=18):
    def f(gray):
        bg = cv2.GaussianBlur(gray, (0, 0), sigma)
        th = cv2.subtract(bg, gray)
        _, m = cv2.threshold(th, contrast, 255, cv2.THRESH_BINARY)
        return blobs(m)
    return f


def box_tophat(k, contrast=18):
    def f(gray):
        bg = cv2.blur(gray, (k, k))
        th = cv2.subtract(bg, gray)
        _, m = cv2.threshold(th, contrast, 255, cv2.THRESH_BINARY)
        return blobs(m)
    return f


def median_tophat(k, contrast=18):
    def f(gray):
        bg = cv2.medianBlur(gray, k if k % 2 else k + 1)
        th = cv2.subtract(bg, gray)
        _, m = cv2.threshold(th, contrast, 255, cv2.THRESH_BINARY)
        return blobs(m)
    return f


def morph_blackhat(k, contrast=18):
    def f(gray):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
        _, m = cv2.threshold(th, contrast, 255, cv2.THRESH_BINARY)
        return blobs(m)
    return f


VARIANTS = [
    ("adaptiveThreshold(31,9)", v_adaptive),
    ("gaussTophat sigma=4", gauss_tophat(4.0)),
    ("gaussTophat sigma=6", gauss_tophat(6.0)),
    ("gaussTophat sigma=12", gauss_tophat(12.0)),
    ("boxTophat k=13", box_tophat(13)),
    ("boxTophat k=25", box_tophat(25)),
    ("boxTophat k=41", box_tophat(41)),
    ("medianTophat k=15", median_tophat(15)),
    ("morphBlackhat k=15", morph_blackhat(15)),
    ("morphBlackhat k=25", morph_blackhat(25)),
]


def main():
    files = sorted(SAMPLES.glob("*.jpg"))
    grays = {}
    for f in files:
        img = cv2.imread(str(f))
        small = cv2.resize(img, (640, 480))
        grays[f.stem] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    probe = grays["weld_clean"]
    names = list(grays)

    print(f"{'variant':<26}{'ms/call':>9}  " + "  ".join(f"{n.replace('weld_',''):>9}" for n in names))
    print("-" * (36 + 11 * len(names)))
    for label, fn in VARIANTS:
        ms = t(lambda: fn(probe))
        counts = []
        for n in names:
            try:
                counts.append(f"{fn(grays[n]):>9}")
            except Exception as e:
                counts.append(f"{'err':>9}")
        print(f"{label:<26}{ms:>9.3f}  " + "  ".join(counts))

    print("\nwant: clean=0, defect>0, damage>0 (recall), blurry/glare small")


if __name__ == "__main__":
    main()