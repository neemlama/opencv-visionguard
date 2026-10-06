"""Generate synthetic inspection samples: clean weld, defect, blurry, glare, damage.

Deterministic by construction. Every pixel is drawn from a fixed RNG seed, so
regenerating on any machine or architecture yields byte-identical images. That
matters for the COOL benchmark: arms A/B/C must all measure the *same* input, or
the comparison is meaningless. Unseeded noise here would have silently produced
different images per instance.

Usage:
    python -m backend.app.make_samples              # write to <repo>/samples
    python make_samples.py /tmp/samples             # write to a given directory
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Fixed seed: identical images on every arm, every run, every architecture.
SEED = 20261006


def base_plate(rng, w=640, h=480):
    img = np.full((h, w, 3), 140, np.uint8)
    noise = rng.integers(-12, 12, (h, w, 1), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_weld(img, y, defect=False):
    row = slice(y, y + 26)
    cv2.rectangle(img, (40, row.start), (600, row.stop), (198, 198, 198), -1)
    for x in range(40, 600, 7):
        cv2.circle(img, (x, y + 13), 5, (170, 170, 170), -1)
    if defect:
        cv2.circle(img, (320, y + 13), 13, (38, 38, 38), -1)
        cv2.circle(img, (320, y + 13), 6, (10, 10, 10), -1)
    return img


def generate(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = add_weld(base_plate(np.random.default_rng(SEED)), 120, defect=False)
    cv2.imwrite(str(out_dir / "weld_clean.jpg"), clean)

    defect = add_weld(base_plate(np.random.default_rng(SEED)), 120, defect=True)
    for i in range(14):
        cv2.circle(defect, (250 + i * 22, 300 + (i % 3) * 9), 7, (60, 60, 60), -1)
    cv2.imwrite(str(out_dir / "weld_defect.jpg"), defect)

    cv2.imwrite(
        str(out_dir / "weld_blurry.jpg"), cv2.GaussianBlur(clean, (0, 0), 7.0)
    )

    glare = clean.copy()
    cv2.circle(glare, (300, 240), 150, (255, 255, 255), -1)
    cv2.imwrite(
        str(out_dir / "weld_glare.jpg"),
        cv2.addWeighted(glare, 0.55, np.full_like(glare, 255), 0.45, 0),
    )

    damage = add_weld(base_plate(np.random.default_rng(SEED)), 120, defect=True)
    for i in range(26):
        cv2.circle(damage, (120 + i * 18, 130 + (i % 5) * 5), 10, (25, 25, 25), -1)
    for i in range(12):
        cv2.ellipse(
            damage, (300 + i * 20, 340), (16, 9), 0, 0, 360, (45, 45, 45), -1
        )
    cv2.imwrite(str(out_dir / "weld_damage.jpg"), damage)

    return sorted(out_dir.glob("*.jpg"))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "samples"
    )
    files = generate(target)
    print("wrote:", ", ".join(f.name for f in files))