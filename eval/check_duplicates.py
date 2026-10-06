"""Check for duplicate / near-duplicate images across NEU-CLS and NEU-DET.

This gates the headline accuracy number. NEU-CLS is 1800 images and NEU-DET is
1800 images from the same source survey. If DET is a re-encode of CLS, then
combining them puts the SAME photograph on both sides of the train/test split
and inflates accuracy. That is exactly what we did when we trained on both.

Method: perceptual hashing rather than exact file hashes, because the two sets
differ in format (BMP vs JPG) and compression even when the photo is identical.

  pHash  : 32x32 DCT -> 64-bit, Hamming distance <= 6 = near-duplicate
  dHash  : 9x8 grayscale -> 64-bit, catches re-encode/gamma differences
  Exact  : md5 of raw pixels

Also reports same-class vs cross-class duplicates, because a duplicate pair with
the SAME label is leakage, while a cross-class duplicate would mean contradictory
labels (a different, more serious problem).

    python eval/check_duplicates.py
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLS_DIR = ROOT / "NEU-CLS" / "NEU-CLS"
DET_DIR = ROOT / "NEU-DET" / "NEU-DET" / "IMAGES"

CLS_PREFIX = ("Cr", "In", "Pa", "PS", "RS", "Sc")
CLASS_NAMES = ("crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches")
PHASH_THRESHOLD = 6
DHASH_THRESHOLD = 6


def phash(img):
    # cv2.dct requires a single channel (CV_32FC1/CV_64FC1), so grayscale first.
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    med = np.median(d[1:, 1:])
    return (d > med).flatten()


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def collect():
    items = []
    for ci, p in enumerate(CLS_PREFIX):
        for f in sorted(CLS_DIR.glob(f"{p}_*.bmp")):
            img = cv2.imread(str(f))
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            items.append({
                "set": "cls", "path": f, "label": CLASS_NAMES[ci],
                "ph": phash(img), "dh": dhash(img),
                "exact": hashlib.md5(g.tobytes()).hexdigest(),
            })
    for ci, p in enumerate(CLASS_NAMES):
        for f in sorted(DET_DIR.glob(f"{p}_*.jpg")):
            img = cv2.imread(str(f))
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            items.append({
                "set": "det", "path": f, "label": CLASS_NAMES[ci],
                "ph": phash(img), "dh": dhash(img),
                "exact": hashlib.md5(g.tobytes()).hexdigest(),
            })
    return items


def main():
    items = collect()
    print(f"loaded {sum(1 for i in items if i['set'] == 'cls')} CLS + "
          f"{sum(1 for i in items if i['set'] == 'det')} DET images")

    # Exact pixel duplicates
    by_exact = defaultdict(list)
    for it in items:
        by_exact[it["exact"]].append(it)
    exact_dupes = [v for v in by_exact.values() if len(v) > 1]
    print(f"\nexact pixel duplicates: {len(exact_dupes)} groups")
    cross_exact = 0
    for g in exact_dupes[:10]:
        sets = {x["set"] for x in g}
        labels = {x["label"] for x in g}
        if len(sets) > 1:
            cross_exact += 1
        print(f"  {len(g)} copies  sets={sorted(sets)}  labels={sorted(labels)}  "
              f"e.g. {g[0]['path'].name}")

    # Perceptual near-duplicates across the two sets (CLS is small, DET is large,
    # so only cross-set comparison is O(n*m) with n=1800, m=1800 = 3.2M - fine).
    cls = [i for i in items if i["set"] == "cls"]
    det = [i for i in items if i["set"] == "det"]
    print(f"\nperceptual comparison: {len(cls)} CLS x {len(det)} DET "
          f"= {len(cls) * len(det):,} pairs ...")
    ph_dupes, dh_dupes, same_label, cross_label = [], [], 0, []
    for a in cls:
        for b in det:
            dp = hamming(a["ph"], b["ph"])
            dd = hamming(a["dh"], b["dh"])
            if dp <= PHASH_THRESHOLD or dd <= DHASH_THRESHOLD:
                ph_dupes.append((dp, dd, a, b))
                if a["label"] == b["label"]:
                    same_label += 1
                else:
                    cross_label.append((a, b, dp, dd))

    ph_dupes.sort(key=lambda t: (t[0] + t[1]))
    strict = [(dp, dd, a, b) for dp, dd, a, b in ph_dupes if dp == 0 and dd == 0]
    strict_same = [(dp, dd, a, b) for dp, dd, a, b in strict
                   if a["label"] == b["label"]]
    print(f"near-duplicate pairs (pHash<={PHASH_THRESHOLD} or "
          f"dHash<={DHASH_THRESHOLD}): {len(ph_dupes)}")
    print(f"  STRICT (pHash==0 AND dHash==0, i.e. same photograph): {len(strict)}")
    print(f"     of which same label: {len(strict_same)}")
    print(f"  same label  (LEAKAGE across the split): {same_label}")
    print(f"  cross label: {len(cross_label)}  <- mostly pHash collisions at this")
    print(f"     threshold; a 64-bit 8x8 DCT hash is not discriminative enough to")
    print(f"     treat a non-zero distance as identity evidence.")
    for dp, dd, a, b in strict[:8]:
        flag = "same-label LEAK" if a["label"] == b["label"] else "cross-label"
        print(f"    STRICT {flag:<18} {a['label']:<16} "
              f"{a['path'].name:<20} <-> {b['path'].name}")

    # Within-set duplicates also split across train/test
    within = 0
    for name, group in (("cls", cls), ("det", det)):
        idx = defaultdict(list)
        for i in group:
            idx[i["exact"]].append(i)
        within += sum(1 for v in idx.values() if len(v) > 1)
    print(f"\nwithin-set exact duplicate groups: {within}")

    out = {
        "cls": len(cls), "det": len(det),
        "exact_duplicate_groups": len(exact_dupes),
        "cross_set_near_duplicates": len(ph_dupes),
        "same_label_near_duplicates": same_label,
        "cross_label_near_duplicates": len(cross_label),
        "within_set_exact_duplicate_groups": within,
        "verdict": ("LEAKAGE: near-duplicate pairs span the two sets with the same "
                    "label, so the combined split is optimistic"
                    if same_label else
                    "no cross-set leakage detected"),
        "examples": [
            {"cls": str(a["path"].name), "det": str(b["path"].name),
             "label": a["label"], "phash": dp, "dhash": dd}
            for dp, dd, a, b in ph_dupes[:25]
        ],
    }
    Path(ROOT / "eval" / "duplicate_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}")
    print("wrote eval/duplicate_check.json")


if __name__ == "__main__":
    main()
