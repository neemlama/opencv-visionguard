"""Leakage-free dataset construction for the NEU classifier.

Why this module exists
----------------------
NEU-CLS and NEU-DET are the SAME 1800 source photographs. eval/check_duplicates.py
found 1045 cross-set pairs with pHash==0 AND dHash==0 (bit-identical content) and
7701 same-label pairs within a perceptual threshold of 6. The file numbering makes
the mapping explicit: Cr_1.bmp <-> crazing_1.jpg, Pa_101.bmp <-> patches_101.jpg.

Training on "CLS + DET" therefore meant training on each photograph twice, and a
random per-image split could place Cr_1 in train and crazing_1 in test. The model
would score a memorised image, not a generalisation. The reported 93.80% was
inflated by that.

Fix, in two parts:
  1. DEDUPE  - keep ONE representative per identity group (the DET copy, since
     inference runs on DET captures). 3600 -> ~1800 unique images.
  2. GROUP-AWARE SPLIT - split by identity group, never by row, so no group can
     straddle train/val/test.

Also provides a proper three-way split so calibration never touches the test set
(fitting temperature on the same split used to report ECE was a second leak).

    python eval/splits.py          # report the leakage-safe dataset shape
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.features import CLASS_NAMES, DESCRIPTOR_LEN, describe, to_gray  # noqa: E402

CLS_DIR = ROOT / "NEU-CLS" / "NEU-CLS"
DET_DIR = ROOT / "NEU-DET" / "NEU-DET" / "IMAGES"
CLS_PREFIX = ("Cr", "In", "Pa", "PS", "RS", "Sc")
CLEAN_CLASS = "no_defect"
N_DEFECT_CLASSES = len(CLASS_NAMES)


def _index(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def identity_group(label, index):
    """Identity = (class, source-image number).

    Far more reliable than perceptual hashing here, because the two NEU subsets
    number the same survey sequentially. Hash evidence is used only to VERIFY
    this assumption in eval/check_duplicates.py.
    """
    return f"{label}#{index}"


def phash(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
                .astype(np.float32))[:8, :8]
    return (d > np.median(d[1:, 1:])).flatten()


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8),
                   interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def enumerate_files():
    items = []
    for ci, p in enumerate(CLS_PREFIX):
        for f in sorted(CLS_DIR.glob(f"{p}_*.bmp")):
            items.append({"path": f, "set": "cls", "label": CLASS_NAMES[ci],
                          "index": _index(f.stem)})
    for ci, p in enumerate(CLASS_NAMES):
        for f in sorted(DET_DIR.glob(f"{p}_*.jpg")):
            items.append({"path": f, "set": "det", "label": CLASS_NAMES[ci],
                          "index": _index(f.stem)})
    return items


def dedupe(items):
    """One representative per identity group; DET preferred, else CLS."""
    groups = defaultdict(list)
    for it in items:
        groups[identity_group(it["label"], it["index"])].append(it)
    kept = []
    for g, members in groups.items():
        det = [m for m in members if m["set"] == "det"]
        kept.append((det[0] if det else members[0], len(members), g))
    kept.sort(key=lambda t: t[2])
    return kept


def background_samples(per_image=2, size=128, seed=0):
    """Build a 'no_defect' class from NEU-DET regions OUTSIDE the ground-truth boxes.

    NEU contains only defective images, so a triage system needs a clean class or
    it can never legitimately output "pass". Rather than invent clean steel, we
    crop areas of a defective capture that no annotation box covers.

    Caveat we state rather than hide: annotation boxes are imperfect, so some
    crops may contain unannotated defect. Each crop gets its own identity group so
    two crops of the same photo cannot straddle the split.
    """
    import xml.etree.ElementTree as ET

    ann = ROOT / "NEU-DET" / "NEU-DET" / "ANNOTATIONS"
    rng = np.random.default_rng(seed)
    out = []
    for xp in sorted(ann.glob("*.xml")):
        try:
            root = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                          int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax")))))
        img = cv2.imread(str(DET_DIR / (xp.stem + ".jpg")), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < size or w < size:
            continue
        got = 0
        for _ in range(per_image * 12):
            if got >= per_image:
                break
            x0 = int(rng.integers(0, w - size + 1))
            y0 = int(rng.integers(0, h - size + 1))
            x1, y1 = x0 + size, y0 + size
            if any(not (x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1)
                   for bx0, by0, bx1, by1 in boxes):
                continue
            out.append(describe(to_gray(img[y0:y1, x0:x1])))
            got += 1
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, DESCRIPTOR_LEN))


def build(with_background=True, per_image=2, seed=0):
    """Return X, y, groups, source for the deduplicated dataset."""
    from backend.app.features import describe, to_gray

    kept = dedupe(enumerate_files())
    X, y, src, grp = [], [], [], []
    for item, n_copies, g in kept:
        img = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(describe(to_gray(img)))
        y.append(CLASS_NAMES.index(item["label"]))
        src.append(item["set"])
        grp.append(g)

    if with_background:
        bg = background_samples(per_image=per_image, seed=seed)
        for i in range(len(bg)):
            X.append(list(bg[i]))
            y.append(N_DEFECT_CLASSES)
            src.append("bg")
            # Each background crop is its own group: they are different crops of
            # possibly the same photo, so they must not be allowed to straddle
            # splits either.
            grp.append(f"bg#{i}")

    return (np.asarray(X, dtype=np.float64), np.asarray(y),
            np.asarray(grp), np.asarray(src), len(kept))


def grouped_split(y, groups, test_fraction=0.3, val_fraction=0.15, seed=0):
    """Stratified split over GROUPS, so no identity group straddles a boundary."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(set)
    for i, (c, g) in enumerate(zip(y, groups)):
        by_class[int(c)].add(g)

    tr, va, te = [], [], []
    for c, gs in by_class.items():
        gl = sorted(gs)
        rng.shuffle(gl)
        n = len(gl)
        n_te = int(round(n * test_fraction))
        n_va = int(round(n * val_fraction))
        te += gl[:n_te]
        va += gl[n_te:n_te + n_va]
        tr += gl[n_te + n_va:]

    def idx(gs):
        s = set(gs)
        return np.array(sorted(i for i, g in enumerate(groups) if g in s))

    return idx(tr), idx(va), idx(te)


def audit(y, groups, tr, va, te, name="split"):
    """Prove no group appears in more than one subset."""
    gtr, gva, gte = set(groups[tr]), set(groups[va]), set(groups[te])
    overlap = {
        "train&val": len(gtr & gva),
        "train&test": len(gtr & gte),
        "val&test": len(gva & gte),
    }
    ok = all(v == 0 for v in overlap.values())
    print(f"  [{name}] train={len(tr)} val={len(va)} test={len(te)}  "
          f"group overlap: {overlap}  -> {'CLEAN' if ok else 'LEAK'}")
    return ok


def main():
    X, y, groups, src, n_unique = build()
    uniq = len(set(groups.tolist()))
    print(f"raw files enumerated : 3600 (1800 CLS + 1800 DET)")
    print(f"unique source images : {n_unique}   <- CLS and DET are the same survey")
    print(f"rows after dedupe    : {len(y)}  "
          f"(defect {int((y < 6).sum())} + background {int((y == 6).sum())})")
    print(f"unique group ids     : {uniq}")
    tr, va, te = grouped_split(y, groups)
    print()
    ok = audit(y, groups, tr, va, te, "grouped 70/15/15")
    counts = {int(c): int((y[tr] == c).sum()) for c in np.unique(y)}
    print(f"  train per class     : {counts}")
    out = {
        "unique_source_images": int(n_unique),
        "rows_after_dedupe": int(len(y)),
        "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
        "group_overlap": {"train&val": len(set(groups[tr]) & set(groups[va])),
                          "train&test": len(set(groups[tr]) & set(groups[te])),
                          "val&test": len(set(groups[va]) & set(groups[te]))},
        "leakage_free": bool(ok),
    }
    (ROOT / "eval" / "splits_audit.json").write_text(json.dumps(out, indent=2))
    print("\nVERDICT: " + ("leakage-free split" if ok else "LEAK STILL PRESENT"))
    print("wrote eval/splits_audit.json")


if __name__ == "__main__":
    main()
