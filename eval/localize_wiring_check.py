"""What did wiring the new localiser into analysis.py actually change?

`backend/app/localize.py` was measured and reported but never imported by the
request path; `analysis.py::localize` was a separate single-threshold
implementation and that was what ran. This script quantifies the consequence of
fixing that, so the change is documented with numbers instead of an assurance.

It runs the full request path - `analyze()` then `agent.decide()` - twice on the
same images: once with the new localiser, once with the pre-2026-10-17
implementation monkeypatched back in. The old code is reproduced verbatim below
rather than imported from git, so this file is self-contained and the comparison
stays runnable.

The number that matters is the veto: `agent.decide` sends a confident
`no_defect` prediction to human review when the localiser also finds structure.
That disagreement check is one of the three mechanisms the README calls agentic,
so if the new localiser stopped firing it, wiring it in would have quietly
removed a feature rather than added one.

    python eval/localize_wiring_check.py --per-class 25
"""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import backend.app.analysis as A  # noqa: E402
from backend.app.agent import decide  # noqa: E402
from backend.app.features import anomaly_map  # noqa: E402

DET_IMAGES = ROOT / "NEU-DET" / "NEU-DET" / "IMAGES"
DET_ANN = ROOT / "NEU-DET" / "NEU-DET" / "ANNOTATIONS"


def old_localize(gray, anom_thresh=8.0, min_area=400, close_k=9, box=21):
    """The implementation analysis.py used until 2026-10-17, verbatim."""
    anom = anomaly_map(gray, box)
    _, binary = cv2.threshold(anom, anom_thresh, 1.0, cv2.THRESH_BINARY)
    mask = (binary * 255).astype(np.uint8)
    if close_k:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = gray.shape[0] * gray.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or (w * h) / total > 0.95:
            continue
        sel = labels[y:y + h, x:x + w] == i
        regions.append({
            "box": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "energy": round(float(anom[y:y + h, x:x + w][sel].mean()), 3),
        })
    regions.sort(key=lambda r: -r["area"])
    return regions, anom


def clean_images(per_image=2, size=128, seed=0):
    """Crops of annotated photographs that no ground-truth box covers.

    Same construction as eval/splits.background_samples, but returning the crop
    instead of its descriptor, because this script needs pixels to run the
    request path on.
    """
    rng = np.random.default_rng(seed)
    out = []
    for xp in sorted(DET_ANN.glob("*.xml"))[:400]:
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
        img = cv2.imread(str(DET_IMAGES / (xp.stem + ".jpg")))
        if img is None:
            continue
        h, w = img.shape[:2]
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
            out.append(img[y0:y1, x0:x1].copy())
            got += 1
    return out


def defect_images(per_class):
    out = defaultdict(list)
    for xp in sorted(DET_ANN.glob("*.xml")):
        cls = ET.parse(xp).getroot().find("object").findtext("name", "?").strip()
        if len(out[cls]) < per_class:
            p = DET_IMAGES / (xp.stem + ".jpg")
            if p.exists():
                out[cls].append(p)
    return {k: v for k, v in out.items()}


def run(images, use_old):
    """Analyze + decide for a list of images under one localiser."""
    original = A.localize
    if use_old:
        A.localize = lambda gray, cfg=None, scorer=None: old_localize(gray)
    try:
        finals = defaultdict(int)
        vetoes = 0
        rc = []
        t0 = time.perf_counter()
        for im in images:
            a = A.analyze(im)
            d = decide(a)
            finals[d["final"]] += 1
            if d.get("disagreement"):
                vetoes += 1
            rc.append(a["localization"]["region_count"])
        dt = (time.perf_counter() - t0) / max(1, len(images))
    finally:
        A.localize = original
    return finals, vetoes, float(np.mean(rc)) if rc else 0.0, dt * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--clean", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "eval" / "localize_wiring_check.json"))
    args = ap.parse_args()

    # Warm the model and scorer caches so the first timed image is not an outlier.
    A.analyze(cv2.imread(str(next(DET_IMAGES.glob("*.jpg")))))

    clean = clean_images(per_image=1)[:args.clean]
    defects = defect_images(args.per_class)
    defect_imgs = [cv2.imread(str(p)) for p in
                   [q for v in defects.values() for q in v] if p.exists()]
    defect_imgs = [i for i in defect_imgs if i is not None]

    print(f"clean crops: {len(clean)}   defect images: {len(defect_imgs)}\n")

    results = {}
    for label, imgs in (("clean", clean), ("defect", defect_imgs)):
        print(f"--- {label} (n={len(imgs)}) ---")
        print(f"{'arm':<8}{'pass':>7}{'vetoes':>8}{'regions/img':>13}"
              f"{'ms/img':>9}   final decisions")
        row = {}
        for arm, use_old in (("old", True), ("new", False)):
            finals, vetoes, rc, ms = run(imgs, use_old)
            row[arm] = {"finals": dict(finals), "vetoes": vetoes,
                        "mean_region_count": round(rc, 3), "ms_per_image": round(ms, 2)}
            top = ", ".join(f"{k}={v}" for k, v in sorted(finals.items(),
                                                         key=lambda t: -t[1])[:4])
            print(f"{arm:<8}{finals.get('pass', 0):>7}{vetoes:>8}{rc:>13.3f}{ms:>9.2f}"
                  f"   {top}")
        results[label] = row

    # Latency is content-dependent: crazing is the noisiest class and produces by
    # far the most candidates, so a whole-dataset average would flatter it.
    print("\n--- latency by class (new localiser) ---")
    per_class_ms = {}
    for cls, paths in sorted(defects.items()):
        imgs = [cv2.imread(str(p)) for p in paths]
        imgs = [i for i in imgs if i is not None]
        _f, _v, _rc, ms_new = run(imgs, False)
        _f, _v, _rc, ms_old = run(imgs, True)
        per_class_ms[cls] = {"old_ms": round(ms_old, 2), "new_ms": round(ms_new, 2)}
        print(f"  {cls:<18}old {ms_old:>6.2f} ms   new {ms_new:>6.2f} ms")
    results["latency_by_class"] = per_class_ms

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()