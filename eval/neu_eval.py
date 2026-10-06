"""Evaluate defect localisation on NEU-DET (1800 images, 1800 XML box sets).

Deliberately runs BEFORE any tuning, so the first numbers are an honest baseline
rather than a tuned showcase. Metric is standard detection-with-boxes:
greedy IoU matching, reported per defect class because NEU's six classes are
visually very different and a single pooled number hides that.

    python eval/neu_eval.py                 # full 1800, default config
    python eval/neu_eval.py --limit 300     # quick pass
    python eval/neu_eval.py --config '{"anom_thresh": 16.0}'
"""
import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.neu import DetectConfig, Region, locate_defects  # noqa: E402

DET_ROOT = ROOT / "NEU-DET" / "NEU-DET"
IMG_DIR = DET_ROOT / "IMAGES"
ANN_DIR = DET_ROOT / "ANNOTATIONS"
# NB: the XML writes this class with a HYPHEN ("rolled-in_scale"), so a naive
# underscore lookup silently evaluates zero images for it.
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")


def parse_annotations(xml_path):
    """NEU XMLs contain a zero-width space inside the <filename> tag."""
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown").strip()
        bb = obj.find("bndbox")
        x0 = int(float(bb.findtext("xmin")))
        y0 = int(float(bb.findtext("ymin")))
        x1 = int(float(bb.findtext("xmax")))
        y1 = int(float(bb.findtext("ymax")))
        out.append((name, Region(x0, y0, x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0), 1.0, 1.0)))
    return out


def match(preds, gts, iou_thresh=0.5):
    """Greedy highest-IoU matching. Returns (tp, fp, fn, matched_ious)."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = _iou(p, g)
            if v >= iou_thresh:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    return len(used_p), len(preds) - len(used_p), len(gts) - len(used_g), ious


def _iou(a, b):
    ix0, iy0 = max(a.x, b.x), max(a.y, b.y)
    ix1, iy1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / (a.w * a.h + b.w * b.h - inter)


def best_iou_any(preds, gts):
    if not preds or not gts:
        return 0.0
    return max(_iou(p, g) for p in preds for g in gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="images per class (0 = all)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--config", default="{}", help="JSON overrides for DetectConfig")
    ap.add_argument("--out", default=str(ROOT / "eval" / "neu_results.json"))
    args = ap.parse_args()

    cfg = DetectConfig(**json.loads(args.config))
    print(f"config: {cfg}")

    xmls = sorted(ANN_DIR.glob("*.xml"))
    by_class = defaultdict(list)
    for x in xmls:
        by_class[parse_annotations(x)[0][0]].append(x)
    if args.limit:
        by_class = {k: v[: args.limit] for k, v in by_class.items()}

    stats = {}
    overall = defaultdict(int)
    all_ious = []
    t0 = time.time()

    for cls in CLASSES:
        tp = fp = fn = 0
        ious, no_pred, detected_at_loose = [], 0, 0
        covs = []
        for xp in by_class.get(cls, []):
            gts = parse_annotations(xp)
            img_path = IMG_DIR / (xp.stem + ".jpg")
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            det = locate_defects(img, cfg)
            preds = det.regions
            if not preds:
                no_pred += 1
            gt_regions = [r for _, r in gts]
            a, b, c, mi = match(preds, gt_regions, args.iou)
            tp += a
            fp += b
            fn += c
            ious.extend(mi)
            if best_iou_any(preds, gt_regions) >= 0.3:
                detected_at_loose += 1
            covs.append(det.quality["coverage"])
            all_ious.extend(mi)

        n = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n if n else 0.0
        stats[cls] = {
            "images": len(by_class.get(cls, [])),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "detected_at_iou0.3": round(detected_at_loose / max(1, len(by_class.get(cls, []))), 4),
            "images_with_no_prediction": no_pred,
            "mean_coverage": round(float(np.mean(covs)), 4) if covs else 0.0,
        }
        overall["tp"] += tp
        overall["fp"] += fp
        overall["fn"] += fn
        print(f"  {cls:<18} P={prec:.3f} R={rec:.3f} F1={stats[cls]['f1']:.3f} "
              f"mIoU={stats[cls]['mean_matched_iou']:.3f} "
              f"@0.3={stats[cls]['detected_at_iou0.3']:.3f} n={stats[cls]['images']}")

    tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    macro_f1 = float(np.mean([s["f1"] for s in stats.values()]))
    elapsed = time.time() - t0

    print(f"\n{'OVERALL':<18} P={prec:.3f} R={rec:.3f} "
          f"F1={2 * prec * rec / (prec + rec) if (prec + rec) else 0:.3f}  macroF1={macro_f1:.3f}")
    print(f"images={sum(s['images'] for s in stats.values())} "
          f"time={elapsed:.1f}s ({sum(s['images'] for s in stats.values()) / max(elapsed, 1e-9):.1f} img/s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NEU-DET",
        "config": cfg.__dict__,
        "iou_threshold": args.iou,
        "per_class": stats,
        "overall": {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "macro_f1": round(macro_f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "images": sum(s["images"] for s in stats.values()),
            "seconds": round(elapsed, 1),
        },
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
