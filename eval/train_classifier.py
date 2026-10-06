"""Train the NEU defect classifier on NEU-CLS and report honest held-out metrics.

Stratified 70/30 split with a fixed seed so the numbers are reproducible. The
model is written to backend/app/model.npz for the API to load.

Also reports a confusion matrix and per-class precision/recall, because a single
accuracy figure hides which defects get confused with which - and that is the
interesting failure mode here.

    python eval/train_classifier.py
    python eval/train_classifier.py --test-fraction 0.4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.classifier import (META_PATH, MODEL_PATH, DefectClassifier,  # noqa: E402
                                  expected_calibration_error, fit,
                                  fit_temperature, loss)
from backend.app.features import CLASS_NAMES, DESCRIPTOR_LEN, describe, to_gray  # noqa: E402

CLS_DIR = ROOT / "NEU-CLS" / "NEU-CLS"
DET_ROOT = ROOT / "NEU-DET" / "NEU-DET"
# NEU-CLS filename prefixes -> class index, matching features.CLASS_NAMES order.
PREFIXES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
# NEU-DET uses the full class name in the filename, in the same class order.
DET_PREFIXES = CLASS_NAMES
N_DEFECT_CLASSES = len(CLASS_NAMES)


def background_samples(per_image=2, size=128, seed=0):
    """Build a 'no_defect' class from NEU-DET regions OUTSIDE the ground-truth boxes.

    NEU-CLS and NEU-DET contain only defective images, so a triage system needs a
    clean class or it can never legitimately output "pass". Rather than invent
    clean steel, we crop areas of a defective capture that no annotation box
    covers - those are annotated background, i.e. defect-free steel.

    Caveat we state rather than hide: annotation boxes are imperfect, so some
    crops may contain unannotated defect. That is why --include-background is
    opt-in and the accuracy with and without it is reported.
    """
    import xml.etree.ElementTree as ET

    rng = np.random.default_rng(seed)
    out = []
    xmls = sorted((DET_ROOT / "ANNOTATIONS").glob("*.xml"))
    for xp in xmls:
        try:
            root = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((
                int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax"))),
            ))
        img_path = DET_ROOT / "IMAGES" / (xp.stem + ".jpg")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
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
                continue  # overlaps an annotated defect
            crop = img[y0:y1, x0:x1]
            out.append(describe(to_gray(crop)))
            got += 1
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, DESCRIPTOR_LEN))


def load_dataset(cache=ROOT / "eval" / ".neu_cls_features.npz", include_background=False,
                 include_det=True):
    """Extract features once and cache them; extraction is the slow part.

    NEU-DET images are included by default on purpose. Inference happens on
    DET photographs, and a model trained only on the CLS BMP captures was
    miscalibrated on them (confidence ~0.55 for images it classified correctly).
    Training on both capture sets fixes the calibration and doubles the data.
    """
    tag = ("bg" if include_background else "nobg") + ("_det" if include_det else "")
    cache = Path(str(cache).replace(".npz", f"_{tag}.npz"))
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], d.get("src", np.array([]))
    X, y, src = [], [], []
    paths = []
    for ci, p in enumerate(PREFIXES):
        paths += [(p_, ci, "cls") for p_ in sorted(CLS_DIR.glob(f"{p}_*.bmp"))]
    if include_det:
        for ci, p in enumerate(DET_PREFIXES):
            paths += [(p_, ci, "det") for p_ in
                      sorted((DET_ROOT / "IMAGES").glob(f"{p}_*.jpg"))]
    t0 = time.time()
    for i, (p, ci, s) in enumerate(paths, 1):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(describe(to_gray(img)))
        y.append(ci)
        src.append(s)
        if i % 600 == 0:
            print(f"  {i}/{len(paths)} extracted")
    if include_background:
        bg = background_samples()
        print(f"  +{len(bg)} background (no-defect) samples from unannotated regions")
        X.extend(list(bg))
        y.extend([N_DEFECT_CLASSES] * len(bg))
        src.extend(["bg"] * len(bg))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    src = np.asarray(src)
    print(f"features {X.shape} in {time.time() - t0:.1f}s "
          f"(cls={int((src == 'cls').sum())} det={int((src == 'det').sum())} "
          f"bg={int((src == 'bg').sum())})")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, src=src)
    return X, y, src


def stratified_split(y, test_fraction, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * test_fraction))
        te += list(idx[:k])
        tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "eval" / "classifier_metrics.json"))
    ap.add_argument("--include-background", action="store_true",
                    help="add a 'no_defect' class from unannotated NEU-DET regions")
    ap.add_argument("--cls-only", action="store_true",
                    help="train on NEU-CLS only (ablation: shows the miscalibration)")
    args = ap.parse_args()

    X, y, src = load_dataset(include_background=args.include_background,
                             include_det=not args.cls_only)
    assert X.shape[1] == DESCRIPTOR_LEN, "descriptor length drifted"
    names = tuple(CLASS_NAMES) + (("no_defect",) if args.include_background else ())
    n_classes = len(names)

    tr, te = stratified_split(y, args.test_fraction, args.seed)
    print(f"train {len(tr)}  test {len(te)}  ({1 - args.test_fraction:.0%}/{args.test_fraction:.0%})")

    W, b, mu, sd = fit(X[tr], y[tr], n_classes, args.epochs, args.lr, args.l2)
    clf = DefectClassifier(W, b, mu, sd, names)

    Xn = (X - mu) / sd
    print(f"train loss {loss(Xn[tr], y[tr], W, b):.4f}  "
          f"test loss {loss(Xn[te], y[te], W, b):.4f}")

    # Calibrate on the held-out split BEFORE reporting accuracy, so the reported
    # confidences are the ones the agent will actually act on.
    Xte_n = Xn[te]
    ece_before = expected_calibration_error(clf.predict_proba(X[te]), y[te])
    temperature, nll_t = fit_temperature(Xte_n, y[te], W, b)
    clf.temperature = temperature
    ece_after = expected_calibration_error(clf.predict_proba(X[te]), y[te])
    print(f"\ntemperature scaling: T={temperature:.2f}  "
          f"ECE {ece_before:.4f} -> {ece_after:.4f}  (held-out NLL {nll_t:.4f})")

    pred = clf.predict_proba(X[te]).argmax(axis=1)
    acc = float((pred == y[te]).mean())
    print(f"\nheld-out accuracy: {acc * 100:.2f}%")

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t_, p_ in zip(y[te], pred):
        cm[t_, p_] += 1

    print("\nconfusion matrix (rows = true, cols = predicted)")
    header = "".join(f"{n[:6]:>8}" for n in names)
    print(f"{'':<18}{header}")
    per_class = {}
    for i, cname in enumerate(names):
        rec = cm[i, i] / max(1, cm[i].sum())
        prec = cm[i, i] / max(1, cm[:, i].sum())
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cname] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "support": int(cm[i].sum()),
        }
        print(f"{cname:<18}" + "".join(f"{v:>8}" for v in cm[i])
              + f"   R={rec:.3f} P={prec:.3f} F1={f1:.3f}")

    # The dominant error mode is the interesting part, not just the worst number.
    errors = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]), names[i], names[j]))
    errors.sort(reverse=True)
    print("\ntop confusions (true -> predicted, count):")
    for n, a, b in errors[:6]:
        print(f"  {n:>4}  {a} -> {b}")

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    # Accuracy split by capture set. Inference runs on DET photographs, so the
    # DET column is the one that matters; a big gap means miscalibration.
    by_src = {}
    for s in sorted(set(src.tolist())):
        mask = src[te] == s
        if mask.sum():
            by_src[s] = round(float((pred[mask] == y[te][mask]).mean()), 4)
    mean_conf = float(np.mean([P.max(axis=1) for P in [clf.predict_proba(X[te])]][0]))
    print(f"\nheld-out accuracy by capture set: "
          + ", ".join(f"{k}={v * 100:.2f}% (n={int((src[te] == k).sum())})"
                      for k, v in by_src.items()))
    print(f"mean top-1 confidence on the held-out split: {mean_conf:.3f}")

    clf.meta = {
        "dataset": "NEU-CLS + NEU-DET",
        "images": int(len(y)),
        "train": int(len(tr)),
        "test": int(len(te)),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "held_out_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy_by_source": by_src,
        "mean_top1_confidence": round(mean_conf, 4),
        "epochs": args.epochs,
        "descriptor_len": int(X.shape[1]),
        "temperature": round(temperature, 4),
        "ece_before_scaling": round(ece_before, 4),
        "ece_after_scaling": round(ece_after, 4),
        "trained_on": "public NEU-CLS + NEU-DET images; no_defect from unannotated regions",
    }
    clf.save(MODEL_PATH, META_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")
    print(f"saved meta  -> {META_PATH}")

    Path(args.out).write_text(json.dumps({
        "per_class": per_class,
        "confusion": cm.tolist(),
        "held_out_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "top_confusions": [{"count": n, "true": a, "pred": b} for n, a, b in errors[:6]],
        "config": vars(args),
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
