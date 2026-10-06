"""Demonstrate the closed inspection loop on real NEU photographs.

Runs in-process (no server needed) and prints the full trace so the loop can be
read as evidence: when the agent asks for a recapture, the capture SETTINGS are
corrected, the scene is re-rendered, and perception runs again on a genuinely
different frame.

    python eval/demo_loop.py
    python eval/demo_loop.py --image scratches_3.jpg --preset sensor_noise
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agent import run_loop  # noqa: E402
from backend.app.analysis import analyze  # noqa: E402
from backend.app.capture import DEGRADATIONS, make  # noqa: E402

IMAGES = ROOT / "NEU-DET" / "NEU-DET" / "IMAGES"

# soft_focus leads because it is the ONE condition where the loop detects a bad
# capture, acts on it, and converges. under_exposed and specular_glare are kept
# deliberately as honest failure cases: the gate has no brightness check, and it
# cannot separate synthetic glare from natural saturated regions on scaled steel,
# so neither recaptures. sensor_noise shows the gate correctly refusing.
CASES = [
    ("inclusion_1.jpg", "soft_focus"),
    ("inclusion_1.jpg", "under_exposed"),
    ("patches_1.jpg", "specular_glare"),
    ("rolled-in_scale_1.jpg", "sensor_noise"),
    ("crazing_1.jpg", "clean"),
    ("scratches_2.jpg", "low_contrast"),
]


def show(name, preset):
    img = cv2.imread(str(IMAGES / name))
    if img is None:
        print(f"missing {IMAGES / name}")
        return
    label = name.rsplit("_", 1)[0].replace("-", "_")
    print("=" * 78)
    print(f"{name}   NEU class: {label}   starting capture: {preset} "
          f"({DEGRADATIONS[preset]})")
    print("=" * 78)

    sim = make(img, preset)
    start = sim.image
    first = analyze(start)
    print(f"  capture as received: pred={first['prediction']['defect_class']} "
          f"conf={first['prediction']['confidence']:.3f} usable={first['quality']['usable']} "
          f"sharpness={first['quality']['sharpness']} glare={first['quality']['glare_ratio']}")

    result = run_loop(scene=img, preset=preset)

    for i, s in enumerate(result["trace"], 1):
        print(f"\n  step {i}")
        print(f"    perception: {s['perception']}")
        print(f"    decision  : {s['decision']}")
        print(f"    action    : {s['action']}")

    print(f"\n  => FINAL: {result['final']}  iterations={result['iterations']}  "
          f"human_required={result['human_required']}  workorder={result['workorder_id']}")
    print(f"  final capture settings: {result['final_settings']}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    ap.add_argument("--preset", default="clean", choices=sorted(DEGRADATIONS))
    args = ap.parse_args()
    if args.image:
        show(args.image, args.preset)
        return
    for name, preset in CASES:
        show(name, preset)


if __name__ == "__main__":
    main()
