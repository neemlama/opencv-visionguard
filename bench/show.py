"""Pretty-print a saved benchmark comparison for the technical report."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/results/comparison-us-east-1.json")
data = json.loads(p.read_text())

for r in data:
    print(f"arm {r['label']}  {r['instance_type']}  {r['cpu'][:46]}")
    print(f"   build={r['build']}  opencv={r['opencv_version']}  arch={r['arch']}")
    print(f"   images={r['images']}  iters={r['iters']}  warmup_discarded={r['warmup_discarded']}")
    print(f"   fps={r['fps']}  ms/frame={r['mean_frame_seconds'] * 1000:.2f}")
    print("   per-op (ms/call):")
    for k, v in r["per_op_ms"].items():
        print(f"     {k:<20}{v:>9.4f}")
    print()

if len(data) >= 2:
    by = {r["label"]: r for r in data}
    if "A" in by and "B" in by:
        print(f"A/B library only        : {by['A']['fps'] / by['B']['fps']:.2f}x")
        print("  per-op A vs B:")
        for k in by["A"]["per_op_ms"]:
            a, b = by["A"]["per_op_ms"][k], by["B"]["per_op_ms"][k]
            print(f"    {k:<20}{b / a:>6.2f}x")
    if "B" in by and "C" in by:
        print(f"B/C hardware only       : {by['B']['fps'] / by['C']['fps']:.2f}x")
        print("  per-op B vs C:")
        for k in by["B"]["per_op_ms"]:
            b, c = by["B"]["per_op_ms"][k], by["C"]["per_op_ms"][k]
            print(f"    {k:<20}{c / b:>6.2f}x")