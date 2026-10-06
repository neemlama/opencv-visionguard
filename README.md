# ForgeSight — Agentic Visual Inspection with OpenCV 5 + COOL on AWS Graviton
Target: OpenCV AI Competition 2026, powered by AWS (deadline Oct 26, 2026 11:45pm PDT)
Goal: Win Overall ($5k/$3k/$2k) + Best Use of COOL ($1k) + Agentic Vision ($1k) = max $7k.
Rule verified: "An entry may receive no more than one Overall Award and may also receive one or both Special Awards."

## What it does
Agentic inspection for welds / PCB / solar / pipe images:
1. OpenCV 5 analyzes image: blur (Laplacian variance), glare (HSV V channel),
   defects (Canny edges **+ adaptive-threshold dark-blob detection**).
2. Agent loop decides next action from vision evidence:
   - blurry -> ask re-shoot with fix settings
   - glare -> ask angle change
   - uncertain -> auto-zoom crop + re-run
   - defect found -> escalate to human approval + create work order
   - clean -> pass
3. Core OpenCV workload runs with **COOL on AWS Graviton**, benchmarked in three
   arms (see below).

## Why this wins
- **Agentic**: vision output changes the next action, and every step is logged as
  a trace. Not a chatbot describing a fixed result.
- **COOL**: three-arm benchmark isolates the library from the hardware.
  Notably `adaptiveThreshold` — one of the ops COOL optimizes — is ~56% of our
  pipeline cost, so the workload genuinely exercises COOL's fast path.
- **Overall rubric**: technical 30% + innovation 20% + impact 20% + UX 10% +
  docs 10% + cloud 10%.

## COOL is an AMI, not a package
COOL ships as an **AWS Marketplace Graviton AMI** (Ubuntu 24.04, KleidiCV,
OpenCV 5-based) — there is no `pip install cool` and nothing to bake into a
Dockerfile. Inside the instance it is activated with:

```bash
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/cool/python_3.12/site-packages/"
python3.12 bench.py
```

Because of that, the benchmark uses three arms rather than two:

| Arm | Instance | Library | Purpose |
|---|---|---|---|
| A | Graviton | COOL | Award eligibility: COOL on Arm |
| B | Graviton, same family | DIY pip OpenCV | isolates the library's effect |
| C | x86 | DIY pip OpenCV | hardware baseline, Arm/x86 coexistence |

A two-way COOL-vs-x86 test would confound library and hardware. See
`docs/COOL-setup.md`.

## Structure
- `backend/app/` FastAPI + OpenCV 5 vision + agent loop (`vision.py`, `agent.py`)
- `frontend/index.html` upload UI, metrics panel, trace timeline, approvals
- `bench/bench.py` per-op + end-to-end benchmark, self-identifies COOL vs DIY
- `aws/run_bench.py` SSM-driven orchestrator that runs all arms and terminates them
- `docs/CHALLENGES.md` engineering log incl. the dark-pit false-negative post-mortem
- `docs/COOL-setup.md` Marketplace steps, env vars, cost model, judge checklist
- `samples/` 5 synthetic weld images driving 5 agent outcomes

## Quick start (local, no AWS)
```powershell
cd C:\Users\wecal\Documents\Codes\opencv-visionguard
pip install opencv-python fastapi "uvicorn[standard]" python-multipart
python -m backend.app.make_samples        # regenerate sample images
uvicorn backend.app.main:app --port 8000
# open http://localhost:8000
```

Local benchmark (proves the harness runs on any machine):
```powershell
python bench\bench.py --label local-dev-arm --instance-type none --iters 8 --warmup 2
```

## COOL benchmark on AWS
```powershell
aws login                                     # session expired on this machine
Copy-Item .env.example .env                   # fill in AMI IDs
python aws\run_bench.py --region ap-south-1 --ami-arm64 $env:COOL_AMI
```

## Submission checklist (from rules)
- [x] OpenCV 5 substantive analysis + AWS component
- [x] agent trace proving vision -> action
- [x] benchmark harness + orchestrator
- [x] eval evidence incl. failure cases (`docs/CHALLENGES.md`)
- [ ] three-arm COOL benchmark executed, results committed
- [ ] technical report (problem, users, arch, eval, limits, responsible use)
- [ ] architecture diagram
- [ ] working public endpoint
- [ ] <=5 min video
