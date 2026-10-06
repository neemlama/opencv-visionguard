# ForgeSight — Agentic Visual Inspection with OpenCV 5 + COOL on AWS Graviton
Target: OpenCV AI Competition 2026, powered by AWS (deadline Oct 26, 2026 11:45pm PDT)
Goal: Win Overall ($5k/$3k/$2k) + Best Use of COOL ($1k) + Agentic Vision ($1k) = max $7k.
Rule verified: "An entry may receive no more than one Overall Award and may also receive one or both Special Awards."

## What it does
Agentic inspection for welds / PCB / solar / pipe images:
1. OpenCV 5 analyzes image: blur (Laplacian), glare (HSV), defects (Canny + contours + DNN).
2. Agent loop decides next action from vision evidence:
   - blurry -> ask re-shoot with fix settings
   - glare -> ask angle change
   - low confidence -> auto-zoom crop + re-run
   - defect found -> escalate to human approval + create work order
3. Core OpenCV workload runs with COOL on AWS Graviton (t4g), benchmarked vs x86.

## Why this wins
- Agentic: vision output changes next tool call / action (not a chatbot). Trace logged.
- COOL: reproducible latency/throughput/cost per 1000 frames, Graviton vs x86.
- Overall rubric: technical 30% + innovation 20% + impact 20% + UX 10% + docs 10% + cloud 10%.

## Structure
- `backend/` FastAPI + OpenCV 5 + agent + MCP tools
- `frontend/` minimal upload + trace viewer + human approval
- `bench/` Graviton vs x86 benchmark
- `aws/` Docker, EC2 user-data, CloudFormation
- `docs/` report, architecture, video script, eval
- `samples/` test images

## Quick start (local, no AWS)
```powershell
cd C:\Users\wecal\Documents\Codes\opencv-visionguard
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload --port 8000
# open http://localhost:8000/docs, POST /inspect with image
```

## Submission checklist (from rules)
- [ ] technical report (problem, users, arch, OpenCV5, AWS, eval, limits, responsible use)
- [ ] code repo + pinned deps + build/test instructions
- [ ] architecture diagram (OpenCV5 + AWS + COOL + agent)
- [ ] working web endpoint OR live screen-share
- [ ] <=5 min video (team, app working, arch, results)
- [ ] eval evidence incl. failure cases
- [ ] COOL evidence: version, instance config, repro method, proof COOL runs core workload
- [ ] Agentic evidence: workflow diagram, trace showing vision -> decision change, task success + failure handling
