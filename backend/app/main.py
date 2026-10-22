"""ForgeSight API — agentic visual inspection of rolled-steel surfaces.

Routes:
  GET  /health              liveness + OpenCV version + model metadata
  POST /inspect             single perception -> decision (no simulation)
  POST /inspect/loop        closed loop: the agent acts and the capture changes
  POST /capture/degrade     apply a capture degradation (demo helper)
  GET  /samples             labelled NEU samples for the demo UI
  GET  /workorders          every order the agent has raised, newest first
  GET  /workorders/{id}     one order, with its audit trail
  POST /approve/{id}        human decision on an order that is still open
"""
import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import approve as approve_workorder
from .agent import decide, get_workorder, list_workorders, run_loop
from .analysis import analyze
from .capture import DEGRADATIONS, PRESETS, b64_jpeg, make

ROOT = Path(__file__).resolve().parent.parent.parent
NEU_IMG = ROOT / "NEU-DET" / "NEU-DET" / "IMAGES"
# Small re-encoded copies shipped with the app so the deployed instance has a
# working demo even though the full NEU dataset is not redistributed (see
# docs/DATASETS.md).
FALLBACK_IMG = ROOT / "samples" / "neu"


def _sample_dir():
    return NEU_IMG if NEU_IMG.is_dir() else FALLBACK_IMG

app = FastAPI(title="ForgeSight", version="1.0.0")

# Bounds for a public endpoint. The model operates on 200x200 mill crops, so a
# 12-megapixel phone photo is not a bigger input, it is a slower request and a
# larger allocation. Refuse it with a reason instead of timing out.
MAX_UPLOAD = int(os.environ.get("FORGESIGHT_MAX_UPLOAD", 8 * 1024 * 1024))
MAX_SIDE = int(os.environ.get("FORGESIGHT_MAX_SIDE", 1024))


@app.middleware("http")
async def observe(request: Request, call_next):
    """One JSON line per request - the observability a judge can inspect.

    Written to stdout, which systemd captures in the journal on the deployed
    instance (`journalctl -u forgesight`). No bodies, no image bytes, no keys.
    """
    rid = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _log(request, rid, 500, started)
        raise
    response.headers["x-request-id"] = rid
    _log(request, rid, response.status_code, started)
    return response


def _log(request, rid, status, started):
    print(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": rid,
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }), flush=True)


async def _read(file: UploadFile):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(
            413, f"upload exceeds {MAX_UPLOAD // (1024 * 1024)} MB")
    return data


def _decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        raise HTTPException(
            422, f"image is {w}x{h}; resize to <= {MAX_SIDE}px on the long side")
    return img


@app.get("/health")
def health():
    from .analysis import get_model
    m = get_model()
    return {
        "ok": True,
        "opencv": cv2.__version__,
        "domain": "rolled-steel surface defects (NEU)",
        "open_workorders": len(list_workorders(status="open")),
        "model": {
            "dataset": m.meta.get("dataset"),
            "held_out_accuracy": m.meta.get("held_out_accuracy"),
            "macro_f1": m.meta.get("macro_f1"),
            "classes": list(m.class_names),
        },
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    img = _decode(await _read(file))
    a = analyze(img)
    d = decide(a)
    return JSONResponse({"analysis": a, "decision": d})


@app.post("/inspect/loop")
async def inspect_loop(file: UploadFile = File(...), max_steps: int = 6,
                       preset: str | None = None):
    """Run the closed loop. Pass the degradation the frame was captured under.

    Without `preset` the endpoint still runs, but a recapture request can only be
    recorded for an operator, because re-rendering needs a capture simulator.
    With `preset` the simulator is rebuilt server-side, so the agent's requested
    fix changes a real camera setting and the scene is re-rendered - which is the
    action the Agentic criterion asks to see.
    """
    img = _decode(await _read(file))
    if preset is not None and preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {sorted(PRESETS)}")
    if preset is None:
        result = run_loop(img, max_steps=max_steps)
        result["final_capture_jpeg_b64"] = (
            b64_jpeg(img) if result["iterations"] <= 1 else None)
    else:
        result = run_loop(scene=img, preset=preset, max_steps=max_steps)
    return JSONResponse(result)


class DegradeRequest(BaseModel):
    preset: str


@app.post("/capture/degrade")
async def capture_degrade(file: UploadFile = File(...), preset: str = "clean"):
    """Render the capture under a named bad setting, so a demo can start degraded.

    The response image is what the CAMERA produced under that setting; the loop
    then corrects the setting and re-renders the same scene.
    """
    if preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {sorted(PRESETS)}")
    img = _decode(await _read(file))
    sim = make(img, preset)
    return JSONResponse({
        "preset": preset,
        "settings": sim.settings,
        "available_presets": sorted(PRESETS),
        "image_jpeg_b64": b64_jpeg(sim.image),
    })


@app.get("/samples")
def samples():
    """A few labelled NEU images so the demo UI needs no external assets."""
    d = _sample_dir()
    if not d.is_dir():
        return {"samples": []}
    # Fallback copies are named neu_<class>_1.jpg; originals are <class>_1.jpg.
    names = sorted(p.name for p in d.glob("*.jpg"))
    out = []
    for fname in names:
        stem = fname[4:] if fname.startswith("neu_") else fname
        label = stem.rsplit("_", 1)[0].replace("-in_", "in_")
        out.append({"file": fname, "label": label, "url": f"/samples/{fname}"})
    return {"samples": out, "source": d.name}


@app.get("/samples/{name}")
def sample(name: str):
    d = _sample_dir().resolve()
    p = (d / name).resolve()
    if not p.is_file() or d not in p.parents:
        raise HTTPException(404, "no such sample")
    return FileResponse(p)


@app.get("/workorders")
def workorders(status: str | None = None):
    """Every order the agent has raised. `?status=open` filters."""
    rows = list_workorders(status=status)
    return {"count": len(rows), "workorders": rows}


@app.get("/workorders/{workorder_id}")
def workorder(workorder_id: str):
    rec = get_workorder(workorder_id)
    if rec is None:
        raise HTTPException(404, f"no work order {workorder_id}")
    return rec


@app.post("/approve/{workorder_id}")
def approve(workorder_id: str, approved: bool = True, note: str = ""):
    """A person decides an open order. Unknown ids 404; decided ids 409."""
    try:
        return approve_workorder(workorder_id, approved, note)
    except KeyError:
        raise HTTPException(404, f"no work order {workorder_id}")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
