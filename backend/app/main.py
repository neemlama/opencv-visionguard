"""ForgeSight API — agentic visual inspection of rolled-steel surfaces.

Routes:
  GET  /health              liveness + OpenCV version + model metadata
  POST /inspect             single perception -> decision (no simulation)
  POST /inspect/loop        closed loop: the agent acts and the capture changes
  POST /capture/degrade     apply a capture degradation (demo helper)
  GET  /samples             labelled NEU samples for the demo UI
  POST /approve/{id}        human-in-the-loop decision on a work order
"""
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import approve as approve_workorder
from .agent import decide, run_loop
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


def _decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return img


@app.get("/health")
def health():
    from .analysis import get_model
    m = get_model()
    return {
        "ok": True,
        "opencv": cv2.__version__,
        "domain": "rolled-steel surface defects (NEU)",
        "model": {
            "dataset": m.meta.get("dataset"),
            "held_out_accuracy": m.meta.get("held_out_accuracy"),
            "macro_f1": m.meta.get("macro_f1"),
            "classes": list(m.class_names),
        },
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    img = _decode(await file.read())
    a = analyze(img)
    d = decide(a)
    return JSONResponse({"analysis": a, "decision": d})


@app.post("/inspect/loop")
async def inspect_loop(file: UploadFile = File(...), max_steps: int = 6):
    img = _decode(await file.read())
    result = run_loop(img, max_steps=max_steps)
    result["final_capture_jpeg_b64"] = b64_jpeg(img) if result["iterations"] <= 1 else None
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
    img = _decode(await file.read())
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


@app.post("/approve/{workorder_id}")
def approve(workorder_id: str, approved: bool = True, note: str = ""):
    return approve_workorder(workorder_id, approved, note)


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
