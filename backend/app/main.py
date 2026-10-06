"""ForgeSight backend — OpenCV 5 inspection + agentic loop."""
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .vision import analyze_image
from .agent import run_agent_loop

ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="ForgeSight", version="0.1.0")


@app.get("/health")
def health():
    import cv2

    return {"ok": True, "opencv": cv2.__version__}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    data = await file.read()
    vision = analyze_image(data)
    trace = run_agent_loop(vision)
    return JSONResponse({"vision": vision, "agent_trace": trace})


@app.post("/approve/{workorder_id}")
def approve(workorder_id: str, approved: bool = True):
    # Human-in-the-loop stub — persisted to DynamoDB on AWS, in-memory locally
    return {"workorder_id": workorder_id, "approved": approved, "status": "closed" if approved else "rework"}


@app.get("/samples/{name}")
def sample(name: str):
    path = (ROOT / "samples" / name).resolve()
    if not path.is_file() or ROOT / "samples" not in path.parents:
        return Response(status_code=404)
    return FileResponse(path)


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")