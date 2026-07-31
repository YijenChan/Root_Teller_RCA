from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import (
    RUNTIME_ROOT,
    SYSTEM_ROOT,
    WORKSPACE,
    apply_feedback,
    diagnose,
    inspect_case_path,
    safe_extract_zip,
)


STATIC = SYSTEM_ROOT / "static"
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()


class InspectRequest(BaseModel):
    path: str


class DiagnoseRequest(BaseModel):
    dataset: str
    variant: dict[str, Any]
    protocol: str = Field(pattern="^(blind|default)$")
    live_llm: bool = True


class FeedbackRequest(BaseModel):
    entity: str
    verdict: str = Field(pattern="^(ACCEPT|REJECT|accept|reject)$")
    message: str = Field(default="", max_length=1200)


app = FastAPI(title="Root-Teller Local System", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "workspace": str(WORKSPACE),
        "gpu": __import__("torch").cuda.is_available(),
    }


@app.post("/api/cases/inspect")
def inspect_case(request: InspectRequest) -> dict[str, Any]:
    try:
        return inspect_case_path(request.path)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/cases/upload")
async def upload_case(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip archive containing one case or dataset capture.")
    upload_id = uuid.uuid4().hex[:12]
    upload_root = RUNTIME_ROOT / "uploads" / upload_id
    upload_root.mkdir(parents=True, exist_ok=True)
    archive = upload_root / Path(file.filename).name
    size = 0
    with archive.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 1_500_000_000:
                handle.close()
                archive.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Archive exceeds the 1.5 GB upload limit.")
            handle.write(chunk)
    try:
        extracted = safe_extract_zip(archive, upload_root / "extracted")
        result = inspect_case_path(str(extracted))
        result["upload_id"] = upload_id
        return result
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _run_job(job_id: str, request: DiagnoseRequest) -> None:
    def progress(message: str, percent: int) -> None:
        with LOCK:
            JOBS[job_id].update({"message": message, "progress": percent, "updated_at": time.time()})
    try:
        result = diagnose(
            job_id=job_id, dataset=request.dataset, variant=request.variant,
            protocol=request.protocol, live_llm=request.live_llm, progress=progress,
        )
        with LOCK:
            JOBS[job_id].update({"status": "complete", "message": "Diagnosis complete", "progress": 100, "result": result})
    except Exception as error:
        with LOCK:
            JOBS[job_id].update({"status": "failed", "message": str(error), "error_type": type(error).__name__})


@app.post("/api/jobs", status_code=202)
def create_job(request: DiagnoseRequest) -> dict[str, Any]:
    if request.dataset not in {"re2_ob", "re2_tt", "eadro_sn"}:
        raise HTTPException(status_code=400, detail="Unsupported dataset.")
    job_id = uuid.uuid4().hex[:16]
    with LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "running", "message": "Queued", "progress": 0, "created_at": time.time()}
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
    if job:
        return job
    artifact = RUNTIME_ROOT / "jobs" / job_id / "result.json"
    if artifact.exists():
        return {"job_id": job_id, "status": "complete", "progress": 100, "message": "Loaded from artifact", "result": json.loads(artifact.read_text(encoding="utf-8"))}
    raise HTTPException(status_code=404, detail="Unknown diagnosis job.")


@app.post("/api/jobs/{job_id}/feedback")
def feedback(job_id: str, request: FeedbackRequest) -> dict[str, Any]:
    job = get_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Diagnosis is not complete.")
    try:
        update = apply_feedback(job["result"], request.entity, request.verdict.upper(), request.message)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id]["result"] = job["result"]
    return update


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str) -> FileResponse:
    path = RUNTIME_ROOT / "jobs" / job_id / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result artifact does not exist.")
    return FileResponse(path, media_type="application/json", filename=f"root-teller-{job_id}.json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
