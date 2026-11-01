"""Inference service for the command parser (Task 3).

Endpoints:
  POST /parse   {"text": "..."} -> 200 with the four-key JSON object (and nothing else),
                headers X-Model-Version and X-Served-By. Invalid model output -> 500
                {"error": "invalid_model_output"} (never broken JSON with 200).
  GET  /healthz -> 200 while the process is alive.
  GET  /readyz  -> 503 until the model is loaded, 200 after.

Configuration (environment variables; AWS_* come from a Kubernetes Secret):
  MODEL_URI, MODEL_VERSION, S3_ENDPOINT, MAX_NEW_TOKENS, and FAULT_INJECT (test hook).
"""
import os
import socket
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import model_runner

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
POD_NAME = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME") or socket.gethostname()

app = FastAPI(title="command-parser")
runner = model_runner.Runner()


def _safe_load():
    try:
        runner.load()
        print(f"model loaded (version {MODEL_VERSION})", flush=True)
    except Exception as e:  # keep the process alive; /readyz stays 503
        print(f"model load failed: {e}", flush=True)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_safe_load, daemon=True).start()


class ParseIn(BaseModel):
    text: str


def _headers():
    return {"X-Model-Version": MODEL_VERSION, "X-Served-By": POD_NAME}


@app.get("/healthz")
def healthz():
    return Response(status_code=200)


@app.get("/readyz")
def readyz():
    return Response(status_code=200 if runner.ready else 503)


@app.post("/parse")
def parse(body: ParseIn):
    if not runner.ready:
        return JSONResponse({"error": "not_ready"}, status_code=503, headers=_headers())
    raw = runner.raw_output(body.text)
    if os.environ.get("FAULT_INJECT") == "1":
        raw = "not json"  # force an invalid output to test the validation path
    obj = model_runner.validate(raw)
    if obj is None:
        return JSONResponse({"error": "invalid_model_output"}, status_code=500, headers=_headers())
    return JSONResponse(obj, status_code=200, headers=_headers())
