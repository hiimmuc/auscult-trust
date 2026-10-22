"""Inference API for the AuscultTrust demo.

    PYTHONPATH=. .venv-train/bin/uvicorn app.api.server:app --host 127.0.0.1 --port 8000

Environment:
    AT_APP_BUNDLE   bundle directory from `app.api.bundle` (default app/bundle)
    AT_APP_TOKEN    shared access token; when set, every /v1 route except /v1/health needs `Authorization: Bearer <token>`
    AT_APP_ORIGINS  comma-separated CORS origins (default: the Vite dev server on localhost and 127.0.0.1)
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.engine import SR, AudioError, Engine, gcc_phat_lag, read_wav, shift, to_model_rate
from src.processing.icbhi import CLASSES

BUNDLE = Path(os.environ.get("AT_APP_BUNDLE", "app/bundle"))
TOKEN = os.environ.get("AT_APP_TOKEN", "")
ORIGINS = [o.strip() for o in os.environ.get("AT_APP_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
MAX_BYTES = 40 * 1024 * 1024
MAX_PENDING = 4  # requests waiting for or holding the model; more get 429
TIMEOUT_SEC = 120
# ponytail: fixed cut on the GCC-PHAT peak z-score, set on 3 synthetic takes (clean 651, noisy 18, white noise 4.3, another
# ICBHI recording 8.1); calibrate on real speaker/microphone takes before trusting the flag
ALIGN_MIN_SCORE = 12.0

state = {}


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(app):
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    state.update(manifest=manifest, engine=Engine(manifest, BUNDLE), pending=0, cache=OrderedDict(),
                 samples={s["id"]: s for s in manifest["samples"]})
    yield


app = FastAPI(title="AuscultTrust inference API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Authorization", "Content-Type"], expose_headers=["X-Request-ID"], max_age=600)


@app.middleware("http")
async def request_id(request: Request, call_next):
    request.state.rid = uuid.uuid4().hex[:16]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.rid
    return response


@app.exception_handler(ApiError)
async def api_error(request: Request, e: ApiError):
    return JSONResponse({"error": {"code": e.code, "message": e.message}, "request_id": request.state.rid}, e.status)


@app.exception_handler(AudioError)
async def audio_error(request: Request, e: AudioError):
    return await api_error(request, ApiError(422, e.code, str(e)))


def auth(request: Request):
    if not TOKEN:
        return
    got = request.headers.get("authorization", "")
    if not hmac.compare_digest(got.encode(), "Bearer {}".format(TOKEN).encode()):
        raise ApiError(401, "unauthorized", "Missing or wrong access token.")


async def guarded(fn, *args):
    """Run blocking inference in a worker thread, with a queue bound and a time limit."""
    if state["pending"] >= MAX_PENDING:
        raise ApiError(429, "busy", "The model is busy. Retry in a few seconds.")
    state["pending"] += 1
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args), TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise ApiError(504, "timeout", "Inference took longer than {} s.".format(TIMEOUT_SEC))
    finally:
        state["pending"] -= 1


def cached(key, fn):
    c = state["cache"]
    if key in c:
        c.move_to_end(key)
        return c[key]
    c[key] = fn()
    if len(c) > 64:
        c.popitem(last=False)
    return c[key]


def sample(sid):
    if sid not in state["samples"]:
        raise ApiError(404, "unknown_sample", "No demo sample '{}'.".format(sid))
    return state["samples"][sid]


def sample_cycles(s):
    return [(c["start"], c["end"], c["label"]) for c in s["cycles"]]


def parse_annotation(text):
    """ICBHI annotation lines `start end crackle wheeze` -> [(start, end, label)]."""
    out = []
    for i, line in enumerate(text.strip().splitlines()):
        p = line.split()
        try:
            s, e, c, w = float(p[0]), float(p[1]), int(p[2]), int(p[3])
        except (IndexError, ValueError):
            raise ApiError(422, "bad_annotation", "Annotation line {} is not 'start end crackle wheeze'.".format(i + 1))
        if not (0 <= s < e and c in (0, 1) and w in (0, 1)):
            raise ApiError(422, "bad_annotation", "Annotation line {} has invalid values.".format(i + 1))
        out.append((s, e, CLASSES[c + 2 * w]))
    return out


def parse_cycles(text):
    try:
        cyc = [(float(c[0]), float(c[1])) for c in json.loads(text)]
    except (ValueError, TypeError, IndexError):
        raise ApiError(422, "bad_cycles", "`cycles` must be a JSON list of [start, end] pairs in seconds.")
    if any(not 0 <= s < e for s, e in cyc) or len(cyc) > 500:
        raise ApiError(422, "bad_cycles", "Every cycle needs 0 <= start < end; at most 500 cycles.")
    return cyc


@app.get("/v1/health")
def health():
    return {"status": "ok" if "engine" in state else "loading", "pipelines": list(state.get("engine").pipelines) if "engine" in state else [],
            "gpu": torch.cuda.is_available(), "pending": state.get("pending", 0)}


@app.get("/v1/model", dependencies=[Depends(auth)])
def model():
    m = state["manifest"]
    keep = ("label", "checkpoint_sha256", "epoch", "encoder", "method", "spectrum_correction", "thresholds", "calibration")
    return {
        "classes": m["classes"], "classes2": m["classes2"], "devices": m["devices"], "alpha": m["alpha"],
        "preprocessing": m["preprocessing"],
        "pipelines": {n: {**{k: p[k] for k in keep}, **({"sc": {k: v for k, v in p["sc"].items() if k != "state"}} if "sc" in p else {})}
                      for n, p in m["pipelines"].items()},
        "limits": {"max_bytes": MAX_BYTES, "min_sec": 2.0, "max_sec": 90.0, "sample_rate": [4000, 48000], "channels": [1, 2]},
        "sc_freqs_hz": [i * SR / 1024 for i in range(513)],
    }


@app.get("/v1/samples", dependencies=[Depends(auth)])
def samples():
    return {"samples": [{k: v for k, v in s.items() if k != "file"} for s in state["manifest"]["samples"]]}


@app.get("/v1/samples/{sid}/audio", dependencies=[Depends(auth)])
def sample_audio(sid: str):
    return FileResponse(BUNDLE / sample(sid)["file"], media_type="audio/wav")


@app.get("/v1/samples/{sid}/result", dependencies=[Depends(auth)])
async def sample_result(request: Request, sid: str):
    s = sample(sid)

    def run():
        x, meta = read_wav((BUNDLE / s["file"]).read_bytes())
        return {"audio": {**meta, "filename": s["id"] + ".wav", "sha256": s["sha256"]},
                "device": {"declared": s["device"]},
                **state["engine"].infer(to_model_rate(x, meta["sample_rate"]), meta["duration_sec"], s["device"], sample_cycles(s))}

    t0 = time.perf_counter()
    res = await guarded(cached, ("sample", sid), run)
    return {"request_id": request.state.rid, "source": {"kind": "sample", "id": sid}, **res,
            "timing_ms": round(1000 * (time.perf_counter() - t0))}


@app.post("/v1/infer", dependencies=[Depends(auth)])
async def infer(
    request: Request,
    file: UploadFile = File(..., description="WAV audio"),
    device: str = Form("", description="ICBHI device name, or empty when unknown"),
    cycles: str = Form("", description="JSON [[start, end], ...] in seconds"),
    annotation: UploadFile | None = File(None, description="ICBHI annotation .txt (cycles with labels)"),
    reference_sample: str = Form("", description="demo sample id: align to it and use its cycles"),
    window_sec: float = Form(8.0),
    hop_sec: float = Form(4.0),
):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ApiError(413, "too_large", "File is larger than {} MB.".format(MAX_BYTES >> 20))
    if not 1.0 <= window_sec <= 8.0 or not 0.25 <= hop_sec <= window_sec:
        raise ApiError(422, "bad_window", "Need 1 <= window_sec <= 8 and 0.25 <= hop_sec <= window_sec.")
    if device and device not in state["manifest"]["devices"]:
        raise ApiError(422, "bad_device", "Unknown device '{}'.".format(device))
    ref = sample(reference_sample) if reference_sample else None
    cyc = parse_annotation((await annotation.read(1 << 20)).decode("utf-8", "replace")) if annotation and annotation.filename else (
        parse_cycles(cycles) if cycles else None)
    x, meta = read_wav(data)
    sha = hashlib.sha256(data).hexdigest()

    def run():
        x16 = to_model_rate(x, meta["sample_rate"])
        out = {"audio": {**meta, "filename": file.filename, "sha256": sha}, "device": {"declared": device or None}}
        segs = cyc
        if ref is not None:  # demo: align the microphone take to the reference, reuse the reference cycles
            r, rmeta = read_wav((BUNDLE / ref["file"]).read_bytes())
            r16 = to_model_rate(r, rmeta["sample_rate"])
            lag, score = gcc_phat_lag(r16[0].numpy(), x16[0].numpy())
            x16 = shift(x16, lag)[:, : r16.shape[1]]
            if x16.shape[1] < r16.shape[1]:
                x16 = torch.nn.functional.pad(x16, (0, r16.shape[1] - x16.shape[1]))
            out["alignment"] = {"reference_sample": ref["id"], "lag_sec": round(lag, 4), "peak_score": round(score, 2),
                                "reliable": score >= ALIGN_MIN_SCORE, "trimmed_to_sec": rmeta["duration_sec"]}
            segs = sample_cycles(ref)
        duration = x16.shape[1] / SR
        return {**out, **state["engine"].infer(x16, duration, device or None, segs, window_sec, hop_sec)}

    t0 = time.perf_counter()
    key = (sha, device, cycles, json.dumps(cyc), reference_sample, window_sec, hop_sec)
    res = await guarded(cached, key, run)
    return {"request_id": request.state.rid, "source": {"kind": "upload"}, **res, "timing_ms": round(1000 * (time.perf_counter() - t0))}
