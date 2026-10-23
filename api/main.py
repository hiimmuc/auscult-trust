"""Inference API for the AuscultTrust demo.

    PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000

Without a bundle or checkpoints the service still starts: `/v1/health` says `unavailable` with the reason
and every route that needs the model answers 503 `model_unavailable`. It never returns a prediction it did not compute.

Environment:
    AT_APP_BUNDLE      bundle directory from `api.bundle` (default app/bundle)
    AT_APP_AUTH        on | off | auto (default). Authentication of every /v1 route except /v1/health, with
                       `Authorization: Bearer <AT_APP_TOKEN>`. auto: enforced when AT_APP_TOKEN is set. on: enforced, the
                       API refuses to start without a token. off: open, even if a token is set
    AT_APP_TOKEN       the shared access token
    AT_APP_RATE_LIMIT  inference requests (/v1/infer and sample results) per minute, all clients together; 0 = unlimited (default)
    AT_APP_SAMPLES     on (default) | off: off serves no demo samples (the ICBHI recordings) and turns Demo mode off
    AT_APP_ORIGINS     comma-separated CORS origins (default: the Vite dev server on localhost and 127.0.0.1)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from api import engine as E

log = logging.getLogger("uvicorn.error")

BUNDLE = Path(os.environ.get("AT_APP_BUNDLE", "app/bundle"))
TOKEN = os.environ.get("AT_APP_TOKEN", "")
AUTH = os.environ.get("AT_APP_AUTH", "auto")
RATE_LIMIT = int(os.environ.get("AT_APP_RATE_LIMIT", "0"))
SAMPLES = os.environ.get("AT_APP_SAMPLES", "on")
if AUTH not in ("on", "off", "auto") or SAMPLES not in ("on", "off") or (AUTH == "on" and not TOKEN):
    raise RuntimeError("Bad config: AT_APP_AUTH is on|off|auto (on needs AT_APP_TOKEN), AT_APP_SAMPLES is on|off.")
ORIGINS = [o.strip() for o in os.environ.get("AT_APP_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
MAX_BYTES = 40 * 1024 * 1024
MAX_PENDING = 4  # requests waiting for or holding the model; more get 429
TIMEOUT_SEC = 120
# ponytail: fixed cut on the GCC-PHAT peak z-score, set on 3 synthetic takes (clean 651, noisy 18, white noise 4.3, another
# ICBHI recording 8.1); calibrate on real speaker/microphone takes before trusting the flag
ALIGN_MIN_SCORE = 12.0

state = {}


def reset():
    state.update(engine=None, manifest=None, samples={}, pending=0, cache=OrderedDict(), unavailable="Starting.")


reset()


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(app):
    reset()
    try:
        manifest = json.loads((BUNDLE / "manifest.json").read_text())
        state.update(manifest=manifest, engine=E.Engine(manifest, BUNDLE), unavailable=None,
                     samples={s["id"]: s for s in manifest["samples"]} if SAMPLES == "on" else {})
    except Exception as e:
        log.exception("model not loaded")
        state["unavailable"] = ("No model bundle found." if isinstance(e, FileNotFoundError) and "manifest" in str(e)
                                else "The model failed to load ({}).".format(type(e).__name__))
    yield


app = FastAPI(title="AuscultTrust inference API", version="1.1", lifespan=lifespan)
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


@app.exception_handler(E.AudioError)
async def audio_error(request: Request, e: E.AudioError):
    return await api_error(request, ApiError(422, e.code, str(e)))


def auth(request: Request):
    if AUTH == "off" or (AUTH == "auto" and not TOKEN):
        return
    got = request.headers.get("authorization", "")
    if not hmac.compare_digest(got.encode(), "Bearer {}".format(TOKEN).encode()):
        raise ApiError(401, "unauthorized", "Missing or wrong access token.")


hits = deque()  # monotonic times of the inference requests in the last minute


def rate_limit():
    if not RATE_LIMIT:
        return
    now = time.monotonic()
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise ApiError(429, "rate_limited", "More than {} inference requests a minute. Retry in {} s.".format(RATE_LIMIT, int(60 - (now - hits[0])) + 1))
    hits.append(now)


def ready():
    """503 unless the model is loaded. Never falls back to a made-up answer."""
    if state["engine"] is None:
        raise ApiError(503, "model_unavailable", state["unavailable"])


async def guarded(fn, *args):
    """Run blocking inference in a worker thread, with a queue bound and a time limit."""
    if state["pending"] >= MAX_PENDING:
        raise ApiError(429, "busy", "The model is busy. Retry in a few seconds.")
    state["pending"] += 1
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args), TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise ApiError(504, "timeout", "Inference took longer than {} s.".format(TIMEOUT_SEC))
    except (ApiError, E.AudioError):
        raise
    except Exception:
        log.exception("inference failed")
        raise ApiError(500, "internal", "Inference failed. Nothing was computed.")
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
    ready()
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
        out.append((s, e, state["manifest"]["classes"][c + 2 * w]))
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
    up = state["engine"] is not None
    return {"status": "ok" if up else "unavailable", "pipelines": list(state["engine"].pipelines) if up else [],
            "gpu": up and E.torch.cuda.is_available(), "pending": state["pending"],
            **({} if up else {"reason": state["unavailable"]})}


@app.get("/v1/model", dependencies=[Depends(auth)])
def model():
    ready()
    m = state["manifest"]
    keep = ("label", "checkpoint_sha256", "epoch", "encoder", "method", "spectrum_correction", "thresholds", "calibration")
    return {
        "classes": m["classes"], "classes2": m["classes2"], "devices": m["devices"], "alpha": m["alpha"],
        "preprocessing": m["preprocessing"],
        "pipelines": {n: {**{k: p[k] for k in keep}, **({"sc": {k: v for k, v in p["sc"].items() if k != "state"}} if "sc" in p else {})}
                      for n, p in m["pipelines"].items()},
        "limits": {"max_bytes": MAX_BYTES, "min_sec": E.MIN_SEC, "max_sec": E.MAX_SEC, "sample_rate": [E.MIN_SR, E.MAX_SR], "channels": [1, 2]},
        "sc_freqs_hz": [i * E.SR / 1024 for i in range(513)],
    }


@app.get("/v1/samples", dependencies=[Depends(auth)])
def samples():
    ready()
    return {"samples": [{k: v for k, v in s.items() if k != "file"} for s in state["samples"].values()]}


@app.get("/v1/samples/{sid}/audio", dependencies=[Depends(auth)])
def sample_audio(sid: str):
    return FileResponse(BUNDLE / sample(sid)["file"], media_type="audio/wav")


@app.get("/v1/samples/{sid}/result", dependencies=[Depends(auth), Depends(rate_limit)])
async def sample_result(request: Request, sid: str):
    s = sample(sid)

    def run():
        x, meta = E.read_wav((BUNDLE / s["file"]).read_bytes())
        return {"audio": {**meta, "filename": s["id"] + ".wav", "sha256": s["sha256"]},
                "device": {"declared": s["device"]},
                **state["engine"].infer(E.to_model_rate(x, meta["sample_rate"]), meta["duration_sec"], s["device"], sample_cycles(s))}

    t0 = time.perf_counter()
    hit = ("sample", sid) in state["cache"]
    res = await guarded(cached, ("sample", sid), run)
    return {"request_id": request.state.rid, "source": {"kind": "sample", "id": sid}, **res,
            "cached": hit, "timing_ms": round(1000 * (time.perf_counter() - t0))}


@app.post("/v1/infer", dependencies=[Depends(auth), Depends(rate_limit)])
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
    ready()
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
    x, meta = E.read_wav(data)
    sha = hashlib.sha256(data).hexdigest()

    def run():
        x16 = E.to_model_rate(x, meta["sample_rate"])
        out = {"audio": {**meta, "filename": file.filename, "sha256": sha}, "device": {"declared": device or None}}
        segs = cyc
        if ref is not None:  # demo: align the microphone take to the reference, reuse the reference cycles
            r, rmeta = E.read_wav((BUNDLE / ref["file"]).read_bytes())
            x16, lag, score = E.align(E.to_model_rate(r, rmeta["sample_rate"]), x16)
            out["alignment"] = {"reference_sample": ref["id"], "lag_sec": round(lag, 4), "peak_score": round(score, 2),
                                "reliable": score >= ALIGN_MIN_SCORE, "trimmed_to_sec": rmeta["duration_sec"]}
            segs = sample_cycles(ref)
        duration = x16.shape[1] / E.SR
        return {**out, **state["engine"].infer(x16, duration, device or None, segs, window_sec, hop_sec)}

    t0 = time.perf_counter()
    key = (sha, device, cycles, json.dumps(cyc), reference_sample, window_sec, hop_sec)
    hit = key in state["cache"]
    res = await guarded(cached, key, run)
    return {"request_id": request.state.rid, "source": {"kind": "upload"}, **res, "cached": hit,
            "timing_ms": round(1000 * (time.perf_counter() - t0))}
