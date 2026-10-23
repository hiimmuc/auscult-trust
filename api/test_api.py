"""End-to-end check of the API on the built bundle (needs the checkpoints and a GPU).

    PYTHONPATH=. .venv-train/bin/python -m pytest -q api/test_api.py
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from scipy.signal import butter, lfilter

from api.main import BUNDLE, app

pytestmark = pytest.mark.skipif(not (BUNDLE / "manifest.json").exists() or not torch.cuda.is_available(),
                                reason="needs the built bundle, its checkpoints and a GPU")


def wav_bytes(x, sr):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def test_api():
    with TestClient(app) as c:
        assert c.get("/v1/health").json()["status"] == "ok"
        m = c.get("/v1/model").json()
        assert set(m["pipelines"]) == {"auscult", "baseline"} and "checkpoint" not in m["pipelines"]["auscult"]
        s = c.get("/v1/samples").json()["samples"][0]
        ref = c.get("/v1/samples/{}/result".format(s["id"])).json()
        assert ref["cached"] is False and c.get("/v1/samples/{}/result".format(s["id"])).json()["cached"] is True
        seg = ref["pipelines"]["auscult"]["segments"]
        assert len(seg) == len(s["cycles"]) and seg[0]["truth4"] == s["cycles"][0]["label"]
        assert ref["pipelines"]["auscult"]["sc"]["coefficient_source"] == "saved:test"

        # the same file uploaded with its device and cycles gives the same answer
        raw = (BUNDLE / "samples" / (s["id"] + ".wav")).read_bytes()
        up = c.post("/v1/infer", files={"file": ("a.wav", raw, "audio/wav")},
                    data={"device": s["device"], "cycles": json.dumps([[x["start"], x["end"]] for x in s["cycles"]])}).json()
        assert [x["probs4"] for x in up["pipelines"]["auscult"]["segments"]] == [x["probs4"] for x in seg]

        # demo: a delayed, band-limited, quieter take at 48 kHz aligns back to the reference
        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
        x48 = np.interp(np.arange(len(x) * 48000 // sr) * sr / 48000, np.arange(len(x)), x)
        b, a = butter(2, [150 / 24000, 3000 / 24000], "band")
        take = np.concatenate([np.zeros(int(0.37 * 48000)), 0.3 * lfilter(b, a, x48), np.zeros(48000)])
        take += 0.002 * np.random.default_rng(0).standard_normal(len(take))
        d = c.post("/v1/infer", files={"file": ("mic.wav", wav_bytes(take, 48000), "audio/wav")},
                   data={"reference_sample": s["id"]}).json()
        assert abs(d["alignment"]["lag_sec"] - 0.37) < 0.002 and d["alignment"]["reliable"], d["alignment"]
        assert d["pipelines"]["auscult"]["sc"]["coefficient_source"] == "estimated:recording"
        assert len(d["pipelines"]["baseline"]["segments"]) == len(s["cycles"])

        noise = 0.05 * np.random.default_rng(1).standard_normal(len(take))
        d = c.post("/v1/infer", files={"file": ("n.wav", wav_bytes(noise, 48000), "audio/wav")}, data={"reference_sample": s["id"]}).json()
        assert not d["alignment"]["reliable"], d["alignment"]

        # validation and errors
        bad = c.post("/v1/infer", files={"file": ("x.wav", b"not a wav file at all", "audio/wav")})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "not_wav" and bad.json()["request_id"]
        short = c.post("/v1/infer", files={"file": ("x.wav", wav_bytes(x[: sr // 2], sr), "audio/wav")})
        assert short.json()["error"]["code"] == "duration"
        win = c.post("/v1/infer", files={"file": ("x.wav", raw, "audio/wav")}).json()["pipelines"]["auscult"]
        assert win["segments"][0]["end"] - win["segments"][0]["start"] == 8.0 and win["sc"]["coefficient_source"] == "estimated:recording"
        assert win["recording"]["triage"] in ("refer", "no_referral", "uncertain", "rerecord")
        assert c.get("/v1/samples/nope/result").status_code == 404
