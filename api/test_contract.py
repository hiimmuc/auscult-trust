"""API contract checks that need no checkpoints or GPU: start-up without a model, validation, auth, error shape.

    PYTHONPATH=. .venv-train/bin/python -m pytest -q api/test_contract.py
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api import engine, main

SAMPLE = {"id": "s1", "class": "normal", "file": "samples/s1.wav", "device": "Meditron", "cycles": []}
MANIFEST = {"classes": ["normal", "crackle", "wheeze", "both"], "classes2": ["normal", "abnormal"], "devices": ["Meditron", "Litt3200"],
            "alpha": 0.1, "preprocessing": {}, "pipelines": {}, "samples": [SAMPLE]}


class Stub:
    """Stands in for the model: returns a fixed, plainly fake result so the upload plumbing can be checked."""

    pipelines = {"stub": None}

    def __init__(self, manifest, bundle):
        pass

    def infer(self, x16, duration, device=None, cycles=None, window_sec=8.0, hop_sec=4.0):
        return {"segmentation": {"mode": "windows"}, "pipelines": {"stub": {"seen_sec": round(duration, 1), "device": device}}}


def wav(seconds=3.0, sr=16000):
    b = io.BytesIO()
    sf.write(b, 0.1 * np.sin(np.arange(int(seconds * sr)) * 0.05).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def upload(c, data=None, **form):
    return c.post("/v1/infer", files={"file": ("x.wav", wav() if data is None else data, "audio/wav")}, data=form)


@pytest.fixture
def empty_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    monkeypatch.setattr(engine, "Engine", Stub)
    monkeypatch.setattr(main, "TOKEN", "")
    monkeypatch.setattr(main, "AUTH", "auto")
    monkeypatch.setattr(main, "RATE_LIMIT", 0)
    monkeypatch.setattr(main, "SAMPLES", "on")
    with TestClient(main.app) as c:
        yield c


def test_no_bundle_is_unavailable_not_fabricated(empty_bundle):
    c = empty_bundle
    h = c.get("/v1/health")
    assert h.status_code == 200 and h.json()["status"] == "unavailable" and h.json()["pipelines"] == []
    assert "/" not in h.json()["reason"]
    for r in (c.get("/v1/model"), c.get("/v1/samples"), c.get("/v1/samples/a/result"), upload(c), upload(c, b"junk")):
        assert r.status_code == 503, r.url
        assert r.json()["error"]["code"] == "model_unavailable" and r.json()["request_id"] and "pipelines" not in r.json()


def test_failed_model_load_hides_paths(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)

    def boom(*a):
        raise FileNotFoundError("/secret/checkpoints/report_epoch_50.pth")

    monkeypatch.setattr(engine, "Engine", boom)
    with TestClient(main.app) as c:
        h = c.get("/v1/health").json()
        assert h["status"] == "unavailable" and "secret" not in json.dumps(h) and "FileNotFoundError" in h["reason"]
        assert upload(c).status_code == 503


def test_upload_validation(stub_client):
    c = stub_client
    assert c.get("/v1/health").json()["status"] == "ok"
    bad = upload(c, b"not a wav file at all")
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "not_wav" and bad.json()["request_id"]
    assert upload(c, wav(0.5)).json()["error"]["code"] == "duration"
    assert upload(c, wav(sr=2000)).json()["error"]["code"] == "sample_rate"
    assert upload(c, device="Nokia").json()["error"]["code"] == "bad_device"
    assert upload(c, window_sec="20").json()["error"]["code"] == "bad_window"
    assert upload(c, cycles="[[3, 1]]").json()["error"]["code"] == "bad_cycles"
    assert upload(c, cycles="nope").json()["error"]["code"] == "bad_cycles"
    r = c.post("/v1/infer", files={"file": ("x.wav", wav(), "audio/wav"), "annotation": ("a.txt", b"0.5 2.0 1 x\n", "text/plain")})
    assert r.json()["error"]["code"] == "bad_annotation"
    assert upload(c, b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40).json()["error"]["code"] == "unreadable"


def test_too_large(stub_client, monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 1000)
    r = upload(stub_client)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_upload_response_shape_and_cache(stub_client):
    c = stub_client
    a = upload(c, device="Meditron")
    assert a.status_code == 200 and a.headers["X-Request-ID"] == a.json()["request_id"]
    j = a.json()
    assert j["source"] == {"kind": "upload"} and j["audio"]["duration_sec"] == 3.0 and j["device"] == {"declared": "Meditron"}
    assert j["pipelines"]["stub"]["device"] == "Meditron" and j["cached"] is False
    assert upload(c, device="Meditron").json()["cached"] is True


def test_inference_crash_is_a_500_without_a_result(stub_client, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("CUDA out of memory at /home/x/model.py")

    monkeypatch.setattr(Stub, "infer", crash)
    r = upload(stub_client)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal" and "/home" not in r.text and "pipelines" not in r.json()


def test_token_and_cors(stub_client, monkeypatch):
    c = stub_client
    monkeypatch.setattr(main, "TOKEN", "s3cret")
    assert c.get("/v1/health").status_code == 200  # open: hosting platforms probe it
    assert c.get("/v1/model").status_code == 401 and c.get("/v1/model").json()["error"]["code"] == "unauthorized"
    assert upload(c).status_code == 401
    assert c.get("/v1/model", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    origin = main.ORIGINS[0]
    pre = c.options("/v1/infer", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "authorization"})
    assert pre.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in c.get("/v1/health", headers={"Origin": "https://evil.example"}).headers


def test_auth_off_ignores_the_token(stub_client, monkeypatch):
    monkeypatch.setattr(main, "TOKEN", "s3cret")
    monkeypatch.setattr(main, "AUTH", "off")
    assert stub_client.get("/v1/model").status_code == 200 and upload(stub_client).status_code == 200


def test_rate_limit(stub_client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    main.hits.clear()
    assert [upload(stub_client, device="Meditron").status_code for _ in range(2)] == [200, 200]
    r = upload(stub_client)
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    assert stub_client.get("/v1/model").status_code == 200  # only inference is limited
    main.hits.clear()


def test_samples_switch(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    monkeypatch.setattr(engine, "Engine", Stub)
    for mode, n in (("on", 1), ("off", 0)):
        monkeypatch.setattr(main, "SAMPLES", mode)
        with TestClient(main.app) as c:
            assert len(c.get("/v1/samples").json()["samples"]) == n
            assert (c.get("/v1/samples/s1/result").status_code == 404) == (n == 0)  # off: unknown sample
