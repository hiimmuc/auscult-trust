import json

import pytest

from src import artifacts


def test_atomic_failure_preserves_previous_artifact(tmp_path):
    path = tmp_path / "report.json"
    artifacts.atomic_json(path, {"old": True})
    with pytest.raises(RuntimeError):
        with artifacts.atomic_path(path) as temporary:
            temporary.write_text("partial")
            raise RuntimeError("interrupted")
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_prediction_certificate_rejects_stale_inputs_and_modified_output(tmp_path):
    output = tmp_path / "predictions.npz"
    output.write_bytes(b"prediction")
    inputs = {"checkpoint": "a", "data": "b", "sc": "c", "code": "d"}
    with pytest.raises(ValueError, match="Missing provenance"):
        artifacts.verify_prediction(output, inputs)
    artifacts.record_prediction(output, inputs)
    artifacts.verify_prediction(output, inputs)
    for key in inputs:
        with pytest.raises(ValueError, match="Stale"):
            artifacts.verify_prediction(output, {**inputs, key: "changed"})
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        artifacts.verify_prediction(output, inputs)


def test_fingerprint_tracks_data_sc_code_and_checkpoint(tmp_path, monkeypatch):
    for name in ("repo/src", "refs", "data", "audio/audio_test_data", "unit"):
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(artifacts, "REPO", tmp_path / "repo")
    monkeypatch.setattr(artifacts, "REPOS", tmp_path / "refs")
    monkeypatch.setattr(artifacts, "DATA", tmp_path / "data")
    monkeypatch.setattr(artifacts, "environment", lambda: {"packages": {"numpy": "fixed"}})
    unit, checkpoint = tmp_path / "unit", tmp_path / "model.pth"
    artifacts.atomic_json(unit / "train_args.json", {"data_folder": str(tmp_path / "audio"), "encoder": "beats"})
    files = [checkpoint, unit / "sc_state.npz", tmp_path / "audio/official_split.txt",
             tmp_path / "audio/audio_test_data/a.wav", tmp_path / "repo/src/preprocess.py"]
    for path in files:
        path.write_bytes(b"original")
    original = artifacts.prediction_inputs(unit, checkpoint, "screening", True)
    for path in files:
        path.write_bytes(b"changed")
        assert artifacts.prediction_inputs(unit, checkpoint, "screening", True) != original
        path.write_bytes(b"original")


def test_pipeline_rejects_modified_weights_before_model_load(tmp_path, monkeypatch):
    from api import engine
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"original")
    expected = artifacts.sha256_file(checkpoint)
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(engine, "load_frozen", lambda *a: pytest.fail("loaded unverified model"))
    with pytest.raises(ValueError, match="integrity"):
        engine.Pipeline("test", {"checkpoint": str(checkpoint), "checkpoint_sha256": expected}, tmp_path)


def test_legacy_bundle_is_rejected(tmp_path):
    from api.engine import Engine
    with pytest.raises(ValueError, match="Legacy"):
        Engine({"pipelines": {}}, tmp_path)
