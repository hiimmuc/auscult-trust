import json


def test_run_layout_resumes_and_never_overwrites(tmp_path, monkeypatch):
    from src import runs
    monkeypatch.setattr(runs, "_KINDS", {"outputs": tmp_path / "outputs", "checkpoints": tmp_path / "checkpoints"})
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setenv("RUN_TAG", "demo")
    p = runs.log_run("exp", "fm+branch", 3, {"a": 1}, {"score": 0.5})
    assert p.parts[-4] == "exp" and p.name == "seed3.json" and p.parent.name == "fm+branch"
    rid = p.parts[-3]
    assert rid.endswith("_demo") and rid[8] == "-"
    assert json.loads((p.parent.parent / "meta.json").read_text())["config"] == {"a": 1}
    assert json.loads(p.read_text())["metrics"]["score"] == 0.5
    u = runs.unit_dir("exp", "P0", "cv0", "checkpoints")
    assert u.parts[-5:] == ("checkpoints", "exp", rid, "P0", "cv0")
    (u / "last.pth").write_text("x")
    assert (runs.unit_dir("exp", "P0", "cv0", "checkpoints") / "last.pth").exists()  # same RUN_ID: resume keeps files
    assert (tmp_path / "outputs" / "exp" / "latest").resolve().name == rid
    monkeypatch.setenv("RUN_ID", "20300101-000000")  # a new id leaves the earlier run untouched
    assert not (runs.unit_dir("exp", "P0", "cv0", "checkpoints") / "last.pth").exists()
    assert runs.sha256_file(u / "last.pth") == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
