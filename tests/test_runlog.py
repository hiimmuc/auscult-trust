import json


def test_run_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setenv("RUN_TAG", "demo")
    from src import runlog
    p = runlog.log_run("exp", "fm+branch", 3, {"a": 1}, "hash", "enc", {"score": 0.5})
    assert p.parts[-4] == "exp"
    assert p.name == "seed3.json" and p.parent.name == "fm+branch"
    rid = p.parts[-3]
    assert rid.endswith("_demo") and rid[8] == "-"
    assert json.loads((p.parent.parent / "meta.json").read_text())["config"] == {"a": 1}
    assert runlog.ckpt_path("exp", "fm+branch", 3).parts[-5:] == ("checkpoints", "exp", rid, "fm+branch", "seed3.pt")
    assert json.loads(p.read_text())["metrics"]["score"] == 0.5
