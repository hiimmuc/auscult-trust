import importlib.util

spec = importlib.util.spec_from_file_location("grid_rerun", "scripts/grid_rerun.py")
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)


def test_split_gives_free_machine_the_work_and_balances_ends():
    smoke = {m: {"long": {"run_h": 3.0}, "short": {"run_h": 0.5}} for m in "AB"}
    reruns = [("long", 0), ("long", 1), ("short", 0)]
    plan, end = gr.split(reruns, {"A": 0.0, "B": 20.0}, smoke)
    assert not plan["B"] and len(plan["A"]) == 3  # B busy for 20 h: A takes everything
    plan, end = gr.split(reruns, {"A": 0.0, "B": 0.0}, smoke)
    assert plan["A"] and plan["B"] and sorted(plan["A"] + plan["B"]) == sorted(reruns)


def test_swap_moves_resumed_aside_copies_rerun_and_is_idempotent(tmp_path):
    import subprocess
    for k in ("outputs", "checkpoints"):
        for run, tag in (("grid-v1", "old"), ("grid-v1-rerun", "new")):
            d = tmp_path / k / "train" / run / "c" / "seed1"
            d.mkdir(parents=True)
            (d / "f").write_text(tag)
    (tmp_path / "outputs/train/grid-v1/x/seed0").mkdir(parents=True)  # resumed unit rerun on the other machine
    cmd = gr.swap_cmd([("c", 1), ("x", 0)], [("c", 1)])
    for _ in range(2):
        subprocess.run(["bash", "-c", cmd], cwd=tmp_path, check=True)
    for k in ("outputs", "checkpoints"):
        assert (tmp_path / k / "train/grid-v1/c/seed1/f").read_text() == "new"
        assert (tmp_path / k / "train/grid-v1-resumed/c/seed1/f").read_text() == "old"
    assert (tmp_path / "outputs/train/grid-v1-resumed/x/seed0").is_dir() and not (tmp_path / "outputs/train/grid-v1/x/seed0").exists()
