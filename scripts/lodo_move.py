"""Move unstarted LODO units from A and B to D once D's own run has ended; then keep the three sweeps alive and collect the
outputs on A. Runs on A, drives B and D through ssh; safe to restart.

    setsid nohup .venv-train/bin/python scripts/lodo_move.py > logs/lodo-v1-move.log 2>&1 < /dev/null &
    .venv-train/bin/python scripts/lodo_move.py --dry --d-free 17   # plan only, as if D were free in 17 h

1. Wait until D's sweep of `--d-run` has stopped with all its units done (or its log says it finished).
2. Stop the A and B sweeps (by PID; their running jobs go on as orphans). Remaining hours per machine = smoke hours x fold
   cost x fraction of epochs left, over the throughput measured since the sweep started (D gets B's: same GPU). Move units
   that have not started, from the end of the busier list, to D while that lowers the latest end.
3. Write `<run>-A2.txt`, `<run>-B2.txt` (old list minus the moved units; the original lists are kept) and `<run>-D.txt`,
   start the three sweeps (tags -A2, -B2, -D: new log directories, old job logs are not overwritten).
4. Restart any of the three sweeps that died without finishing (power cut); when all are done, copy B's and D's outputs to A.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from grid import LODO_DEVICES

REPO = Path(os.environ.get("AT_REPO") or Path(__file__).resolve().parents[1])  # AT_REPO: the checkout whose logs/outputs to use
HOST = {"B": "namdp15@10.148.183.8", "D": "namdp15@10.148.183.171"}
REMOTE = "Workspace/Personal/KHKT-CVA-2026/auscult-trust"
CAMP = "logs/campaigns"
PY = ".venv-train/bin/python"


def sh(m, cmd, check=False, input=None):
    """Run a shell command in the repo of machine m; returns stdout."""
    full = ["bash", "-c", cmd] if m == "A" else ["ssh", HOST[m], f"cd {REMOTE} && bash -c {shlex.quote(cmd)}"]
    p = subprocess.run(full, cwd=REPO, capture_output=True, text=True, timeout=3600, input=input)
    if check and p.returncode:
        raise RuntimeError(f"{m}: {cmd[:80]}: exit {p.returncode}: {p.stderr[-500:]}")
    return p.stdout


def log(*a):
    print(time.strftime("%d/%m %H:%M"), *a, flush=True)


def unit(line):
    r = re.search(r"--cell (\S+) .*--seed (\d+)", line)
    return (r[1], int(r[2])) if r else None


def lines(m, name):
    return [l for l in sh(m, f"cat {CAMP}/{name}.txt").splitlines() if l.strip()]


def done(m, run):
    return {(p.split("/")[3], int(p.split("/")[4][4:])) for p in sh(m, f"ls outputs/train/{run}/*/seed*/report.json 2>/dev/null").split()}


def sweep_pids(m, lst):
    """PIDs of the sweep scheduler of this list (matched on its own command line, never on a shell's)."""
    out = sh(m, "ps -eo pid,args")
    return [int(l.split()[0]) for l in out.splitlines() if l.split(None, 1)[-1].startswith(f"{PY} scripts/sweep.py {lst} ")]


def progress(m, tag):
    """Last epoch reached per unit, from the job logs of sweep tag `tag`."""
    out = sh(m, f"for f in logs/{tag}/*.log; do grep -a -m1 -o 'cell=[^,]*, unit=seed[0-9]*' $f | tr '\\n' ' '; "
                "grep -a '^Train epoch' $f | tail -1 | cut -d, -f1; echo; done")
    p = {}
    for r in re.finditer(r"cell=(\S+), unit=seed(\d+) Train epoch (\d+)", out):
        p[(r[1], int(r[2]))] = max(int(r[3]), p.get((r[1], int(r[2])), 0))
    return p


def hours(smoke, u):
    """Sequential smoke hours of one LODO unit."""
    base, dev = u[0].split("+lodo_")
    return smoke[base]["run_h"] * LODO_DEVICES[dev]


def state(m, run, smoke):
    """(remaining smoke hours, measured throughput, units not started in list order) of machine m's list `<run>-m`."""
    ls = [unit(l) for l in lines(m, f"{run}-{m}")]
    fin, prog = done(m, run), progress(m, f"{run}-{m}")
    epochs = lambda u: smoke[u[0].split("+lodo_")[0]]["epochs"]
    did = sum(hours(smoke, u) for u in ls if u in fin) + sum(hours(smoke, u) * min(e, epochs(u)) / epochs(u)
                                                              for u, e in prog.items() if u not in fin)
    t0 = int(sh(m, f"stat -c %W logs/{run}-{m}/0.log").strip())  # birth of the first job log = sweep start
    speed = did / max((time.time() - t0) / 3600, 0.5)
    left = sum(hours(smoke, u) * (1 - prog.get(u, 0) / epochs(u)) for u in ls if u not in fin)
    return left, speed, [u for u in ls if u not in fin and u not in prog]


def plan(run, d_free):
    """Units to move per machine, throughput, and the expected end (hours from now) of every machine before and after."""
    smoke = {m: json.loads((REPO / f"logs/grid_smoke_{m}.json").read_text()) for m in "AB"}
    st = {m: state(m, run, smoke[m]) for m in "AB"}
    speed = {"A": st["A"][1], "B": st["B"][1], "D": st["B"][1]}
    end = {"A": st["A"][0] / speed["A"], "B": st["B"][0] / speed["B"], "D": d_free}
    before, movable, moved = dict(end), {m: list(st[m][2]) for m in "AB"}, {"A": [], "B": []}
    while True:
        m = max("AB", key=end.get)
        if not movable[m]:
            break
        u = movable[m][-1]
        new = {**end, m: end[m] - hours(smoke[m], u) / speed[m], "D": end["D"] + hours(smoke["B"], u) / speed["D"]}
        if max(new.values()) >= max(end.values()):
            break
        end = new
        moved[m].append(movable[m].pop())
    return moved, speed, before, end


def start(m, run, tag, workers=4):
    sh(m, f"RUN_ID={run} SWEEP_TAG=-{tag} setsid nohup {PY} scripts/sweep.py {CAMP}/{run}-{tag}.txt --workers {workers} --ram-gb 6 "
          f">> logs/{run}-{tag}.log 2>&1 < /dev/null &", check=True)
    log(f"{m}: sweep {run}-{tag} started")


def finished_ok(m, run, tag):
    return {unit(l) for l in lines(m, f"{run}-{tag}")} <= done(m, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lodo-v1")
    ap.add_argument("--d-run", default="pcmcl-v1")
    ap.add_argument("--dry", action="store_true", help="print the plan for a D free in --d-free hours, change nothing")
    ap.add_argument("--d-free", type=float, default=0.0)
    a = ap.parse_args()
    run, tags = a.run, {"A": "A2", "B": "B2", "D": "D"}
    if a.dry:
        moved, speed, before, end = plan(run, a.d_free)
        print(json.dumps({"speed": speed, "end_h_before": before, "end_h_after": end, "moved": {m: len(v) for m, v in moved.items()}}, indent=1))
        return
    if not (REPO / CAMP / f"{run}-D.txt").exists():
        d_list = f"{CAMP}/{a.d_run}-D.txt"
        while sweep_pids("D", d_list) or not (finished_ok("D", a.d_run, "D") or "all jobs finished" in sh("D", f"tail -1 logs/{a.d_run}-D.log")):
            time.sleep(300)
        log(f"D: {a.d_run} ended")
        for m in "AB":
            for pid in sweep_pids(m, f"{CAMP}/{run}-{m}.txt"):
                sh(m, f"kill {pid}", check=True)
                log(f"{m}: stopped sweep PID {pid} (its running jobs continue)")
        moved, speed, before, end = plan(run, 0.0)
        log("plan:", json.dumps({"speed": speed, "end_h_before": before, "end_h_after": end, "moved": moved}))
        to_d = []
        for m in "AB":
            keep = []
            for l in lines(m, f"{run}-{m}"):
                (to_d if unit(l) in moved[m] else keep).append(l)
            sh(m, f"cat > {CAMP}/{run}-{tags[m]}.txt", check=True, input="\n".join(keep) + "\n")
        sh("D", f"cat > {CAMP}/{run}-D.txt", check=True, input="\n".join(to_d) + "\n")
        (REPO / CAMP / f"{run}-D.txt").write_text("\n".join(to_d) + "\n")  # local copy marks the move as done (restart-safe)
        for m in "ABD":
            start(m, run, tags[m])
    while True:  # keep alive: restart a sweep that died without its final line (power cut, crash)
        todo = [m for m in "ABD" if not finished_ok(m, run, tags[m])]
        dead = [m for m in todo if not sweep_pids(m, f"{CAMP}/{run}-{tags[m]}.txt")]
        failed = [m for m in dead if "all jobs finished" in sh(m, f"tail -1 logs/{run}-{tags[m]}.log")]
        if len(failed) == len(todo):
            break
        for m in set(dead) - set(failed):
            log(f"{m}: sweep {run}-{tags[m]} not running, restarting")
            start(m, run, tags[m])
        time.sleep(600)
    for m in failed:
        log(f"{m}: sweep {run}-{tags[m]} ended with units missing, see logs/{run}-{tags[m]}.log")
    for m in "BD":
        subprocess.run(["rsync", "-a", f"{HOST[m]}:{REMOTE}/outputs/train/{run}/", f"outputs/train/{run}/"], cwd=REPO, check=True)
    log(f"ALL DONE: A, B and D units of {run} in outputs/train/{run}/ on A")


if __name__ == "__main__":
    main()
