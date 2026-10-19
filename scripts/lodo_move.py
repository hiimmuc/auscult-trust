"""Re-split the unstarted units of running sweeps over a group of machines, then keep the new sweeps alive and collect the
outputs on A. Runs on A, drives the other machines through ssh; safe to restart (the plan is saved).

    # LODO without SC shared by A, B and E, now:
    setsid nohup .venv-train/bin/python scripts/lodo_move.py --src A:lodo-v1-A B:lodo-v1-B --to A B E \
        > logs/lodo-v1-share-ABE.log 2>&1 < /dev/null &
    # trained SC shared by C and D, once D's pcmcl-v1 sweep has ended:
    setsid nohup .venv-train/bin/python scripts/lodo_move.py --src C:lodo-v1-C --to C D --after D:pcmcl-v1 \
        > logs/lodo-v1-share-CD.log 2>&1 < /dev/null &
    # plan only (D busy for 18 more hours), nothing changed:
    .venv-train/bin/python scripts/lodo_move.py --src C:lodo-v1-C --to C D --dry --delay D=18

1. With `--after M:RUN`: wait until M's sweep of RUN has stopped with all its units done (or its final line written).
2. Stop the source schedulers by PID. Their running jobs go on and stay on their machine, as do interrupted units
   (their last.pth is there).
3. Every unit not started yet goes, longest first, to the machine that would finish it first. Hours = smoke hours on B x
   fold cost x SC cost; throughput = B-smoke hours per wall hour, measured on each machine's `<run>-<M>` list since it
   started (`--speed M=x` overrides; a machine without a list gets B's).
4. New lists `<run>-<M>2.txt` (the source lists are kept). A machine's new sweep (tag -<M>2) starts once none of its
   running jobs is left, so nothing is stopped and nothing runs twice.
5. Restart a sweep that died without its final line (power cut); at the end copy the other machines' outputs to A.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from grid import LODO_DEVICES, SC_COST

REPO = Path(os.environ.get("AT_REPO") or Path(__file__).resolve().parents[1])  # AT_REPO: the checkout whose logs/outputs to use
HOST = {"B": "namdp15@10.148.183.8", "C": "namdp15@10.148.183.119", "D": "namdp15@10.148.183.171", "E": "namdp15@10.148.182.179"}
REMOTE = "Workspace/Personal/KHKT-CVA-2026/auscult-trust"
CAMP = "logs/campaigns"
PY = ".venv-train/bin/python"
SWEEP_ARGS = {"C": "--workers 3 --gpu-gb 29"}  # C: the user's share of its GPU
SMOKE = json.loads((REPO / "logs/grid_smoke_B.json").read_text())


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
    return [l for l in sh(m, f"cat {CAMP}/{name}.txt 2>/dev/null").splitlines() if l.strip()]


def done(m, run):
    return {(p.split("/")[3], int(p.split("/")[4][4:])) for p in sh(m, f"ls outputs/train/{run}/*/seed*/report.json 2>/dev/null").split()}


def sweep_pids(m, lst):
    """PIDs of the sweep scheduler of this list (matched on its own command line, never on a shell's)."""
    out = sh(m, "ps -eo pid,args")
    return [int(l.split()[0]) for l in out.splitlines() if l.split(None, 1)[-1].startswith(f"{PY} scripts/sweep.py {CAMP}/{lst}.txt ")]


def training(m):
    """Units with a training process on machine m."""
    return {unit(l) for l in sh(m, "ps -eo args").splitlines() if "patchmix_cl.main" in l and unit(l)}


def progress(m, lst):
    """Last epoch reached per unit, from the job logs of list `lst`."""
    out = sh(m, f"for f in logs/{lst}/*.log; do grep -a -m1 -o 'cell=[^,]*, unit=seed[0-9]*' $f | tr '\\n' ' '; "
                "grep -a '^Train epoch' $f | tail -1 | cut -d, -f1; echo; done 2>/dev/null")
    p = {}
    for r in re.finditer(r"cell=(\S+), unit=seed(\d+) Train epoch (\d+)", out):
        p[(r[1], int(r[2]))] = max(int(r[3]), p.get((r[1], int(r[2])), 0))
    return p


def base(u):
    return re.split(r"\+sc|\+lodo_", u[0])[0]


def hours(u):
    """Sequential smoke hours (machine B) of one unit."""
    dev = u[0].split("+lodo_")[1] if "+lodo_" in u[0] else None
    return SMOKE[base(u)]["run_h"] * (LODO_DEVICES[dev] if dev else 1.0) * (SC_COST if "+sc" in u[0] else 1.0)


def left(u, prog):
    return hours(u) * (1 - min(prog.get(u, 0), SMOKE[base(u)]["epochs"]) / SMOKE[base(u)]["epochs"])


def speed(m, lst):
    """B-smoke hours done per wall hour by machine m on list `lst` since its first job started."""
    ls, fin, prog = [unit(l) for l in lines(m, lst)], done(m, lst.rsplit("-", 1)[0]), progress(m, lst)
    did = sum(hours(u) for u in ls if u in fin) + sum(hours(u) - left(u, prog) for u in prog if u in ls and u not in fin)
    t0 = int(sh(m, f"stat -c %W logs/{lst}/0.log").strip())
    return did / max((time.time() - t0) / 3600, 0.5)


def plan(run, src, to, speeds, delay):
    """Assign units: running and interrupted ones stay; the rest longest first to the machine that ends it first."""
    sp = {m: speeds.get(m) or (speed(m, l) if m in src else None) for m, l in [(m, src.get(m)) for m in to]}
    b = sp.get("B") or speeds.get("B") or speed("B", f"{run}-B")
    sp = {m: v or b for m, v in sp.items()}
    keep, free, lines_of, end = {m: [] for m in to}, [], {}, {}
    for m in to:
        end[m] = delay.get(m, 0.0)
        if m not in src:
            continue
        ls = lines(m, src[m])
        fin, prog, run_now = done(m, run), progress(m, src[m]), training(m)
        orphans = [u for u in map(unit, ls) if u in run_now and u not in fin]
        if orphans:  # the new sweep waits for all of them; meanwhile they share the machine
            end[m] += max(left(u, prog) for u in orphans) * len(orphans) / sp[m]
        for l in ls:
            u = unit(l)
            lines_of[u] = l
            if u in fin:
                continue
            if u in run_now or u in prog:
                keep[m].append(u)
                if u not in run_now:
                    end[m] += left(u, prog) / sp[m]
            else:
                free.append(u)
    for u in sorted(free, key=lambda u: -hours(u)):
        m = min(to, key=lambda m: end[m] + hours(u) / sp[m])
        end[m] += hours(u) / sp[m]
        keep[m].append(u)
    return keep, lines_of, sp, end


def start(m, lst):
    sh(m, f"RUN_ID={lst.rsplit('-', 1)[0]} SWEEP_TAG=-{lst.rsplit('-', 1)[1]} setsid nohup {PY} scripts/sweep.py {CAMP}/{lst}.txt "
          f"{SWEEP_ARGS.get(m, '--workers 4')} --ram-gb 6 >> logs/{lst}.log 2>&1 < /dev/null &", check=True)
    log(f"{m}: sweep {lst} started")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lodo-v1")
    ap.add_argument("--src", nargs="+", required=True, help="M:list of the running sweeps whose units are re-split")
    ap.add_argument("--to", nargs="+", required=True, help="machines that share the units")
    ap.add_argument("--after", default=None, help="M:RUN, start once M's sweep of RUN has ended")
    ap.add_argument("--speed", nargs="*", default=[], help="M=x: throughput of a machine without history")
    ap.add_argument("--delay", nargs="*", default=[], help="M=h: hours until M is free (dry run)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run, src = a.run, dict(s.split(":") for s in a.src)
    speeds = {k: float(v) for k, v in (s.split("=") for s in a.speed)}
    delay = {k: float(v) for k, v in (s.split("=") for s in a.delay)}
    new = {m: f"{run}-{m}2" for m in a.to}
    saved = REPO / CAMP / f"{run}-share-{''.join(a.to)}.json"
    if a.dry:
        keep, _, sp, end = plan(run, src, a.to, speeds, delay)
        print(json.dumps({"speed": sp, "end_h": end, "units": {m: len(v) for m, v in keep.items()}}, indent=1))
        return
    if not saved.exists():
        if a.after:
            m, r = a.after.split(":")
            while sweep_pids(m, f"{r}-{m}") or not ({unit(l) for l in lines(m, f"{r}-{m}")} <= done(m, r)
                                                    or "all jobs finished" in sh(m, f"tail -1 logs/{r}-{m}.log")):
                time.sleep(300)
            log(f"{m}: {r} ended")
        for m, lst in src.items():
            for pid in sweep_pids(m, lst):
                sh(m, f"kill {pid}", check=True)
                log(f"{m}: stopped sweep {lst} (PID {pid}); its running jobs continue")
        keep, lines_of, sp, end = plan(run, src, a.to, speeds, {})
        for m in a.to:
            sh(m, f"mkdir -p {CAMP} && cat > {CAMP}/{new[m]}.txt", check=True, input="".join(lines_of[u] + "\n" for u in keep[m]))
        saved.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "speed": sp, "end_h": end,
                                     "units": {m: [list(u) for u in keep[m]] for m in a.to}}, indent=1))
        log("plan:", json.dumps({"speed": sp, "end_h": end, "units": {m: len(v) for m, v in keep.items()}}))
    units = {m: {tuple(u) for u in v} for m, v in json.loads(saved.read_text())["units"].items()}
    started = set()
    while True:
        todo = [m for m in a.to if not units[m] <= done(m, run)]
        for m in todo:
            if sweep_pids(m, new[m]):
                started.add(m)
                continue
            final = "all jobs finished" in sh(m, f"tail -1 logs/{new[m]}.log 2>/dev/null")
            if m in started and final:
                continue  # ended with failed units: reported below
            if m not in started and units[m] & training(m):
                continue  # its old jobs are still running
            if m in started:
                log(f"{m}: sweep {new[m]} not running, restarting")
            start(m, new[m])
            started.add(m)
        if all(m in started and not sweep_pids(m, new[m]) for m in todo):
            break
        time.sleep(300)
    for m in todo:
        log(f"{m}: sweep {new[m]} ended with units missing, see logs/{new[m]}.log")
    for m in a.to:
        if m != "A":
            subprocess.run(["rsync", "-a", f"{HOST[m]}:{REMOTE}/outputs/train/{run}/", f"outputs/train/{run}/"], cwd=REPO, check=True)
    log(f"ALL DONE: {', '.join(a.to)} units of {run} in outputs/train/{run}/ on A")


if __name__ == "__main__":
    main()
