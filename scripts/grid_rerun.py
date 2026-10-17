"""Rerun the grid units that were resumed from last.pth, without interruption, on whichever machine is free first; then
swap the reruns into the grid and finish (audit, sync, tables). Runs on A, drives B through ssh; safe to restart.

    setsid nohup .venv-train/bin/python scripts/grid_rerun.py > logs/grid-v1-rerun-dispatch.log 2>&1 < /dev/null &

1. Wait until one machine's grid sweep has stopped with all its units done ("free").
2. Split the reruns once (saved in logs/campaigns/grid-v1-rerun-plan.json): longest first, each to the machine that would
   end it earlier, given the busy machine's remaining grid hours (smoke hours of its unfinished units, minus the epochs
   already done, over its measured throughput). Start the free machine's rerun sweep at once; the other machine's when it
   is free.
3. When every rerun is done: on both machines move the resumed units to `<outputs|checkpoints>/train/grid-v1-resumed/`
   and copy the reruns into grid-v1; audit each machine's final units with --hash; copy B's outputs to A; audit all
   outputs on A; build both tables. Results are moved, never deleted.
"""
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B = "namdp15@10.148.183.8"
REMOTE_REPO = "~/Workspace/Personal/KHKT-CVA-2026/auscult-trust"
SPEED = {"A": 1.17, "B": 1.73}  # sequential smoke-hours done per hour with parallel jobs, measured 16/10 15:00-17/10 10:43
CAMP = REPO / "logs" / "campaigns"
PLAN = CAMP / "grid-v1-rerun-plan.json"
PY = ".venv-train/bin/python"


def sh(m, cmd, check=False, input=None):
    """Run a shell command in the repo of machine m; returns stdout."""
    full = ["bash", "-c", cmd] if m == "A" else ["ssh", B, f"cd {REMOTE_REPO} && bash -c {shlex.quote(cmd)}"]
    p = subprocess.run(full, cwd=REPO, capture_output=True, text=True, timeout=3600, input=input)
    if check and p.returncode:
        raise RuntimeError(f"{m}: {cmd[:80]}: exit {p.returncode}: {p.stderr[-500:]}")
    return p.stdout


def units(text):
    return [(c, int(s)) for c, s in re.findall(r"--cell (\S+) .*--seed (\d+)", text)]


def rerun_lines():
    return {u: l for l in (CAMP / "grid-v1-rerun-A.txt").read_text().splitlines() + (CAMP / "grid-v1-rerun-B.txt").read_text().splitlines()
            if l.strip() for u in units(l)}


def done(m, run):
    out = sh(m, f"ls outputs/train/{run}/*/seed*/report.json 2>/dev/null")
    return {(p.split("/")[3], int(p.split("/")[4][4:])) for p in out.split()}


def sweep_running(m, job_list):
    return f"scripts/sweep.py {job_list}" in sh(m, "ps -eo args")


def grid_hours_left(m):
    """Remaining hours of m's grid sweep: unfinished units' smoke hours times the fraction of epochs not yet done."""
    smoke = json.loads((REPO / f"logs/grid_smoke_{m}.json").read_text())
    todo = set(units(sh(m, f"cat logs/campaigns/grid-v1-{m}.txt"))) - done(m, "grid-v1")
    progress = {}  # last epoch reached per unit, from the job logs
    for line in sh(m, f"for f in logs/grid-v1-{m}/*.log; do grep -a -m1 -o 'cell=[^,]*, unit=seed[0-9]*' $f | tr '\\n' ' '; "
                      "grep -a '^Train epoch' $f | tail -1 | cut -d, -f1; echo; done").splitlines():
        r = re.match(r"cell=(\S+), unit=seed(\d+) Train epoch (\d+)", line)
        if r:
            progress[(r[1], int(r[2]))] = max(int(r[3]), progress.get((r[1], int(r[2])), 0))
    return sum(smoke[c]["run_h"] * (1 - progress.get((c, s), 0) / smoke[c]["epochs"]) for c, s in todo) / SPEED[m]


def split(reruns, busy_hours, smoke):
    """Assign units (longest first) to the machine that would end each one earlier. busy_hours: m -> hours until free."""
    end, plan = dict(busy_hours), {m: [] for m in busy_hours}
    for u in sorted(reruns, key=lambda u: -max(smoke[m][u[0]]["run_h"] for m in busy_hours)):
        m = min(end, key=lambda m: end[m] + smoke[m][u[0]]["run_h"] / SPEED[m])
        end[m] += smoke[m][u[0]]["run_h"] / SPEED[m]
        plan[m].append(u)
    return plan, end


def free(m):
    return not sweep_running(m, f"logs/campaigns/grid-v1-{m}.txt") and not set(units(sh(m, f"cat logs/campaigns/grid-v1-{m}.txt"))) - done(m, "grid-v1")


def start_reruns(m, plan):
    lst = f"logs/campaigns/grid-v1-rerun-{m}.assigned.txt"
    if not plan[m] or set(map(tuple, plan[m])) <= done(m, "grid-v1-rerun") or sweep_running(m, lst):
        return
    print(time.strftime("%d/%m %H:%M"), f"{m}: starting rerun sweep ({len(plan[m])} units)", flush=True)
    sh(m, f"RUN_ID=grid-v1-rerun SWEEP_TAG=-{m} setsid nohup {PY} scripts/sweep.py {lst} --workers 4 --ram-gb 6 "
          f">> logs/grid-v1-rerun-{m}.log 2>&1 < /dev/null &", check=True)


def swap_cmd(every, mine):
    """Shell command: move every resumed unit aside (idempotent), then copy this machine's reruns into grid-v1."""
    cmds = [f"for k in outputs checkpoints; do g=$k/train/grid-v1/{c}/seed{s}; r=$k/train/grid-v1-resumed/{c}/seed{s}; "
            f"if [ -e $g ] && [ ! -e $r ]; then mkdir -p $(dirname $r) && mv $g $r; fi; done" for c, s in every]
    cmds += [f"for k in outputs checkpoints; do mkdir -p $k/train/grid-v1/{c}; [ -e $k/train/grid-v1/{c}/seed{s} ] || "
             f"cp -a $k/train/grid-v1-rerun/{c}/seed{s} $k/train/grid-v1/{c}/seed{s}; done" for c, s in mine]
    return " && ".join(cmds)


def finish(plan):
    every = [tuple(u) for m in plan for u in plan[m]]
    for m in "AB":  # resumed copies aside (A also holds copies of B's units from earlier syncs), then the reruns in
        sh(m, swap_cmd(every, plan[m]), check=True)
        other = {tuple(u) for u in plan["B" if m == "A" else "A"]}
        mine = [u for u in units(sh(m, f"cat logs/campaigns/grid-v1-{m}.txt")) if u not in other and u not in map(tuple, plan[m])]
        lines = "\n".join(f"--cell {c} --seed {s}" for c, s in mine + [tuple(u) for u in plan[m]])
        sh(m, f"cat > logs/campaigns/grid-v1-final-{m}.txt", check=True, input=lines + "\n")
        out = sh(m, f"PYTHONPATH=. {PY} scripts/grid.py audit --list logs/campaigns/grid-v1-final-{m}.txt --hash 2>&1 | grep -v -i 'warn\\|autocast'")
        print(out, flush=True)
        if " 0 problems" not in out:
            sys.exit(f"{m}: audit found problems; stopped before the tables")
    for c, s in plan["A"]:  # B keeps a finished copy of its own units rerun on A, so its grid list stays complete
        sh("B", f"mkdir -p outputs/train/grid-v1/{c}", check=True)
        subprocess.run(["rsync", "-a", f"outputs/train/grid-v1/{c}/seed{s}", f"{B}:Workspace/Personal/KHKT-CVA-2026/auscult-trust/outputs/train/grid-v1/{c}/"],
                       cwd=REPO, check=True)
    subprocess.run(["rsync", "-a", f"{B}:Workspace/Personal/KHKT-CVA-2026/auscult-trust/outputs/train/grid-v1/", "outputs/train/grid-v1/"],
                   cwd=REPO, check=True)
    out = sh("A", f"PYTHONPATH=. {PY} scripts/grid.py audit 2>&1 | grep -v -i 'warn\\|autocast'")
    print(out, flush=True)
    if "350 finished units checked, 0 not finished" not in out or " 0 problems" not in out:
        sys.exit("A: full audit after the sync is not clean; stopped before the tables")
    for e in ("last", "best_test"):
        sh("A", f"PYTHONPATH=. {PY} scripts/grid.py table --epoch {e} > /dev/null", check=True)
    print(time.strftime("%d/%m %H:%M"), "ALL DONE: outputs/train/grid-v1/table_last.md, table_best_test.md", flush=True)


def main():
    reruns = list(rerun_lines())
    smoke = {m: json.loads((REPO / f"logs/grid_smoke_{m}.json").read_text()) for m in "AB"}
    while not PLAN.exists():
        f = {m: free(m) for m in "AB"}
        if any(f.values()):
            busy = {m: 0.0 if f[m] else grid_hours_left(m) for m in "AB"}
            plan, end = split(reruns, busy, smoke)
            PLAN.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "busy_hours": busy, "expected_end_hours": end,
                                        **{m: plan[m] for m in "AB"}}, indent=1))
            lines = rerun_lines()
            for m in "AB":
                text = "".join(lines[tuple(u)] + "\n" for u in plan[m])
                (CAMP / f"grid-v1-rerun-{m}.assigned.txt").write_text(text)
                if m == "B":
                    sh("B", "cat > logs/campaigns/grid-v1-rerun-B.assigned.txt", check=True, input=text)
            print(time.strftime("%d/%m %H:%M"), "plan:", PLAN.read_text(), flush=True)
        else:
            time.sleep(300)
    plan = json.loads(PLAN.read_text())
    plan = {m: [tuple(u) for u in plan[m]] for m in "AB"}
    while True:
        for m in "AB":
            if free(m):
                start_reruns(m, plan)  # also restarts a rerun sweep that died (power cut) with units left
        if all(set(plan[m]) <= done(m, "grid-v1-rerun") for m in "AB"):
            break
        time.sleep(300)
    finish(plan)


if __name__ == "__main__":
    main()
