"""Run a list of training commands in parallel on a shared GPU, starting each only when enough GPU memory is free.

    python scripts/sweep.py jobs.txt --workers 3 --mem-gb 8

`jobs.txt` has one shell command per line (written by `PARALLEL=3 bash scripts/run.sh ...`). A job starts when fewer than
`--workers` jobs run, the GPU has at least `--mem-gb` GB free and the machine has at least `--ram-gb` GB of available RAM
(other users of the machine are respected, nothing is killed). Each job gets `--threads` math threads and `--num-workers`
loader workers so that parallel jobs do not oversubscribe the CPU. A job that dies from lack of memory (CUDA "out of memory"
or the kernel's kill, exit 137) is queued again, up to 3 tries.
Resume: run the same command again with the same RUN_ID. Units with a report.json are skipped here without starting
anything; an interrupted unit continues from its last.pth. One log per job: logs/<RUN_ID><SWEEP_TAG>/<n>.log.
"""
import argparse
import os
import re
import subprocess
import time
from pathlib import Path


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return min(int(x) for x in out.stdout.split()) / 1024 if out.returncode == 0 and out.stdout.strip() else 0.0


def free_ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def finished(cmd):
    """True when the unit of this training command already has its report.json (written last)."""
    cell, seed, fold = (re.search(p, cmd) for p in (r"--cell (\S+)", r"--seed (\d+)", r"--cv_fold (\d+)"))
    if not cell:
        return False
    unit = "cv" + fold.group(1) if fold and "--cv_folds" in cmd else "seed" + (seed.group(1) if seed else "0")
    return Path("outputs/train", os.environ.get("RUN_ID", "sweep"), cell.group(1), unit, "report.json").exists()


def running_elsewhere(cmd):
    """True when a process of another scheduler (or a shell) is already training this unit."""
    cell, seed, fold = (re.search(p, cmd) for p in (r"--cell (\S+)", r"--seed (\d+)", r"--cv_fold (\d+)"))
    if not cell:
        return False
    flag = f"--cv_fold {fold.group(1)}" if fold and "--cv_folds" in cmd else f"--seed {seed.group(1) if seed else 0}"
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("patchmix_cl.main" in l and f"--cell {cell.group(1)} " in l and re.search(re.escape(flag) + r"( |$)", l) for l in ps.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mem-gb", type=float, default=8.0, help="GPU memory one job needs")
    ap.add_argument("--ram-gb", type=float, default=6.0, help="available RAM a new job needs (about 2 with the spectrogram cache)")
    ap.add_argument("--ramp", type=int, default=90, help="seconds a new job needs to reach its memory use; the next one starts after")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--threads", type=int, default=3, help="math threads per job (OMP/MKL); jobs share the CPU")
    ap.add_argument("--num-workers", type=int, default=4, help="data loader workers per job")
    a = ap.parse_args()
    jobs = [(i, line.strip()) for i, line in enumerate(Path(a.jobs).read_text().splitlines()) if line.strip()]
    queue = [(i, cmd, 0) for i, cmd in jobs if not finished(cmd) and not running_elsewhere(cmd)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(queue)} finished or running elsewhere, {len(queue)} to run", flush=True)
    tag = os.environ.get("SWEEP_TAG", "")  # distinguishes a second scheduler of the same RUN_ID
    log_dir = Path("logs") / (os.environ.get("RUN_ID", "sweep") + tag)
    log_dir.mkdir(parents=True, exist_ok=True)
    running, failed, last_start = [], [], 0.0
    while queue or running:
        for job in running[:]:
            i, cmd, tries, proc = job
            if proc.poll() is None:
                continue
            running.remove(job)
            log = (log_dir / f"{i}.log").read_text(errors="ignore")
            if proc.returncode != 0:
                if ("out of memory" in log or proc.returncode in (137, -9)) and tries < 3:
                    queue.append((i, cmd, tries + 1))
                    print(f"job {i}: out of memory (exit {proc.returncode}), queued again", flush=True)
                else:
                    failed.append(i)
                    print(f"job {i}: FAILED, see {log_dir / f'{i}.log'}", flush=True)
        while queue and (finished(queue[0][1]) or running_elsewhere(queue[0][1])):  # taken over by someone else meanwhile
            queue.pop(0)
        if queue and len(running) < a.workers and time.time() - last_start > a.ramp and free_gb() >= a.mem_gb and free_ram_gb() >= a.ram_gb:
            i, cmd, tries = queue.pop(0)
            env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads)}
            proc = subprocess.Popen(cmd + f" --num_workers {a.num_workers}", env=env, shell=True, stdout=open(log_dir / f"{i}.log", "w"), stderr=subprocess.STDOUT)
            running.append((i, cmd, tries, proc))
            last_start = time.time()
            print(f"job {i} started ({len(running)} running, {len(queue)} waiting): {cmd[-90:]}", flush=True)
        time.sleep(a.poll)
    print(f"all jobs finished, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
