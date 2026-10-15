"""Run a list of training commands in parallel on a shared GPU, starting each only when enough GPU memory is free.

    python scripts/sweep.py jobs.txt --workers 3 --mem-gb 8

`jobs.txt` has one shell command per line (written by `PARALLEL=3 bash scripts/run.sh ...`). A job starts when fewer than
`--workers` jobs run, the GPU has at least `--mem-gb` GB free (or the job's own `MEM_GB=<n>` prefix), the reservations of all
units of the list that run now plus this one fit in `--gpu-gb` (default: GPU total minus 1 GB; a job's free memory at start
says nothing about how far the running jobs will still grow) and the machine has at least `--ram-gb` GB of available RAM
(other users of the machine are respected, nothing is killed). Each job gets `--threads` math threads and `--num-workers`
loader workers so that parallel jobs do not oversubscribe the CPU. A job that dies from lack of memory (CUDA "out of memory"
or the kernel's kill, exit 137) is queued again, up to `--retries` times.
Resume: run the same command again with the same RUN_ID. Units with a report.json are skipped here without starting
anything; an interrupted unit continues from its last.pth. One log per job: logs/<RUN_ID><SWEEP_TAG>/<n>.log.
"""
import argparse
import os
import re
import subprocess
import time
from pathlib import Path


def free_gb(field="free"):
    out = subprocess.run(["nvidia-smi", f"--query-gpu=memory.{field}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
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
    run = re.search(r"(?:^|\s)RUN_ID=(\S+)", cmd)  # a job line may set its own RUN_ID (mixed campaigns in one list)
    run = run.group(1) if run else os.environ.get("RUN_ID", "sweep")
    return Path("outputs/train", run, cell.group(1), unit, "report.json").exists()


def mem_gb(cmd, default):
    m = re.search(r"(?:^|\s)MEM_GB=(\S+)", cmd)
    return float(m.group(1)) if m else default


def ps_args():
    return subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


def running_elsewhere(cmd, ps=None):
    """True when a process (this scheduler's, another one's or a shell's) is training this unit."""
    cell, seed, fold = (re.search(p, cmd) for p in (r"--cell (\S+)", r"--seed (\d+)", r"--cv_fold (\d+)"))
    if not cell:
        return False
    flag = f"--cv_fold {fold.group(1)}" if fold and "--cv_folds" in cmd else f"--seed {seed.group(1) if seed else 0}"
    ps = ps_args() if ps is None else ps
    return any("patchmix_cl.main" in l and f"--cell {cell.group(1)} " in l and re.search(re.escape(flag) + r"( |$)", l) for l in ps.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mem-gb", type=float, default=8.0, help="GPU memory one job needs")
    ap.add_argument("--ram-gb", type=float, default=6.0, help="available RAM a new job needs (about 2 with the spectrogram cache)")
    ap.add_argument("--ramp", type=int, default=90, help="seconds a new job needs to reach its memory use; the next one starts after")
    ap.add_argument("--retries", type=int, default=30, help="tries after a memory failure; a full GPU can crash several starts in a row")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--threads", type=int, default=3, help="math threads per job (OMP/MKL); jobs share the CPU")
    ap.add_argument("--num-workers", type=int, default=4, help="data loader workers per job")
    ap.add_argument("--gpu-gb", type=float, default=None, help="GPU memory the reservations of running units may sum to")
    a = ap.parse_args()
    budget = a.gpu_gb if a.gpu_gb is not None else free_gb("total") - 1
    jobs = [(i, line.strip()) for i, line in enumerate(Path(a.jobs).read_text().splitlines()) if line.strip()]
    # units already running (a previous scheduler's orphans) stay queued: they count against the budget while they run and
    # are resumed from last.pth if they die unfinished
    queue = [(i, cmd, 0) for i, cmd in jobs if not finished(cmd)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(queue)} finished, {len(queue)} to run, GPU budget {budget:.1f} GB", flush=True)
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
                if ("out of memory" in log or proc.returncode in (137, -9)) and tries < a.retries:
                    queue.append((i, cmd, tries + 1))
                    print(f"job {i}: out of memory (exit {proc.returncode}), queued again", flush=True)
                else:
                    failed.append(i)
                    print(f"job {i}: FAILED, see {log_dir / f'{i}.log'}", flush=True)
        queue = [q for q in queue if not finished(q[1])]
        ps = ps_args()
        reserved = sum(mem_gb(cmd, a.mem_gb) for _, cmd in jobs if running_elsewhere(cmd, ps))
        ready = [q for q in queue if not running_elsewhere(q[1], ps)]
        need = mem_gb(ready[0][1], a.mem_gb) if ready else 0
        if (ready and len(running) < a.workers and time.time() - last_start > a.ramp and reserved + need <= budget
                and free_gb() >= need and free_ram_gb() >= a.ram_gb):
            i, cmd, tries = ready[0]
            queue.remove(ready[0])
            env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            proc = subprocess.Popen(cmd + f" --num_workers {a.num_workers}", env=env, shell=True, stdout=open(log_dir / f"{i}.log", "w"), stderr=subprocess.STDOUT)
            running.append((i, cmd, tries, proc))
            last_start = time.time()
            print(f"job {i} started ({len(running)} running, {len(queue)} waiting, {reserved + need:.0f} GB reserved): {cmd[-90:]}", flush=True)
        time.sleep(a.poll)
    print(f"all jobs finished, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
