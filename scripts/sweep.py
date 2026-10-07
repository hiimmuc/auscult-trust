"""Run a list of training commands in parallel on a shared GPU, starting each only when enough GPU memory is free.

    python scripts/sweep.py jobs.txt --workers 3 --mem-gb 8

`jobs.txt` has one shell command per line (written by `PARALLEL=3 bash scripts/run.sh ...`). A job starts when fewer than
`--workers` jobs run and the GPU has at least `--mem-gb` GB free (other users of the GPU are respected, nothing is
killed). A job that dies with "out of memory" is queued again (up to 3 tries). Finished runs are skipped by the training
code itself, so the same list can be submitted again to resume. One log per job: logs/<RUN_ID>/<n>.log.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

RAMP_SECONDS = 90  # a new job needs this long to reach its memory use; do not start the next one before


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return min(int(x) for x in out.stdout.split()) / 1024 if out.returncode == 0 and out.stdout.strip() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mem-gb", type=float, default=8.0, help="GPU memory one job needs")
    ap.add_argument("--poll", type=int, default=20)
    a = ap.parse_args()
    queue = [(i, line.strip(), 0) for i, line in enumerate(Path(a.jobs).read_text().splitlines()) if line.strip()]
    log_dir = Path("logs") / os.environ.get("RUN_ID", "sweep")
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
                if "out of memory" in log and tries < 3:
                    queue.append((i, cmd, tries + 1))
                    print(f"job {i}: out of memory, queued again", flush=True)
                else:
                    failed.append(i)
                    print(f"job {i}: FAILED, see {log_dir / f'{i}.log'}", flush=True)
        if queue and len(running) < a.workers and time.time() - last_start > RAMP_SECONDS and free_gb() >= a.mem_gb:
            i, cmd, tries = queue.pop(0)
            proc = subprocess.Popen(cmd, shell=True, stdout=open(log_dir / f"{i}.log", "w"), stderr=subprocess.STDOUT)
            running.append((i, cmd, tries, proc))
            last_start = time.time()
            print(f"job {i} started ({len(running)} running, {len(queue)} waiting): {cmd[-90:]}", flush=True)
        time.sleep(a.poll)
    print(f"all jobs finished, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
