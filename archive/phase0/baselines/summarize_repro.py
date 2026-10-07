"""Per-seed ICBHI Score from logs/seed<N>.log: best epoch on test (paper protocol) and last epoch.

Usage: .venv/bin/python summarize_repro.py
"""
import glob
import re

for f in sorted(glob.glob("logs/seed[0-9].log")):
    sc = [float(x) for x in re.findall(r"\* S_p: [\d.]+, S_e: [\d.]+, Score: ([\d.]+)", open(f).read())]
    if sc:
        b = max(range(len(sc)), key=sc.__getitem__)
        print(f"{f}: epochs {len(sc)}, best-on-test {sc[b]:.2f} (epoch {b + 1}), last {sc[-1]:.2f}")
