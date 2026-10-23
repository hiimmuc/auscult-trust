"""E1 (BEATs PAFA, SG-SCL, Patch-Mix CL) with trained SC dynamic: the tables of scripts/ensemble.py and
scripts/ensemble_sesp.py, restricted to E1 because only the BEATs cells have SC dynamic units.

    PYTHONPATH=. .venv-train/bin/python scripts/ensemble_e1_dynamic.py > ensemble_e1_dynamic.md
"""
import sys

sys.path.insert(0, "scripts")
import ensemble as E
import ensemble_sesp as S

E.ENS = {"E1 BEATs": E.E1}
E.E2 = E.E3 = []
E.SC_ARMS = {"trained SC dynamic": "+sc"}
S.ENS, S.SC_ARMS = E.ENS, E.SC_ARMS
S.GROUPS = [("ref", [E.REF])] + list(E.ENS.items())
E.main()
S.main()
