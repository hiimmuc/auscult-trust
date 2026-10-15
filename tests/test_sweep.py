import importlib.util

spec = importlib.util.spec_from_file_location("sweep", "scripts/sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_reservations_sum_running_units_only():
    jobs = ["RUN_ID=g MEM_GB=10 python -m src.training.patchmix_cl.main --cell ast+ft+ce --seed 1",
            "RUN_ID=g MEM_GB=13 python -m src.training.patchmix_cl.main --cell ast+ft+ce --seed 12",
            "RUN_ID=g python -m src.training.patchmix_cl.main --cell ast+ft+pafa --seed 0"]
    ps = ("python -m src.training.patchmix_cl.main --cell ast+ft+ce --seed 1 --num_workers 4\n"
          "python -m src.training.patchmix_cl.main --cell ast+ft+pafa --seed 0 --num_workers 4\n")
    assert sum(sweep.mem_gb(c, 8.0) for c in jobs if sweep.running_elsewhere(c, ps)) == 18.0  # seed 12 is not seed 1
