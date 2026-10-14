import pytest

from src.training.patchmix_cl.config import parse_args

C = "experiments/train/"


def test_lung_srad_settings():
    a = parse_args(["--config", C + "base.yaml," + C + "encoder_dass.yaml," + C + "lung_srad.yaml"])
    assert (a.cl_mode, a.gaussian_blur, a.target_type, a.batch_size, a.temperature) == ("dual", True, "project_flow", 16, 0.2)
    assert a.cell == "encoder_dass+lung_srad"


def test_lung_srad_needs_dass():
    with pytest.raises(AssertionError):
        parse_args(["--config", C + "base.yaml," + C + "lung_srad.yaml"])
