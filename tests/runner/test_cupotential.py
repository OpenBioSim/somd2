import tempfile
from pathlib import Path

import pytest

from somd2.config import Config
from somd2.runner import Runner

from tests.conftest import has_cuda

try:
    import nv_openmm_plugin  # noqa: F401

    has_cupotential = True
except Exception:
    has_cupotential = False


def test_cupotential_config_options():
    """Validate the parsing of the cuPotential option."""
    assert not Config().cupotential

    with pytest.raises(ValueError):
        Config(cupotential="yes")


def test_cupotential_requires_cuda(ethane_methanol):
    """Validate that cuPotential is rejected on platforms other than CUDA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(platform="cpu", output_directory=tmpdir, cupotential=True)

        with pytest.raises(ValueError, match="CUDA"):
            Runner(ethane_methanol, config)


def test_cupotential_rejects_double_precision(ethane_methanol):
    """Validate that cuPotential is rejected with double precision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cuda",
            precision="double",
            output_directory=tmpdir,
            cupotential=True,
        )

        with pytest.raises(ValueError, match="double precision"):
            Runner(ethane_methanol, config)


@pytest.mark.skipif(
    not (has_cuda and has_cupotential), reason="CUDA or cuPotential not available."
)
def test_cupotential_options_passed(ethane_methanol):
    """
    Validate that cuPotential replaces the NonbondedForce and integrator, with a
    fixed PME tolerance and no PME tuning.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cuda",
            output_directory=tmpdir,
            cupotential=True,
        )

        runner = Runner(ethane_methanol, config)

        extra_args = runner._config._extra_args
        assert extra_args["use_cupotential"]
        assert extra_args["tolerance"] == pytest.approx(5e-4)
        assert "pme_grid" not in extra_args
        assert not (Path(tmpdir) / "pme_parameters.yaml").exists()

        d = runner._system.dynamics(**runner._dynamics_kwargs)
        names = [f.getName() for f in d._d._omm_mols.getSystem().getForces()]

        assert "CuPotentialNonbondedForce" in names
        assert "NonbondedForce" not in names
        assert "CMMotionRemover" not in names
        assert type(d._d._omm_mols.getIntegrator()).__module__ == "nvplugin"
