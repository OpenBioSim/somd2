import os
import tempfile
from pathlib import Path

import pytest
import yaml

from somd2.config import Config
from somd2.runner import Runner


def _pme_parameters(runner):
    from openmm import NonbondedForce, unit

    d = runner._system.dynamics(**runner._dynamics_kwargs)

    for force in d._d._omm_mols.getSystem().getForces():
        if isinstance(force, NonbondedForce):
            alpha, *grid = force.getPMEParameters()
            return alpha.value_in_unit(unit.nanometer**-1), grid


def _short_run(tmpdir, **options):
    config = {
        "runtime": "12fs",
        "output_directory": tmpdir,
        "energy_frequency": "4fs",
        "checkpoint_frequency": "4fs",
        "frame_frequency": "4fs",
        "platform": "CPU",
        "max_threads": 1,
        "num_lambda": 2,
    }
    config.update(options)
    return Config(**config)


def test_pme_config_options():
    """Validate the parsing of the PME options."""
    assert Config().tune_pme

    assert Config(pme_grid=64).pme_grid == [64]
    assert Config(pme_grid=["64", "64", "72"]).pme_grid == [64, 64, 72]
    assert Config(pme_alpha="3.47").pme_alpha == pytest.approx(3.47)
    assert Config(pme_spacing="0.12 nm").pme_spacing.value() == pytest.approx(1.2)

    for options in [
        {"pme_grid": [64, 64]},
        {"pme_grid": 4},
        {"pme_alpha": -1.0},
        {"pme_spacing": "1 ps"},
    ]:
        with pytest.raises(ValueError):
            Config(**options)


def test_pme_options_passed(ethane_methanol):
    """Validate that explicit PME options reach the OpenMM context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cpu", output_directory=tmpdir, pme_alpha=3.4, pme_grid=32
        )

        alpha, grid = _pme_parameters(Runner(ethane_methanol, config))

        assert alpha == pytest.approx(3.4)
        assert grid == [32, 32, 32]


def test_pme_restart(ethane_methanol):
    """
    Validate that a restart reuses saved PME parameters, and that a new run
    removes stale ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Runner(ethane_methanol, _short_run(tmpdir))
        runner.run()
        del runner

        params = {"pme_alpha": 3.4, "pme_grid": [32, 32, 32]}
        pme_file = Path(tmpdir) / "pme_parameters.yaml"

        with open(pme_file, "w") as f:
            yaml.safe_dump(params, f)

        runner = Runner(
            ethane_methanol,
            _short_run(tmpdir, runtime="24fs", restart=True, overwrite=True),
        )

        alpha, grid = _pme_parameters(runner)

        assert alpha == pytest.approx(3.4)
        assert grid == [32, 32, 32]

        del runner

        Runner(ethane_methanol, _short_run(tmpdir, overwrite=True))

        assert not pme_file.exists()


def _has_cuda():
    try:
        import openmm

        openmm.Platform.getPlatformByName("CUDA")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA platform is not available")
def test_pme_tuning(ethane_methanol, monkeypatch):
    """Validate that a new CUDA run tunes and saves the PME parameters."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Runner(
            ethane_methanol, Config(platform="cuda", output_directory=tmpdir)
        )

        pme_file = Path(tmpdir) / "pme_parameters.yaml"

        if pme_file.exists():
            with open(pme_file) as f:
                params = yaml.safe_load(f)

            alpha, grid = _pme_parameters(runner)

            assert alpha == pytest.approx(params["pme_alpha"])
            assert grid == params["pme_grid"]
