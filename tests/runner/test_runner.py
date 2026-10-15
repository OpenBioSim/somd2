from pathlib import Path

import numpy as np
import pytest
import tempfile

from somd2.runner import Runner
from somd2.config import Config
from somd2.io import parquet_to_dataframe


def test_runtime_remainder(ethane_methanol):
    """
    Validate that a runtime that isn't a multiple of the checkpoint frequency
    finishes with a block for the remaining time.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "20fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "12fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        runner = Runner(ethane_methanol, Config(**config))
        runner.run()

        for lam in ["0.00000", "1.00000"]:
            energy_traj, _ = parquet_to_dataframe(
                Path(tmpdir) / f"energy_traj_{lam}.parquet"
            )
            times = energy_traj.index.get_level_values(0)
            assert list(np.round(times, 6)) == [0.004, 0.008, 0.012, 0.016, 0.020]

            checkpoint_state = np.load(Path(tmpdir) / f"checkpoint_{lam}.npz")
            assert checkpoint_state["time_ps"].item() == pytest.approx(0.020)


def test_runtime_remainder_with_terminal_flips(phenethyl_mols):
    """
    Validate that blocks which aren't a whole number of terminal flip intervals
    don't overrun, and that the remainder is run with terminal flips.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "20fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "12fs",
            "frame_frequency": "4fs",
            "terminal_flip_frequency": "8fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        runner = Runner(phenethyl_mols, Config(**config))
        runner.run()

        energy_traj, _ = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )
        times = energy_traj.index.get_level_values(0)
        assert list(np.round(times, 6)) == [0.004, 0.008, 0.012, 0.016, 0.020]

        # Flips are attempted every 8 fs, at 0, 8 and 16 fs, for each window.
        log = (Path(tmpdir) / "log.txt").read_text()
        assert log.count("Performing terminal flip move at") == 6


def test_terminal_flip_frequency_across_blocks(phenethyl_mols):
    """
    Validate that terminal flips keep their frequency when the checkpoint
    frequency isn't a multiple of it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "48fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "12fs",
            "frame_frequency": "4fs",
            "terminal_flip_frequency": "8fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 1,
        }

        runner = Runner(phenethyl_mols, Config(**config))
        runner.run()

        # Every 8 fs, at 0, 8, 16, 24, 32 and 40 fs.
        log = (Path(tmpdir) / "log.txt").read_text()
        assert log.count("Performing terminal flip move at") == 6


@pytest.mark.parametrize(
    "option, value",
    [
        ("runtime", "18fs"),
        ("checkpoint_frequency", "10fs"),
        ("energy_frequency", "6fs"),
        ("terminal_flip_frequency", "6fs"),
    ],
)
def test_times_must_be_whole_steps(ethane_methanol, option, value):
    """
    Validate that times which aren't a whole number of integration steps are
    rejected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "24fs",
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "12fs",
            "frame_frequency": "12fs",
            "timestep": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
            option: value,
        }

        with pytest.raises(ValueError, match=f"'{option}' must be a multiple"):
            Runner(ethane_methanol, Config(**config))


def test_energy_components_after_restart(ethane_methanol):
    """
    Validate that energy components are time stamped correctly when a restart
    has less than a checkpoint interval left to run.
    """
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "12fs",
            "frame_frequency": "4fs",
            "save_energy_components": True,
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        runner = Runner(ethane_methanol, Config(**config))
        runner.run()
        del runner

        config["runtime"] = "20fs"
        config["restart"] = True
        runner = Runner(ethane_methanol, Config(**config))
        runner.run()

        components = pd.read_parquet(Path(tmpdir) / "energy_components_0.00000.parquet")
        # The times are in ns.
        times = list(np.round(components["time"] * 1e6, 6))
        assert times == [4, 8, 12, 16, 20]
