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


def test_equilibration_time_per_window(ethane_methanol):
    """Validate that only windows with a non-zero time are equilibrated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "4fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "equilibration_time": ["0fs", "4fs"],
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        runner = Runner(ethane_methanol, Config(**config))
        runner.run()

        log = (Path(tmpdir) / runner._config.log_file).read_text()
        assert "Equilibrating at λ = 0.00000" not in log
        assert "Equilibrating at λ = 1.00000" in log


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


def test_sire_thread_limit_with_repeated_runners(monkeypatch):
    """
    Validate that the Sire thread limit is worked out from the threads that
    were available before the first runner, keeping a limit set by the user,
    rather than from the limit left by an earlier runner in the same process.
    """
    import sire as sr

    from somd2.runner._base import RunnerBase

    sr.set_default_num_threads()
    if sr.get_max_num_threads() < 4:
        pytest.skip("Needs at least four CPU threads.")

    # A limit set by the user before the first runner is created.
    user_threads = sr.get_max_num_threads() // 2

    monkeypatch.setattr(RunnerBase, "_total_sire_threads", None)
    monkeypatch.setattr(RunnerBase, "_get_gpu_devices", staticmethod(lambda *args: [0]))

    gpu_runner = object.__new__(RunnerBase)
    gpu_runner._is_gpu = True
    gpu_runner._config = Config(max_gpus=1, oversubscription_factor=2)

    cpu_runner = object.__new__(RunnerBase)
    cpu_runner._is_gpu = False

    try:
        sr.set_max_num_threads(user_threads)

        for _ in range(2):
            gpu_runner._limit_sire_threads()
            assert gpu_runner._sire_threads == user_threads // 2
            assert gpu_runner._checkpoint_sire_threads == user_threads

        # A CPU runner isn't left with the limit of the GPU runner before it.
        cpu_runner._limit_sire_threads()
        assert sr.get_max_num_threads() == user_threads
    finally:
        sr.set_default_num_threads()
