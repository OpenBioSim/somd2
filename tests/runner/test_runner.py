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
