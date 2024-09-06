from pathlib import Path

import tempfile
import pytest

import sire as sr

from somd2.runner import Runner
from somd2.config import Config
from somd2.io import *


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_hmr"])
def test_remainder(mols, request):
    """
    Validate that a simulations whose runtime % checkpoint frequency != 0 runs for the correct amount of time
    and records the correct data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mols = request.getfixturevalue(mols)

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "8fs",
            "checkpoint_frequency": "8fs",
            "frame_frequency": "8fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        # Instantiate a runner using the config defined above.
        runner = Runner(mols, Config(**config))

        # Run the simulation.
        runner.run()

        # Load the energy trajectory.
        energy_traj_1, meta_1 = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )

        num_entries = len(energy_traj_1.index)

        assert num_entries == 2

        # Load the trajectory.
        traj_1 = sr.load(
            [str(Path(tmpdir) / "system0.prm7"), str(Path(tmpdir) / "traj_0.00000.dcd")]
        )

        assert traj_1.num_frames() == 2
