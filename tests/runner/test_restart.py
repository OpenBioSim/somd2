import tempfile
from somd2.runner import Runner
from somd2.config import Config
from somd2.io import *
from pathlib import Path
import sire as sr


def test_restart():
    """Validate that a simulation can be run from a checkpoint file,
    with all trajcetories preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
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
            Path(tmpdir) / "energy_traj_0.parquet"
        )

        num_entries = len(energy_traj_1.index)

        del runner

        config_new = {
            "runtime": "24fs",
            "restart": True,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
            "supress_overwrite_warning": True,
        }

        runner2 = Runner(mols, Config(**config_new))

        # Run the simulation.
        runner2.run()

        # Load the energy trajectory.
        energy_traj_2, meta_2 = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.parquet"
        )

        # Check that first half of energy trajectory is the same
        assert energy_traj_1.equals(energy_traj_2.iloc[:num_entries])
        # Check that second energy trajectory is twice as long as the first
        assert len(energy_traj_2.index) == 2 * num_entries
        # Check that a second trajectory was written and that the first still exists
        assert Path.exists(Path(tmpdir) / "traj_0.dcd")
        assert Path.exists(Path(tmpdir) / "traj_0_1.dcd")


if __name__ == "__main__":
    test_restart()
    print("test_restart.py: All tests passed.")
