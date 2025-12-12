from pathlib import Path

import tempfile
import pytest

import sire as sr

from somd2.runner import Runner
from somd2.config import Config
from somd2.io import *


def test_lambda_values(ethane_methanol):
    """
    Validate that a simulation can be run with a custom list of lambda values.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        mols = ethane_methanol.clone()

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "lambda_values": [0.0, 0.5, 1.0],
        }

        # Instantiate a runner using the config defined above.
        runner = Runner(mols, Config(**config))

        # Run the simulation.
        runner.run()

        # Load the energy trajectory.
        energy_traj, meta = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )

        # Make sure the energy trajectory has the expected columns.
        cols = energy_traj.columns
        found = 0
        for col in cols:
            if col in config["lambda_values"]:
                found += 1
        assert found == len(config["lambda_values"])

        # Make sure the second dimension of the energy trajectory is the correct
        # size. This is one for the current lambda value, one for its gradient,
        # and two for the additional values in the lambda_values list.
        assert energy_traj.shape[1] == 4


def test_lambda_energy(ethane_methanol):
    """
    Validate that a simulation can sample energies at a different set of
    lambda values.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        mols = ethane_methanol.clone()

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "lambda_values": [0.0, 1.0],
            "lambda_energy": [0.5],
        }

        # Instantiate a runner using the config defined above.
        runner = Runner(mols, Config(**config))

        # Run the simulation.
        runner.run()

        # Load the energy trajectory.
        energy_traj, meta = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )

        # Make sure the energy trajectory has the expected columns.
        cols = energy_traj.columns
        found = 0
        for col in cols:
            if col in config["lambda_values"] or col in config["lambda_energy"]:
                found += 1
        assert found == len(config["lambda_values"]) + len(config["lambda_energy"])

        # Make sure the second dimension of the energy trajectory is the correct.
        # This is the sampled lambda values, i.e. unique entries from lambda_values
        # and lambda_energy, plus the gradient for TI.
        assert energy_traj.shape[1] == 4
