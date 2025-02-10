from pathlib import Path

import numpy as np
import pytest
import tempfile

from somd2.runner import RepexRunner
from somd2.runner._base import RunnerBase
from somd2.config import Config

from conftest import has_cuda


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_output(ethane_methanol):
    """
    Validate that repex specific simulation output is generated.
    """
    with tempfile.TemporaryDirectory() as tmpdir:

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": 2,
            "replica_exchange": True,
        }

        # Instantiate a runner using the config defined above.
        runner = RepexRunner(ethane_methanol, Config(**config))

        # Run the simulation.
        runner.run()

        # Make sure that the replica exchange transition matrix is written.
        assert (Path(tmpdir) / "repex_matrix.txt").exists()


def test_repex_mixing():
    """
    Validate that replicas are mixed correctly.
    """

    # Create a uniform energy matrix. (All state have the same energy.)
    energy_matrix = np.ones((10, 10), dtype=np.float32)

    # Create matrices for the proposed and accepted swaps.
    proposed = np.zeros((10, 10), dtype=np.int32)
    accepted = np.zeros((10, 10), dtype=np.int32)

    # Perform the mixing.
    states = RepexRunner._mix_replicas(10, energy_matrix, proposed, accepted)

    # Make sure that exchanges are always accepted.
    assert (proposed == accepted).all()

    # Create a matrix where states are uncorrelated. All off diagonal elements
    # have a large energy (1000) and the diagonals are the same.
    energy_matrix = 10000 * np.ones((10, 10), dtype=np.float32)
    np.fill_diagonal(energy_matrix, 1)

    # Create matrices for the proposed and accepted swaps.
    proposed = np.zeros((10, 10), dtype=np.int32)
    accepted = np.zeros((10, 10), dtype=np.int32)

    # Perform the mixing.
    states = RepexRunner._mix_replicas(10, energy_matrix, proposed, accepted)

    # Get the off-diagonal elements of the accepted matrix.
    off_diagonal = accepted - np.diag(np.diag(accepted))

    # Make sure that all off-diagonal elements are 0.
    assert (off_diagonal == 0).all()


@pytest.mark.parametrize(
    "rest2_scale, is_valid",
    [
        (10, True),
        ([0.1, 0.2], False),
        ([1.0, 2.0, 1.0], True),
        ([2.0, 3.0, 1.0], False),
        ([1.0, 3.0, 2.0], False),
    ],
)
def test_rest2_scale(ethane_methanol, rest2_scale, is_valid):
    """Validate the REST2 scale factor handling."""

    with tempfile.TemporaryDirectory() as tmpdir:

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 3,
            "replica_exchange": True,
            "rest2_scale": rest2_scale,
        }

        # Instantiate a runner using the config defined above.
        if is_valid:
            runner = RepexRunner(ethane_methanol, Config(**config))
        else:
            with pytest.raises(ValueError):
                runner = RepexRunner(ethane_methanol, Config(**config))


@pytest.mark.parametrize(
    "rest2_selection, is_valid",
    [
        ("resname LIG", True),
        ("resname CAT", False),
        ("residx 1", False),
        ("residx 1000", False),
        ("residx 0", False),
        ("molidx 0", True),
    ],
)
def test_rest2_selection(ethane_methanol, rest2_selection, is_valid):
    """Validate the REST2 selection handling."""

    with tempfile.TemporaryDirectory() as tmpdir:

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 3,
            "replica_exchange": True,
            "rest2_selection": rest2_selection,
        }

        # Instantiate a runner using the config defined above.
        if is_valid:
            runner = RepexRunner(ethane_methanol, Config(**config))
        else:
            with pytest.raises(ValueError):
                runner = RepexRunner(ethane_methanol, Config(**config))
