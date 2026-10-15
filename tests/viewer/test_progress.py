import pickle

import numpy as np
import pytest
import sire as sr

from somd2._viewer._data import Simulation
from somd2.runner._repex import DynamicsCache

LAMBDAS = [0.0, 0.5, 1.0]

# A hot window at either end and a cold one in the middle.
WINDOWS = [
    {"lambda": 0.0, "samples": 2000},
    {"lambda": 0.5, "samples": 800},
    {"lambda": 1.0, "samples": 2000},
]


def _simulation(tmp_path, time):
    """
    Create a simulation whose replica exchange state is at the given time.
    """
    n = len(LAMBDAS)
    cache = object.__new__(DynamicsCache)
    cache.__setstate__(
        {
            "_lambdas": LAMBDAS,
            "_rest2_scale_factors": [1.0] * n,
            "_num_replicas": n,
            "_num_slots": n,
            "_update_constraints": True,
            "_states": np.arange(n),
            "_time": sr.u(time),
            "_clocks": [None] * n,
            "_hot_replicas": [0, 2],
            "_cold_votes": [0] * n,
            "_adaptive_data": None,
            "_openmm_states": [None] * n,
            "_virtual_sites": None,
            "_gcmc_samplers": [None] * n,
            "_gcmc_states": [None] * n,
            "_gcmc_stats": None,
            "_terminal_flip_stats": [[0, 0]] * n,
            "_num_proposed": np.zeros((n, n)),
            "_num_accepted": np.zeros((n, n)),
            "_num_swaps": np.zeros((n, n)),
        }
    )

    with open(tmp_path / "repex_state.pkl", "wb") as f:
        pickle.dump(cache, f)

    return Simulation(tmp_path, "test")


def _config(adaptive):
    return {
        "replica_exchange": True,
        "adaptive_repex": adaptive,
        "runtime": "5 ns",
        "energy_frequency": "1 ps",
        "checkpoint_frequency": "100 ps",
    }


def test_adaptive_progress_uses_simulation_time(tmp_path):
    """
    Validate that the progress of an adaptive replica exchange simulation is
    the simulation time of the hot windows, not the samples of every window.
    """
    simulation = _simulation(tmp_path, "2 ns")
    progress = simulation._progress(_config(True), LAMBDAS, WINDOWS, None)

    assert progress["fraction"] == pytest.approx(0.4)
    assert progress["completed_ns"] == pytest.approx(6.0)

    # Cold windows have fewer samples by design, so there is no expectation.
    assert progress["expected_samples"] is None


def test_adaptive_progress_finishes_without_a_log(tmp_path):
    """
    Validate that a finished adaptive replica exchange simulation is reported
    as finished when there is no log, even though its cold windows have fewer
    samples than the runtime implies.
    """
    simulation = _simulation(tmp_path, "5 ns")
    windows = [dict(w, samples=5000 if w["samples"] == 2000 else 1500) for w in WINDOWS]
    progress = simulation._progress(_config(True), LAMBDAS, windows, None)

    assert progress["fraction"] == pytest.approx(1.0)
    assert progress["status"] == "finished"


def test_standard_progress_uses_samples(tmp_path):
    """
    Validate that the progress of a standard replica exchange simulation is
    still taken from the samples of every window.
    """
    simulation = _simulation(tmp_path, "2 ns")
    progress = simulation._progress(_config(False), LAMBDAS, WINDOWS, None)

    assert progress["fraction"] == pytest.approx(4.8 / 15)
    assert progress["expected_samples"] == 5000
