from pathlib import Path

import numpy as np
import pytest
import tempfile

from somd2.runner import RepexRunner
from somd2.runner._adaptive import WindowRanker, cold_rate, statistical_inefficiency
from somd2.runner._base import RunnerBase
from somd2.runner._repex import _mix_subset
from somd2.config import Config

from tests.conftest import has_cuda


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
    states = _mix_subset(np.arange(10), energy_matrix, proposed, accepted)

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
    states = _mix_subset(np.arange(10), energy_matrix, proposed, accepted)

    # Get the off-diagonal elements of the accepted matrix.
    off_diagonal = accepted - np.diag(np.diag(accepted))

    # Make sure that all off-diagonal elements are 0.
    assert (off_diagonal == 0).all()


def test_repex_mixing_moves_configurations():
    """
    Validate that each configuration is moved to the state that the mixing
    accepted, for a mix that can only be a 3-cycle.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    num_replicas = 3

    # The configuration from replica i is only favourable in state i + 1.
    energy_matrix = 10000 * np.ones((num_replicas, num_replicas))
    for i in range(num_replicas):
        energy_matrix[i, (i + 1) % num_replicas] = -10000

    proposed = np.zeros((num_replicas, num_replicas), dtype=np.int32)
    accepted = np.zeros((num_replicas, num_replicas), dtype=np.int32)

    np.random.seed(42)
    states = _mix_subset(np.arange(num_replicas), energy_matrix, proposed, accepted)

    cache = object.__new__(DynamicsCache)
    cache._openmm_states = list(range(num_replicas))
    cache._gcmc_states = list(range(num_replicas))
    cache._state_moved = [False] * num_replicas
    cache._num_swaps = np.zeros((num_replicas, num_replicas))
    cache._states = states
    cache.mix_states()

    # Each state now holds the configuration that is favourable there.
    for state, config in enumerate(cache._openmm_states):
        assert energy_matrix[config, state] == -10000

    # The swap matrix records each configuration's move.
    for config in range(num_replicas):
        assert cache._num_swaps[config, (config + 1) % num_replicas] == 1


class _FakeState:
    def __init__(self, context):
        self._positions = context.positions.copy()
        self._velocities = context.velocities.copy()
        self._box = context.box

    def getPositions(self, asNumpy=True):
        return self._positions

    def getVelocities(self, asNumpy=True):
        return self._velocities

    def getPeriodicBoxVectors(self, asNumpy=True):
        return self._box


class _FakeContext:
    def __init__(self):
        # Positions hold [configuration, steps run], with -1 the input system.
        self.positions = np.array([-1, 0])
        self.velocities = np.array([-1])
        self.box = (1, 2, 3)
        self.lam = None

    def setPeriodicBoxVectors(self, *box):
        self.box = box

    def setPositions(self, positions):
        self.positions = positions.copy()

    def setVelocities(self, velocities):
        self.velocities = velocities.copy()

    def getState(self, getPositions=True, getVelocities=True):
        return _FakeState(self)


class _FakeDynamics:
    def __init__(self):
        self._context = _FakeContext()

    def context(self):
        return self._context

    def clear_energy_cache(self):
        pass

    def set_lambda(self, lam, rest2_scale=None, update_constraints=None):
        self._context.lam = lam

    def _get_clock(self):
        return None

    def _set_clock(self, clock):
        pass

    def set_energy_trajectory(self, trajectory):
        pass


class _FakeGCMCSampler:
    def __init__(self, num_waters):
        self.water = np.zeros(num_waters, dtype=int)

    def water_state(self):
        return self.water.copy()

    def _set_water_state(self, context, indices=None, states=None, force=False):
        self.water[indices] = states

    def push(self):
        pass

    def pop(self):
        pass

    def set_lambda(self, lam, rest2_scale):
        pass

    def set_ghost_file(self, ghost_file):
        pass

    def num_waters(self, context=None):
        return int(self.water.sum())


def _fake_slots(cache, num_slots):
    cache._num_slots = num_slots
    cache._build_slot_layout()
    cache._dynamics = [_FakeDynamics() for _ in range(num_slots)]
    cache._gcmc_samplers = [
        _FakeGCMCSampler(cache._num_replicas) for _ in range(num_slots)
    ]


@pytest.mark.parametrize(
    "slot_schedule, inverted",
    [
        ([4], False),
        ([2], False),
        ([1], False),
        ([4, 2, 3, 1], False),
        ([2], True),
    ],
)
def test_repex_mixing_samples_boltzmann(slot_schedule, inverted):
    """
    Validate that repeated mixing, pushed through replica loading and storing,
    samples the Boltzmann distribution over assignments of configurations to
    lambda windows. Each entry in the slot schedule is the number of contexts
    for an equal share of the cycles, with a checkpoint restart in between.
    Applying the inverse permutation must be detected.
    """
    import itertools
    import pickle
    from types import SimpleNamespace

    from somd2.runner._dynamics_cache import DynamicsCache

    n = 4
    num_cycles = 20000

    # Reduced energies of order kT, so that accepted swaps often chain.
    U = np.random.default_rng(1234).normal(0.0, 1.0, (n, n))

    cache = object.__new__(DynamicsCache)
    cache._lambdas = list(np.linspace(0, 1, n))
    cache._rest2_scale_factors = [1.0] * n
    cache._num_replicas = n
    cache._update_constraints = True
    cache._strided = False
    cache._states = np.arange(n)
    cache._time = None
    cache._clocks = [None] * n
    cache._hot_replicas = None
    cache._cold_votes = [0] * n
    cache._adaptive_data = None
    cache._virtual_sites = None
    cache._gcmc_stats = None
    cache._terminal_flip_stats = [[0, 0] for _ in range(n)]
    cache._energy_trajectories = [None] * n
    cache._ghost_files = [None] * n
    cache._gcmc_num_waters = [None] * n
    cache._state_moved = [False] * n
    cache._num_proposed = np.zeros((n, n))
    cache._num_accepted = np.zeros((n, n))
    cache._num_swaps = np.zeros((n, n))
    _fake_slots(cache, slot_schedule[0])

    # Each replica starts with its own configuration and water occupancy.
    cache._openmm_states = [
        {"positions": np.array([r, 0]), "velocities": np.array([r]), "box": (1, 2, 3)}
        for r in range(n)
    ]
    cache._gcmc_states = [np.eye(n, dtype=int)[r] for r in range(n)]

    # The configuration expected in each window.
    expected = list(range(n))

    counts = {}
    cycles_per_restart = num_cycles // len(slot_schedule)

    for cycle in range(num_cycles):
        if cycle > 0 and cycle % cycles_per_restart == 0:
            state = pickle.loads(pickle.dumps(cache.__getstate__()))
            cache = object.__new__(DynamicsCache)
            cache.__setstate__(state)
            _fake_slots(cache, slot_schedule[cycle // cycles_per_restart])

        energies = np.zeros((n, n))
        runner = SimpleNamespace(_dynamics_cache=cache)

        for batch in RepexRunner._replica_passes(runner, cycle):
            for replica in batch:
                cache.load_replica(replica)
                slot = cache.slot_for(replica)
                context = cache._dynamics[slot].context()
                config = int(context.positions[0])
                water = cache._gcmc_samplers[slot].water

                assert context.lam == cache._lambdas[replica]
                assert config == expected[replica]
                assert water.sum() == 1 and water[config] == 1

                context.positions = context.positions + np.array([0, 1])
                energies[replica] = U[:, config]
                cache.store_replica(replica)

        states = _mix_subset(
            np.arange(n), energies, cache._num_proposed, cache._num_accepted
        )
        if inverted:
            states = np.argsort(states)

        num_swaps = cache._num_swaps.copy()
        cache.set_states(states)
        cache.mix_states()

        for window, source in enumerate(states):
            assert cache._num_swaps[source, window] == num_swaps[source, window] + 1

        expected = [expected[source] for source in states]
        counts[tuple(expected)] = counts.get(tuple(expected), 0) + 1

    # Compare with the exact distribution using the total variation distance.
    perms = list(itertools.permutations(range(n)))
    weights = np.array([np.exp(-sum(U[k, p[k]] for k in range(n))) for p in perms])
    weights /= weights.sum()
    tvd = 0.5 * sum(
        abs(counts.get(p, 0) / num_cycles - w) for p, w in zip(perms, weights)
    )

    if inverted:
        assert tvd > 0.05
    else:
        assert tvd < 0.03


def test_mix_subset_samples_boltzmann():
    """
    Validate that mixing only the hot replicas on most cycles, with a hot set
    that changes during the run and every replica mixed periodically, samples
    the same Boltzmann distribution over assignments as mixing them all.
    """
    import itertools

    n = 4
    num_cycles = 40000
    mix_interval = 4
    hot_sets = [[0, 1, 2], [1, 2, 3], [0, 1], [2, 3], [1, 2]]

    rng = np.random.default_rng(1234)
    U = rng.normal(0.0, 1.0, (n, n))
    proposed = np.zeros((n, n))
    accepted = np.zeros((n, n))

    # The configuration in each window.
    expected = list(range(n))

    counts = {}
    for cycle in range(num_cycles):
        if (cycle + 1) % mix_interval == 0:
            active = list(range(n))
        else:
            active = hot_sets[(cycle // 1000) % len(hot_sets)]

        # Only the rows of the replicas being mixed may be used.
        energies = rng.normal(0.0, 50.0, (n, n))
        for replica in active:
            energies[replica] = U[:, expected[replica]]

        states = _mix_subset(np.array(active), energies, proposed, accepted)
        for window in set(range(n)) - set(active):
            assert states[window] == window

        expected = [expected[source] for source in states]
        counts[tuple(expected)] = counts.get(tuple(expected), 0) + 1

    perms = list(itertools.permutations(range(n)))
    weights = np.array([np.exp(-sum(U[k, p[k]] for k in range(n))) for p in perms])
    weights /= weights.sum()

    def tvd(target):
        return 0.5 * sum(
            abs(counts.get(p, 0) / num_cycles - w) for p, w in zip(perms, target)
        )

    assert tvd(weights) < 0.03

    # The energies must be far enough from uniform for the test to have power.
    assert tvd(np.full(len(perms), 1 / len(perms))) > 0.1


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
            runner = RunnerBase(ethane_methanol, Config(**config))
        else:
            with pytest.raises(ValueError):
                runner = RunnerBase(ethane_methanol, Config(**config))


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
            runner = RunnerBase(ethane_methanol, Config(**config))
        else:
            with pytest.raises(ValueError):
                runner = RunnerBase(ethane_methanol, Config(**config))


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("max_contexts", [1, 2, 3, 4])
def test_repex_bounded_contexts(ethane_methanol, max_contexts):
    """
    Validate that a replica exchange simulation runs when there are fewer
    OpenMM contexts than replicas, so that each context is re-used to
    propagate several replicas per cycle.
    """
    num_lambda = 4

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
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        # Only the requested number of contexts should have been created.
        assert len(runner._dynamics_cache._dynamics) == max_contexts

        # Every replica must be assigned to exactly one slot.
        groups = runner._dynamics_cache._groups
        assert sorted(r for group in groups for r in group) == list(range(num_lambda))

        runner.run()

        # Output is per replica, regardless of how many contexts were used.
        assert (Path(tmpdir) / "repex_matrix.txt").exists()
        for i in range(num_lambda):
            assert Path(runner._filenames[i]["energy_traj"]).exists()


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_frame_frequency_constraint(ethane_methanol):
    """
    Validate that frames can only be saved on checkpoint cycles when contexts
    are re-used across lambda values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "8fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": 4,
            "replica_exchange": True,
            "max_contexts": 2,
        }

        with pytest.raises(ValueError, match="frame_frequency"):
            RepexRunner(ethane_methanol, Config(**config))


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("update_constraints", [True, False])
def test_repex_update_constraints(ethane_methanol, update_constraints):
    """
    Validate both constraint modes. Ethane to methanol does perturb a
    constrained bond length, so update_constraints=True forces the context to
    be reinitialised on every lambda change.
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
            "num_lambda": 4,
            "replica_exchange": True,
            "max_contexts": 2,
            "update_constraints": update_constraints,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_bounded_contexts_output_equivalence(ethane_methanol):
    """
    Validate that re-using contexts produces the same output structure as one
    context per replica. Energies are not compared: a shared context consumes
    the integrator's random number stream in a different order, so the
    trajectories legitimately differ.
    """
    import pandas as pd

    num_lambda = 4

    def run(max_contexts, tmpdir):
        config = {
            "runtime": "16fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
        }
        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()
        return [
            pd.read_parquet(runner._filenames[i]["energy_traj"])
            for i in range(num_lambda)
        ], list(runner._lambda_values)

    with tempfile.TemporaryDirectory() as tmpdir:
        full, lambdas = run(num_lambda, tmpdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        cached, _ = run(1, tmpdir)

    # One record per energy_frequency interval, starting at the first. Pinning
    # the values rather than only comparing the two runs catches a clock that
    # is wrong the same way in both.
    expected_times = [0.004, 0.008, 0.012, 0.016]

    for i in range(num_lambda):
        assert len(cached[i]) == len(full[i])
        assert list(cached[i].columns) == list(full[i].columns)
        assert cached[i].index.equals(full[i].index)

        for records in (full[i], cached[i]):
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == expected_times, f"replica {i} recorded times {times}"

            # Each replica records energies at every lambda value.
            assert list(records.columns) == lambdas


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_bounded_contexts_restart(ethane_methanol):
    """
    Validate that a replica exchange simulation using fewer contexts than
    replicas can be restarted, that each replica resumes from the state it
    stopped at, and that the energy trajectory is extended rather than
    restarted.
    """
    import pandas as pd

    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": 2,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()

        num_rows = [
            len(pd.read_parquet(runner._filenames[i]["energy_traj"]))
            for i in range(num_lambda)
        ]

        # The state each replica finished at, which the checkpoint holds.
        import openmm.unit as omm_unit

        stopped = [
            {
                "positions": state["positions"].value_in_unit(omm_unit.nanometer),
                "velocities": state["velocities"].value_in_unit(
                    omm_unit.nanometer / omm_unit.picosecond
                ),
                "box": state["box"].value_in_unit(omm_unit.nanometer),
            }
            for state in runner._dynamics_cache._openmm_states
        ]

        # Restart, extending the runtime.
        config["runtime"] = "16fs"
        config["restart"] = True

        runner = RepexRunner(ethane_methanol, Config(**config))

        # Every replica must resume from where it stopped. The contexts are
        # created from the input system, so the only thing carrying the
        # simulated state across a restart is the checkpoint.
        for i in range(num_lambda):
            state = runner._dynamics_cache._openmm_states[i]
            for key, unit in (
                ("positions", omm_unit.nanometer),
                ("velocities", omm_unit.nanometer / omm_unit.picosecond),
                ("box", omm_unit.nanometer),
            ):
                assert np.allclose(
                    state[key].value_in_unit(unit), stopped[i][key], atol=1e-6
                ), f"replica {i} {key} not restored"

        # The input coordinates must not be what was restored, otherwise the
        # checks above would pass even if the checkpoint were ignored.
        import sire as sr

        inputs = sr.io.get_coords_array(runner._system)
        restored = runner._dynamics_cache._openmm_states[0]["positions"].value_in_unit(
            omm_unit.angstrom
        )
        assert not np.allclose(restored, inputs, atol=1e-3)

        # Restoring the checkpoint into the cache is not enough: the contexts
        # are created from the input system, so the state has to reach them
        # too. Loading a replica is what pushes it.
        for i in range(num_lambda):
            runner._dynamics_cache.load_replica(i)
            dynamics, _ = runner._dynamics_cache.get(runner._dynamics_cache.slot_for(i))
            positions = (
                dynamics.context()
                .getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(omm_unit.nanometer)
            )
            assert np.allclose(positions, stopped[i]["positions"], atol=1e-5), (
                f"replica {i} positions not pushed into its context"
            )

        runner.run()

        # 8 fs at 4 fs intervals, extended to 16 fs. The clock has to continue
        # from where it stopped rather than restarting at zero, so the records
        # must run to the new runtime with no repeats or gaps.
        expected_times = [0.004, 0.008, 0.012, 0.016]

        for i in range(num_lambda):
            extended = pd.read_parquet(runner._filenames[i]["energy_traj"])
            assert len(extended) > num_rows[i]

            times = [round(t, 6) for t in extended.index.get_level_values(0)]
            assert times == expected_times, f"replica {i} recorded times {times}"


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("max_contexts", [1, 4])
def test_repex_checkpoint_single_lock(ethane_methanol, max_contexts):
    """
    Validate that every checkpoint file is written within a single acquisition
    of the file lock, so that a process streaming the output off the machine
    always sees a coherent set rather than a mixture of new and old files.
    """
    import somd2.runner._repex as repex_module

    num_lambda = 4
    acquisitions = []

    real_filelock = repex_module._FileLock

    class CountingFileLock(real_filelock):
        def acquire(self, *args, **kwargs):
            acquisitions.append(1)
            return super().acquire(*args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        repex_module._FileLock = CountingFileLock
        try:
            runner.run()
        finally:
            repex_module._FileLock = real_filelock

    # Two checkpoint cycles, each taking the lock once for the checkpoint files
    # and the repex state together. This must not scale with the number of passes.
    assert len(acquisitions) == 2


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_checkpoint_failure_keeps_lock(ethane_methanol, monkeypatch):
    """
    Validate that when one replica's checkpoint fails, the others finish
    writing while the file lock is still held, before the error is raised.
    """
    import time

    from filelock import FileLock, Timeout

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "4fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": 4,
            "replica_exchange": True,
            # Enough workers to checkpoint every replica at once.
            "oversubscription_factor": 4,
            "num_checkpoint_workers": 4,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        held = []

        def checkpoint(index, *args, **kwargs):
            if index == 0:
                return index, RuntimeError("checkpoint failed")
            time.sleep(0.5)
            lock = FileLock(runner._lock_file)
            try:
                lock.acquire(timeout=0)
                lock.release()
                held.append(False)
            except Timeout:
                held.append(True)
            return index, None

        monkeypatch.setattr(runner, "_checkpoint", checkpoint)

        with pytest.raises(RuntimeError, match="checkpoint failed"):
            runner.run()

    assert held == [True, True, True]


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize(
    "runtime, checkpoint_frequency, expected",
    [("8fs", "4fs", [False, False]), ("12fs", "8fs", [False, True])],
)
def test_repex_state_saved_once(
    ethane_methanol, runtime, checkpoint_frequency, expected
):
    """
    Validate that the replica exchange state is saved once per checkpoint
    cycle, with a separate final save only when the last cycle is not a
    checkpoint cycle.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": runtime,
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": checkpoint_frequency,
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": 2,
            "replica_exchange": True,
        }
        runner = RepexRunner(ethane_methanol, Config(**config))

        saves = []
        save = runner._save_repex_state

        def counting_save(final=False):
            saves.append(final)
            return save(final=final)

        runner._save_repex_state = counting_save
        runner.run()

        assert saves == expected
        assert (Path(tmpdir) / "repex_state.pkl").exists()


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("max_contexts", [1, 4])
def test_repex_gcmc_bounded_contexts(ethane_methanol, max_contexts):
    """
    Validate that GCMC sampling works when contexts are re-used across lambda
    values, so a slot's single sampler is re-parameterised and re-pointed at
    the ghost file of whichever replica it hosts.
    """
    pytest.importorskip("loch")

    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
            # Checkpoint replicas concurrently.
            "oversubscription_factor": 2,
            "gcmc": True,
            "gcmc_selection": "resname LIG",
            "gcmc_frequency": "4fs",
        }

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()

        # One ghost file per lambda, each with a line per saved frame. A slot
        # writing to the wrong file would leave these unbalanced.
        counts = []
        for lam in runner._lambda_values:
            ghost_file = Path(tmpdir) / f"gcmc_ghosts_{lam:.5f}.txt"
            assert ghost_file.exists()
            counts.append(len(ghost_file.read_text().strip().splitlines()))

        assert len(set(counts)) == 1, f"unbalanced ghost files: {counts}"
        assert counts[0] > 0

        # Each ghost line pairs with a trajectory frame.
        import sire as sr

        for lam, count in zip(runner._lambda_values, counts):
            traj = sr.load(
                str(Path(tmpdir) / "system0.prm7"),
                str(Path(tmpdir) / f"traj_{lam:.5f}.dcd"),
            )
            assert traj.num_frames() == count


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_gcmc_without_a_selection(ethane_methanol):
    """
    Validate GCMC sampling with no 'gcmc_selection', where moves are attempted
    within the entire simulation box rather than a region around a selection.

    The sampler then has no reference, so it cannot count the waters within a
    region, and every move samples the whole box. Counting the waters on each
    replica handover has to account for that.
    """
    pytest.importorskip("loch")

    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": 2,
            "gcmc": True,
            "gcmc_frequency": "4fs",
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        # The bulk-only path is the point of this test, so make sure it can't
        # stop being exercised without the test failing.
        assert runner._dynamics_cache._gcmc_samplers[0]._reference is None

        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()
        for lam in runner._lambda_values:
            assert (Path(tmpdir) / f"gcmc_ghosts_{lam:.5f}.txt").exists()


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("max_contexts", [2, None])
def test_repex_concurrent_slots(ethane_methanol, max_contexts):
    """
    Validate that replicas sharing a slot are never propagated concurrently.
    Oversubscribing exercises this on a single GPU, since the worker count is
    the number of GPUs times the oversubscription factor.

    This is also the only test that equilibrates, so it covers moving replicas
    in and out of their slots during equilibration, the context rebuild when
    the constraints change, and the post-equilibration checkpoint.
    """
    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "equilibration_time": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
            "oversubscription_factor": 2,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        # Guard against the equilibration coverage being lost silently.
        assert runner._is_equilibration

        # Minimising without constraints and equilibrating with them means the
        # contexts are rebuilt part way through, which is the path being
        # covered here.
        assert not runner._config.minimisation_constraints
        assert runner._config.equilibration_constraints

        # Every batch of slots set up at once must hold distinct slots, at most
        # oversubscription_factor on each GPU.
        cache = runner._dynamics_cache
        for batch in runner._setup_batches(runner._replica_devices()):
            slots = [cache.slot_for(group[0]) for group in batch]
            assert len(slots) == len(set(slots)), f"batch {batch} repeats a slot"
            devices = [cache.device_for(slot) for slot in slots]
            for device in set(devices):
                assert devices.count(device) <= config["oversubscription_factor"]

        # Every pass of commits must contain at most one replica per slot.
        for batch in runner._replica_passes(0) + runner._replica_passes(1):
            slots = [cache.slot_for(r) for r in batch]
            assert len(slots) == len(set(slots)), f"batch {batch} shares a slot"

        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()

        # Equilibration doesn't advance any replica's clock, so each one
        # records its first energies after one energy_frequency.
        import pandas as pd

        for i in range(num_lambda):
            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == [0.004, 0.008, 0.012], f"replica {i} recorded {times}"


@pytest.mark.parametrize(
    "gpu_devices, expected",
    [
        # Visible set starts at zero and is contiguous, so the OpenMM index and
        # the physical device agree.
        (["0", "1", "2"], ["0", "1", "2"]),
        # Offset visible set: OpenMM index 0 is physical device 1.
        (["1", "2"], ["1", "2"]),
        # A single device that is not device zero.
        (["3"], ["3"]),
        # CUDA_VISIBLE_DEVICES may hold UUIDs rather than indices.
        (["GPU-abc123", "GPU-def456"], ["GPU-abc123", "GPU-def456"]),
        # Unknown visible set falls back to the OpenMM index.
        (None, [0, 1]),
    ],
)
def test_physical_device_mapping(gpu_devices, expected):
    """
    Validate that an OpenMM device index is mapped to the physical device
    backing it, since OpenMM numbers devices relative to the visible set
    whereas pynvml and pyopencl enumerate all of them.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    cache = object.__new__(DynamicsCache)
    cache._gpu_devices = gpu_devices

    assert [cache._physical_device(i) for i in range(len(expected))] == expected


def test_max_contexts_advice():
    """
    Validate the advice given when the replicas don't fit in GPU memory. It
    must name a number that the user can pass to 'max_contexts', and mention
    the frame frequency constraint that comes with it.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    advice = DynamicsCache._max_contexts_advice(6)
    assert "'max_contexts' to 6 or fewer" in advice
    assert "frame_frequency" in advice and "checkpoint_frequency" in advice

    # Nothing fits, so there is no number to suggest.
    advice = DynamicsCache._max_contexts_advice(0)
    assert "max_contexts" not in advice
    assert "does not fit" in advice


def test_gcmc_state_follows_replica():
    """
    Validate that the GCMC water occupancy travels with the configuration it
    belongs to when replicas are mixed.

    A slot holds one GCMC sampler but hosts several replicas, and load_replica()
    installs a replica's water state by diffing it against whatever the last
    resident left in the sampler. If the occupancy did not follow the positions
    through a mix, a replica would run with another replica's waters, which
    gives plausible numbers rather than an obviously wrong output file.

    The ghost files and the sampling statistics belong to the lambda window
    rather than the configuration, so they must not be permuted.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    num_replicas = 4

    cache = object.__new__(DynamicsCache)

    # Label both states with the replica they came from, so that a replica
    # holding mismatched positions and waters is detectable.
    cache._openmm_states = list(range(num_replicas))
    cache._gcmc_states = list(range(num_replicas))
    cache._ghost_files = [f"ghosts_{i}.txt" for i in range(num_replicas)]
    cache._state_moved = [False] * num_replicas
    cache._num_swaps = np.zeros((num_replicas, num_replicas))

    # Mix twice, since a slot is re-used within a cycle.
    for states in ([2, 0, 3, 1], [1, 3, 0, 2]):
        expected = [cache._gcmc_states[state] for state in states]

        cache._states = states
        cache.mix_states()

        # The water occupancy follows the same permutation as the positions.
        assert cache._gcmc_states == expected

        # Every replica holds the positions and waters of the same origin.
        assert cache._openmm_states == cache._gcmc_states

    # The ghost files stay with the lambda window.
    assert cache._ghost_files == [f"ghosts_{i}.txt" for i in range(num_replicas)]


@pytest.mark.parametrize(
    "device, key, value",
    [
        ("2", "index", 2),
        (2, "index", 2),
        ("GPU-abc123", "uuid", b"GPU-abc123"),
    ],
)
def test_check_device_memory_queries_requested_device(monkeypatch, device, key, value):
    """
    Validate that the memory query is made against the device it was asked
    for, by index or by UUID.
    """
    import sys
    import types

    from somd2.runner._dynamics_cache import DynamicsCache

    pynvml = pytest.importorskip("pynvml")

    # Force the OpenCL branch to fail so the pynvml path is always taken.
    broken = types.SimpleNamespace()

    def get_platforms():
        raise RuntimeError("no OpenCL")

    broken.get_platforms = get_platforms
    monkeypatch.setitem(sys.modules, "pyopencl", broken)

    requested = {}

    class Memory:
        used, free, total = 1, 2, 3

    def by_index(index):
        requested["index"] = index
        return "handle"

    def by_uuid(uuid):
        requested["uuid"] = uuid
        return "handle"

    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", by_index)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", by_uuid)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", lambda handle: Memory)

    assert DynamicsCache._check_device_memory(device) == (1, 2, 3)
    assert requested == {key: value}


def test_legacy_checkpoint_restore():
    """
    Validate that a checkpoint written before slots existed can still be
    loaded. These stored the replica states unpermuted, with the states array
    holding the mapping to apply on restart.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    n = 4
    legacy = {
        "_lambdas": [0.0, 0.33, 0.67, 1.0],
        "_rest2_scale_factors": [1.0] * n,
        "_states": np.array([2, 0, 1, 3]),
        "_time": None,
        "_openmm_states": [f"state{i}" for i in range(n)],
        "_gcmc_samplers": [None] * n,
        "_gcmc_states": [f"water{i}" for i in range(n)],
        "_gcmc_stats": [None] * n,
        "_terminal_flip_stats": [[0, 0]] * n,
        "_num_proposed": np.zeros((n, n)),
        "_num_accepted": np.zeros((n, n)),
        "_num_swaps": np.zeros((n, n)),
    }

    cache = object.__new__(DynamicsCache)
    cache.__setstate__(dict(legacy))

    # Converted to the current convention: each replica's own state, with the
    # last mix applied. The legacy states hold each configuration's destination.
    assert cache._openmm_states == ["state1", "state2", "state0", "state3"]
    assert cache._gcmc_states == ["water1", "water2", "water0", "water3"]

    # Every replica is seeded from its stored state on a restart.
    assert cache._state_moved == [True] * n

    # Attributes postdating the checkpoint are defaulted, one slot per replica.
    assert cache._num_slots == n
    assert cache._groups == [[0], [1], [2], [3]]
    assert cache._energy_trajectories == [None] * n
    assert cache._ghost_files == [None] * n


def test_checkpoint_from_before_the_cache_moved(monkeypatch):
    """
    Validate that a checkpoint pickled when the dynamics cache was defined in
    _repex can still be loaded.
    """
    import pickle

    from somd2.runner._dynamics_cache import DynamicsCache

    n = 2
    cache = object.__new__(DynamicsCache)
    cache.__setstate__(
        {
            "_lambdas": [0.0, 1.0],
            "_rest2_scale_factors": [1.0] * n,
            "_num_slots": n,
            "_states": np.arange(n),
            "_time": None,
            "_openmm_states": [None] * n,
            "_gcmc_samplers": [None] * n,
            "_gcmc_states": [None] * n,
            "_gcmc_stats": None,
            "_terminal_flip_stats": [[0, 0]] * n,
            "_num_proposed": np.zeros((n, n)),
            "_num_accepted": np.zeros((n, n)),
            "_num_swaps": np.zeros((n, n)),
        }
    )

    with monkeypatch.context() as m:
        m.setattr(DynamicsCache, "__module__", "somd2.runner._repex")
        data = pickle.dumps(cache)

    assert b"somd2.runner._repex" in data
    assert isinstance(pickle.loads(data), DynamicsCache)


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_max_contexts_change_on_restart(ethane_methanol):
    """
    Validate that the number of contexts can change on restart. The slot
    layout is rebuilt from the configuration rather than restored.
    """
    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": 2,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))
        assert len(runner._dynamics_cache._dynamics) == 2
        runner.run()

        # Restart with a different number of contexts.
        config["runtime"] = "16fs"
        config["restart"] = True
        config["max_contexts"] = 4

        runner = RepexRunner(ethane_methanol, Config(**config))
        assert len(runner._dynamics_cache._dynamics) == 4
        assert runner._dynamics_cache._groups == [[0], [1], [2], [3]]
        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_gcmc_lambda_cache_warm(ethane_methanol, monkeypatch):
    """
    Validate that a GCMC sampler builds exactly one OpenMM context, scanning
    it over every lambda it will host, and none once running. A mismatch
    between the cached lambdas and those passed to set_lambda would show up
    here as an extra build.
    """
    loch = pytest.importorskip("loch")

    num_lambda = 4
    calls = []

    real_precompute = loch.GCMCSampler._precompute_lambdas

    def counting_precompute(self, lambda_values, rest2_scales):
        # Only record calls with work to do. Deduplicated, since the caller
        # may name the same lambda twice but the scan extracts it once.
        missing = sorted(
            {
                (float(lam), float(scale))
                for lam, scale in zip(lambda_values, rest2_scales)
                if (float(lam), float(scale)) not in self._lambda_params
            }
        )
        if missing:
            calls.append(missing)
        return real_precompute(self, lambda_values, rest2_scales)

    monkeypatch.setattr(loch.GCMCSampler, "_precompute_lambdas", counting_precompute)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "8fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": 1,
            "gcmc": True,
            "gcmc_selection": "resname LIG",
            "gcmc_frequency": "4fs",
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        # A single context build, covering every lambda value in one scan.
        assert len(calls) == 1
        assert sorted(lam for lam, _ in calls[0]) == sorted(runner._lambda_values)

        runner.run()

        # No further context may be built once the simulation is running.
        assert len(calls) == 1

        # An uncached lambda still works, building and caching on demand.
        _, sampler = runner._dynamics_cache.get(0)
        uncached = 0.123456
        assert uncached not in runner._lambda_values

        sampler.push()
        try:
            sampler.set_lambda(uncached)
        finally:
            sampler.pop()

        assert len(calls) == 2
        assert (uncached, sampler._rest2_scale) in sampler._lambda_params


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("swap_end_states", [False, True])
def test_repex_perturbed_system_seeding(ethane_methanol, swap_end_states):
    """
    Validate that the end states are seeded from the right coordinates when
    contexts are shared.

    A context is created from a single system and every replica it hosts starts
    from that context, so the end state a replica starts from is chosen from
    the middle of the group rather than its first replica. That keeps any
    mismatch next to the lambda value at which the end state switches, instead
    of it depending on where the groups happen to fall.

    Swapping the end states reverses the lambda schedule, so the perturbed end
    state moves to lambda = 0 and the seeding must mirror with it.

    Only the perturbable molecule is displaced, so the assertions also cover
    the property path that its coordinates travel along: read from
    'coordinates1' via link_to_perturbed, written to 'coordinates0', then read
    back via link_to_reference.
    """
    import sire as sr

    # A perturbed end state, with the perturbable molecule displaced so that
    # its coordinates are distinct.
    perturbed = ethane_methanol.clone()
    perturbed.set_property("space", ethane_methanol.property("space"))
    coords = sr.io.get_coords_array(ethane_methanol)
    from sire.legacy.IO import setCoordinates

    # Flag the atoms of the perturbable molecules. get_coords_array returns the
    # atoms in molecule order, so the offset tracks the array index.
    is_perturbable = np.zeros(len(coords), dtype=bool)
    offset = 0
    for mol in ethane_methanol.molecules():
        num_atoms = mol.num_atoms()
        if mol.has_property("is_perturbable"):
            is_perturbable[offset : offset + num_atoms] = True
        offset += num_atoms
    assert is_perturbable.any(), "no perturbable molecules in the test system"

    coords[is_perturbable] += 1.0

    # Write to 'coordinates1', which is the property that the runner reads via
    # link_to_perturbed.
    perturbed = sr.system.System(
        setCoordinates(perturbed._system, coords.tolist(), True)
    )

    # Ten replicas across three contexts, a layout in which the switch falls
    # inside a group.
    num_lambda = 10

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "4fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": 3,
            "perturbed_system": perturbed,
            "swap_end_states": swap_end_states,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        reference = sr.io.get_coords_array(runner._system)
        target = sr.io.get_coords_array(runner._perturbed_system)

        import openmm.unit as omm_unit

        seeded = []
        for i in range(num_lambda):
            positions = runner._dynamics_cache._openmm_states[i][
                "positions"
            ].value_in_unit(omm_unit.angstrom)
            from_reference = np.allclose(positions, reference, atol=1e-3)
            from_target = np.allclose(positions, target, atol=1e-3)
            assert from_reference != from_target, f"replica {i} matches neither"
            seeded.append("perturbed" if from_target else "reference")

        # The end states themselves must always be right. Swapping the end
        # states puts the perturbed one at lambda = 0.
        if swap_end_states:
            assert seeded[0] == "perturbed"
            assert seeded[-1] == "reference"
        else:
            assert seeded[0] == "reference"
            assert seeded[-1] == "perturbed"

        # Both systems must be used, otherwise the option does nothing.
        assert set(seeded) == {"reference", "perturbed"}

        # Only the group containing the switch can be seeded from the wrong end
        # state, and then for no more than half of it. Choosing the end state
        # from the first replica of a group rather than its middle breaks this.
        lambdas = runner._lambda_values
        for group in runner._dynamics_cache._groups:
            wrong = [
                i
                for i in group
                if (seeded[i] == "perturbed") != ((lambdas[i] > 0.5) != swap_end_states)
            ]
            assert len(wrong) <= len(group) // 2, (
                f"group {group} has {len(wrong)} replicas seeded from the "
                f"wrong end state: {wrong}"
            )


@pytest.mark.parametrize(
    "last_cycle, cycle, cycles_per_move, expected",
    [
        (2, 3, 4, True),
        (1, 2, 4, False),
        # A move that fell due while the replica wasn't propagated.
        (0, 5, 4, True),
        (4, 5, 4, False),
        (-1, 0, 1, True),
    ],
)
def test_is_due(last_cycle, cycle, cycles_per_move, expected):
    """
    Validate that periodic moves fall due on schedule, and that a replica
    catches up on one that fell due while it wasn't propagated.
    """
    assert RepexRunner._is_due(last_cycle, cycle, cycles_per_move) == expected


def test_mix_subset_leaves_others_in_place():
    """
    Validate that mixing a subset of the replicas only swaps between them.
    """
    n = 6
    subset = np.array([1, 2, 4])
    energy_matrix = np.zeros((n, n))
    proposed = np.zeros((n, n))
    accepted = np.zeros((n, n))

    # Every swap is accepted, so the subset is thoroughly shuffled.
    states = _mix_subset(subset, energy_matrix, proposed, accepted)

    for r in set(range(n)) - set(subset):
        assert states[r] == r
    assert sorted(states[subset]) == sorted(subset)

    # Only swaps between members of the subset are recorded.
    outside = [r for r in range(n) if r not in subset]
    assert proposed[outside, :].sum() == 0
    assert proposed[:, outside].sum() == 0


def test_strided_slot_layout():
    """
    Validate that a strided layout spreads any contiguous range of replicas
    evenly over the slots.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    cache = object.__new__(DynamicsCache)
    cache._num_replicas = 10
    cache._num_slots = 4
    cache._strided = True
    cache._build_slot_layout()

    assert cache._groups == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]

    for start in range(7):
        slots = [cache.slot_for(r) for r in range(start, start + 4)]
        assert sorted(slots) == [0, 1, 2, 3]


def test_statistical_inefficiency():
    """
    Validate the statistical inefficiency against an AR(1) process, for which
    it is (1 + phi) / (1 - phi).
    """
    rng = np.random.default_rng(42)

    x = rng.normal(size=20000)
    assert statistical_inefficiency(x) == pytest.approx(1.0, abs=0.1)

    phi = 0.8
    y = np.zeros(20000)
    for i in range(1, len(y)):
        y[i] = phi * y[i - 1] + rng.normal()
    expected = (1 + phi) / (1 - phi)
    assert statistical_inefficiency(y) == pytest.approx(expected, rel=0.15)


def _ranking_runner(num_lambda, is_ring_break=False, is_cached=False, **config):
    """
    Create a runner holding only what is needed to rank windows.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    runner = object.__new__(RepexRunner)
    runner._config = Config(num_lambda=num_lambda, **config)
    runner._lambda_values = list(np.linspace(0, 1, num_lambda))
    zeros = np.zeros((num_lambda, num_lambda))

    cache = object.__new__(DynamicsCache)
    cache._num_replicas = num_lambda
    cache._num_proposed = zeros
    cache._num_accepted = zeros
    cache._hot_replicas = None
    cache._cold_votes = [0] * num_lambda
    cache._adaptive_data = None
    runner._dynamics_cache = cache

    runner._ranker = WindowRanker(
        num_lambda,
        runner._config.adaptive_min_samples,
        zeros,
        zeros,
        is_ring_break=is_ring_break,
        is_cached=is_cached,
    )
    return runner


def _rank(runner, num_gpus, window_samples=None, cycles_left=10**6, capacity=None):
    """
    Rank the windows, with window r on GPU r % num_gpus. By default every
    window has the same samples, the run is far from its end, so that only
    the sampling rates matter, and each GPU can propagate as many replicas at
    once as the oversubscription factor.
    """
    num_lambda = runner._config.num_lambda
    if window_samples is None:
        window_samples = [200] * num_lambda
    if capacity is None:
        capacity = runner._config.oversubscription_factor
    devices = [r % num_gpus for r in range(num_lambda)]
    cold_rate = 1.0 / runner._config.adaptive_mix_interval
    runner._rank_windows(
        devices, [capacity] * num_gpus, window_samples, cycles_left, cold_rate
    )


def _rank_twice(runner, num_gpus, capacity=None):
    """
    Rank the windows at two consecutive checkpoints, so that the windows found
    cold at both stop being hot.
    """
    _rank(runner, num_gpus=num_gpus, capacity=capacity)
    _rank(runner, num_gpus=num_gpus, capacity=capacity)


def _fill_ranking_data(runner, hard_pair, num_samples=200, seed=7):
    """
    Record converged energy differences between every pair of neighbouring
    windows, apart from one pair with a large spread.
    """
    rng = np.random.default_rng(seed)
    num_lambda = runner._config.num_lambda

    for k in range(num_lambda - 1):
        sigma = 1.5 if k == hard_pair else 0.1
        runner._ranker.upper[k] = list(rng.normal(0.5, sigma, num_samples))
        runner._ranker.lower[k + 1] = list(rng.normal(-0.5, sigma, num_samples))


def test_rank_windows():
    """
    Validate that the hot replicas are the windows either side of the pair with
    the largest uncertainty, padded by a window each side, and that windows
    only go cold after being found cold at two consecutive rankings.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)
    cache = runner._dynamics_cache

    # Starting from every window hot, those found cold once stay hot.
    _fill_ranking_data(runner, hard_pair=3)
    _rank(runner, num_gpus=2)
    assert cache._hot_replicas is None

    _rank(runner, num_gpus=2)
    assert cache._hot_replicas == [2, 3, 4, 5]

    # Windows 2 and 3 are found cold once, so stay hot.
    _fill_ranking_data(runner, hard_pair=5)
    _rank(runner, num_gpus=2)
    assert cache._hot_replicas == [2, 3, 4, 5, 6, 7]

    # Found cold a second time.
    _fill_ranking_data(runner, hard_pair=5)
    _rank(runner, num_gpus=2)
    assert cache._hot_replicas == [4, 5, 6, 7]


def test_rank_windows_balances_gpus():
    """
    Validate that the hot replicas are padded so that every GPU has the same
    number, and that every replica being hot is recorded as None.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)

    _fill_ranking_data(runner, hard_pair=3)
    _rank_twice(runner, num_gpus=3)
    hot = runner._dynamics_cache._hot_replicas
    assert len(hot) == 6
    assert sorted(r % 3 for r in hot) == [0, 0, 1, 1, 2, 2]

    runner._dynamics_cache._hot_replicas = None
    _fill_ranking_data(runner, hard_pair=3)
    _rank(runner, num_gpus=8)
    assert runner._dynamics_cache._hot_replicas is None


@pytest.mark.parametrize(
    "runtime, checkpoint_frequency, burn_in, rank_frequency, warns",
    [
        ("5 ns", "100 ps", "500 ps", None, False),
        # The only checkpoint is the end of the run.
        ("1 ns", "2 ns", "0 ps", None, True),
        # The first checkpoint after the burn-in is the end of the run.
        ("12 ns", "4 ns", "9 ns", None, True),
        # The second checkpoint after the burn-in is the end of the run, so
        # no window goes cold.
        ("1 ns", "500 ps", "400 ps", None, True),
        ("1.2 ns", "500 ps", "400 ps", None, False),
        ("5 ns", "0 ps", "500 ps", None, True),
        # Checkpoints are no further apart than 'adaptive_mix_interval'.
        ("5 ns", "4 ps", "500 ps", None, True),
        # Ranking more often than checkpointing leaves time to go cold.
        ("1 ns", "500 ps", "400 ps", "100 ps", False),
        ("5 ns", "0 ps", "500 ps", "100 ps", False),
    ],
)
def test_check_adaptive_config(
    monkeypatch, runtime, checkpoint_frequency, burn_in, rank_frequency, warns
):
    """
    Validate that settings which stop adaptive replica exchange having an
    effect are warned about.
    """
    import sire as sr

    from somd2.runner import _repex

    warnings = []
    monkeypatch.setattr(_repex._logger, "warning", warnings.append)

    runner = object.__new__(RepexRunner)
    runner._config = Config(
        runtime=runtime,
        energy_frequency="1 ps",
        checkpoint_frequency=checkpoint_frequency,
        adaptive_burn_in=burn_in,
        adaptive_mix_interval=4,
        adaptive_rank_frequency=rank_frequency,
    )
    runner._start_time = sr.u("0 ps")
    runner._check_adaptive_config()

    assert bool(warnings) == warns


def test_rank_windows_minimises_variance_for_compute():
    """
    Validate that a pair with a large uncertainty is made hot without being
    flagged, and that pairs which add more cost than they remove variance are
    left cold.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)

    _fill_ranking_data(runner, hard_pair=-1)

    # Forward and reverse estimates that agree, so the pair isn't flagged.
    rng = np.random.default_rng(11)
    runner._ranker.upper[3] = list(rng.normal(1.0, 1.0, 200))
    runner._ranker.lower[4] = list(rng.normal(0.0, 1.0, 200))
    _rank_twice(runner, num_gpus=2)

    assert runner._dynamics_cache._hot_replicas == [2, 3, 4, 5]


def test_rank_windows_fill_windows_stay_cold_candidates():
    """
    Validate that a window made hot only to fill the workers keeps counting
    as found cold, rather than as chosen.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)

    # The flagged pair 0 gives windows 0, 1 and 2, so one window is filled.
    _fill_ranking_data(runner, hard_pair=0)
    _rank_twice(runner, num_gpus=2)

    hot = runner._dynamics_cache._hot_replicas
    assert hot is not None and len(hot) == 4
    (filler,) = set(hot) - {0, 1, 2}
    assert runner._dynamics_cache._cold_votes[filler] == 2


def test_rank_windows_accounts_for_samples_collected():
    """
    Validate that, of two pairs with the same uncertainty per sample, only the
    one with fewer samples so far is made hot, since more sampling of the
    other would barely change the final free energy.
    """

    def rank(samples):
        runner = _ranking_runner(8, adaptive_min_samples=10)
        _fill_ranking_data(runner, hard_pair=-1)

        # Forward and reverse estimates that agree, so neither pair is flagged.
        rng = np.random.default_rng(11)
        for k in (1, 5):
            runner._ranker.upper[k] = list(rng.normal(1.0, 1.0, 200))
            runner._ranker.lower[k + 1] = list(rng.normal(0.0, 1.0, 200))

        for _ in range(2):
            _rank(runner, 2, window_samples=samples, cycles_left=1000)
        return runner._dynamics_cache._hot_replicas

    # With the same samples, both pairs are worth sampling, i.e. every window.
    assert rank([1000] * 8) is None

    # Pair 1 has been sampled more, so only pair 5 is.
    assert rank([1000, 4000, 4000, 1000, 1000, 1000, 1000, 1000]) == [4, 5, 6, 7]


def test_rank_windows_at_end_of_run():
    """
    Validate that a ranking with no cycles left keeps the hot replicas and the
    cold votes, so that an extended run carries on with them.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)
    runner._dynamics_cache._hot_replicas = [2, 3, 4, 5]
    runner._dynamics_cache._cold_votes = [1, 0, 0, 0, 0, 0, 0, 1]

    _fill_ranking_data(runner, hard_pair=-1)
    _rank(runner, 2, cycles_left=0)

    assert runner._dynamics_cache._hot_replicas == [2, 3, 4, 5]
    assert runner._dynamics_cache._cold_votes == [1, 0, 0, 0, 0, 0, 0, 1]


def _loop_schedule(num_cycles, cycles_per_checkpoint, mix):
    """
    The checkpoint cycles and the cycles that propagate every window, from
    stepping through the cycles with a running checkpoint threshold, kept
    exact so that it doesn't drift, with the same tolerance as the runner.
    """
    from fractions import Fraction

    checkpoints, full = [], []
    cycles_per_checkpoint = Fraction(cycles_per_checkpoint)
    tolerance = Fraction(1, 10**10)
    threshold = cycles_per_checkpoint
    for i in range(num_cycles):
        is_checkpoint = (i + 1) >= threshold - tolerance
        if is_checkpoint:
            checkpoints.append(i + 1)
            threshold += cycles_per_checkpoint
        if (i + 1) % mix == 0 or is_checkpoint or i == num_cycles - 1:
            full.append(i + 1)
    return checkpoints, full


def _full_cycle_share(first, num_cycles, cycles_per_checkpoint, mix):
    """
    The share of the cycles after 'first' that propagate every window.
    """
    _, full = _loop_schedule(num_cycles, cycles_per_checkpoint, mix)
    return sum(c > first for c in full) / (num_cycles - first)


SCHEDULES = [
    # Every checkpoint cycle is also a mix cycle.
    (100, 1000, 100, 4),
    (100, 1000, 10, 4),
    (0, 1000, 100, 1),
    # Not a whole number of cycles between checkpoints.
    (5, 1000, 2.5, 4),
    (100, 997, 33.3, 7),
    (0, 100000, 33.3, 3),
    # Multiples that are whole numbers, but not exactly so in binary.
    (0, 1000, 1.1, 4),
    # A checkpoint interval longer than the run.
    (0, 250, 250.0, 4),
    # Close to the end, where the final cycle matters.
    (990, 1000, 100, 4),
    (998, 1000, 100, 4),
]


@pytest.mark.parametrize("first, num_cycles, cycles_per_checkpoint, mix", SCHEDULES)
def test_checkpoint_cycles(first, num_cycles, cycles_per_checkpoint, mix):
    """
    Validate the checkpoint cycles against a running checkpoint threshold.
    """
    expected, _ = _loop_schedule(num_cycles, cycles_per_checkpoint, mix)
    cycles = RepexRunner._checkpoint_cycles(num_cycles, cycles_per_checkpoint)
    assert cycles.tolist() == expected


def test_checkpoint_cycles_without_cycles():
    """
    Validate that a run with no cycles, and so no checkpoint interval, has no
    checkpoints.
    """
    assert RepexRunner._checkpoint_cycles(0, 0.0).tolist() == []


def test_rank_cycles_across_restart():
    """
    Validate that windows ranked on their own schedule are ranked at the same
    simulation times whether or not the run is restarted.
    """
    # Ranked every 300 cycles over 1100 cycles: at 300, 600 and 900.
    assert RepexRunner._rank_cycles(1100, 0, 300).tolist() == [300, 600, 900]

    # Restarted after 500 cycles, the same rankings fall 100 and 400 cycles in.
    assert RepexRunner._rank_cycles(600, 500, 300).tolist() == [100, 400]


@pytest.mark.parametrize(
    "rank_frequency, error",
    [
        # Not a whole number of cycles.
        ("10.5 ps", "a multiple of 'energy_frequency'"),
        # Fewer cycles apart than 'adaptive_mix_interval'.
        ("2 ps", "at least 'adaptive_mix_interval' cycles"),
    ],
)
def test_rank_frequency_is_validated(rank_frequency, error):
    """
    Validate that a ranking interval that isn't a whole number of cycles, or
    is shorter than 'adaptive_mix_interval' cycles, is rejected.
    """
    import sire as sr

    runner = object.__new__(RepexRunner)
    runner._config = Config(
        runtime="5 ns",
        energy_frequency="1 ps",
        checkpoint_frequency="500 ps",
        adaptive_mix_interval=4,
        adaptive_rank_frequency=rank_frequency,
    )
    runner._start_time = sr.u("0 ps")

    with pytest.raises(ValueError, match=error):
        runner._check_adaptive_config()


@pytest.mark.parametrize("first, num_cycles, cycles_per_checkpoint, mix", SCHEDULES)
def test_cold_rate(first, num_cycles, cycles_per_checkpoint, mix):
    """
    Validate the share of cycles that propagate the cold windows against the
    run loop's schedule.
    """
    checkpoints = RepexRunner._checkpoint_cycles(num_cycles, cycles_per_checkpoint)
    assert cold_rate(first, num_cycles, checkpoints, mix) == pytest.approx(
        _full_cycle_share(first, num_cycles, cycles_per_checkpoint, mix)
    )


def test_rank_windows_without_gain():
    """
    Validate that every replica is made hot straight away when no subset is
    better than every replica being hot, rather than keeping the previous hot
    replicas for another ranking.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)
    runner._dynamics_cache._hot_replicas = [2, 3, 4, 5]
    runner._dynamics_cache._cold_votes = [1, 1, 0, 0, 0, 0, 1, 1]

    # Every pair has the same uncertainty.
    _fill_ranking_data(runner, hard_pair=-1)
    _rank(runner, num_gpus=2)

    assert runner._dynamics_cache._hot_replicas is None
    assert runner._dynamics_cache._cold_votes == [0] * 8


def test_rank_windows_without_uncertainty():
    """
    Validate that every replica stays hot when no window contributes any
    uncertainty, rather than none of them.
    """
    runner = _ranking_runner(4, adaptive_min_samples=10)

    for k in range(3):
        runner._ranker.upper[k] = [0.5] * 100
        runner._ranker.lower[k + 1] = [-0.5] * 100
    _rank(runner, num_gpus=1)

    assert runner._dynamics_cache._hot_replicas is None


def test_adaptive_data_is_restored():
    """
    Validate that the data used to rank the windows survives being stored in
    the dynamics cache and restored, as it is across a restart.
    """
    import pickle

    runner = _ranking_runner(8, adaptive_min_samples=10)
    _fill_ranking_data(runner, hard_pair=3)
    runner._ranker.distance[2] = [1.0, 2.0]

    expected = (runner._ranker.lower, runner._ranker.upper)
    data = pickle.loads(pickle.dumps(runner._ranker.to_dict()))

    zeros = np.zeros((8, 8))
    runner._ranker.reset(zeros, zeros)
    runner._ranker.restore(data)

    assert runner._ranker.lower == expected[0]
    assert runner._ranker.upper == expected[1]
    assert runner._ranker.distance[2] == [1.0, 2.0]


@pytest.mark.parametrize("burn_in, is_restored", [("0 ps", True), ("1 ns", False)])
def test_start_adaptive_restores_data(burn_in, is_restored):
    """
    Validate that a restart restores the data used to rank the windows from
    the dynamics cache, unless the burn-in has been raised beyond the current
    time, which makes every window hot again.
    """
    import sire as sr

    saved = _ranking_runner(8, adaptive_min_samples=10)
    _fill_ranking_data(saved, hard_pair=3)

    runner = _ranking_runner(8, adaptive_repex=True, adaptive_burn_in=burn_in)
    runner._ring_pair = None
    runner._is_cached = False
    runner._start_time = sr.u("100 ps")
    runner._dynamics_cache._adaptive_data = saved._ranker.to_dict()
    runner._dynamics_cache._hot_replicas = [2, 3, 4]
    runner._dynamics_cache._cold_votes = [1] * 8

    runner._start_adaptive()

    if is_restored:
        assert runner._ranker.upper == saved._ranker.upper
        assert runner._dynamics_cache._hot_replicas == [2, 3, 4]
    else:
        assert runner._ranker.upper == [[]] * 8
        assert runner._dynamics_cache._hot_replicas is None
        assert runner._dynamics_cache._cold_votes == [0] * 8


def test_rank_windows_needs_enough_samples():
    """
    Validate that windows aren't ranked until every one has enough samples,
    and that the data keeps accumulating until they do.
    """
    runner = _ranking_runner(8, adaptive_min_samples=50)

    _fill_ranking_data(runner, hard_pair=3, num_samples=20)
    _rank(runner, num_gpus=2)

    assert runner._dynamics_cache._hot_replicas is None
    assert len(runner._ranker.upper[0]) == 20


def test_rank_windows_keeps_undersampled_windows_hot():
    """
    Validate that a window with too few independent samples stays hot, even
    though its uncertainty is small.
    """
    runner = _ranking_runner(8, adaptive_min_samples=50)

    _fill_ranking_data(runner, hard_pair=3)

    # A slowly decorrelating series with a small spread.
    rng = np.random.default_rng(5)
    y = np.zeros(200)
    for i in range(1, len(y)):
        y[i] = 0.95 * y[i - 1] + rng.normal(0.0, 0.01)
    runner._ranker.upper[6] = list(0.5 + y)
    _rank_twice(runner, num_gpus=2)

    assert 6 in runner._dynamics_cache._hot_replicas


def test_rank_windows_keeps_slow_bond_distance_hot():
    """
    Validate that a window in which the distance of the bond that is broken
    decorrelates slowly stays hot, even though its energies are converged.
    """
    runner = _ranking_runner(8, is_ring_break=True, adaptive_min_samples=50)

    _fill_ranking_data(runner, hard_pair=3)

    rng = np.random.default_rng(9)
    for k in range(8):
        runner._ranker.distance[k] = list(rng.normal(1.5, 0.05, 200))

    y = np.zeros(200)
    for i in range(1, len(y)):
        y[i] = 0.95 * y[i - 1] + rng.normal()
    runner._ranker.distance[7] = list(10.0 + y)
    _rank_twice(runner, num_gpus=2)

    assert 7 in runner._dynamics_cache._hot_replicas
    assert 0 not in runner._dynamics_cache._hot_replicas


@pytest.mark.parametrize(
    "oversubscription_factor, capacity, num_hot",
    [
        (1, 1, 4),
        (2, 2, 8),
        # Two replicas per GPU at once, but only one context on each.
        (2, 1, 4),
    ],
)
def test_rank_windows_fills_whole_batches(oversubscription_factor, capacity, num_hot):
    """
    Validate that the hot replicas fill whole batches on every GPU, of as many
    replicas as the GPU can propagate at once.
    """
    runner = _ranking_runner(
        8, adaptive_min_samples=10, oversubscription_factor=oversubscription_factor
    )

    # The flagged pair 0 gives windows 0, 1 and 2, one on each of three GPUs.
    _fill_ranking_data(runner, hard_pair=0)
    _rank_twice(runner, num_gpus=4, capacity=capacity)

    hot = runner._dynamics_cache._hot_replicas
    hot = list(range(8)) if hot is None else hot
    assert len(hot) == num_hot
    assert {0, 1, 2}.issubset(hot)
    assert sorted(r % 4 for r in hot) == sorted(list(range(4)) * (num_hot // 4))


def test_rank_windows_unfilled_with_shared_contexts():
    """
    Validate that the hot replicas aren't filled when contexts are shared,
    since a window added on a busy context needs another pass.
    """
    runner = _ranking_runner(8, is_cached=True, adaptive_min_samples=10)

    # The flagged pair 0 gives windows 0, 1 and 2, one on each of three GPUs.
    _fill_ranking_data(runner, hard_pair=0)
    _rank_twice(runner, num_gpus=4)

    assert runner._dynamics_cache._hot_replicas == [0, 1, 2]


def test_run_waits_for_blocks_on_error(monkeypatch):
    """
    Validate that an error during a run waits for the blocks still running
    in the thread pools before it is raised.
    """
    import time

    runner = object.__new__(RepexRunner)
    runner._config = Config()
    runner._num_gpus = 1

    finished = []

    def fail(dynamics_executor, checkpoint_executor):
        dynamics_executor.submit(lambda: (time.sleep(0.2), finished.append(True)))
        raise RuntimeError("dynamics failed")

    monkeypatch.setattr(runner, "_run", fail)

    with pytest.raises(RuntimeError, match="dynamics failed"):
        runner.run()

    assert finished == [True]


@pytest.mark.parametrize("is_raised", [False, True])
def test_map_replicas_finishes_every_step(is_raised):
    """
    Validate that a failing step is only raised once every other step has
    finished, whether the step returns the error or raises it.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    runner = object.__new__(RepexRunner)
    runner._lambda_values = [0.0, 0.5, 1.0]

    finished = []

    def step(replica):
        if replica == 0:
            error = RuntimeError("checkpoint failed")
            if is_raised:
                raise error
            return replica, error
        time.sleep(0.2)
        finished.append(replica)
        return replica, None

    with ThreadPoolExecutor(max_workers=3) as executor:
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            runner._map_replicas(executor, "Checkpoint", step, [0, 1, 2])

        assert sorted(finished) == [1, 2]


def _layout(num_replicas, num_slots, num_gpus):
    """
    A dynamics cache holding only a strided layout of replicas over slots, and
    slots over GPUs.
    """
    from somd2.runner._dynamics_cache import DynamicsCache

    cache = object.__new__(DynamicsCache)
    cache._replica_slot = [r % num_slots for r in range(num_replicas)]
    cache._slot_devices = [s % num_gpus for s in range(num_slots)]
    return cache


@pytest.mark.parametrize(
    "num_slots, num_gpus, oversubscription_factor, expected",
    [
        (25, 3, 2, [2, 2, 2]),
        # One context per GPU, so only one replica at once.
        (4, 4, 2, [1, 1, 1, 1]),
        (5, 4, 2, [2, 1, 1, 1]),
    ],
)
def test_device_capacity(num_slots, num_gpus, oversubscription_factor, expected):
    """
    Validate that a GPU propagates no more replicas at once than it has
    contexts.
    """
    runner = object.__new__(RepexRunner)
    runner._config = Config(oversubscription_factor=oversubscription_factor)
    runner._num_gpus = num_gpus
    runner._dynamics_cache = _layout(num_slots, num_slots, num_gpus)

    assert runner._device_capacity() == expected


def test_replica_devices_with_shared_contexts():
    """
    Validate that a replica runs on the GPU of the context it shares.
    """
    runner = object.__new__(RepexRunner)
    runner._num_gpus = 2
    runner._num_replicas = 10
    runner._dynamics_cache = _layout(10, 4, 2)

    assert runner._replica_devices() == [(r % 4) % 2 for r in range(10)]


def test_device_batches():
    """
    Validate that a batch never has more replicas on a GPU than the
    oversubscription factor, so that hot replicas that aren't consecutive
    still spread over the GPUs.
    """
    runner = object.__new__(RepexRunner)
    runner._config = Config(num_lambda=25)
    runner._num_gpus = 3
    runner._num_replicas = 25
    runner._dynamics_cache = _layout(25, 25, 3)
    devices = runner._replica_devices()

    # Every replica is batched as consecutive groups of three.
    assert runner._device_batches(list(range(25)), devices) == [
        list(range(i, min(i + 3, 25))) for i in range(0, 25, 3)
    ]

    hot = [6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    batches = runner._device_batches(hot, devices)
    assert sorted(r for batch in batches for r in batch) == hot
    assert len(batches) == 5
    for batch in batches:
        assert len({r % 3 for r in batch}) == len(batch)


def _ring_pair_runner(system, virtual_sites=None):
    """
    Create a runner holding only what is needed to find the ring pair.
    """
    runner = object.__new__(RepexRunner)
    runner._system = system
    runner._virtual_sites = virtual_sites
    return runner


def test_find_ring_pair(syk_ring_break_mols):
    """
    Validate that the bond that is broken is found from the perturbable
    molecule, and isn't shifted by its own virtual sites, which come after its
    atoms, or by those of later molecules.
    """
    runner = _ring_pair_runner(syk_ring_break_mols)
    assert runner._find_ring_pair() == (1, 7)

    runner = _ring_pair_runner(syk_ring_break_mols, np.array([[0, 3], [100, 4]]))
    assert runner._find_ring_pair() == (1, 7)

    # Legacy restarts load a system per replica.
    runner = _ring_pair_runner([syk_ring_break_mols, syk_ring_break_mols])
    assert runner._find_ring_pair() == (1, 7)


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize(
    "rank_frequency, cycles_left",
    [
        # Ranked at each checkpoint, every 4 cycles, apart from the last.
        (None, [8, 4]),
        # Ranked every 2 cycles, from the end of the burn-in.
        ("8fs", [8, 6, 4, 2]),
    ],
)
def test_adaptive_repex_ranking_arguments(
    ethane_methanol, monkeypatch, rank_frequency, cycles_left
):
    """
    Validate that each ranking is given the samples each window has, the
    cycles left, and the share of those cycles that propagate every window,
    that swaps are only counted from the end of the burn-in, and that the
    windows are ranked on their own schedule if one is set.
    """
    num_lambda = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "48fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "16fs",
            "frame_frequency": "16fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "adaptive_repex": True,
            "adaptive_mix_interval": 2,
            "adaptive_burn_in": "8fs",
            "adaptive_rank_frequency": rank_frequency,
        }

        runner = RepexRunner(ethane_methanol, Config(**config))

        calls = []
        baselines = []

        def rank_windows(*args):
            calls.append(args)
            baselines.append(runner._ranker.proposed.sum())

        monkeypatch.setattr(runner, "_rank_windows", rank_windows)
        runner.run()

    # 12 cycles with a checkpoint every 4, and every window always hot.
    assert [c[3] for c in calls] == cycles_left

    # The swaps proposed in the two burn-in cycles aren't counted.
    assert baselines[0] > 0
    assert len(set(baselines)) == 1
    for devices, capacity, samples, cycles_left, cold_rate in calls:
        # A context per replica, assigned to the GPUs in turn.
        assert devices == [r % runner._num_gpus for r in range(num_lambda)]
        assert capacity == [1] * runner._num_gpus
        first = 12 - cycles_left
        assert samples == [first] * num_lambda
        expected = _full_cycle_share(first, 12, 4, 2)
        assert cold_rate == pytest.approx(expected)


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_adaptive_repex_frames_every_cycle(ethane_methanol, monkeypatch):
    """
    Validate that frequent frames don't stop cold replicas being skipped, and
    that a cold replica saves the frame it missed when it is next propagated.
    """
    import pandas as pd
    import sire as sr

    num_lambda = 4
    hot = [0, 1]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "48fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "16fs",
            "frame_frequency": "4fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "adaptive_repex": True,
            "adaptive_mix_interval": 2,
            "adaptive_burn_in": "0 ps",
        }

        # Never rank, so that the hot replicas stay as set.
        monkeypatch.setattr(RepexRunner, "_rank_windows", lambda *args: None)

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner._dynamics_cache._hot_replicas = hot
        runner.run()

        # Cold replicas only run on even cycles, saving a frame each time.
        for i in range(num_lambda):
            expected = 12 if i in hot else 6

            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            assert len(records) == expected, f"replica {i}"

            mols = sr.load(
                [runner._filenames["topology0"], runner._filenames[i]["trajectory"]]
            )
            assert mols.num_frames() == expected, f"replica {i}"


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_adaptive_repex_records_bond_distance(syk_ring_break_mols):
    """
    Validate that the distance of the bond that is broken is recorded in each
    window during an adaptive replica exchange simulation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "40fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "40fs",
            "frame_frequency": "40fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": 3,
            "replica_exchange": True,
            "lambda_schedule": "ring_break_morph",
            "adaptive_repex": True,
            "adaptive_burn_in": "0 ps",
        }

        runner = RepexRunner(syk_ring_break_mols, Config(**config))
        assert runner._ring_pair == (1, 7)
        runner.run()

        for distances in runner._ranker.distance:
            assert len(distances) == 10

        # The S-C bond is intact at lambda = 0.
        assert np.mean(runner._ranker.distance[0]) == pytest.approx(1.8, abs=0.2)


def test_find_ring_pair_needs_a_single_bond(syk_ring_break_mols):
    """
    Validate that the distance isn't used when more than one bond is broken or
    made.
    """
    mols = syk_ring_break_mols.clone()
    mol = mols["property is_perturbable"].molecules()[0]
    mol = mol.edit().set_property("ring_making_bonds", [2, 8]).molecule().commit()
    mols.update(mol)

    assert _ring_pair_runner(mols)._find_ring_pair() is None


def test_rank_windows_flags_unconverged_pairs():
    """
    Validate that a pair whose forward and reverse estimates disagree is made
    hot, even though its uncertainty is small.
    """
    runner = _ranking_runner(8, adaptive_min_samples=10)

    _fill_ranking_data(runner, hard_pair=3)
    rng = np.random.default_rng(3)
    runner._ranker.lower[1] = list(rng.normal(-3.0, 0.1, 200))
    _rank_twice(runner, num_gpus=2)

    hot = runner._dynamics_cache._hot_replicas
    assert hot is not None and {0, 1, 2}.issubset(hot)


def test_checkpoint_without_adaptive_state():
    """
    Validate that a checkpoint written before adaptive replica exchange was
    added restores with every replica hot and no clocks of its own.
    """
    from somd2.runner._dynamics_cache import DynamicsCache
    from tests.runner.test_gcmc_stats import TestLegacyRepexCheckpoint

    cache = object.__new__(DynamicsCache)
    cache.__setstate__(TestLegacyRepexCheckpoint.make_state(None))

    assert cache._hot_replicas is None
    assert cache._clocks == [None] * 3
    assert cache._cold_votes == [0] * 3
    assert cache._adaptive_data is None
    assert cache._strided is False


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
@pytest.mark.parametrize("max_contexts", [None, 2])
def test_adaptive_repex_hot_replicas(ethane_methanol, max_contexts, monkeypatch):
    """
    Validate that only the hot replicas are propagated between full cycles,
    that each replica records energies against its own clock without gaps or
    repeats, that frames are saved for every replica on the same cycles, that
    a restart carries on with the same hot replicas, that a restart can't
    change the energy frequency, and that raising the burn-in on restart makes
    every replica hot again.
    """
    import pandas as pd
    import sire as sr

    num_lambda = 4
    hot = [0, 1]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "48fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "16fs",
            "frame_frequency": "16fs",
            "platform": "cuda",
            "max_threads": 1,
            "num_lambda": num_lambda,
            "replica_exchange": True,
            "max_contexts": max_contexts,
            "adaptive_repex": True,
            "adaptive_mix_interval": 2,
            "adaptive_burn_in": "0 ps",
        }

        # Never rank, so that the hot replicas stay as set.
        monkeypatch.setattr(RepexRunner, "_rank_windows", lambda *args: None)

        runner = RepexRunner(ethane_methanol, Config(**config))

        # Not a ring-breaking perturbation, so there is no bond distance.
        assert runner._ring_pair is None

        runner._dynamics_cache._hot_replicas = hot
        runner.run()

        # Hot replicas run every cycle. The others only run on even cycles,
        # which are every 'adaptive_mix_interval', or a checkpoint.
        def expected_times(num_hot, num_cold, r):
            num = num_hot if r in hot else num_cold
            return [round(0.004 * (k + 1), 6) for k in range(num)]

        for i in range(num_lambda):
            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == expected_times(12, 6, i), f"replica {i}: {times}"

            # A frame on every checkpoint cycle, for every replica.
            mols = sr.load(
                [runner._filenames["topology0"], runner._filenames[i]["trajectory"]]
            )
            assert mols.num_frames() == 3

        assert runner._dynamics_cache._time.to("fs") == pytest.approx(48)

        # Restart, extending the runtime.
        config["runtime"] = "64fs"
        config["restart"] = True

        runner = RepexRunner(ethane_methanol, Config(**config))
        assert runner._dynamics_cache._hot_replicas == hot
        runner.run()

        for i in range(num_lambda):
            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == expected_times(16, 8, i), f"replica {i}: {times}"

        # The sample counts depend on the energy frequency.
        config["runtime"] = "80fs"
        changed = {**config, "energy_frequency": "8fs"}
        with pytest.raises(ValueError, match="'energy_frequency' can't be changed"):
            RepexRunner(ethane_methanol, Config(**changed))

        # Restart with a burn-in that hasn't been reached, so that every
        # replica is propagated every cycle.
        config["adaptive_burn_in"] = "1 ns"

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()

        assert runner._dynamics_cache._hot_replicas is None

        for i in range(num_lambda):
            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == expected_times(20, 12, i), f"replica {i}: {times}"

        # Restart with adaptive replica exchange switched off.
        config["runtime"] = "96fs"
        config["adaptive_repex"] = False

        runner = RepexRunner(ethane_methanol, Config(**config))
        runner.run()

        assert runner._dynamics_cache._hot_replicas is None

        for i in range(num_lambda):
            records = pd.read_parquet(runner._filenames[i]["energy_traj"])
            times = [round(t, 6) for t in records.index.get_level_values(0)]
            assert times == expected_times(24, 16, i), f"replica {i}: {times}"

            # Frames are scheduled by each replica's own clock again.
            mols = sr.load(
                [runner._filenames["topology0"], runner._filenames[i]["trajectory"]]
            )
            assert mols.num_frames() == 6
