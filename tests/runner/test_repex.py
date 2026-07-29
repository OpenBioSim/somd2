from pathlib import Path

import numpy as np
import pytest
import tempfile

from somd2.runner import RepexRunner
from somd2.runner._base import RunnerBase
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
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        full = run(num_lambda, tmpdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        cached = run(1, tmpdir)

    for i in range(num_lambda):
        assert len(cached[i]) == len(full[i])
        assert list(cached[i].columns) == list(full[i].columns)
        assert cached[i].index.equals(full[i].index)


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

        for i in range(num_lambda):
            extended = pd.read_parquet(runner._filenames[i]["energy_traj"])
            assert len(extended) > num_rows[i]


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

    # Two cycles, each taking the lock once for the checkpoint files and once
    # for the repex state, plus a final acquisition. This must not scale with
    # the number of passes.
    assert len(acquisitions) == 5


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


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_repex_concurrent_slots(ethane_methanol):
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
            "max_contexts": 2,
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

        # Every batch must contain at most one replica per slot.
        num_workers = runner._num_gpus * config["oversubscription_factor"]
        for batch in runner._safe_batches(num_workers):
            slots = [runner._dynamics_cache.slot_for(r) for r in batch]
            assert len(slots) == len(set(slots)), f"batch {batch} shares a slot"

        runner.run()

        assert (Path(tmpdir) / "repex_matrix.txt").exists()


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
    from somd2.runner._repex import DynamicsCache

    cache = object.__new__(DynamicsCache)
    cache._gpu_devices = gpu_devices

    assert [cache._physical_device(i) for i in range(len(expected))] == expected


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

    from somd2.runner._repex import DynamicsCache

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
    from somd2.runner._repex import DynamicsCache

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
    # last mix applied.
    assert cache._openmm_states == ["state2", "state0", "state1", "state3"]
    assert cache._gcmc_states == ["water2", "water0", "water1", "water3"]

    # Every replica is seeded from its stored state on a restart.
    assert cache._state_moved == [True] * n

    # Attributes postdating the checkpoint are defaulted, one slot per replica.
    assert cache._num_slots == n
    assert cache._groups == [[0], [1], [2], [3]]
    assert cache._energy_trajectories == [None] * n
    assert cache._ghost_files == [None] * n


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
def test_repex_perturbed_system_seeding(ethane_methanol):
    """
    Validate that the end states are seeded from the right coordinates when
    contexts are shared.

    A context is created from a single system and every replica it hosts starts
    from that context, so the end state a replica starts from is chosen from
    the middle of the group rather than its first replica. That keeps any
    mismatch next to the lambda value at which the end state switches, instead
    of it depending on where the groups happen to fall.
    """
    import sire as sr

    # A perturbed end state, displaced so that its coordinates are distinct.
    perturbed = ethane_methanol.clone()
    perturbed.set_property("space", ethane_methanol.property("space"))
    coords = sr.io.get_coords_array(ethane_methanol)
    from sire.legacy.IO import setCoordinates

    perturbed = sr.system.System(
        setCoordinates(perturbed._system, (coords + 1.0).tolist())
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

        # The end states themselves must always be right.
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
                i for i in group if (seeded[i] == "perturbed") != (lambdas[i] > 0.5)
            ]
            assert len(wrong) <= len(group) // 2, (
                f"group {group} has {len(wrong)} replicas seeded from the "
                f"wrong end state: {wrong}"
            )
