from pathlib import Path
import tempfile

import pyarrow.parquet as pq
import pytest

from somd2.runner import Runner, RepexRunner
from somd2.config import Config

from tests.conftest import has_cuda


def _config(tmpdir, runtime, platform, repex, restart=False):
    return Config(
        runtime=runtime,
        restart=restart,
        output_directory=tmpdir,
        energy_frequency="4fs",
        checkpoint_frequency="12fs",
        frame_frequency="12fs",
        platform=platform,
        max_threads=1,
        num_lambda=2,
        replica_exchange=repex,
        save_energy_components=True,
    )


def _times(tmpdir, lam):
    table = pq.read_table(Path(tmpdir) / f"energy_components_{lam}.parquet")
    return table.column("time").to_pylist()


@pytest.mark.parametrize(
    "runner_class, platform",
    [
        (Runner, "CPU"),
        pytest.param(
            RepexRunner,
            "cuda",
            marks=pytest.mark.skipif(not has_cuda, reason="CUDA not available."),
        ),
    ],
)
def test_energy_components_rows(ethane_methanol, runner_class, platform):
    """
    Validate that the energy components file holds one row per energy save,
    including across a restart.
    """
    repex = runner_class is RepexRunner
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = runner_class(ethane_methanol, _config(tmpdir, "12fs", platform, repex))
        runner.run()

        for lam in ("0.00000", "1.00000"):
            times = _times(tmpdir, lam)
            assert len(times) == 3
            assert times == sorted(set(times))

        runner = runner_class(
            ethane_methanol, _config(tmpdir, "24fs", platform, repex, restart=True)
        )
        runner.run()

        for lam in ("0.00000", "1.00000"):
            times = _times(tmpdir, lam)
            assert len(times) == 6
            assert times == sorted(set(times))


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_energy_components_written_at_checkpoint(ethane_methanol):
    """
    Validate that energy components are buffered between checkpoints rather
    than written on every energy save.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = RepexRunner(ethane_methanol, _config(tmpdir, "24fs", "cuda", True))

        saves = []
        flushes = []
        save = runner._save_energy_components
        flush = runner._flush_energy_components

        def counting_save(index, context, time_ns):
            saves.append(index)
            return save(index, context, time_ns)

        def counting_flush(index):
            flushes.append(index)
            return flush(index)

        runner._save_energy_components = counting_save
        runner._flush_energy_components = counting_flush

        runner.run()

        # Six energy saves and two checkpoints per replica.
        assert saves.count(0) == 6
        assert flushes.count(0) == 2
        assert len(_times(tmpdir, "0.00000")) == 6


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_energy_components_buffer_limit(ethane_methanol):
    """
    Validate that the buffer is written out when it reaches its size limit,
    so that memory is bounded when checkpoints are rare.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = RepexRunner(ethane_methanol, _config(tmpdir, "24fs", "cuda", True))
        runner._max_ec_rows = 2

        flushes = []
        flush = runner._flush_energy_components

        def counting_flush(index):
            flushes.append(index)
            return flush(index)

        runner._flush_energy_components = counting_flush

        runner.run()

        # The limit is reached after the second and fifth saves, since the
        # checkpoint after the third empties the buffer, plus one flush at each
        # of the two checkpoints.
        assert flushes.count(0) == 4
        times = _times(tmpdir, "0.00000")
        assert len(times) == 6
        assert times == sorted(set(times))
