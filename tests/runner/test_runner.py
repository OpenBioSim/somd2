from pathlib import Path

import pytest
import tempfile

from somd2.runner import Runner
from somd2.config import Config


def test_failed_window(ethane_methanol, monkeypatch):
    """
    Validate that a failed window doesn't stop the others from running, and
    that the run raises an error naming the failed window.
    """
    import concurrent.futures

    # Run windows in threads so that the patched _run is used.
    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        lambda max_workers, mp_context: concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ),
    )

    run = Runner._run

    def failing_run(self, system, index, **kwargs):
        if index == 0:
            raise RuntimeError("Forced failure")
        return run(self, system, index, **kwargs)

    monkeypatch.setattr(Runner, "_run", failing_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "4fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        runner = Runner(ethane_methanol, Config(**config))
        with pytest.raises(RuntimeError, match="Failed for λ = 0.0"):
            runner.run()

        assert not (Path(tmpdir) / "energy_traj_0.00000.parquet").exists()
        assert (Path(tmpdir) / "energy_traj_1.00000.parquet").exists()
