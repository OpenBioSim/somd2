"""
Tests for GCMC sampling via the regular (non replica exchange) runner.
"""

import pytest
import re
import tempfile

from pathlib import Path

from somd2.config import Config
from somd2.runner import Runner

from tests.conftest import has_cuda


@pytest.mark.skipif(not has_cuda, reason="CUDA not available.")
def test_runner_gcmc_without_a_selection(ethane_methanol):
    """
    Validate GCMC sampling with no 'gcmc_selection', where moves are attempted
    within the entire simulation box rather than a region around a selection.

    The sampler then has no reference, so it cannot count the waters within a
    region. Reporting the water count has to account for that, which is what
    this exercises: counting it raised before there was a path for the
    reference-free case.
    """
    pytest.importorskip("loch")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            runtime="8fs",
            output_directory=tmpdir,
            energy_frequency="4fs",
            checkpoint_frequency="4fs",
            frame_frequency="4fs",
            platform="cuda",
            max_threads=1,
            num_lambda=2,
            gcmc=True,
            gcmc_frequency="4fs",
        )

        # The bulk-only path is the point of this test, so make sure it can't
        # stop being exercised without the test failing.
        assert config.gcmc_selection is None

        runner = Runner(ethane_methanol, config)
        runner.run()

        # GCMC ran, so the ghost residues were written.
        for lam in runner._lambda_values:
            assert (Path(tmpdir) / f"gcmc_ghosts_{lam:.5f}.txt").exists()

        # With no region the count is the number of non-ghost waters in the
        # box, which is never zero for a solvated system.
        log = (Path(tmpdir) / config.log_file).read_text()
        counts = [
            int(x)
            for x in re.findall(r"number of waters in GCMC volume.*? is (\d+)", log)
        ]
        assert counts, "no water count was logged"
        assert all(count > 0 for count in counts), f"zero water count logged: {counts}"
