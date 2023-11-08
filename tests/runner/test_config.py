import tempfile
import sire as sr

from somd2.config import Config
from somd2.runner import Runner

import platform
import pytest


def test_dynamics_options():
    """Validate that dynamics options are set correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))

        # Instantiate a runner using the default config.
        # (All default options, other than platform="cpu".)
        runner = Runner(mols, Config(platform="cpu", output_directory=tmpdir))

        # Initalise a fake simulation.
        runner._initialise_simulation(runner._system.clone(), 0.0)

        # Setup a dynamics object for equilibration.
        runner._sim._setup_dynamics(equilibration=True)

        # Store the config object.
        config_inp = runner._config

        # Store the dynamics object.
        d = runner._sim._dyn

        # Timestep.
        assert config_inp.equilibration_timestep == d.timestep()

        # Setup a dynamics object for production.
        runner._sim._setup_dynamics(equilibration=False)

        # Store the dynamics object.
        d = runner._sim._dyn

        # Timestep.
        assert str(config_inp.timestep).lower() == str(d.timestep()).lower()

        # Schedule.
        assert (
            config_inp.lambda_schedule.to_string().lower()
            == d.get_schedule().to_string().lower()
        )

        # Cutoff-type.
        assert config_inp.cutoff_type.lower() == d.info().cutoff_type().lower()

        # Platform.
        assert config_inp.platform.lower() == d.platform().lower()

        # Temperature and pressure.
        if not d.ensemble().is_micro_canonical():
            assert (
                str(config_inp.temperature).lower()
                == str(d.ensemble().temperature()).lower()
            )
            assert (
                str(config_inp.pressure).lower() == str(d.ensemble().pressure()).lower()
            )

        # Constraint.
        assert config_inp.constraint.lower() == d.constraint().lower()

        # Integrator.
        assert config_inp.integrator.lower().replace(
            "_", ""
        ) == d.integrator().__class__.__name__.lower().replace("integrator", "")


# Skip on windows due to file permission issues
# @pytest.mark.skipif(
#    platform.system() == "Windows", reason="File permission issues on Windows"
# )
def test_logfile_creation():
    # Test that the logfile is created by either the initialisation of the runner or of a config
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))
        from pathlib import Path

        # Instantiate a runner using the default config.
        # (All default options, other than platform="cpu".)
        config = Config(output_directory=tmpdir, log_file="test.log")
        assert config.log_file is not None
        assert Path.exists(config.output_directory / config.log_file)
        """
        # Instantiate a runner using the default config.
        # (All default options, other than platform="cpu".)
        runner = Runner(mols, Config(output_directory=tmpdir, log_file="test1.log"))
        assert runner._config.log_file is not None
        assert Path.exists(runner._config.output_directory / runner._config.log_file)
        """
