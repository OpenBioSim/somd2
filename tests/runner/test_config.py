import pytest
import tempfile

import sire as sr

import somd2

from somd2.config import Config
from somd2.runner import Runner


def test_dynamics_options():
    """Validate that dynamics options are set correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))

        # Create a config object.
        config = Config(platform="cpu", output_directory=tmpdir)

        # Instantiate a runner using the default config.
        # (All default options, other than platform="cpu".)
        runner = Runner(mols, config)

        # Initalise a fake simulation.
        d = runner._system.dynamics(**runner._dynamics_kwargs)

        # Timestep.
        assert str(config.timestep).lower() == str(d.timestep()).lower()

        # Schedule.
        assert (
            config.lambda_schedule.to_string().lower()
            == d.get_schedule().to_string().lower()
        )

        # Cutoff-type.
        assert config.cutoff_type.lower() == d.info().cutoff_type().lower()

        # Platform.
        assert config.platform.lower() == d.platform().lower()

        # Temperature and pressure.
        if not d.ensemble().is_micro_canonical():
            assert (
                str(config.temperature).lower()
                == str(d.ensemble().temperature()).lower()
            )
            assert str(config.pressure).lower() == str(d.ensemble().pressure()).lower()

        # Constraint.
        assert config.constraint.lower() == d.constraint().lower()

        # Perturbable_constraint.
        assert (
            config.perturbable_constraint.lower() == d.perturbable_constraint().lower()
        )

        # Integrator.
        assert config.integrator.lower().replace(
            "_", ""
        ) == d.integrator().__class__.__name__.lower().replace("integrator", "")


def test_logfile_creation():
    # Test that the logfile is created by either the initialisation of the runner or of a config
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))
        from pathlib import Path

        # Test that a logfile is created once a config object is initialised
        config = Config(output_directory=tmpdir, log_file="test.log")
        assert config.log_file is not None
        assert Path.exists(config.output_directory / config.log_file)

        # Test that a logfile is created once a runner object is initialised
        runner = Runner(mols, Config(output_directory=tmpdir, log_file="test1.log"))
        assert runner._config.log_file is not None
        assert Path.exists(runner._config.output_directory / runner._config.log_file)

        somd2._logger.remove()
