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
    # Test that the logfile is only created once a runner is initialised, not
    # by the config alone - this is deferred so that a user can change
    # output_directory after constructing a Config (e.g. via the Python API)
    # without leaving behind a stale directory/duplicate log sink from the
    # default value.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))
        from pathlib import Path

        # A config object alone should not create the logfile.
        config = Config(output_directory=tmpdir, log_file="test.log")
        assert config.log_file is not None
        assert not Path.exists(config.output_directory / config.log_file)

        # Test that a logfile is created once a runner object is initialised.
        runner = Runner(mols, Config(output_directory=tmpdir, log_file="test1.log"))
        assert runner._config.log_file is not None
        assert Path.exists(runner._config.output_directory / runner._config.log_file)

        somd2._logger.remove()


def test_morse_restraint_options():
    """Validate that the Morse restraint options are parsed correctly."""
    import math

    import pytest

    # The defaults are parsed as Sire units.
    config = Config()
    assert config.morse_hard_well_depth == sr.u("150 kcal mol-1")
    assert config.morse_soft_well_depth == sr.u("50 kcal mol-1")
    assert config.morse_soft_force_constant == sr.u("125 kcal mol-1 A-2")

    # Equivalent units are accepted, and converted.
    config = Config(morse_hard_well_depth="418.4 kJ mol-1")
    assert math.isclose(
        config.morse_hard_well_depth.to(sr.units.kcal_per_mol), 100.0, rel_tol=1e-6
    )

    # Well depths must be energies.
    for option in ("morse_hard_well_depth", "morse_soft_well_depth"):
        with pytest.raises(TypeError):
            Config(**{option: 150})

        with pytest.raises(ValueError, match="Unable to parse"):
            Config(**{option: "not a unit"})

        with pytest.raises(ValueError, match="units are invalid"):
            Config(**{option: "150 kcal mol-1 A-2"})

    # The force constant must be an energy per unit area.
    with pytest.raises(TypeError):
        Config(morse_soft_force_constant=125)

    with pytest.raises(ValueError, match="Unable to parse"):
        Config(morse_soft_force_constant="not a unit")

    with pytest.raises(ValueError, match="units are invalid"):
        Config(morse_soft_force_constant="125 kcal mol-1")
