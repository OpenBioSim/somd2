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


def test_lambda_schedule_input_forms():
    """Validate that all supported lambda schedule input forms are accepted."""
    import os

    import pytest

    schedule = sr.cas.LambdaSchedule.standard_morph()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schedule.s3")
        sr.stream.save(schedule, path)

        # A named schedule, which is case insensitive.
        config = Config(lambda_schedule="DECOUPLE")
        assert config._lambda_schedule_name == "decouple"

        # The path to a stream file.
        config = Config(lambda_schedule=path)
        assert isinstance(config.lambda_schedule, sr.cas.LambdaSchedule)
        assert config._lambda_schedule_name is None

        # A hex string of the serialised object.
        config = Config(lambda_schedule=Config._to_hex(schedule))
        assert isinstance(config.lambda_schedule, sr.cas.LambdaSchedule)

        # The object itself.
        config = Config(lambda_schedule=schedule)
        assert isinstance(config.lambda_schedule, sr.cas.LambdaSchedule)

        # Anything else is rejected.
        with pytest.raises(ValueError, match="Unable to interpret"):
            Config(lambda_schedule="not_a_schedule")

        # A stream file holding the wrong type of object.
        wrong_path = os.path.join(tmpdir, "wrong.s3")
        sr.stream.save(sr.cas.Symbol("x"), wrong_path)
        with pytest.raises(ValueError, match="not a 'LambdaSchedule'"):
            Config(lambda_schedule=wrong_path)


def test_restraints_input_forms():
    """Validate that all supported restraint input forms are accepted."""
    import os

    import pytest

    mols = sr.load_test_files("ala.top", "ala.crd")
    restraint0 = sr.restraints.positional(mols, atoms="atomidx 0")
    restraint1 = sr.restraints.positional(mols, atoms="atomidx 1")

    with tempfile.TemporaryDirectory() as tmpdir:
        path0 = os.path.join(tmpdir, "restraint0.s3")
        both_path = os.path.join(tmpdir, "both.s3")
        sr.stream.save(restraint0, path0)
        sr.stream.save([restraint0, restraint1], both_path)

        # A single object, or a list of objects.
        assert len(Config(restraints=restraint0).restraints) == 1
        assert len(Config(restraints=[restraint0, restraint1]).restraints) == 2

        # The path to a stream file, or a list of paths.
        assert len(Config(restraints=path0).restraints) == 1
        assert len(Config(restraints=[path0, path0]).restraints) == 2

        # A stream file holding a list of sets of restraints.
        assert len(Config(restraints=both_path).restraints) == 2

        # A hex string of the serialised object.
        assert len(Config(restraints=Config._to_hex(restraint0)).restraints) == 1

        # Objects and paths can be mixed, and all are retained.
        config = Config(restraints=[restraint0, path0])
        assert len(config.restraints) == 2
        assert all(
            isinstance(restraint, sr.mm._MM.Restraints)
            for restraint in config.restraints
        )

        # Anything else is rejected.
        with pytest.raises(ValueError, match="Unable to interpret"):
            Config(restraints="not_a_restraint")

        # A stream file holding the wrong type of object.
        wrong_path = os.path.join(tmpdir, "wrong.s3")
        sr.stream.save(sr.cas.LambdaSchedule.standard_morph(), wrong_path)
        with pytest.raises(ValueError, match="must be a sire.mm._MM.Restraints"):
            Config(restraints=wrong_path)


def test_help_text_scraping():
    """Validate that help text isn't truncated by the parameter name."""
    parser = Config._create_parser()

    for action in parser._actions:
        if action.dest == "restraints":
            break

    # The description wraps onto a line starting with the parameter name, which
    # must not be mistaken for the start of the next parameter.
    assert "applied to the atoms" in action.help
    assert "a list of sets" in action.help
