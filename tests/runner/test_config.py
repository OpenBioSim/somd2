import sire as sr
import tempfile

from somd2.config import Config
from somd2.runner import Runner


def test_config():
    """Validate that default options are set correctly."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the demo stream file.
        mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))

        # Instantiate a runner using the default config.
        runner = Runner(mols, Config())

        # Initalise a fake simulation.
        runner._initialise_simulation(runner._system.clone(), 0.0)

        # Setup a dynamics object for equilibration.
        runner._sim._setup_dynamics(equilibration=True)

        # Store the config object.
        config_inp = runner._config

        # Store the dynamics object.
        d = runner._sim._dyn

        assert config_inp.equilibration_timestep == d.timestep()

        # Setup a dynamics object for production.
        runner._sim._setup_dynamics(equilibration=False)

        # Store the dynamics object.
        d = runner._sim._dyn

        assert str(config_inp.timestep).lower() == str(d.timestep()).lower()
        assert (
            str(config_inp.temperature).lower()
            == str(d.ensemble().temperature()).lower()
        )
        assert str(config_inp.pressure).lower() == str(d.ensemble().pressure()).lower()
        assert (
            config_inp.lambda_schedule.to_string().lower()
            == d.get_schedule().to_string().lower()
        )
        assert config_inp.cutoff_type.lower() == d.info().cutoff_type().lower()
        assert config_inp.platform.lower() == d.platform().lower()
