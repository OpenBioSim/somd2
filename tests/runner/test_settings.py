import pytest
import sire as sr

from somd2.config import Config
from somd2.runner import Runner


def test_config():
    # Testing that default options are set correctly
    runner = Runner("ethane_methanol.bss", Config())
    runner._initialise_simulation(runner._system.clone(), 0.0)
    runner._sim._setup_dynamics(equilibration=True)
    config_inp = runner._config
    d = runner._sim._dyn
    assert config_inp.equilibration_timestep == d.timestep()
    runner._sim._setup_dynamics(equilibration=False)
    d = runner._sim._dyn
    config_inp = runner._config
    assert str(config_inp.timestep).lower() == str(d.timestep()).lower()
    assert (
        str(config_inp.temperature).lower() == str(d.ensemble().temperature()).lower()
    )
    assert str(config_inp.pressure).lower() == str(d.ensemble().pressure()).lower()
    assert (
        config_inp.lambda_schedule.to_string().lower()
        == d.get_schedule().to_string().lower()
    )
    assert config_inp.cutoff_type.lower() == d.info().cutoff_type().lower()
    assert config_inp.platform.lower() == d.platform().lower()
