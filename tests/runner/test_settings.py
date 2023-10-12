from somd2.runner import Controller
import sire as sr
import pytest


def test_config():
    # Testing that default options are set correctly
    mols = sr.stream.load("Methane_Ethane_solv.bss")
    for mol in mols.molecules("molecule property is_perturbable"):
        mols.update(mol.perturbation().link_to_reference().commit())

    runner = Controller(mols)
    runner.configure({})
    runner._initialise_simulation(runner._system.clone(), 0.0)
    runner._sim._setup_dynamics(equilibration=True)
    config_inp = runner.config
    d = runner._sim._dyn
    assert config_inp.equilibration_timestep == d.timestep()
    runner._sim._setup_dynamics(equilibration=False)
    d = runner._sim._dyn
    config_inp = runner.config
    assert config_inp.timestep == d.timestep()
    assert config_inp.temperature == d.ensemble().temperature()
    assert config_inp.pressure == d.ensemble().pressure()
    assert config_inp.lambda_schedule.to_string() == d.get_schedule().to_string()
    assert config_inp.cutoff_type == d.info().cutoff_type()
    assert config_inp.platform == d.platform()
