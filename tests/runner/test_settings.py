from somd2.runner import Controller
from sire import stream
from sire import u
import tempfile
import pytest


def test_setup():
    temp_dir = tempfile.TemporaryDirectory()
    simulation_options = {
        "no bookkeeping time": "1ps",
        "runtime": "1ns",
        "frame frequency": "0.5ps",
        "energy frequency": "0.1ps",
        "pressure": "1atm",
        "output directory": temp_dir.name,
    }
    options_units = [
        "no bookkeeping time",
        "runtime",
        "frame frequency",
        "energy frequency",
        "pressure",
    ]
    system = stream.load("./src/somd2/tests/test_systems/merged_molecule.s3")
    runner = Controller(system, platform="CPU", num_lambda=11)
    runner.create_sim_options(simulation_options)
    options = runner.get_options()
    for key in simulation_options.keys():
        if key in options_units:
            try:
                assert options[key] == u(simulation_options[key])
            except AssertionError:
                print(f"{options[key]} != {simulation_options[key]}")
                raise

        else:
            assert options[key] == simulation_options[key]
    temp_dir.cleanup()


def test_run():
    temp_dir = tempfile.TemporaryDirectory()
    simulation_options = {
        "no bookkeeping time": "1ps",
        "runtime": "1ps",
        "frame frequency": "0.5ps",
        "energy frequency": "0.1ps",
        "pressure": "1atm",
        "output directory": temp_dir.name,
    }
    system = stream.load("./src/somd2/tests/test_systems/merged_molecule.s3")
    runner = Controller(system, platform="CPU", num_lambda=2)
    runner.create_sim_options(simulation_options)
    runner.run_simulations()
    temp_dir.cleanup()
