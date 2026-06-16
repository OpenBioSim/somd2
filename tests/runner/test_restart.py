from pathlib import Path

import tempfile
import pytest

import sire as sr

from somd2.runner import Runner
from somd2.config import Config
from somd2.io import *


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_hmr"])
def test_restart(mols, request):
    """
    Validate that a simulation can be run from a checkpoint file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mols = request.getfixturevalue(mols)

        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        # Instantiate a runner using the config defined above.
        runner = Runner(mols, Config(**config))

        # Run the simulation.
        runner.run()

        # Load the energy trajectory.
        energy_traj_1, meta_1 = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )

        num_entries = len(energy_traj_1.index)

        # Load the trajectory.
        traj_1 = sr.load(
            [str(Path(tmpdir) / "system0.prm7"), str(Path(tmpdir) / "traj_0.00000.dcd")]
        )

        # Check that the compact numpy checkpoint file was written.
        import numpy as np

        checkpoint_state = np.load(str(Path(tmpdir) / "checkpoint_0.00000.npz"))
        assert "positions" in checkpoint_state
        assert "velocities" in checkpoint_state
        assert "time_ps" in checkpoint_state

        del runner

        config_new = {
            "runtime": "24fs",
            "restart": True,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
            "overwrite": True,
            "log_level": "DEBUG",
        }

        runner2 = Runner(mols, Config(**config_new))

        # Run the simulation.
        runner2.run()

        # Load the energy trajectory.
        energy_traj_2, meta_2 = parquet_to_dataframe(
            Path(tmpdir) / "energy_traj_0.00000.parquet"
        )

        # Check that first half of energy trajectory is the same
        assert energy_traj_1.equals(energy_traj_2.iloc[:num_entries])

        # Check that second energy trajectory is twice as long as the first
        assert len(energy_traj_2.index) == 2 * num_entries

        # Reload the trajectory.
        traj_2 = sr.load(
            [str(Path(tmpdir) / "system0.prm7"), str(Path(tmpdir) / "traj_0.00000.dcd")]
        )

        # Check that the trajectory is twice as long as the first.
        assert traj_2.num_frames() == 2 * traj_1.num_frames()

        config_difftimestep = config_new.copy()
        config_difftimestep["runtime"] = "36fs"
        config_difftimestep["timestep"] = "2fs"

        with pytest.raises(ValueError):
            runner_timestep = Runner(mols, Config(**config_difftimestep))

        config_difftemperature = config_new.copy()
        config_difftemperature["runtime"] = "36fs"
        config_difftemperature["temperature"] = "200K"

        with pytest.raises(ValueError):
            runner_temperature = Runner(mols, Config(**config_difftemperature))

        config_diffscalefactor = config_new.copy()
        config_diffscalefactor["runtime"] = "36fs"
        config_diffscalefactor["charge_scale_factor"] = 0.5

        with pytest.raises(ValueError):
            runner_scalefactor = Runner(mols, Config(**config_diffscalefactor))

        config_diffconstraint = config_new.copy()
        config_diffconstraint["runtime"] = "36fs"
        config_diffconstraint["constraint"] = "bonds"

        with pytest.raises(ValueError):
            runner_constraints = Runner(mols, Config(**config_diffconstraint))

        config_diffcutofftype = config_new.copy()
        config_diffcutofftype["runtime"] = "36fs"
        config_diffcutofftype["cutoff_type"] = "rf"

        with pytest.raises(ValueError):
            runner_cutofftype = Runner(mols, Config(**config_diffcutofftype))

        config_diffhmassfactor = config_new.copy()
        config_diffhmassfactor["runtime"] = "36fs"
        config_diffhmassfactor["h_mass_factor"] = 2.0

        with pytest.raises(ValueError):
            runner_hmassfactor = Runner(mols, Config(**config_diffhmassfactor))

        config_diffintegrator = config_new.copy()
        config_diffintegrator["runtime"] = "36fs"
        config_diffintegrator["integrator"] = "verlet"

        with pytest.raises(ValueError):
            runner_integrator = Runner(mols, Config(**config_diffintegrator))

        config_difflambdaschedule = config_new.copy()
        config_difflambdaschedule["runtime"] = "36fs"
        config_difflambdaschedule["charge_scale_factor"] = 0.5
        config_difflambdaschedule["lambda_schedule"] = "charge_scaled_morph"

        with pytest.raises(ValueError):
            runner_lambdaschedule = Runner(mols, Config(**config_difflambdaschedule))

        config_diffnumlambda = config_new.copy()
        config_diffnumlambda["runtime"] = "36fs"
        config_diffnumlambda["num_lambda"] = 3

        with pytest.raises(ValueError):
            runner_numlambda = Runner(mols, Config(**config_diffnumlambda))

        config_diffoutputdirectory = config_new.copy()
        config_diffoutputdirectory["runtime"] = "36fs"
        with tempfile.TemporaryDirectory() as tmpdir2:
            config_diffoutputdirectory["output_directory"] = tmpdir2

            with pytest.raises(OSError):
                runner_outputdirectory = Runner(
                    mols, Config(**config_diffoutputdirectory)
                )

        config_diffperturbableconstraint = config_new.copy()
        config_diffperturbableconstraint["runtime"] = "36fs"
        config_diffperturbableconstraint["perturbable_constraint"] = "bonds"

        with pytest.raises(ValueError):
            runner_perturbableconstraint = Runner(
                mols, Config(**config_diffperturbableconstraint)
            )

        config_diffpressure = config_new.copy()
        config_diffpressure["runtime"] = "36fs"
        config_diffpressure["pressure"] = "1.5 atm"

        with pytest.raises(ValueError):
            runner_pressure = Runner(mols, Config(**config_diffpressure))

        config_diffshiftdelta = config_new.copy()
        config_diffshiftdelta["runtime"] = "36fs"
        config_diffshiftdelta["shift_delta"] = "3 Angstrom"

        with pytest.raises(ValueError):
            runner_shiftdelta = Runner(mols, Config(**config_diffshiftdelta))

        config_diffswapendstates = config_new.copy()
        config_diffswapendstates["runtime"] = "36fs"
        config_diffswapendstates["swap_end_states"] = True

        with pytest.raises(ValueError):
            runner_swapendstates = Runner(mols, Config(**config_diffswapendstates))

        # Removing the config yaml should raise an OSError since the new-format
        # checkpoint stores no config (the yaml is the sole validation source).
        for file in Path(tmpdir).glob("*.yaml"):
            file.unlink()

        with pytest.raises(OSError):
            runner_noconfig = Runner(mols, Config(**config_new))

        # Write a config yaml with a wrong pressure value and verify restart fails.
        import yaml

        bad_config = config_new.copy()
        bad_config["pressure"] = "0.5 atm"
        with open(Path(tmpdir) / "config.yaml", "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError):
            runner_badconfig = Runner(mols, Config(**config_new))


def test_restart_custom_schedule(ethane_methanol):
    """
    Test that a restart works when using a non-standard lambda schedule.
    """
    mols = ethane_methanol.clone()
    schedule = sr.cas.LambdaSchedule.standard_decouple()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "runtime": "12fs",
            "restart": False,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "lambda_schedule": schedule,
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
        }

        # Instantiate a runner using the config defined above.
        runner = Runner(mols, Config(**config))

        del runner

        config_new = {
            "runtime": "24fs",
            "restart": True,
            "output_directory": tmpdir,
            "energy_frequency": "4fs",
            "checkpoint_frequency": "4fs",
            "frame_frequency": "4fs",
            "lambda_schedule": schedule,
            "platform": "CPU",
            "max_threads": 1,
            "num_lambda": 2,
            "overwrite": True,
            "log_level": "DEBUG",
        }

        runner2 = Runner(mols, Config(**config_new))

        # Run the simulation.
        runner2.run()
