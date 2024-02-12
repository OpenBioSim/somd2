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
    Validate that a simulation can be run from a checkpoint file,
    with all trajcetories preserved.
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
            Path(tmpdir) / "energy_traj_0.parquet"
        )

        num_entries = len(energy_traj_1.index)

        # Check that both config and lambda have been written
        # as properties to the streamed checkpoint file.
        checkpoint = sr.stream.load(str(Path(tmpdir) / "checkpoint_0.s3"))
        props = checkpoint.property_keys()
        assert "config" in props
        assert "lambda" in props

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
            Path(tmpdir) / "energy_traj_0.parquet"
        )

        # Check that first half of energy trajectory is the same
        assert energy_traj_1.equals(energy_traj_2.iloc[:num_entries])
        # Check that second energy trajectory is twice as long as the first
        assert len(energy_traj_2.index) == 2 * num_entries
        # Check that a second trajectory was written and that the first still exists
        assert Path.exists(Path(tmpdir) / "traj_0.dcd")
        assert Path.exists(Path(tmpdir) / "traj_0_1.dcd")

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

        config_diffcoulombpower = config_new.copy()
        config_diffcoulombpower["runtime"] = "36fs"
        config_diffcoulombpower["coulomb_power"] = 0.5

        with pytest.raises(ValueError):
            runner_coulombpower = Runner(mols, Config(**config_diffcoulombpower))

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

        # Need to test restart from sire checkpoint file
        # this needs to be done last as it requires unlinking the config files
        for file in Path(tmpdir).glob("*.yaml"):
            file.unlink()

        # This should work as the config is read from the lambda=0 checkpoint file
        runner_noconfig = Runner(mols, Config(**config_new))

        # remove config again
        for file in Path(tmpdir).glob("*.yaml"):
            file.unlink()

        # Load the checkpoint file using sire and change the pressure option
        sire_checkpoint = sr.stream.load(str(Path(tmpdir) / "checkpoint_0.s3"))
        cfg = sire_checkpoint.property("config")
        cfg["pressure"] = "0.5 atm"
        sire_checkpoint.set_property("config", cfg)
        sr.stream.save(sire_checkpoint, str(Path(tmpdir) / "checkpoint_0.s3"))

        # Load the new checkpoint file and make sure the restart fails
        with pytest.raises(ValueError):
            runner_badconfig = Runner(mols, Config(**config_new))
