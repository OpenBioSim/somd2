######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# SOMD2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SOMD2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SOMD2. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = ["Dynamics"]

import platform as _platform
from pathlib import Path as _Path

from ..config import Config as _Config
from ..io import dataframe_to_parquet as _dataframe_to_parquet
from ..io import parquet_append as _parquet_append

from somd2 import _logger

from ._runner import _lam_sym


class Dynamics:
    """
    Class for controlling the running and bookkeeping of a single lambda value
    simulation.Currently just a wrapper around Sire dynamics.
    """

    def __init__(
        self,
        system,
        lambda_val,
        lambda_array,
        config,
        increment=0.001,
        device=None,
        has_space=True,
    ):
        """
        Constructor

        Parameters
        ----------

        system : Sire System
            Sire system containing at least one perturbable molecule

        lambda_val : float
            Lambda value for the simulation

        lambda_array : list
            List of lambda values to be used for perturbation, if none won't return
            reduced perturbed energies

        increment : float
            Increment of lambda value - used for calculating the gradient

        config : somd2 Config object
            Config object containing simulation options

        device : int
            GPU device number to use  - does nothing if running on CPU (default None)

        has_space : bool
            Whether this simulation has a periodic space or not. Disable NPT if
            no space is present.

        """

        try:
            system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        self._system = system

        if not isinstance(config, _Config):
            raise TypeError("config must be a Config object")

        self._config = config
        # If resarting, subtract the time already run from the total runtime
        if self._config.restart:
            self._config.runtime = str(self._config.runtime - self._system.time())
        self._lambda_val = lambda_val
        self._lambda_array = lambda_array
        self._increment = increment
        self._device = device
        self._has_space = has_space
        self._filenames = self.create_filenames(
            self._lambda_array,
            self._lambda_val,
            self._config.output_directory,
            self._config.restart,
        )

    @staticmethod
    def create_filenames(lambda_array, lambda_value, output_directory, restart=False):
        # Create incremental file - used for writing trajectory files
        def increment_filename(base_filename, suffix):
            file_number = 0
            file_path = _Path(output_directory)
            while True:
                filename = (
                    f"{base_filename}_{file_number}.{suffix}"
                    if file_number > 0
                    else f"{base_filename}.{suffix}"
                )
                full_path = file_path / filename
                if not full_path.exists():
                    return filename
                file_number += 1

        if lambda_value not in lambda_array:
            raise ValueError("lambda_value not in lambda_array")
        filenames = {}
        index = lambda_array.index(lambda_value)
        filenames["checkpoint"] = f"checkpoint_{index}.s3"
        filenames["energy_traj"] = f"energy_traj_{index}.parquet"
        if restart:
            filenames["trajectory"] = increment_filename(f"traj_{index}", "dcd")
            filenames["config"] = increment_filename("config", "yaml")
        else:
            filenames["trajectory"] = f"traj_{index}.dcd"
            filenames["config"] = "config.yaml"
        return filenames

    def _setup_dynamics(self, equilibration=False):
        """
        Setup the dynamics object.

        Parameters
        ----------

        lam_val_min : float
            Lambda value at which to run minimisation,
            if None run at pre-set lambda_val

        equilibration : bool
            If True, use equilibration settings, otherwise use production settings
        """

        # Don't use NPT for vacuum simulations.
        if self._has_space:
            pressure = self._config.pressure
        else:
            pressure = None

        try:
            map = self._config._extra_args
        except:
            map = None

        self._dyn = self._system.dynamics(
            temperature=self._config.temperature,
            pressure=pressure,
            timestep=self._config.equilibration_timestep
            if equilibration
            else self._config.timestep,
            lambda_value=self._lambda_val,
            cutoff_type=self._config.cutoff_type,
            schedule=self._config.lambda_schedule,
            platform=self._config.platform,
            device=self._device,
            constraint=None if equilibration else self._config.constraint,
            perturbable_constraint=None
            if equilibration
            else self._config.perturbable_constraint,
            vacuum=not self._has_space,
            map=map,
        )

    def _minimisation(self, lambda_min=None):
        """
        Minimisation of self._system.

        Parameters
        ----------

        lambda_min : float
            Lambda value at which to run minimisation, if None run at pre-set
            lambda_val.
        """
        if lambda_min is None:
            _logger.info(f"Minimising at {_lam_sym} = {self._lambda_val}")
            try:
                m = self._system.minimisation(
                    cutoff_type=self._config.cutoff_type,
                    schedule=self._config.lambda_schedule,
                    lambda_value=self._lambda_val,
                    platform=self._config.platform,
                    vacuum=not self._has_space,
                    map=self._config._extra_args,
                )
                m.run()
                self._system = m.commit()
            except:
                raise
        else:
            _logger.info(f"Minimising at {_lam_sym} = {lambda_min}")
            try:
                m = self._system.minimisation(
                    cutoff_type=self._config.cutoff_type,
                    schedule=self._config.lambda_schedule,
                    lambda_value=lambda_min,
                    platform=self._config.platform,
                    vacuum=not self._has_space,
                    map=self._config._extra_args,
                )
                m.run()
                self._system = m.commit()
            except:
                raise

    # combine these - just equil time
    # reset timer to zero when bookeeping starts
    def _equilibration(self):
        """
        Per-window equilibration.
        Currently just runs dynamics without any saving
        """

        _logger.info(f"Equilibrating at {_lam_sym} = {self._lambda_val}")
        self._setup_dynamics(equilibration=True)
        self._dyn.run(
            self._config.equilibration_time,
            frame_frequency=0,
            energy_frequency=0,
            save_velocities=False,
            auto_fix_minimise=False,
        )
        self._system = self._dyn.commit()
        # Run minimisation at the end of equilibration as the timestep may be changing
        self._minimisation()

    def _run(self, lambda_minimisation=None, is_restart=False):
        """
        Run the simulation with bookkeeping.

        Returns
        -------

        df : pandas dataframe
            Dataframe containing the sire energy trajectory.
        """
        from sire import u as _u
        from sire import stream as _stream

        def generate_lam_vals(lambda_base, increment):
            """Generate lambda values for a given lambda_base and increment"""
            if lambda_base + increment > 1.0 and lambda_base - increment < 0.0:
                raise ValueError("Increment too large")
            if lambda_base + increment > 1.0:
                lam_vals = [lambda_base - increment]
            elif lambda_base - increment < 0.0:
                lam_vals = [lambda_base + increment]
            else:
                lam_vals = [lambda_base - increment, lambda_base + increment]
            return lam_vals

        if self._config.minimise:
            self._minimisation(lambda_minimisation)

        if (self._config.equilibration_time.value() > 0.0) and not is_restart:
            self._equilibration()
            # Reset the timer to zero
            self._system.set_time(_u("0ps"))

        # Perform HMR after minimisation and equilibration
        from ._runner import Runner
        from math import isclose

        h_mass_factor = Runner._get_h_mass_factor(self._system)

        if not isclose(h_mass_factor, 1.0, abs_tol=1e-4):
            _logger.debug(
                f"Existing repartitioning found in the {_lam_sym}={self._lambda_val} system"
            )
            if not isclose(h_mass_factor, self._config.h_mass_factor, abs_tol=1e-4):
                new_factor = self._config.h_mass_factor / h_mass_factor
                _logger.warning(
                    f"Existing hydrogen mass repartitioning factor of {h_mass_factor:.3f} "
                    f"does not match the requested value of {self._config.h_mass_factor:.3f}. "
                    f"A new factor of {new_factor:.3f} will be applied to the {_lam_sym} = {self._lambda_val} system."
                )
                self._system = Runner._repartition_h_mass(self._system, new_factor)

        else:
            self._system = Runner._repartition_h_mass(
                self._system, self._config.h_mass_factor
            )
        self._setup_dynamics(equilibration=False)
        # Work out the lambda values for finite-difference gradient analysis.
        self._lambda_grad = generate_lam_vals(self._lambda_val, self._increment)

        if self._lambda_array is None:
            lam_arr = self._lambda_grad
        else:
            lam_arr = self._lambda_array + self._lambda_grad

        _logger.info(f"Running dynamics at {_lam_sym} = {self._lambda_val}")

        if self._config.checkpoint_frequency.value() > 0.0:
            ### Calc number of blocks and remainder (surely there's a better way?)###
            num_blocks = 0
            rem = self._config.runtime
            while True:
                if rem > self._config.checkpoint_frequency:
                    num_blocks += 1
                    rem -= self._config.checkpoint_frequency
                else:
                    break
            # Append only this number of lines from the end of the dataframe during checkpointing
            energy_per_block = (
                self._config.checkpoint_frequency / self._config.energy_frequency
            )
            sire_checkpoint_name = str(
                _Path(self._config.output_directory) / self._filenames["checkpoint"]
            )
            # Run num_blocks dynamics and then run a final block if rem > 0
            for x in range(int(num_blocks)):
                try:
                    self._dyn.run(
                        self._config.checkpoint_frequency,
                        energy_frequency=self._config.energy_frequency,
                        frame_frequency=self._config.frame_frequency,
                        lambda_windows=lam_arr,
                        save_velocities=self._config.save_velocities,
                        auto_fix_minimise=False,
                    )
                except:
                    raise
                try:
                    self._system = self._dyn.commit()
                    _stream.save(self._system, str(sire_checkpoint_name))
                    df = self._system.energy_trajectory(to_alchemlyb=True)
                    if x == 0:
                        # Not including speed in checkpoints for now.
                        f = _dataframe_to_parquet(
                            df,
                            metadata={
                                "attrs": df.attrs,
                                "lambda": str(self._lambda_val),
                                "lambda_array": lam_arr,
                                "lambda_grad": self._lambda_grad,
                                "temperature": str(self._config.temperature.value()),
                            },
                            filepath=self._config.output_directory,
                            filename=self._filenames["energy_traj"],
                        )
                        # Also want to add the simulation config to the
                        # system properties once a block has been successfully run.
                        self._system.set_property(
                            "config", self._config.as_dict(sire_compatible=True)
                        )
                        # Finally, encode lambda value in to properties.
                        self._system.set_property("lambda", self._lambda_val)
                    else:
                        _parquet_append(
                            f,
                            df.iloc[-int(energy_per_block) :],
                        )
                    _logger.info(
                        f"Finished block {x+1} of {num_blocks} for {_lam_sym} = {self._lambda_val}"
                    )
                except:
                    raise
            # No need to checkpoint here as it is the final block.
            if rem > 0:
                try:
                    self._dyn.run(
                        rem,
                        energy_frequency=self._config.energy_frequency,
                        frame_frequency=self._config.frame_frequency,
                        lambda_windows=lam_arr,
                        save_velocities=self._config.save_velocities,
                        auto_fix_minimise=False,
                    )
                except:
                    raise
                self._system = self._dyn.commit()
        else:
            try:
                self._dyn.run(
                    self._config.checkpoint_frequency,
                    energy_frequency=self._config.energy_frequency,
                    frame_frequency=self._config.frame_frequency,
                    lambda_windows=lam_arr,
                    save_velocities=self._config.save_velocities,
                    auto_fix_minimise=False,
                )
            except:
                raise
            self._system = self._dyn.commit()

        if self._config.save_trajectories:
            traj_filename = str(
                self._config.output_directory / self._filenames["trajectory"]
            )
            from sire import save as _save

            _save(self._system.trajectory(), traj_filename, format=["DCD"])
        # dump final system to checkpoint file
        _stream.save(self._system, sire_checkpoint_name)
        df = self._system.energy_trajectory(to_alchemlyb=True)
        return df

    def get_timing(self):
        return self._dyn.time_speed()

    def _cleanup(self):
        del self._dyn
