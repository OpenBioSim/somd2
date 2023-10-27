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

from pathlib import Path as _Path

from ..config import Config as _Config
from ..io import dataframe_to_parquet as _dataframe_to_parquet
from ..io import parquet_append as _parquet_append


class Dynamics:
    """
    Class for controlling the running and bookkeeping of a single lambda value
    simulation.Currently just a wrapper around sire dynamics

    Simulation options are held within a Config object
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
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        if not isinstance(config, _Config):
            raise TypeError("config must be a Config object")
        self._config = config
        self._lambda_val = lambda_val
        self._lambda_array = lambda_array
        self._increment = increment
        self._device = device
        self._has_space = has_space

    def _setup_dynamics(self, equilibration=False):
        """
        Minimise if needed and then setup dynamics object

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
            map = self._config.extra_args
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
            constraint="none" if equilibration else self._config.constraint,
            perturbable_constraint="none"
            if equilibration
            else self._config.perturbable_constraint,
            map=map,
        )

    def _minimisation(self, lambda_min=None):
        """
        Minimisation of self._system

        Parameters
        ----------

        lambda_min : float
            Lambda value at which to run minimisation,
            if None run at pre-set lambda_val
        """
        if lambda_min is None:
            try:
                m = self._system.minimisation(
                    cutoff_type=self._config.cutoff_type,
                    schedule=self._config.lambda_schedule,
                    lambda_value=self._lambda_val,
                    platform=self._config.platform,
                    map=self._config.extra_args,
                )
                m.run()
                self._system = m.commit()
            except:
                raise
        else:
            try:
                m = self._system.minimisation(
                    cutoff_type=self._config.cutoff_type,
                    schedule=self._config.lambda_schedule,
                    lambda_value=lambda_min,
                    platform=self._config.platform,
                    map=self._config.extra_args,
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
        self._setup_dynamics(equilibration=True)
        self._dyn.run(
            self._config.equilibration_time,
            frame_frequency=0,
            energy_frequency=0,
            save_velocities=False,
            auto_fix_minimise=False,
        )
        self._system = self._dyn.commit()

    def _run(self, lambda_minimisation=None):
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

        if self._config.equilibration_time.value() > 0.0:
            self._equilibration()
            # Reset the timer to zero
            self._system.set_time(_u("0ps"))

        self._setup_dynamics(equilibration=False)
        # Work out the lambda values for finite-difference gradient analysis.
        self._lambda_grad = generate_lam_vals(self._lambda_val, self._increment)

        if self._lambda_array is None:
            lam_arr = self._lambda_grad
        else:
            lam_arr = self._lambda_array + self._lambda_grad

        if self._config.checkpoint:
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
            sire_checkpoint_name = (
                _Path(self._config.output_directory)
                / f"checkpoint_{self._lambda_val}.s3"
            )
            # Run num_blocks dynamics and then run a final block if rem > 0
            for _ in range(int(num_blocks)):
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
                    if _ == 0:
                        # Not inlcuding speed in checkpoints for now

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
                        )
                    else:
                        _parquet_append(
                            f,
                            df.iloc[-int(energy_per_block) :],
                        )
                except:
                    raise
            # No need to checkpoint here as it is the final block
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
            traj_filename = self._config.output_directory / f"traj_{self._lambda_val}"
            from sire import save as _save

            _save(self._system.trajectory(), traj_filename, format=["DCD"])
        df = self._system.energy_trajectory(to_alchemlyb=True)
        return df

    def get_timing(self):
        return self._dyn.time_speed()

    def _cleanup(self):
        del self._dyn
