######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2024
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
        # If restarting, subtract the time already run from the total runtime
        if self._config.restart:
            self._config.runtime = str(self._config.runtime - self._system.time())

            # Work out the current block number.
            self._current_block = int(
                round(
                    self._system.time().value()
                    / self._config.checkpoint_frequency.value(),
                    12,
                )
            )
        else:
            self._current_block = 0

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
        # Create incremental file name for current restart.
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
        filenames["topology"] = "system.prm7"
        filenames["checkpoint"] = f"checkpoint_{index}.s3"
        filenames["energy_traj"] = f"energy_traj_{index}.parquet"
        filenames["trajectory"] = f"traj_{index}.dcd"
        filenames["trajectory_chunk"] = f"traj_{index}_"
        if restart:
            filenames["config"] = increment_filename("config", "yaml")
        else:
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

        self._dyn = self._system.dynamics(
            temperature=self._config.temperature,
            pressure=pressure,
            barostat_frequency=self._config.barostat_frequency,
            timestep=(
                self._config.equilibration_timestep
                if equilibration
                else self._config.timestep
            ),
            lambda_value=self._lambda_val,
            cutoff_type=self._config.cutoff_type,
            cutoff=self._config.cutoff,
            schedule=self._config.lambda_schedule,
            platform=self._config.platform,
            device=self._device,
            constraint=(
                "none"
                if equilibration and not self._config.equilibration_constraints
                else self._config.constraint
            ),
            perturbable_constraint=(
                "none"
                if equilibration and not self._config.equilibration_constraints
                else self._config.perturbable_constraint
            ),
            include_constrained_energies=self._config.include_constrained_energies,
            dynamic_constraints=self._config.dynamic_constraints,
            swap_end_states=self._config.swap_end_states,
            com_reset_frequency=self._config.com_reset_frequency,
            vacuum=not self._has_space,
            map=self._config._extra_args,
        )

    def _minimisation(
        self, lambda_min=None, constraint="none", perturbable_constraint="none"
    ):
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
                    cutoff=self._config.cutoff,
                    schedule=self._config.lambda_schedule,
                    lambda_value=self._lambda_val,
                    platform=self._config.platform,
                    vacuum=not self._has_space,
                    constraint=constraint,
                    perturbable_constraint=perturbable_constraint,
                    include_constrained_energies=self._config.include_constrained_energies,
                    dynamic_constraints=self._config.dynamic_constraints,
                    swap_end_states=self._config.swap_end_states,
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
                    cutoff=self._config.cutoff,
                    schedule=self._config.lambda_schedule,
                    lambda_value=lambda_min,
                    platform=self._config.platform,
                    vacuum=not self._has_space,
                    constraint=constraint,
                    perturbable_constraint=perturbable_constraint,
                    include_constrained_energies=self._config.include_constrained_energies,
                    dynamic_constraints=self._config.dynamic_constraints,
                    swap_end_states=self._config.swap_end_states,
                    map=self._config._extra_args,
                )
                m.run()
                self._system = m.commit()
            except:
                raise

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

    def _run(self, lambda_minimisation=None, is_restart=False):
        """
        Run the simulation with bookkeeping.

        Returns
        -------

        df : pandas dataframe
            Dataframe containing the sire energy trajectory.
        """

        import sire as sr

        # Save the system topology to a PRM7 file that can be used to load the
        # trajectory.
        topology = str(self._config.output_directory / self._filenames["topology"])
        sr.save(self._system, topology)

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
            # Minimise with no constraints if we need to equilibrate first.
            # This seems to improve the stability of the equilibration.
            if self._config.equilibration_time.value() > 0.0 and not is_restart:
                constraint = "none"
                perturbable_constraint = "none"
            else:
                constraint = self._config.constraint
                perturbable_constraint = self._config.perturbable_constraint

            self._minimisation(
                lambda_minimisation,
                constraint=constraint,
                perturbable_constraint=perturbable_constraint,
            )

        if self._config.equilibration_time.value() > 0.0 and not is_restart:
            self._equilibration()

            # Reset the timer to zero
            self._system.set_time(sr.u("0ps"))

            # Perform minimisation at the end of equilibration only if the
            # timestep is increasing, or the constraint is changing.
            if (self._config.timestep > self._config.equilibration_timestep) or (
                not self._config.equilibration_constraints
                and self._config.perturbable_constraint != "none"
            ):
                self._minimisation(
                    lambda_min=self._lambda_val,
                    constraint=self._config.constraint,
                    perturbable_constraint=self._config.perturbable_constraint,
                )

        # Setup the dynamics object for production.
        self._setup_dynamics(equilibration=False)

        # Work out the lambda values for finite-difference gradient analysis.
        self._lambda_grad = generate_lam_vals(self._lambda_val, self._increment)

        if self._lambda_array is None:
            lam_arr = self._lambda_grad
        else:
            lam_arr = self._lambda_array + self._lambda_grad

        _logger.info(f"Running dynamics at {_lam_sym} = {self._lambda_val}")

        if self._config.checkpoint_frequency.value() > 0.0:
            # Calculate the number of blocks and the remaineder time.
            frac = (
                self._config.runtime.value() / self._config.checkpoint_frequency.value()
            )
            num_blocks = int(frac)
            rem = frac - num_blocks

            # Append only this number of lines from the end of the dataframe during checkpointing
            energy_per_block = (
                self._config.checkpoint_frequency / self._config.energy_frequency
            )
            sire_checkpoint_name = str(
                _Path(self._config.output_directory) / self._filenames["checkpoint"]
            )

            # Run num_blocks dynamics and then run a final block if rem > 0
            for x in range(int(num_blocks)):
                # Add the current block number.
                x += self._current_block

                # Run the dynamics.
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

                # Checkpoint.
                try:
                    # Set to the current block number if this is a restart.
                    if x == 0:
                        x = self._current_block

                    # Commit the current system and save it to a checkpoint file.
                    self._system = self._dyn.commit()
                    sr.stream.save(self._system, str(sire_checkpoint_name))

                    # Save the current trajectory chunk to file.
                    if self._config.save_trajectories:
                        traj_filename = (
                            str(
                                self._config.output_directory
                                / self._filenames["trajectory_chunk"]
                            )
                            + f"{x}.dcd"
                        )
                        sr.save(
                            self._system.trajectory(), traj_filename, format=["DCD"]
                        )

                        # Delete the trajectory from memory.
                        self._system.delete_all_frames()

                    # Save the energy trajectory to a parquet file.
                    df = self._system.energy_trajectory(
                        to_alchemlyb=True, energy_unit="kT"
                    )
                    if x == self._current_block:
                        # Not including speed in checkpoints for now.
                        parquet = _dataframe_to_parquet(
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
                            parquet,
                            df.iloc[-int(energy_per_block) :],
                        )
                    _logger.info(
                        f"Finished block {x+1} of {self._current_block + num_blocks + int(rem > 0)} "
                        f"for {_lam_sym} = {self._lambda_val}"
                    )
                except:
                    raise
            # No need to checkpoint here as it is the final block.
            if rem > 0:
                x += 1
                try:
                    self._dyn.run(
                        rem,
                        energy_frequency=self._config.energy_frequency,
                        frame_frequency=self._config.frame_frequency,
                        lambda_windows=lam_arr,
                        save_velocities=self._config.save_velocities,
                        auto_fix_minimise=False,
                    )

                    # Save the current trajectory chunk to file.
                    if self._config.save_trajectories:
                        traj_filename = (
                            str(
                                self._config.output_directory
                                / self._filenames["trajectory_chunk"]
                            )
                            + f"{x}.dcd"
                        )
                        sr.save(
                            self._system.trajectory(), traj_filename, format=["DCD"]
                        )

                        # Delete the trajectory from memory.
                        self._system.delete_all_frames()

                    _logger.info(
                        f"Finished block {x+1} of {self._current_block + num_blocks + int(rem > 0)} "
                        f"for {_lam_sym} = {self._lambda_val}"
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

        # Assemble and save the final energy trajectory.
        if self._config.save_trajectories:
            # Create the final trajectory file name.
            traj_filename = str(
                self._config.output_directory / self._filenames["trajectory"]
            )

            # Glob for the trajectory chunks.
            from glob import glob

            traj_chunks = sorted(
                glob(
                    str(
                        self._config.output_directory
                        / f"{self._filenames['trajectory_chunk']}*"
                    )
                )
            )

            # If this is a restart, then we need to check for an existing
            # trajectory file with the same name. If it exists and is non-empty,
            # then copy it to a backup file and prepend it to the list of chunks.
            if self._config.restart:
                path = _Path(traj_filename)
                if path.exists() and path.stat().st_size > 0:
                    from shutil import copyfile

                    copyfile(traj_filename, f"{traj_filename}.bak")
                    traj_chunks = [f"{traj_filename}.bak"] + traj_chunks

            # Load the topology and chunked trajectory files.
            system = sr.load([topology] + traj_chunks)

            # Save the final trajectory to a single file.
            traj_filename = str(
                self._config.output_directory / self._filenames["trajectory"]
            )
            sr.save(system.trajectory(), traj_filename, format=["DCD"])

            # Delete the trajectory from memory.
            self._system.delete_all_frames()

            # Now remove the chunked trajectory files.
            for chunk in traj_chunks:
                _Path(chunk).unlink()

        # Add config and lambda value to the system properties.
        self._system.add_shared_property(
            "config", self._config.as_dict(sire_compatible=True)
        )
        self._system.add_shared_property("lambda", self._lambda_val)

        # Save the final system to checkpoint file.
        sr.stream.save(self._system, sire_checkpoint_name)
        df = self._system.energy_trajectory(to_alchemlyb=True, energy_unit="kT")

        return df

    def get_timing(self):
        return self._dyn.time_speed()

    def _cleanup(self):
        del self._dyn
