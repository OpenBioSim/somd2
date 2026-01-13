######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
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

__all__ = ["Runner"]

from filelock import FileLock as _FileLock
from time import time as _timer

import numpy as _np

import sire as _sr

from somd2 import _logger

from .._utils import _lam_sym

from ._base import RunnerBase as _RunnerBase


class Runner(_RunnerBase):
    """
    Standard simulation runner class. (Uncoupled simulations.)
    """

    from multiprocessing import Manager

    _manager = Manager()
    _lock = _manager.Lock()
    _queue = _manager.Queue()

    def __init__(self, system, config):
        """
        Constructor.

        Parameters
        ----------

        system: str, :class: `System <sire.system.System>`
            The perturbable system to be simulated. This can be either a path
            to a stream file, or a Sire system object.

        config: :class: `Config <somd2.config.Config>`
            The configuration options for the simulation.
        """

        # No support for replica exchange.
        if config.replica_exchange:
            msg = (
                "The Runner class does not support replica exchange simulations. "
                "Please set replica_exchange=False, or use the RepexRunner class."
            )
            _logger.error(msg)
            raise ValueError(msg)

        # Call the base class constructor.
        super().__init__(system, config)

        # Store the array of lambda values for energy sampling.
        if self._config.lambda_energy is not None:
            self._lambda_energy = self._config.lambda_energy.copy()
        else:
            self._lambda_energy = self._lambda_values.copy()

    def __str__(self):
        """Return a string representation of the object."""
        return f"Runner(system={self._system}, config={self._config})"

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def _create_shared_resources(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        if self._is_gpu:
            devices = self._get_gpu_devices(
                self._config.platform, self._config.oversubscription_factor
            )
            if self._config.max_gpus is not None:
                if self._config.max_gpus > len(devices):
                    _logger.warning(
                        f"Requested {self._config.max_gpus} GPUs, but only {len(devices)} are available."
                    )
                num_devices = min(len(devices), self._config.max_gpus)
            else:
                num_devices = len(devices)

            # Create the GPU pool from the available devices.
            self._gpu_pool = self._manager.list(
                self._initialise_gpu_devices(
                    num_devices,
                    self._config.oversubscription_factor,
                )
            )

    def _update_gpu_pool(self, gpu_num):
        """
        Updates the GPU pool to add the GPU assigned to a worker that has finished.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be added to the pool.
        """
        self._gpu_pool.append(gpu_num)

    def _remove_gpu_from_pool(self, gpu_num):
        """
        Removes a GPU from the GPU pool when it is assigned to a worker.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be removed from the pool.
        """
        self._gpu_pool.remove(gpu_num)

    @staticmethod
    def _initialise_gpu_devices(num_devices, oversubscription_factor=1):
        """
        Create the list of avaiable GPU devices.

        Parameters
        ----------

        num_devices: int
            The number of GPU devices to use.

        oversubscription_factor: int
            The oversubscription factor for the GPUs. This is the number of
            workers that can use a single GPU at the same time.

        Returns
        -------

        devices : [(str, int)]
            List of available device numbers with oversubscription factor.
        """
        devices = []
        for i in range(oversubscription_factor):
            for j in range(num_devices):
                devices.append((str(j), i))
        return devices

    def run(self):
        """
        Use concurrent.futures to run lambda windows in parallel
        """

        # Record the start time.
        start = _timer()

        # Create shared resources.
        self._create_shared_resources()

        # Work out the number of workers and the resources for each.

        # CPU platform.
        if self._config.platform == "cpu":
            # If the number of lambda windows is greater than the number of threads, then
            # the number of threads per worker is set to 1.
            if self._config.num_lambda > self._config.max_threads:
                self.max_workers = self._config.max_threads
                threads_per_worker = 1
            # Otherwise, divide the number of threads equally between workers.
            else:
                self.max_workers = self._config.num_lambda
                threads_per_worker = self._config.max_threads // self._config.num_lambda
            self._config._extra_args["threads"] = threads_per_worker

        # GPU platform.
        elif self._is_gpu:
            self.max_workers = len(self._gpu_pool)

        # All other platforms.
        else:
            self._max_workers = 1

        import concurrent.futures as _futures
        import multiprocessing as _mp

        success = True
        with _futures.ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=_mp.get_context("spawn")
        ) as executor:
            jobs = {}
            for index, lambda_value in enumerate(self._lambda_values):
                jobs[executor.submit(self.run_window, index)] = lambda_value
            try:
                for job in _futures.as_completed(jobs):
                    lambda_value = jobs[job]
                    try:
                        success, time = job.result()
                    except Exception as e:
                        _logger.error(
                            f"Exception raised for {_lam_sym} = {lambda_value}: {e}"
                        )
                        success = False

            # Kill all current and future jobs if keyboard interrupt.
            except KeyboardInterrupt:
                _logger.error("Cancelling job...")
                for pid in executor._processes:
                    executor._processes[pid].terminate()
                success = False

        if success:
            # Record the end time.
            end = _timer()

            # Work how many fractional days the simulation took.
            days = (end - start) / 86400

            # Calculate the speed in nanoseconds per day.
            speed = time.to("ns") / days

            # Log the speed.
            _logger.info(f"Overall performance: {speed:.2f} ns day-1")

            # Log the run time in minutes.
            _logger.success(
                f"Simulation finished. Run time: {(end - start) / 60:.2f} minutes"
            )

            # Cleanup backup files.
            self._cleanup()

    def run_window(self, index):
        """
        Run a single lamdba window.

        Parameters
        ----------

        index: int
            The index of the window.

        Returns
        -------

        success: bool
            Whether the simulation was successful.

        time: sire.units.GeneralUnit
            The duration of the simulation.
        """

        # Since this method is called in a separate process with the "spawn"
        # method, we need to re-set the logger.
        self._config._reset_logger(_logger)

        # Get the lambda value.
        lambda_value = self._lambda_values[index]

        if self._is_restart:
            _logger.debug(f"Restarting {_lam_sym} = {lambda_value} from file")
            system = self._system[index].clone()

            time = system.time()
            if time > self._config.runtime - self._config.timestep:
                _logger.success(
                    f"{_lam_sym} = {lambda_value} already complete. Skipping."
                )
                return True, time
            else:
                _logger.info(
                    f"Restarting {_lam_sym} = {lambda_value} at time {time}, "
                    f"time remaining = {self._config.runtime - time}"
                )
        else:
            system = self._system.clone()

        # GPU platform.
        if self._is_gpu:
            # Get a GPU from the pool.
            with self._lock:
                gpu = self._gpu_pool[0]
                gpu_num = gpu[0]
                self._remove_gpu_from_pool(gpu)
                if lambda_value is not None:
                    _logger.info(
                        f"Running {_lam_sym} = {lambda_value} on GPU {gpu_num}"
                    )

            # Run the simulation.
            try:
                time = self._run(
                    system,
                    index,
                    device=gpu_num,
                    is_restart=self._is_restart,
                )

                with self._lock:
                    self._update_gpu_pool(gpu)
            except Exception as e:
                with self._lock:
                    self._update_gpu_pool(gpu)
                _logger.error(f"Error running {_lam_sym} = {lambda_value}: {e}")
                return False, _sr.u("0ps")

        # All other platforms.
        else:
            _logger.info(f"Running {_lam_sym} = {lambda_value}")

            # Run the simulation.
            try:
                time = self._run(system, index, is_restart=self._is_restart)
            except Exception as e:
                _logger.error(f"Error running {_lam_sym} = {lambda_value}: {e}")
                return False, _sr.u("0ps")

        return True, time

    def _run(
        self, system, index, device=None, lambda_minimisation=None, is_restart=False
    ):
        """
        Run the simulation with bookkeeping.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        index: int
            The index of the lambda window.

        device: int
            The GPU device number to be used for the simulation.

        lambda_minimisation: float
            The lambda value for the minimisation.

        is_restart: bool
            Whether this is a restart simulation.

        Returns
        -------

        time: sire.units.GeneralUnit
            The duration of the simulation.
        """

        # Get the lambda value.
        lambda_value = self._lambda_values[index]

        # Get the index in the lambda_energy array.
        nrg_index = self._lambda_energy.index(lambda_value)

        # Get the REST2 scaling factor.
        rest2_scale = self._rest2_scale_factors[nrg_index]

        # Check for completion if this is a restart.
        if is_restart:
            time = system.time()
            if time > self._config.runtime - self._config.timestep:
                _logger.success(
                    f"{_lam_sym} = {lambda_value} already complete. Skipping."
                )
                return _sr.u("0ps")

            # Work out the current block number.
            if self._config.checkpoint_frequency.value() > 0.0:
                self._start_block = int(
                    round(time.value() / self._config.checkpoint_frequency.value(), 12)
                )
            else:
                self._start_block = 0

            # Subtract the current time from the runtime.
            time = self._config.runtime - time
        else:
            self._start_block = 0
            time = self._config.runtime

        def generate_lam_vals(lambda_base, increment=0.001):
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

        # Prepare the GCMC sampler.
        if self._config.gcmc:
            _logger.info(f"Preparing GCMC sampler at {_lam_sym} = {lambda_value:.5f}")

            try:
                from loch import GCMCSampler
            except:
                msg = "loch is not installed. GCMC sampling cannot be performed."
                _logger.error(msg)

            gcmc_sampler = GCMCSampler(
                system,
                device=int(device),
                lambda_value=lambda_value,
                rest2_scale=rest2_scale,
                ghost_file=self._filenames[index]["gcmc_ghosts"],
                **self._gcmc_kwargs,
            )

            # Get the GCMC system.
            system = gcmc_sampler.system()

            # Log the initial position of the GCMC sphere.
            if gcmc_sampler._reference is not None:
                positions = _sr.io.get_coords_array(system)
                target = gcmc_sampler._get_target_position(positions)
                _logger.info(
                    f"Initial GCMC sphere centre at {_lam_sym} = {lambda_value:.5f}: "
                    f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] A"
                )

        else:
            gcmc_sampler = None

        # Minimisation.
        if self._config.minimise:
            constraint = self._config.constraint
            perturbable_constraint = self._config.perturbable_constraint

            # Don't use constraints during minimisation.
            if not self._config.minimisation_constraints:
                constraint = "none"
                perturbable_constraint = "none"
            # We will be performing an equilibration stage.
            elif not is_restart and self._config.equilibration_time.value() > 0.0:
                # Don't use constraints during equilibration.
                if not self._config.equilibration_constraints:
                    constraint = "none"
                    perturbable_constraint = "none"

            try:
                system = self._minimisation(
                    system,
                    lambda_value,
                    rest2_scale=rest2_scale,
                    device=device,
                    constraint=constraint,
                    perturbable_constraint=perturbable_constraint,
                )
            except Exception as e:
                msg = f"Minimisation failed for {_lam_sym} = {lambda_value:.5f}: {e}"
                if self._config.minimisation_errors:
                    raise RuntimeError(msg)
                else:
                    _logger.warning(msg)

        # Equilibration.
        is_equilibrated = False
        if not is_restart and self._config.equilibration_time.value() > 0.0:
            is_equilibrated = True
            try:
                # Run without saving energies or frames.
                _logger.info(f"Equilibrating at {_lam_sym} = {lambda_value:.5f}")

                # Copy the dynamics kwargs.
                dynamics_kwargs = self._dynamics_kwargs.copy()

                # Overload the dynamics kwargs with the simulation options.
                dynamics_kwargs.update(
                    {
                        "device": device,
                        "lambda_value": lambda_value,
                        "rest2_scale": rest2_scale,
                        "timestep": self._config.equilibration_timestep,
                        "constraint": (
                            "none"
                            if not self._config.equilibration_constraints
                            else self._config.constraint
                        ),
                        "perturbable_constraint": (
                            "none"
                            if not self._config.equilibration_constraints
                            else self._config.perturbable_constraint
                        ),
                    }
                )

                # Create the dynamics object.
                dynamics = system.dynamics(**dynamics_kwargs)

                # Equilibrate with GCMC moves.
                if gcmc_sampler is not None:
                    # Bind the GCMC sampler to the dynamics object.
                    gcmc_sampler.bind_dynamics(dynamics)

                    _logger.info(
                        f"Equilibrating with GCMC moves at {_lam_sym} = {lambda_value:.5f}"
                    )

                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())

                # Run without saving energies or frames.
                dynamics.run(
                    self._config.equilibration_time,
                    energy_frequency=0,
                    frame_frequency=0,
                    save_velocities=False,
                    auto_fix_minimise=True,
                    save_crash_report=self._config.save_crash_report,
                )

                # Commit the system.
                system = dynamics.commit()

                # Reset the timer.
                if self._initial_time[index].value() != 0:
                    system.set_time(self._initial_time[index])
                else:
                    system.set_time(_sr.u("0ps"))

            except Exception as e:
                try:
                    self._save_energy_components(index, dynamics.context())
                except:
                    pass
                raise RuntimeError(f"Equilibration failed: {e}")

        # Work out the lambda values for finite-difference gradient analysis.
        lambda_grad = generate_lam_vals(lambda_value)

        # Create the array of lambda values for energy sampling.
        lambda_energy = self._lambda_energy.copy()

        # Sort the lambda values.
        lambda_energy = sorted(lambda_energy)

        # Create the lambda array.
        lambda_array = lambda_energy.copy()

        # Add the lambda values for finite-difference gradient analysis.
        lambda_array.extend(lambda_grad)

        # Add additional REST2 scaling factors.
        rest2_scale_factors = self._rest2_scale_factors.copy()
        rest2_scale_factors.extend([rest2_scale] * len(lambda_grad))

        # Get the indices of the lambda values in sorted order.
        sorted_indices = [
            i for i, _ in sorted(enumerate(lambda_array), key=lambda x: x[1])
        ]

        # Sort the lambda values.
        lambda_array = sorted(lambda_array)

        # Now sort the scaling factors.
        rest2_scale_factors = [rest2_scale_factors[i] for i in sorted_indices]

        _logger.info(f"Running dynamics at {_lam_sym} = {lambda_value:.5f}")

        # Copy the dynamics kwargs.
        dynamics_kwargs = self._dynamics_kwargs.copy()

        # Overload the dynamics kwargs with the simulation options.
        dynamics_kwargs.update(
            {
                "device": device,
                "lambda_value": lambda_value,
                "rest2_scale": rest2_scale,
            }
        )

        # Create the dynamics object.
        dynamics = system.dynamics(**dynamics_kwargs)

        # Reset the GCMC sampler. This resets the sampling statistics and clears
        # the associated OpenMM forces.
        if gcmc_sampler is not None:
            gcmc_sampler.reset()

            # Bind the GCMC sampler to the dynamics object.
            gcmc_sampler.bind_dynamics(dynamics)

            # If this is a restart, then we need to reset the GCMC water state
            # to match that of the restart system.
            if self._is_restart:
                from openmm.unit import angstrom

                # First set all waters to non-ghosts.
                gcmc_sampler._set_water_state(
                    dynamics.context(),
                    states=_np.ones(len(gcmc_sampler._water_indices)),
                    force=True,
                )

                # Now set the ghost waters.
                gcmc_sampler._set_water_state(
                    dynamics.context(),
                    self._restart_ghost_waters[index],
                    states=_np.zeros(len(gcmc_sampler._water_indices)),
                    force=True,
                )

                # Finally, reset the context positions to match the restart system.
                dynamics.context().setPositions(
                    self._restart_positions[index] * angstrom
                )

            # Otherwise, if we've performed equilibration, then we need to reset
            # the water state in the new context to match the equilibrated system.
            elif is_equilibrated:
                # Reset the water state.
                gcmc_sampler._set_water_state(
                    dynamics.context(),
                    force=True,
                )

        # Set the number of neighbours used for the energy calculation.
        # If not None, then we add one to account for the extra windows
        # used for finite-difference gradient analysis.
        if self._config.num_energy_neighbours is not None:
            num_energy_neighbours = self._config.num_energy_neighbours + 1
        else:
            num_energy_neighbours = None

        # Store the checkpoint time in nanoseconds.
        checkpoint_interval = self._config.checkpoint_frequency.to("ns")

        # Store the start time.
        start = _timer()

        # Run the simulation, checkpointing in blocks.
        if self._config.checkpoint_frequency.value() > 0.0:

            # Calculate the number of blocks and the remainder time.
            frac = (time / self._config.checkpoint_frequency).value()

            # Handle the case where the runtime is less than the checkpoint frequency.
            if frac < 1.0:
                frac = 1.0
                self._config.checkpoint_frequency = f"{time} ps"

            num_blocks = int(frac)
            rem = round(frac - num_blocks, 12)

            # Run the dynamics in blocks.
            for block in range(int(num_blocks)):
                # Add the start block number.
                block += self._start_block

                # Record the start time.
                block_start = _timer()

                # Run the dynamics.
                try:
                    # GCMC specific handling. Note that the frame and checkpoint
                    # frequencies are multiples of the energy frequency so we can
                    # run in energy frequency blocks with no remainder.
                    if self._config.gcmc:
                        # Initialise the run time and time at which the next frame is saved.
                        runtime = _sr.u("0ps")
                        save_frames = self._config.frame_frequency > 0
                        next_frame = self._config.frame_frequency

                        # Loop until we reach the runtime.
                        while runtime <= self._config.checkpoint_frequency:
                            # Run the dynamics in blocks of the GCMC frequency.
                            dynamics.run(
                                self._config.gcmc_frequency,
                                energy_frequency=self._config.energy_frequency,
                                frame_frequency=self._config.frame_frequency,
                                lambda_windows=lambda_array,
                                rest2_scale_factors=rest2_scale_factors,
                                save_velocities=self._config.save_velocities,
                                auto_fix_minimise=True,
                                num_energy_neighbours=num_energy_neighbours,
                                null_energy=self._config.null_energy,
                                save_crash_report=self._config.save_crash_report,
                                # GCMC specific options.
                                excess_chemical_potential=(
                                    self._mu_ex if gcmc_sampler is not None else None
                                ),
                                num_waters=(
                                    _np.sum(gcmc_sampler.water_state())
                                    if gcmc_sampler is not None
                                    else None
                                ),
                            )

                            # Perform a GCMC move.
                            _logger.info(
                                f"Performing GCMC move at {_lam_sym} = {lambda_value:.5f}"
                            )
                            gcmc_sampler.move(dynamics.context())

                            # Update the runtime.
                            runtime += self._config.energy_frequency

                            # If a frame is saved, then we need to save current indices
                            # of the ghost water residues.
                            if save_frames and runtime >= next_frame:
                                gcmc_sampler.write_ghost_residues()
                                next_frame += self._config.frame_frequency

                    else:
                        dynamics.run(
                            self._config.checkpoint_frequency,
                            energy_frequency=self._config.energy_frequency,
                            frame_frequency=self._config.frame_frequency,
                            lambda_windows=lambda_array,
                            rest2_scale_factors=rest2_scale_factors,
                            save_velocities=self._config.save_velocities,
                            auto_fix_minimise=True,
                            num_energy_neighbours=num_energy_neighbours,
                            null_energy=self._config.null_energy,
                            save_crash_report=self._config.save_crash_report,
                        )
                except Exception as e:
                    try:
                        self._save_energy_components(index, dynamics.context())
                    except:
                        pass
                    raise RuntimeError(
                        f"Dynamics block {block+1} for {_lam_sym} = {lambda_value:.5f} failed: {e}"
                    )

                # Checkpoint.
                try:
                    # Save the energy contribution for each force.
                    if self._config.save_energy_components:
                        self._save_energy_components(index, dynamics.context())

                    # Commit the current system.
                    system = dynamics.commit()

                    # If performing GCMC, then we need to flag the ghost waters.
                    if gcmc_sampler is not None:
                        system = gcmc_sampler._flag_ghost_waters(system)

                    # Record the end time.
                    block_end = _timer()

                    # Work how many fractional days the block took.
                    block_time = (block_end - block_start) / 86400

                    # Calculate the speed in nanoseconds per day.
                    speed = checkpoint_interval / block_time

                    # Check if this is the final block.
                    is_final_block = (
                        block - self._start_block
                    ) == num_blocks - 1 and rem == 0

                    # Create the lock.
                    lock = _FileLock(self._lock_file)

                    # Acquire the file lock to ensure that the checkpoint files are
                    # in a consistent state if read by another process.
                    with lock.acquire(timeout=self._config.timeout.to("seconds")):
                        # Backup any existing checkpoint files.
                        index, error = self._backup_checkpoint(index)

                        if error is not None:
                            raise error

                        # Write the checkpoint files.
                        index, error = self._checkpoint(
                            system,
                            index,
                            block,
                            speed,
                            lambda_energy=lambda_energy,
                            lambda_grad=lambda_grad,
                            is_final_block=is_final_block,
                        )

                        if error is not None:
                            raise error

                    # Delete all trajectory frames from the Sire system within the
                    # dynamics object.
                    dynamics._d._sire_mols.delete_all_frames()

                    _logger.info(
                        f"Finished block {block+1} of {self._start_block + num_blocks} "
                        f"for {_lam_sym} = {lambda_value:.5f}"
                    )

                    # Log the number of waters within the GCMC sampling volume.
                    if gcmc_sampler is not None:
                        _logger.info(
                            f"Current number of waters in GCMC volume at {_lam_sym} = {lambda_value:.5f} "
                            f"is {gcmc_sampler.num_waters()}"
                        )

                    if is_final_block:
                        _logger.success(
                            f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Checkpoint failed for {_lam_sym} = {lambda_value:.5f}: {e}"
                    )

            # Handle the remainder time. (There will be no remainer when GCMC sampling.)
            if rem > 0:
                block += 1
                block_start = _timer()
                try:
                    dynamics.run(
                        rem,
                        energy_frequency=self._config.energy_frequency,
                        frame_frequency=self._config.frame_frequency,
                        lambda_windows=lambda_array,
                        rest2_scale_factors=rest2_scale_factors,
                        save_velocities=self._config.save_velocities,
                        auto_fix_minimise=True,
                        num_energy_neighbours=num_energy_neighbours,
                        null_energy=self._config.null_energy,
                        save_crash_report=self._config.save_crash_report,
                    )

                    # Save the energy contribution for each force.
                    if self._config.save_energy_components:
                        self._save_energy_components(index, dynamics.context())

                    # Commit the current system.
                    system = dynamics.commit()

                    # Record the end time.
                    block_end = _timer()

                    # Work how many fractional days the block took.
                    block_time = (block_end - block_start) / 86400

                    # Calculate the speed in nanoseconds per day.
                    speed = checkpoint_interval / block_time

                    # Create the lock.
                    lock = _FileLock(self._lock_file)

                    # Acquire the file lock to ensure that the checkpoint files are
                    # in a consistent state if read by another process.
                    with lock.acquire(timeout=self._config.timeout.to("seconds")):
                        self._checkpoint(
                            system,
                            index,
                            block,
                            speed,
                            lambda_energy=lambda_energy,
                            lambda_grad=lambda_grad,
                            is_final_block=True,
                        )

                    # Delete all trajectory frames from the Sire system within the
                    # dynamics object.
                    dynamics._d._sire_mols.delete_all_frames()

                    _logger.info(
                        f"Finished block {block+1} of {self._start_block + num_blocks} "
                        f"for {_lam_sym} = {lambda_value:.5f}"
                    )

                    _logger.success(
                        f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
                    )
                except Exception as e:
                    try:
                        self._save_energy_components(index, dynamics.context())
                    except:
                        pass
                    raise RuntimeError(
                        f"Final dynamics block for {lam_sym} = {lambda_value:.5f} failed: {e}"
                    )
        else:
            try:
                if gcmc_sampler is not None:
                    # Initialise the run time and time at which the next frame is saved.
                    runtime = _sr.u("0ps")
                    save_frames = self._config.frame_frequency > 0
                    next_frame = self._config.frame_frequency

                    # Loop until we reach the runtime.
                    while runtime <= time:
                        # Run the dynamics in blocks of the GCMC frequency.
                        dynamics.run(
                            self._config.gcmc_frequency,
                            energy_frequency=self._config.energy_frequency,
                            frame_frequency=self._config.frame_frequency,
                            lambda_windows=lambda_array,
                            rest2_scale_factors=rest2_scale_factors,
                            save_velocities=self._config.save_velocities,
                            auto_fix_minimise=True,
                            num_energy_neighbours=num_energy_neighbours,
                            null_energy=self._config.null_energy,
                            save_crash_report=self._config.save_crash_report,
                        )

                        # Perform a GCMC move.
                        _logger.info(
                            f"Performing GCMC move at {_lam_sym} = {lambda_value:.5f}"
                        )
                        gcmc_sampler.move(dynamics.context())

                        # Update the runtime.
                        runtime += self._config.energy_frequency

                        # If a frame is saved, then we need to save current indices
                        # of the ghost water residues.
                        if save_frames and runtime >= next_frame:
                            gcmc_sampler.write_ghost_residues()
                            next_frame += self._config.frame_frequency
                else:
                    dynamics.run(
                        time,
                        energy_frequency=self._config.energy_frequency,
                        frame_frequency=self._config.frame_frequency,
                        lambda_windows=lambda_array,
                        rest2_scale_factors=rest2_scale_factors,
                        save_velocities=self._config.save_velocities,
                        auto_fix_minimise=True,
                        num_energy_neighbours=num_energy_neighbours,
                        null_energy=self._config.null_energy,
                        save_crash_report=self._config.save_crash_report,
                    )
            except Exception as e:
                try:
                    self._save_energy_components(index, dynamics.context())
                except:
                    pass
                raise RuntimeError(
                    f"Dynamics for {_lam_sym} = {lambda_value:.5f} failed: {e}"
                )

            # Commit the current system.
            system = dynamics.commit()

            # Record the end time.
            end = _timer()

            # Work how many fractional days the simulation took.
            days = (end - start) / 86400

            # Calculate the speed in nanoseconds per day.
            speed = time.to("ns") / days

            # Create the lock.
            lock = _FileLock(self._lock_file)

            # Acquire the file lock to ensure that the checkpoint files are
            # in a consistent state if read by another process.
            with lock.acquire(timeout=self._config.timeout.to("seconds")):
                # Backup any existing checkpoint files.
                index, error = self._backup_checkpoint(index)

                if error is not None:
                    msg = f"Checkpoint backup failed for {_lam_sym} = {lambda_value:.5f}: {error}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

                # Write the checkpoint files.
                index, error = self._checkpoint(
                    system,
                    index,
                    0,
                    speed,
                    lambda_energy=lambda_energy,
                    lambda_grad=lambda_grad,
                    is_final_block=True,
                )

                if error is not None:
                    msg = f"Checkpoint failed for {_lam_sym} = {lambda_value:.5f}: {error}"
                    _logger.error(msg)
                    raise RuntimeError(msg)

            _logger.success(
                f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
            )

        return time

    def _minimisation(
        self,
        system,
        lambda_value,
        rest2_scale=1.0,
        device=None,
        constraint="none",
        perturbable_constraint="none",
    ):
        """
        Minimise a system.

        Parameters
        ----------

        system: str, :class: `System <sire.system.System>`
            The system to minimise.

        lambda_value: float
            Lambda value at which to run minimisation.

        rest2_scale: float
            The scaling factor for the REST2 region.

        device: int
            The GPU device number to be used for the simulation.

        constraint: str
            The constraint for non-perturbable molecules.

        perturbable_constraint: str
            The constraint for perturbable molecules.

        Returns
        -------

        system: :class: `System <sire.system.System>`
            The minimised system.
        """

        _logger.info(f"Minimising at {_lam_sym} = {lambda_value:.5f}")

        # Copy the dynamics kwargs.
        dynamics_kwargs = self._dynamics_kwargs.copy()

        # Overload the dynamics kwargs with the minimisation options.
        dynamics_kwargs.update(
            {
                "device": device,
                "lambda_value": lambda_value,
                "rest2_scale": rest2_scale,
                "constraint": constraint,
                "perturbable_constraint": perturbable_constraint,
            }
        )

        try:
            # Create a dynamics object.
            dynamics = system.dynamics(**dynamics_kwargs)

            # Run the minimisation.
            dynamics.minimise(timeout=self._config.timeout)

            # Commit the system.
            system = dynamics.commit()
        except:
            raise

        return system
