######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2025
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
            if self._config.max_gpus is None:
                self._gpu_pool = self._manager.list(
                    self._zero_gpu_devices(self._get_gpu_devices(self._config.platform))
                )
            else:
                self._gpu_pool = self._manager.list(
                    self._zero_gpu_devices(
                        self._get_gpu_devices(self._config.platform)[
                            : self._config.max_gpus
                        ]
                    )
                )

    def _update_gpu_pool(self, gpu_num):
        """
        Updates the GPU pool to remove the GPU that has been assigned to a worker.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be added to the pool.
        """
        self._gpu_pool.append(gpu_num)

    def _remove_gpu_from_pool(self, gpu_num):
        """
        Removes a GPU from the GPU pool.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be removed from the pool.
        """
        self._gpu_pool.remove(gpu_num)

    @staticmethod
    def _zero_gpu_devices(devices):
        """
        Set all device numbers relative to the lowest (the device number becomes
        equal to its index in the list).

        Returns
        -------

        devices : [int]
            List of zeroed available device numbers.
        """
        return [str(devices.index(value)) for value in devices]

    def run(self):
        """
        Use concurrent.futures to run lambda windows in parallel
        """

        from time import time

        # Record the start time.
        start = time()

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

        with _futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            jobs = {}
            for index, lambda_value in enumerate(self._lambda_values):
                jobs[executor.submit(self.run_window, index)] = lambda_value
            try:
                for job in _futures.as_completed(jobs):
                    lambda_value = jobs[job]
                    try:
                        result = job.result()
                    except Exception as e:
                        result = False
                        _logger.error(
                            f"Exception raised for {_lam_sym} = {lambda_value}: {e}"
                        )

            # Kill all current and future jobs if keyboard interrupt.
            except KeyboardInterrupt:
                _logger.error("Cancelling job...")
                for pid in executor._processes:
                    executor._processes[pid].terminate()

        # Record the end time.
        end = time()

        # Log the run time in minutes.
        _logger.success(
            f"Simulation finished. Run time: {(end - start) / 60:.2f} minutes"
        )

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
        """

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
                return True
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
                gpu_num = self._gpu_pool[0]
                self._remove_gpu_from_pool(gpu_num)
                if lambda_value is not None:
                    _logger.info(
                        f"Running {_lam_sym} = {lambda_value} on GPU {gpu_num}"
                    )

            # Run the smullation.
            try:
                self._run(
                    system,
                    index,
                    device=gpu_num,
                    is_restart=self._is_restart,
                )

                with self._lock:
                    self._update_gpu_pool(gpu_num)
            except Exception as e:
                with self._lock:
                    self._update_gpu_pool(gpu_num)
                _logger.error(f"Error running {_lam_sym} = {lambda_value}: {e}")

        # All other platforms.
        else:
            _logger.info(f"Running {_lam_sym} = {lambda_value}")

            # Run the simulation.
            try:
                self._run(system, index, is_restart=self._is_restart)
            except Exception as e:
                _logger.error(f"Error running {_lam_sym} = {lambda_value}: {e}")

        return True

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

        df : pandas dataframe
            Dataframe containing the sire energy trajectory.
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
                return

            # Work out the current block number.
            self._start_block = int(
                round(time.value() / self._config.checkpoint_frequency.value(), 12)
            )

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

        # Minimisation.
        if self._config.minimise:
            # Minimise with no constraints if we need to equilibrate first.
            # This seems to improve the stability of the equilibration.
            if self._config.equilibration_time.value() > 0.0 and not is_restart:
                constraint = "none"
                perturbable_constraint = "none"
            else:
                constraint = self._config.constraint
                perturbable_constraint = self._config.perturbable_constraint

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
                raise RuntimeError(f"Minimisation failed: {e}")

        # Equilibration.
        if self._config.equilibration_time.value() > 0.0 and not is_restart:
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

                # Run without saving energies or frames.
                dynamics.run(
                    self._config.equilibration_time,
                    energy_frequency=0,
                    frame_frequency=0,
                    save_velocities=False,
                    auto_fix_minimise=True,
                )

                # Commit the system.
                system = dynamics.commit()

                # Reset the timer to zero.
                system.set_time(_sr.u("0ps"))

                # Perform minimisation at the end of equilibration only if the
                # timestep is increasing, or the constraint is changing.
                if (self._config.timestep > self._config.equilibration_timestep) or (
                    not self._config.equilibration_constraints
                    and self._config.perturbable_constraint != "none"
                ):
                    self._minimisation(
                        system,
                        lambda_value=lambda_value,
                        rest2_scale=rest2_scale,
                        device=device,
                        constraint=self._config.constraint,
                        perturbable_constraint=self._config.perturbable_constraint,
                    )
            except Exception as e:
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

        # Set the number of neighbours used for the energy calculation.
        # If not None, then we add one to account for the extra windows
        # used for finite-difference gradient analysis.
        if self._config.num_energy_neighbours is not None:
            num_energy_neighbours = self._config.num_energy_neighbours + 1
        else:
            num_energy_neighbours = None

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

                # Run the dynamics.
                try:
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
                    )
                except Exception as e:
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

                    # Get the simulation speed.
                    speed = dynamics.time_speed()

                    # Check if this is the final block.
                    is_final_block = (
                        block - self._start_block
                    ) == num_blocks - 1 and rem == 0

                    # Checkpoint.
                    self._checkpoint(
                        system,
                        index,
                        block,
                        speed,
                        lambda_energy=lambda_energy,
                        lambda_grad=lambda_grad,
                        is_final_block=is_final_block,
                    )

                    # Delete all trajectory frames from the Sire system within the
                    # dynamics object.
                    dynamics._d._sire_mols.delete_all_frames()

                    _logger.info(
                        f"Finished block {block+1} of {self._start_block + num_blocks} "
                        f"for {_lam_sym} = {lambda_value:.5f}"
                    )

                    if is_final_block:
                        _logger.success(
                            f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Checkpoint failed for {_lam_sym} = {lambda_value:.5f}: {e}"
                    )

            # Handle the remainder time.
            if rem > 0:
                block += 1
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
                    )

                    # Save the energy contribution for each force.
                    if self._config.save_energy_components:
                        self._save_energy_components(index, dynamics.context())

                    # Commit the current system.
                    system = dynamics.commit()

                    # Get the simulation speed.
                    speed = dynamics.time_speed()

                    # Checkpoint.
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
                    raise RuntimeError(
                        f"Final dynamics block for {lam_sym} = {lambda_value:.5f} failed: {e}"
                    )
        else:
            try:
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
                )
            except Exception as e:
                raise RuntimeError(
                    f"Dynamics for {_lam_sym} = {lambda_value:.5f} failed: {e}"
                )

            # Commit the current system.
            system = dynamics.commit()

            # Get the simulation speed.
            speed = dynamics.time_speed()
            # Checkpoint.
            self._checkpoint(
                system,
                index,
                0,
                speed,
                is_final_block=True,
                lambda_grad=lambda_grad,
                lambda_energy=lambda_energy,
            )

            _logger.success(
                f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
            )

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
