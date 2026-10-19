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

from dataclasses import dataclass as _dataclass
from filelock import FileLock as _FileLock
from time import time as _timer

import numpy as _np

import sire as _sr

from somd2 import _logger

from .._utils import _lam_sym

from ._base import RunnerBase as _RunnerBase


@_dataclass
class _Window:
    """
    The state of a lambda window that is shared by the stages of its
    simulation.
    """

    index: int
    lambda_value: float
    rest2_scale: float
    device: int = None
    gcmc_sampler: object = None
    terminal_flip_sampler: object = None
    dynamics: object = None
    lambda_energy: list = None
    lambda_grad: list = None
    lambda_array: list = None
    rest2_scale_factors: list = None
    num_energy_neighbours: int = None
    # The integration step reached, counted from the start of the simulation.
    step: int = 0
    # The number of energy samples recorded when the last checkpoint was written.
    num_energy_samples: int = 0


class Runner(_RunnerBase):
    """
    Standard simulation runner class. (Uncoupled simulations.)
    """

    _manager = None

    @classmethod
    def _init_manager(cls):
        """
        Initialise the shared-memory Manager the first time a Runner is
        constructed in the parent process. Deferred from class definition time
        so that importing this module does not fork a manager process before
        OpenMM threads have been started.
        """
        if cls._manager is None:
            from multiprocessing import Manager

            cls._manager = Manager()

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

        if config.adaptive_repex:
            msg = "'adaptive_repex' requires 'replica_exchange' to be enabled."
            _logger.error(msg)
            raise ValueError(msg)

        # Production is run in whole integration steps. This is checked before
        # the base class writes the config file, so a rejected restart doesn't
        # leave one behind.
        from math import isclose

        for option in (
            "runtime",
            "checkpoint_frequency",
            "energy_frequency",
            "frame_frequency",
            "gcmc_frequency",
            "terminal_flip_frequency",
        ):
            value = getattr(config, option)
            if value is None:
                continue
            ratio = (value / config.timestep).value()
            if not isclose(ratio, round(ratio), abs_tol=1e-4):
                msg = f"'{option}' must be a multiple of 'timestep'."
                _logger.error(msg)
                raise ValueError(msg)

        # Call the base class constructor.
        super().__init__(system, config)

        # Initialise the shared-memory manager lazily so that importing this
        # module does not fork a manager process before OpenMM threads exist.
        Runner._init_manager()

        # Create Lock and Queue as instance attributes so that they are
        # pickled as manager proxies and shared correctly across all spawned
        # worker processes, preventing race conditions on the GPU pool.
        self._lock = Runner._manager.Lock()
        self._queue = Runner._manager.Queue()

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
            devices = self._gpu_devices
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

        devices: [(str, int)]
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

        # Auto-generate a Boresch restraint for ABFE runs with no user-supplied restraint.
        if self._is_abfe_bound and self._config.restraints is None:
            device = self._gpu_pool[0] if self._is_gpu else None
            try:
                restraints = self._generate_boresch_restraint(device=device)
            except Exception as e:
                msg = f"Unable to generate Boresch restraint for ABFE simulation: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)
            self._config.restraints = restraints
            self._dynamics_kwargs["restraints"] = restraints

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

    def _check_restart(self):
        """
        Check the output directory for a valid restart state.

        Detects new-format (.npz) checkpoints and falls back to the legacy
        .s3 stream file format when only old checkpoints are present.
        """
        from pathlib import Path as _Path

        npz_path = _Path(self._filenames[0]["checkpoint_state"])
        s3_path = _Path(self._filenames[0]["checkpoint"])

        if npz_path.exists():
            return True, self._system
        elif s3_path.exists():
            _logger.info("Restarting from legacy stream file checkpoint.")
            return super()._check_restart()
        else:
            return False, self._system

    def _write_checkpoint_system(self, system, index, context=None, gcmc_sampler=None):
        """
        Write the system state to a compact numpy checkpoint file.

        Saves positions, velocities, box vectors, simulation time, and (for
        GCMC) ghost water indices to a .npz file. The legacy .s3 stream file
        is not written.

        If no context is provided (should not happen in normal operation),
        falls back to the base .s3 implementation.
        """
        if context is None:
            super()._write_checkpoint_system(system, index)
            return

        import openmm.unit as _omm_unit

        state = context.getState(getPositions=True, getVelocities=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(_omm_unit.nanometer)
        vel = state.getVelocities(asNumpy=True).value_in_unit(
            _omm_unit.nanometer / _omm_unit.picosecond
        )
        time_ps = system.time().to("ps")

        save_kwargs = {
            "positions": pos,
            "velocities": vel,
            "time_ps": _np.array([time_ps]),
        }
        # Left out if unknown, since None can only be saved by pickling it.
        if self._virtual_sites is not None:
            save_kwargs["virtual_sites"] = self._virtual_sites

        box = state.getPeriodicBoxVectors(asNumpy=True)
        if box is not None:
            save_kwargs["box"] = box.value_in_unit(_omm_unit.nanometer)

        if gcmc_sampler is not None:
            # water_state() returns 1 for active, 0 for ghost.
            water_state = gcmc_sampler.water_state()
            save_kwargs["ghost_water_indices"] = _np.where(water_state == 0)[0].astype(
                _np.int32
            )

        _np.savez(self._filenames[index]["checkpoint_state"], **save_kwargs)

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
        # method, we need to re-set the logger and silence the progress bar.
        self._config._reset_logger(_logger)
        _sr.base.ProgressBar.set_silent()

        # The thread limit set in the parent process doesn't carry over.
        if self._sire_threads is not None:
            _sr.legacy.Base.set_max_num_threads(self._sire_threads)

        # Get the lambda value.
        lambda_value = self._lambda_values[index]

        if self._is_restart:
            _logger.debug(f"Restarting {_lam_sym} = {lambda_value} from file")
            if isinstance(self._system, list):
                # Old format: system with saved positions loaded from .s3 stream file.
                system = self._system[index].clone()
                time = system.time()
            else:
                # New format: original input system; time stored in .npz checkpoint.
                system = self._system.clone()
                time = _sr.u(
                    f"{float(_np.load(self._filenames[index]['checkpoint_state'])['time_ps'].item()):.6f} ps"
                )
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

        # Delete an existing trajectory frames.
        system.delete_all_frames()

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

        lambda_value = self._lambda_values[index]
        window = _Window(
            index=index,
            lambda_value=lambda_value,
            rest2_scale=self._rest2_scale_factors[
                self._lambda_energy.index(lambda_value)
            ],
            device=device,
        )

        time = self._remaining_time(system, window, is_restart)
        if time is None:
            return _sr.u("0ps")

        system = self._create_samplers(system, window)

        # Don't minimise on restart.
        if self._config.minimise and not is_restart:
            system = self._minimise_window(system, window)

        is_equilibrated = (
            not is_restart and self._config.equilibration_time.value() > 0.0
        )
        if is_equilibrated:
            system = self._equilibrate_window(system, window)

        self._set_energy_lambdas(window)
        self._create_production_dynamics(system, window, is_restart, is_equilibrated)

        # Write a checkpoint immediately after equilibration so that a restart
        # after an early production crash doesn't need to re-equilibrate.
        if is_equilibrated:
            self._write_post_equilibration_checkpoint(system, window)

        window.step = self._num_steps(self._config.runtime - time)
        window.num_energy_samples = self._num_energy_samples(window)

        if self._config.checkpoint_frequency.value() > 0.0:
            self._run_checkpoint_blocks(window, time)
        else:
            self._run_without_checkpoints(window, time)

        return time

    def _remaining_time(self, system, window, is_restart):
        """
        Work out the simulation time remaining for a window, and the block
        that it starts from.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        window: _Window
            The window being simulated.

        is_restart: bool
            Whether this is a restart simulation.

        Returns
        -------

        time: sire.units.GeneralUnit
            The time remaining, or None if the window is already complete.
        """
        if not is_restart:
            self._start_block = 0
            return self._config.runtime

        if isinstance(self._system, list):
            time = system.time()
        else:
            # New format: time stored in .npz, not in the Sire system.
            time = _sr.u(
                f"{float(_np.load(self._filenames[window.index]['checkpoint_state'])['time_ps'].item()):.6f} ps"
            )
            system.set_time(time)
        if time > self._config.runtime - self._config.timestep:
            _logger.success(
                f"{_lam_sym} = {window.lambda_value} already complete. Skipping."
            )
            return None

        # Work out the current block number.
        if self._config.checkpoint_frequency.value() > 0.0:
            self._start_block = int(
                round(time.value() / self._config.checkpoint_frequency.value(), 12)
            )
        else:
            self._start_block = 0

        return self._config.runtime - time

    @staticmethod
    def _gradient_lambdas(lambda_base, increment=0.001):
        """
        The lambda values either side of a window that are used for
        finite-difference gradient analysis.

        Parameters
        ----------

        lambda_base: float
            The lambda value of the window.

        increment: float
            The distance to each neighbouring lambda value.

        Returns
        -------

        list of float
            The lambda values.
        """
        if lambda_base + increment > 1.0 and lambda_base - increment < 0.0:
            raise ValueError("Increment too large")
        if lambda_base + increment > 1.0:
            lam_vals = [lambda_base - increment]
        elif lambda_base - increment < 0.0:
            lam_vals = [lambda_base + increment]
        else:
            lam_vals = [lambda_base - increment, lambda_base + increment]
        return lam_vals

    def _create_samplers(self, system, window):
        """
        Create the GCMC and terminal flip samplers for a window, if required.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        window: _Window
            The window being simulated, which holds the samplers.

        Returns
        -------

        system: :class: `System <sire.system.System>`
            The system to simulate, which includes the GCMC waters.
        """
        lambda_value = window.lambda_value

        # Prepare the GCMC sampler.
        if self._config.gcmc:
            _logger.info(f"Preparing GCMC sampler at {_lam_sym} = {lambda_value:.5f}")

            try:
                from loch import GCMCSampler
            except:
                msg = "loch is not installed. GCMC sampling cannot be performed."
                _logger.error(msg)

            window.gcmc_sampler = GCMCSampler(
                system,
                device=int(window.device),
                lambda_value=lambda_value,
                rest2_scale=window.rest2_scale,
                ghost_file=self._filenames[window.index]["gcmc_ghosts"],
                **self._gcmc_kwargs,
            )

            # Get the GCMC system.
            system = window.gcmc_sampler.system()

        # Create the terminal flip sampler (if terminal groups were detected).
        if self._terminal_groups:
            from ._samplers import TerminalFlipSampler

            window.terminal_flip_sampler = TerminalFlipSampler(
                self._terminal_groups,
                float(self._config.temperature.value()),
            )
            _logger.info(
                f"Terminal flip sampler ready at {_lam_sym} = {lambda_value:.5f} "
                f"(every {self._config.terminal_flip_frequency})"
            )

        return system

    def _minimise_window(self, system, window):
        """
        Minimise a window before it is simulated.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to minimise.

        window: _Window
            The window being simulated.

        Returns
        -------

        system: :class: `System <sire.system.System>`
            The minimised system, or the original if minimisation failed and
            errors are not raised.
        """
        constraint = self._config.constraint
        perturbable_constraint = self._config.perturbable_constraint

        # Don't use constraints during minimisation.
        if not self._config.minimisation_constraints:
            constraint = "none"
            perturbable_constraint = "none"
        # We will be performing an equilibration stage.
        elif self._config.equilibration_time.value() > 0.0:
            # Don't use constraints during equilibration.
            if not self._config.equilibration_constraints:
                constraint = "none"
                perturbable_constraint = "none"

        try:
            system = self._minimisation(
                system,
                window.lambda_value,
                rest2_scale=window.rest2_scale,
                device=window.device,
                constraint=constraint,
                perturbable_constraint=perturbable_constraint,
                gcmc_sampler=window.gcmc_sampler,
            )
        except Exception as e:
            msg = f"Minimisation failed for {_lam_sym} = {window.lambda_value:.5f}: {e}"
            if self._config.minimisation_errors:
                raise RuntimeError(msg)
            else:
                _logger.warning(msg)

        return system

    def _equilibrate_window(self, system, window):
        """
        Equilibrate a window without saving energies or frames.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to equilibrate.

        window: _Window
            The window being simulated.

        Returns
        -------

        system: :class: `System <sire.system.System>`
            The equilibrated system, with its time reset.
        """
        lambda_value = window.lambda_value
        gcmc_sampler = window.gcmc_sampler

        try:
            _logger.info(f"Equilibrating at {_lam_sym} = {lambda_value:.5f}")

            # Copy the dynamics kwargs.
            dynamics_kwargs = self._dynamics_kwargs.copy()

            # Overload the dynamics kwargs with the simulation options.
            dynamics_kwargs.update(
                {
                    "device": window.device,
                    "lambda_value": lambda_value,
                    "rest2_scale": window.rest2_scale,
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

            # Write the OpenMM XML file to the output directory.
            if self._config.save_xml:
                _logger.info(f"Writing OpenMM XML for {_lam_sym} = {lambda_value:.5f}")
                dynamics.to_xml(self._filenames[window.index]["xml"])

            # Equilibrate with GCMC moves.
            if gcmc_sampler is not None:
                # Bind the GCMC sampler to the dynamics object.
                gcmc_sampler.bind_dynamics(dynamics)

                # This context is built from the system as it was committed by
                # minimisation, so it doesn't carry the water state that the
                # moves performed there left behind.
                gcmc_sampler.push()
                try:
                    gcmc_sampler._set_water_state(dynamics.context(), force=True)
                finally:
                    gcmc_sampler.pop()

                _logger.info(
                    f"Equilibrating with GCMC moves at {_lam_sym} = {lambda_value:.5f}"
                )

                gcmc_sampler.push()
                try:
                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())
                finally:
                    gcmc_sampler.pop()

            # Run without saving energies or frames.
            dynamics.run(
                self._config.equilibration_time,
                energy_frequency=0,
                frame_frequency=0,
                save_velocities=False,
                auto_fix_minimise=self._config.auto_fix_minimise,
                save_crash_report=self._config.save_crash_report,
            )

            # Commit the system.
            system = dynamics.commit()

            # Reset the timer.
            if self._initial_time[window.index].value() != 0:
                system.set_time(self._initial_time[window.index])
            else:
                system.set_time(_sr.u("0ps"))

        except Exception as e:
            raise RuntimeError(f"Equilibration failed: {e}")

        return system

    def _set_energy_lambdas(self, window):
        """
        Work out the lambda values, and their REST2 scaling factors, at which
        energies are sampled for a window.

        Parameters
        ----------

        window: _Window
            The window being simulated, which holds the lambda values.
        """
        # Work out the lambda values for finite-difference gradient analysis.
        window.lambda_grad = self._gradient_lambdas(window.lambda_value)

        # Create the sorted array of lambda values for energy sampling.
        window.lambda_energy = sorted(self._lambda_energy.copy())

        # Create the lambda array.
        lambda_array = window.lambda_energy.copy()

        # Add the lambda values for finite-difference gradient analysis.
        lambda_array.extend(window.lambda_grad)

        # Add additional REST2 scaling factors.
        rest2_scale_factors = self._rest2_scale_factors.copy()
        rest2_scale_factors.extend([window.rest2_scale] * len(window.lambda_grad))

        # Get the indices of the lambda values in sorted order.
        sorted_indices = [
            i for i, _ in sorted(enumerate(lambda_array), key=lambda x: x[1])
        ]

        # Sort the lambda values, and the scaling factors to match.
        window.lambda_array = sorted(lambda_array)
        window.rest2_scale_factors = [rest2_scale_factors[i] for i in sorted_indices]

        # Set the number of neighbours used for the energy calculation. If not
        # None, then we add one to account for the extra windows used for
        # finite-difference gradient analysis.
        if self._config.num_energy_neighbours is not None:
            window.num_energy_neighbours = self._config.num_energy_neighbours + 1

    def _create_production_dynamics(self, system, window, is_restart, is_equilibrated):
        """
        Create the dynamics object for production, restoring the state of a
        restart and the state of the samplers.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        window: _Window
            The window being simulated, which holds the dynamics object.

        is_restart: bool
            Whether this is a restart simulation.

        is_equilibrated: bool
            Whether the window was equilibrated, in which case the OpenMM XML
            file has already been written.
        """
        index = window.index
        lambda_value = window.lambda_value
        gcmc_sampler = window.gcmc_sampler

        _logger.info(f"Running dynamics at {_lam_sym} = {lambda_value:.5f}")

        # Copy the dynamics kwargs.
        dynamics_kwargs = self._dynamics_kwargs.copy()

        # Overload the dynamics kwargs with the simulation options.
        dynamics_kwargs.update(
            {
                "device": window.device,
                "lambda_value": lambda_value,
                "rest2_scale": window.rest2_scale,
            }
        )

        # Create the dynamics object.
        dynamics = system.dynamics(**dynamics_kwargs)
        window.dynamics = dynamics

        # Write the OpenMM XML file to the output directory (only if not already
        # written during equilibration).
        if self._config.save_xml and not is_restart and not is_equilibrated:
            _logger.info(f"Writing OpenMM XML for {_lam_sym} = {lambda_value:.5f}")
            dynamics.to_xml(self._filenames[index]["xml"])

        # For new-format restarts, apply saved positions/velocities/box to context.
        _new_format_restart = is_restart and not isinstance(self._system, list)
        if _new_format_restart:
            import openmm.unit as _omm_unit

            _npz_state = _np.load(self._filenames[index]["checkpoint_state"])

            # Set the box before the positions, since a barostat may have
            # changed it between the state being saved and restored.
            if "box" in _npz_state:
                from openmm import Vec3 as _Vec3

                _box = _npz_state["box"]
                dynamics.context().setPeriodicBoxVectors(
                    _Vec3(*_box[0]) * _omm_unit.nanometer,
                    _Vec3(*_box[1]) * _omm_unit.nanometer,
                    _Vec3(*_box[2]) * _omm_unit.nanometer,
                )

            dynamics.context().setPositions(
                _npz_state["positions"] * _omm_unit.nanometer
            )
            dynamics.context().setVelocities(
                _npz_state["velocities"] * _omm_unit.nanometer / _omm_unit.picosecond
            )

        # Reset the GCMC sampler. This resets the sampling statistics and clears
        # the associated OpenMM forces.
        if gcmc_sampler is not None:
            gcmc_sampler.reset()

            # Bind the GCMC sampler to the dynamics object.
            gcmc_sampler.bind_dynamics(dynamics)

            # If this is a restart, then we need to reset the GCMC water state
            # to match that of the restart system.
            if self._is_restart:
                if isinstance(self._system, list):
                    # Old format: restore ghost waters from cached indices/positions.
                    from openmm.unit import angstrom

                    gcmc_sampler.push()
                    try:
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
                    finally:
                        gcmc_sampler.pop()

                    # Finally, reset the context positions to match the restart system.
                    dynamics.context().setPositions(
                        self._restart_positions[index] * angstrom
                    )
                else:
                    # New format: positions already applied; restore ghost water state.
                    ghost_idxs = _npz_state["ghost_water_indices"].tolist()
                    gcmc_sampler.push()
                    try:
                        gcmc_sampler._set_water_state(
                            dynamics.context(),
                            states=_np.ones(len(gcmc_sampler._water_indices)),
                            force=True,
                        )
                        if ghost_idxs:
                            gcmc_sampler._set_water_state(
                                dynamics.context(),
                                ghost_idxs,
                                states=_np.zeros(len(gcmc_sampler._water_indices)),
                                force=True,
                            )
                    finally:
                        gcmc_sampler.pop()

            # Otherwise, reset the water state in the new context to match the
            # system that the preparation stages left behind. Both minimisation
            # and equilibration perform GCMC moves, so the state held by the
            # sampler is ahead of the one the context was built with.
            else:
                # Reset the water state.
                gcmc_sampler.push()
                try:
                    gcmc_sampler._set_water_state(
                        dynamics.context(),
                        force=True,
                    )
                finally:
                    gcmc_sampler.pop()

        # Restore sampler statistics from a previous run.
        if self._is_restart:
            stats = self._load_sampler_stats(index)
            if stats is not None:
                if gcmc_sampler is not None and "gcmc" in stats:
                    gcmc_stats = stats["gcmc"]
                    if self._is_legacy_gcmc_stats(gcmc_stats):
                        gcmc_stats = self._convert_legacy_gcmc_stats(
                            gcmc_stats, self._lambda_values[index]
                        )
                    gcmc_sampler.restore_stats(gcmc_stats)
                terminal_flip_sampler = window.terminal_flip_sampler
                if terminal_flip_sampler is not None and "terminal_flip" in stats:
                    attempted, accepted = stats["terminal_flip"]
                    terminal_flip_sampler.reset(attempted, accepted)

        # Log the GCMC sphere centre using the actual context positions
        # (accurate for both fresh runs and restarts).
        if gcmc_sampler is not None and gcmc_sampler._reference is not None:
            import openmm.unit as _omm_unit

            state = dynamics.context().getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(
                _omm_unit.angstrom
            )
            target = gcmc_sampler._get_target_position(positions)
            _logger.info(
                f"Initial GCMC sphere centre at {_lam_sym} = {lambda_value:.5f}: "
                f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] A"
            )

    def _write_post_equilibration_checkpoint(self, system, window):
        """
        Write a checkpoint for a window straight after equilibration.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The equilibrated system.

        window: _Window
            The window being simulated.
        """
        lock = _FileLock(self._lock_file)
        with lock.acquire(timeout=self._config.timeout.to("seconds")):
            _, error = self._checkpoint(
                system,
                window.index,
                block=-1,
                speed=0.0,
                lambda_energy=window.lambda_energy,
                lambda_grad=window.lambda_grad,
                context=window.dynamics.context(),
                gcmc_sampler=window.gcmc_sampler,
            )
            if error is not None:
                msg = (
                    f"Post-equilibration checkpoint failed for {_lam_sym} = "
                    f"{window.lambda_value:.5f}:\n{error}"
                )
                _logger.error(msg)
                raise error
            _logger.info(
                f"Writing post-equilibration checkpoint "
                f"for {_lam_sym} = {window.lambda_value:.5f}"
            )

    def _run_checkpoint_blocks(self, window, time):
        """
        Run production dynamics in blocks, writing a checkpoint after each.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        time: sire.units.GeneralUnit
            The simulation time remaining.
        """
        lambda_value = window.lambda_value
        dynamics = window.dynamics

        # The number of steps in each block. The last is shorter if the time
        # remaining isn't a whole number of checkpoint intervals.
        num_steps = self._num_steps(time)
        checkpoint_steps = self._num_steps(self._config.checkpoint_frequency)
        blocks = [checkpoint_steps] * (num_steps // checkpoint_steps)
        if num_steps % checkpoint_steps > 0:
            blocks.append(num_steps % checkpoint_steps)
        total_blocks = self._start_block + len(blocks)

        for i, block_steps in enumerate(blocks):
            block = self._start_block + i
            is_final_block = i == len(blocks) - 1
            block_start = _timer()

            try:
                self._run_block(window, block_steps)
            except Exception as e:
                raise RuntimeError(
                    f"Dynamics block {block + 1} for {_lam_sym} = {lambda_value:.5f} failed: {e}"
                )

            try:
                # Commit the current system.
                system = dynamics.commit()

                # Calculate the speed in nanoseconds per day.
                duration = (block_steps * self._config.timestep).to("ns")
                speed = duration / ((_timer() - block_start) / 86400)

                self._write_checkpoint(window, system, block, speed, is_final_block)

                # Delete all trajectory frames from the Sire system within the
                # dynamics object.
                dynamics._d._sire_mols.delete_all_frames()

                _logger.info(
                    f"Finished block {block + 1} of {total_blocks} "
                    f"for {_lam_sym} = {lambda_value:.5f}"
                )

                self._log_samplers(window, system)

                if is_final_block:
                    _logger.success(
                        f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Checkpoint failed for {_lam_sym} = {lambda_value:.5f}: {e}"
                )

    def _run_without_checkpoints(self, window, time):
        """
        Run production dynamics in a single block, writing one checkpoint at
        the end.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        time: sire.units.GeneralUnit
            The simulation time remaining.
        """
        lambda_value = window.lambda_value

        start = _timer()

        try:
            self._run_block(window, self._num_steps(time))
        except Exception as e:
            raise RuntimeError(
                f"Dynamics for {_lam_sym} = {lambda_value:.5f} failed: {e}"
            )

        # Commit the current system.
        system = window.dynamics.commit()

        # Calculate the speed in nanoseconds per day.
        speed = time.to("ns") / ((_timer() - start) / 86400)

        try:
            self._write_checkpoint(window, system, 0, speed, is_final_block=True)
        except Exception as e:
            msg = f"Checkpoint failed for {_lam_sym} = {lambda_value:.5f}: {e}"
            _logger.error(msg)
            raise RuntimeError(msg)

        _logger.success(
            f"{_lam_sym} = {lambda_value:.5f} complete, speed = {speed:.2f} ns day-1"
        )

    def _run_block(self, window, num_steps):
        """
        Run a block of production dynamics. This is split into sub-blocks
        when Monte Carlo moves are performed or energy components are saved.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        num_steps: int
            The number of integration steps in the block.
        """
        index = window.index
        lambda_value = window.lambda_value
        dynamics = window.dynamics
        gcmc_sampler = window.gcmc_sampler
        terminal_flip_sampler = window.terminal_flip_sampler

        if (
            gcmc_sampler is None
            and terminal_flip_sampler is None
            and not self._config.save_energy_components
        ):
            self._run_dynamics(window, num_steps)
            window.step += num_steps
            return

        # The number of steps between each MC move, or between energy samples
        # when only saving energy components.
        gcmc_steps = (
            self._num_steps(self._config.gcmc_frequency)
            if gcmc_sampler is not None
            else None
        )
        flip_steps = (
            self._num_steps(self._config.terminal_flip_frequency)
            if terminal_flip_sampler is not None
            else None
        )
        intervals = [n for n in (gcmc_steps, flip_steps) if n is not None]
        if not intervals:
            intervals = [self._num_steps(self._config.energy_frequency)]

        save_frames = gcmc_sampler is not None and self._config.frame_frequency > 0
        frame_steps = self._num_steps(self._config.frame_frequency)

        end = window.step + num_steps

        while window.step < end:
            # Run to the next move, counting from the start of the simulation so
            # that the moves keep their frequency across blocks.
            step = min(min(n - window.step % n for n in intervals), end - window.step)

            needs_pre_run_snapshot = False

            # GCMC move.
            if gcmc_sampler is not None and window.step % gcmc_steps == 0:
                _logger.info(f"Performing GCMC move at {_lam_sym} = {lambda_value:.5f}")
                gcmc_sampler.push()
                try:
                    gcmc_sampler.move(dynamics.context())
                finally:
                    gcmc_sampler.pop()
                needs_pre_run_snapshot = self._config.auto_fix_minimise

            # Terminal flip move.
            if terminal_flip_sampler is not None and window.step % flip_steps == 0:
                _logger.info(
                    f"Performing terminal flip move at {_lam_sym} = {lambda_value:.5f}"
                )
                flip_accepted = terminal_flip_sampler.move(dynamics.context())
                if flip_accepted:
                    if self._config.auto_fix_minimise:
                        needs_pre_run_snapshot = True
                    if self._config.randomise_velocities:
                        dynamics.randomise_velocities()

            # Snapshot the context state for crash recovery if any MC move
            # changed positions.
            if needs_pre_run_snapshot:
                dynamics._d._pre_run_state = dynamics.context().getState(
                    getPositions=True, getVelocities=True
                )

            # Record ghost residues immediately before the dynamics block if a
            # frame will be saved within it.
            if (
                save_frames
                and (window.step + step) // frame_steps > window.step // frame_steps
            ):
                self._save_ghost_residues(index, gcmc_sampler)

            # Run the dynamics block.
            self._run_dynamics(
                window,
                step,
                excess_chemical_potential=(
                    self._mu_ex if gcmc_sampler is not None else None
                ),
                num_waters=(
                    _np.sum(gcmc_sampler.water_state())
                    if gcmc_sampler is not None
                    else None
                ),
            )

            window.step += step

            # Save energy components.
            if self._config.save_energy_components:
                self._save_energy_components(
                    index, dynamics.context(), dynamics.current_time().to("ns")
                )

    def _num_steps(self, duration):
        """
        The number of integration steps in a length of time.

        Parameters
        ----------

        duration: sire.units.GeneralUnit
            The length of time.

        Returns
        -------

        int
            The number of steps, to the nearest whole step.
        """
        return round((duration / self._config.timestep).value())

    @staticmethod
    def _num_energy_samples(window):
        """
        The number of energy samples recorded by a window's dynamics.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        Returns
        -------

        int
            The number of samples.
        """
        energy_trajectory = window.dynamics._d._energy_trajectory
        return 0 if energy_trajectory is None else len(energy_trajectory)

    def _run_dynamics(self, window, num_steps, **kwargs):
        """
        Run production dynamics, sampling energies at every lambda value.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        num_steps: int
            The number of integration steps to run. A number of steps is passed
            to Sire, since it truncates a time that is a rounding error short
            of a whole step.

        kwargs: dict
            Additional keyword arguments for the dynamics run.
        """
        window.dynamics.run(
            num_steps,
            energy_frequency=self._config.energy_frequency,
            frame_frequency=self._config.frame_frequency,
            lambda_windows=window.lambda_array,
            rest2_scale_factors=window.rest2_scale_factors,
            save_velocities=self._config.save_velocities,
            auto_fix_minimise=self._config.auto_fix_minimise,
            num_energy_neighbours=window.num_energy_neighbours,
            null_energy=self._config.null_energy,
            save_crash_report=self._config.save_crash_report,
            **kwargs,
        )

    def _write_checkpoint(self, window, system, block, speed, is_final_block):
        """
        Back up the previous checkpoint of a window, then write the new one and
        the sampler statistics, all under the file lock so that another process
        always reads a consistent set of files.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        system: :class: `System <sire.system.System>`
            The system to checkpoint.

        block: int
            The index of the block that was run.

        speed: float
            The speed of the block in ns per day.

        is_final_block: bool
            Whether this is the last block of the simulation.
        """
        # Only the energy samples recorded since the last checkpoint are new.
        num_energy_samples = self._num_energy_samples(window)

        lock = _FileLock(self._lock_file)
        with lock.acquire(timeout=self._config.timeout.to("seconds")):
            # Backup any existing checkpoint files.
            _, error = self._backup_checkpoint(window.index)
            if error is not None:
                raise error

            # Write the checkpoint files.
            _, error = self._checkpoint(
                system,
                window.index,
                block,
                speed,
                lambda_energy=window.lambda_energy,
                lambda_grad=window.lambda_grad,
                is_final_block=is_final_block,
                context=window.dynamics.context(),
                gcmc_sampler=window.gcmc_sampler,
                num_energy_rows=num_energy_samples - window.num_energy_samples,
            )
            if error is not None:
                raise error
            window.num_energy_samples = num_energy_samples

            # Save sampler statistics alongside the checkpoint.
            self._save_sampler_stats(
                window.index, window.gcmc_sampler, window.terminal_flip_sampler
            )

    def _log_samplers(self, window, system):
        """
        Log the number of waters in the GCMC sampling volume and the terminal
        flip acceptance rate after a block.

        Parameters
        ----------

        window: _Window
            The window being simulated.

        system: :class: `System <sire.system.System>`
            The system after the block.
        """
        lambda_value = window.lambda_value
        gcmc_sampler = window.gcmc_sampler
        terminal_flip_sampler = window.terminal_flip_sampler

        # Log the number of waters within the GCMC sampling volume.
        if gcmc_sampler is not None:
            gcmc_sampler.push()
            try:
                n_moves = gcmc_sampler._num_moves
                acc_str = (
                    f", acceptance rate = {gcmc_sampler.move_acceptance_probability():.3f}"
                    f" (ins = {gcmc_sampler.num_insertions()}, del = {gcmc_sampler.num_deletions()})"
                    if n_moves > 0
                    else ""
                )
                # Count against the context, since dynamics have run since the
                # last move.
                num_waters = gcmc_sampler.num_waters(context=window.dynamics.context())
                _logger.info(
                    f"Current number of waters in GCMC volume at {_lam_sym} = {lambda_value:.5f} "
                    f"(time = {system.time().to('ns'):.4f} ns) is {num_waters}{acc_str}"
                )
            finally:
                gcmc_sampler.pop()

        # Log terminal flip acceptance rate.
        if terminal_flip_sampler is not None:
            _logger.info(
                f"Terminal flip acceptance rate at "
                f"{_lam_sym} = {lambda_value:.5f}: "
                f"{terminal_flip_sampler.acceptance_rate:.3f} "
                f"({terminal_flip_sampler.num_accepted}/"
                f"{terminal_flip_sampler.num_attempted})"
            )

    def _save_sampler_stats(self, index, gcmc_sampler, terminal_flip_sampler):
        """
        Save GCMC and terminal flip sampler statistics to a pickle file.

        Parameters
        ----------

        index : int
            The index of the lambda value.

        gcmc_sampler : GCMCSampler or None
            The GCMC sampler for this replica.

        terminal_flip_sampler : TerminalFlipSampler or None
            The terminal flip sampler for this replica.
        """
        import pickle as _pickle

        stats = {}
        if gcmc_sampler is not None:
            stats["gcmc"] = gcmc_sampler.get_stats()
        if terminal_flip_sampler is not None:
            stats["terminal_flip"] = [
                terminal_flip_sampler.num_attempted,
                terminal_flip_sampler.num_accepted,
            ]
        with open(self._filenames[index]["sampler_stats"], "wb") as f:
            _pickle.dump(stats, f)

    def _load_sampler_stats(self, index):
        """
        Load sampler statistics from a pickle file.

        Parameters
        ----------

        index : int
            The index of the lambda value.

        Returns
        -------

        dict or None
            The sampler statistics, or None if the file does not exist.
        """
        import pickle as _pickle
        from pathlib import Path as _Path

        path = _Path(self._filenames[index]["sampler_stats"])
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return _pickle.load(f)
        except Exception as e:
            _logger.warning(f"Could not load sampler stats for index {index}: {e}")
            return None

    def _minimisation(
        self,
        system,
        lambda_value,
        rest2_scale=1.0,
        device=None,
        constraint="none",
        perturbable_constraint="none",
        gcmc_sampler=None,
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

        gcmc_sampler: :class: `GCMCSampler <loch.GCMCSampler>`
            A GCMC sampler to pre-equilibrate the water with before minimising.
            If None, then no GCMC moves are performed.

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

            # Pre-equilibrate the water before minimising, so that a dry pocket
            # is filled before the geometry relaxes into it. The context is
            # created from the sampler's own system, so its water state already
            # matches and only needs binding.
            if gcmc_sampler is not None:
                gcmc_sampler.bind_dynamics(dynamics)

                _logger.info(
                    f"Pre-equilibrating with GCMC moves at {_lam_sym} = {lambda_value:.5f}"
                )

                gcmc_sampler.push()
                try:
                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())
                finally:
                    gcmc_sampler.pop()

            # Run the minimisation.
            dynamics.minimise(timeout=self._config.timeout)

            # Commit the system.
            system = dynamics.commit()
        except:
            raise

        return system
