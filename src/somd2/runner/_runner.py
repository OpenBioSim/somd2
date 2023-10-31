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

__all__ = ["Runner"]


import sire as _sr

from ..config import Config as _Config
from ..io import dataframe_to_parquet as _dataframe_to_parquet
from ..io import dict_to_yaml as _dict_to_yaml

# Can't create a self.log variable in this class as it causes errors when passed to a processpool
from ..log import loguru_setup as _loguru_setup


class Runner:
    """
    Controls the initiation of simulations as well as the assigning of
    resources.
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

        num_lambda: int
            The number of lambda windows to be simulated.

        platform: str
            The platform to be used for simulations.

        """

        if not isinstance(system, (str, _sr.system.System)):
            raise TypeError("'system' must be of type 'str' or 'sire.system.System'")

        if isinstance(system, str):
            # Try to load the stream file.
            try:
                self._system = _sr.stream.load(system)
            except:
                raise IOError(f"Unable to load system from stream file: '{system}'")
        else:
            self._system = system

        # Make sure the system contains perturbable molecules.
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        # Link properties to the lambda = 0 end state.
        for mol in self._system.molecules("molecule property is_perturbable"):
            self._system.update(mol.perturbation().link_to_reference().commit())

        # Check for a periodic space.
        self._check_space()

        # Validate the configuration.
        if not isinstance(config, _Config):
            raise TypeError("'config' must be of type 'somd2.config.Config'")
        self._config = config

        # Set the lambda values.
        self._lambda_values = [
            round(i / (self._config.num_lambda - 1), 5)
            for i in range(0, self._config.num_lambda)
        ]

        from sire.mol import Element
        from math import isclose

        # Store the expected hydrogen mass.
        expected_h_mass = Element("H").mass().value()

        # Get the hydrogen mass.
        h_mass = self._system.molecules("property is_perturbable")["element H"][
            0
        ].mass()

        if not isclose(h_mass.value(), expected_h_mass, rel_tol=1e-3):
            raise ValueError(
                "The hydrogen mass in the system is not the expected value of 1.008 g/mol"
            )

        # Repartition hydrogen masses if required.
        if self._config.h_mass_factor > 1:
            self._repartition_h_mass()

        # Save config whenever 'configure' is called to keep it up to date
        if self._config.write_config:
            _dict_to_yaml(self._config.as_dict(), self._config.output_directory)

        # Flag whether this is a GPU simulation.
        self._is_gpu = self._config.platform in ["cuda", "opencl", "hip"]

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
                    self.zero_gpu_devices(self.get_gpu_devices(self._config.platform))
                )
            else:
                self._gpu_pool = self._manager.list(
                    self.zero_gpu_devices(
                        self.get_gpu_devices(self._config.platform)[
                            : self._config.max_gpus
                        ]
                    )
                )

    def _check_space(self):
        """
        Check if the system has a periodic space.
        """
        if (
            self._system.has_property("space")
            and self._system.property("space").is_periodic()
        ):
            self._has_space = True
        else:
            self._has_space = False

    def get_options(self):
        """
        Returns a dictionary of simulation options.

        returns:
        --------
        options: dict
            Dictionary of simulation options.
        """
        return self._config.as_dict()

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

    def _set_lambda_schedule(self, schedule):
        """
        Sets the lambda schedule.

        Parameters
        ----------

        schedule: sr.cas.LambdaSchedule
            Lambda schedule to be set.
        """
        self._config.lambda_schedule = schedule

    @staticmethod
    def get_gpu_devices(platform, _log_level="INFO"):
        """
        Get list of available GPUs from CUDA_VISIBLE_DEVICES,
        OPENCL_VISIBLE_DEVICES, or HIP_VISIBLE_DEVICES.

        Parameters
        ----------

        platform: str
            The GPU platform to be used for simulations.

        Returns
        --------

        available_devices : [int]
            List of available device numbers.
        """
        _logger = _loguru_setup(level=_log_level)

        if not isinstance(platform, str):
            raise TypeError("'platform' must be of type 'str'")

        platform = platform.lower().replace(" ", "")

        if platform not in ["cuda", "opencl", "hip"]:
            raise ValueError("'platform' must be one of 'cuda', 'opencl', or 'hip'.")

        import os as _os

        if platform == "cuda":
            if _os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                raise ValueError("CUDA_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                _logger.info("CUDA_VISIBLE_DEVICES set to", available_devices)
        elif platform == "opencl":
            if _os.environ.get("OPENCL_VISIBLE_DEVICES") is None:
                raise ValueError("OPENCL_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("OPENCL_VISIBLE_DEVICES").split(",")
                _logger.info("OPENCL_VISIBLE_DEVICES set to", available_devices)
        elif platform == "hip":
            if _os.environ.get("HIP_VISIBLE_DEVICES") is None:
                raise ValueError("HIP_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("HIP_VISIBLE_DEVICES").split(",")
                _logger.info("HIP_VISIBLE_DEVICES set to", available_devices)

        num_gpus = len(available_devices)
        _logger.info("Number of GPUs available:", num_gpus)

        return available_devices

    @staticmethod
    def zero_gpu_devices(devices):
        """
        Set all device numbers relative to the lowest
        (the device number becomes equal to its index in the list).

        Returns
        -------

        devices : [int]
            List of zeroed available device numbers.
        """
        return [str(devices.index(value)) for value in devices]

    def _repartition_h_mass(self):
        """
        Reprartition hydrogen masses.
        """

        from sire.morph import (
            repartition_hydrogen_masses as _repartition_hydrogen_masses,
        )

        self._system = _repartition_hydrogen_masses(
            self._system, mass_factor=self._config.h_mass_factor
        )

    def _initialise_simulation(self, system, lambda_value, device=None):
        """
        Create simulation object.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        lambda_value: float
            The lambda value for the simulation.

        device: int
            The GPU device number to be used for the simulation.
        """
        from ._dynamics import Dynamics

        _logger = _loguru_setup(level=self._config.log_level)
        try:
            self._sim = Dynamics(
                system,
                lambda_val=lambda_value,
                lambda_array=self._lambda_values,
                config=self._config,
                device=device,
                has_space=self._has_space,
            )
        except:
            _logger.warning(f"System creation at {lambda_value} failed")
            raise

    def _cleanup_simulation(self):
        """
        Used to delete simulation objects once the required data has been extracted.
        """
        self._sim._cleanup()

    def run(self):
        """
        Use concurrent.futures to run lambda windows in parallel

        Returns
        --------

        results : [str]
            List of simulation results.
        """
        # self._logger = _loguru_setup(level=self._config.log_level)
        results = []
        if self._config.run_parallel and (self._config.num_lambda is not None):
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
                    threads_per_worker = (
                        self._config.max_threads // self._config.num_lambda
                    )
                self._config.extra_args = {"threads": threads_per_worker}

            # (Multi-)GPU platform.
            elif self._is_gpu:
                self.max_workers = len(self._gpu_pool)
                self._config.extra_args = {}

            # All other platforms.
            else:
                self._max_workers = 1
                self._config.extra_args = {}

            import concurrent.futures as _futures

            with _futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for lambda_value in self._lambda_values:
                    kwargs = {"lambda_value": lambda_value}
                    jobs = {executor.submit(self.run_window, **kwargs): lambda_value}
                try:
                    for job in _futures.as_completed(jobs):
                        lam = jobs[job]
                        try:
                            result = job.result()
                        except Exception as e:
                            print(e)
                            pass
                        else:
                            results.append(result)
                # Kill all current and future jobs if keyboard interrupt.
                except KeyboardInterrupt:
                    for pid in executor._processes:
                        executor._processes[pid].terminate()

        # Serial configuration.
        elif self._config.num_lambda is not None:
            if self._config.platform == "cpu":
                self._config.extra_args = {"threads": self._config.max_threads}
            self._lambda_values = [
                round(i / (self._config.num_lambda - 1), 5)
                for i in range(0, self._config.num_lambda)
            ]
            for lambda_value in self._lambda_values:
                result = self.run_window(lambda_value)
                results.append(result)

        else:
            raise ValueError(
                "Vanilla MD not currently supported. Please set num_lambda > 1."
            )

        return results

    def run_window(self, lambda_value):
        """
        Run a single simulation.

        Parameters
        ----------

        lambda_value: float
            The lambda value for the simulation.

        temperature: float
            The temperature for the simulation.

        Returns
        -------

        result: str
            The result of the simulation.
        """
        _logger = _loguru_setup(level=self._config.log_level)
        _logger.info(f"Running lambda = {lambda_value}")

        def _run(sim):
            # This function is complex due to the mixture of options for minimisation and dynamics
            if self._config.minimise:
                try:
                    df = sim._run()
                    lambda_grad = sim._lambda_grad
                    speed = sim.get_timing()
                    return df, lambda_grad, speed
                except Exception as e:
                    _logger.warning(
                        f"Minimisation/dynamics at lambda = {lambda_value} failed with the "
                        f"following exception {e}, trying again with minimsation at lambda = 0."
                    )
                    try:
                        df = sim._run(lambda_minimisation=0.0)
                        lambda_grad = sim._lambda_grad
                        speed = sim.get_timing()
                        return df, lambda_grad, speed
                    except Exception as e:
                        _logger.error(
                            f"Minimisation/dynamics at lambda = {lambda_value} failed, even after "
                            f"minimisation at lambda = 0. The following warning was raised: {e}."
                        )
                        raise
            else:
                try:
                    df = sim._run()
                    lambda_grad = sim._lambda_grad
                    speed = sim.get_timing()
                    return df, lambda_grad, speed
                except Exception as e:
                    _logger.error(
                        f"Dynamics at lambda = {lambda_value} failed. The following warning was "
                        f"raised: {e}. This may be due to a lack of minimisation."
                    )

        system = self._system.clone()

        # GPU platform.
        if self._is_gpu:
            if self._config.run_parallel:
                with self._lock:
                    gpu_num = self._gpu_pool[0]
                    self._remove_gpu_from_pool(gpu_num)
                    if lambda_value is not None:
                        print(f"Running lambda = {lambda_value} on GPU {gpu_num}")
            # Assumes that device for non-parallel GPU jobs is 0
            else:
                gpu_num = 0
            self._initialise_simulation(system, lambda_value, device=gpu_num)
            try:
                df, lambda_grad, speed = _run(self._sim)
            except:
                if self._config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
                raise
            else:
                if self._config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
            self._sim._cleanup()

        # All other platforms.
        else:
            self._initialise_simulation(system, lambda_value)
            try:
                df, lambda_grad, speed = _run(self._sim)
            except:
                raise
            self._sim._cleanup()

        _ = _dataframe_to_parquet(
            df,
            metadata={
                "attrs": df.attrs,
                "lambda": str(lambda_value),
                "lambda_array": self._lambda_values,
                "lambda_grad": lambda_grad,
                "speed": speed,
                "temperature": str(self._config.temperature.value()),
            },
            filepath=self._config.output_directory,
        )
        del system
        _logger.success("Lambda = {} complete".format(lambda_value))
        return f"Lambda = {lambda_value} complete"
