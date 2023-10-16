__all__ = ["Controller"]
from ..config import Config as _Config
from ..io import *


class Controller:
    """
    Controls the initiation of simulations as well as the assigning of
    resources.
    """

    def __init__(self, system, input_options={}):
        """
        Constructor.

        Parameters:
        -----------

        system: :class: `System <Sire.System>`| :class: `System <BioSimSpace._SireWrapper.System>`
            The perturbable system to be simulated.

        num_lambda: int
            The number of lambda windows to be simulated.

        platform: str
            The platform to be used for simulations.

        """
        self._system = system
        self.configure(input_options)
        # query status of processes
        # checkpointing
        from sire.system import System as sire_system

        if not isinstance(self._system, (sire_system)):
            raise TypeError("System must be Sire system")

        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        if self.config.repartition_h_mass:
            self.repartition_h_mass()

    def configure(self, input):
        """
        Configure simulation options.
        This is called upon construction of the Controller object,
        use this if settings need to be changed after construction.

        Parameters:
        -----------
        input: dict
            Dictionary of simulation options.
        """
        self.config = _Config(**input)
        self._lambda_values = [
            round(i / (self.config.num_lambda - 1), 5)
            for i in range(0, self.config.num_lambda)
        ]
        # Save config whenever 'configure' is called to keep it up to date
        if self.config.config_to_file:
            dict_to_yaml(self.config.as_dict(), self.config.output_directory)

    def _create_shared_resources(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        if self.config.platform == "CUDA":
            from multiprocessing import Manager

            self._manager = Manager()
            self._lock = self._manager.Lock()
            if self.config.max_GPUS is None:
                self._gpu_pool = self._manager.list(
                    self.zero_cuda_devices(self.get_cuda_devices())
                )
            else:
                self._gpu_pool = self._manager.list(
                    self.zero_cuda_devices(
                        self.get_cuda_devices()[: self.config.max_GPUS]
                    )
                )

    def _check_space_options(self):
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
        return self.config.as_dict()

    def _update_gpu_pool(self, gpu_num):
        """
        Updates the GPU pool to remove the GPU that has been assigned to a worker.

        Parameters:
        -----------
        gpu_num: str
            The GPU number to be added to the pool.
        """
        self._gpu_pool.append(gpu_num)

    def _remove_gpu_from_pool(self, gpu_num):
        """
        Removes a GPU from the GPU pool.

        Parameters:
        -----------
        gpu_num: str
            The GPU number to be removed from the pool.
        """
        self._gpu_pool.remove(gpu_num)

    def _set_lambda_schedule(self, schedule):
        """
        Sets the lambda schedule.

        Parameters:
        -----------
        schedule: sr.cas.LambdaSchedule
            Lambda schedule to be set.
        """
        self.config.lambda_schedule = schedule

    @staticmethod
    def get_cuda_devices():
        """
        Get list of available GPUs from CUDA_VISIBLE_DEVICES.

        Returns:
        --------
        available_devices (list): List of available device numbers.
        """
        import os as _os

        if _os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            raise ValueError("CUDA_VISIBLE_DEVICES not set")
        else:
            available_devices = _os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
            print("CUDA_VISIBLE_DEVICES set to", available_devices)
            num_gpus = len(available_devices)
            print("Number of GPUs available:", num_gpus)
            return available_devices

    @staticmethod
    def zero_cuda_devices(devices):
        """
        Set all device numbers relative to the lowest
        (the device number becomes equal to its index in the list).

        Returns:
        --------
        devices (list): List of zeroed available device numbers.
        """
        return [str(devices.index(value)) for value in devices]

    def repartition_h_mass(self):
        """
        Perform HMR on the input system.
        Performs a short minimisation before repartitioning for increased stability
        """
        print("Repartitioning hydrogen masses")
        from sire.morph import (
            repartition_hydrogen_masses as _repartition_hydrogen_masses,
        )

        self._system = (
            self._system.minimisation(map={"platform": self.config.platform})
            .run()
            .commit()
        )
        mol = self._system.molecule("molecule property is_perturbable")
        mol = _repartition_hydrogen_masses(mol, mass_factor=self.config.h_mass_factor)
        self._system.update(mol)

    def _initialise_simulation(self, system, lambda_value, device=None):
        """
        Create simulation object.

        Parameters:
        -----------
        system: sire sytem
            The system to be simulated.

        lambda_value: float
            The lambda value for the simulation.

        device: int
            The GPU device number to be used for the simulation.
        """
        from ._run_single_pert import RunSingleWindow
        from loguru import logger as _logger

        try:
            self._sim = RunSingleWindow(
                system,
                lambda_val=lambda_value,
                lambda_array=self._lambda_values,
                config=self.config,
                device=device,
            )
        except Exception:
            _logger.warning(f"System creation at {lambda_value} failed")
            raise

    def _cleanup_simulation(self):
        """
        Used to delete simulation objects once the required data has been extracted.
        """
        self._sim._cleanup()

    def run_simulations(self):
        """
        Use concurrent.futures to run lambda windows in parallel

        Returns:
        --------
        results (list): List of simulation results.
        """
        results = []
        if self.config.run_parallel and (self.config.num_lambda is not None):
            self._create_shared_resources()
            import concurrent.futures as _futures

            # figure out max workers and number of CPUs depending on platform and number of lambda windows
            if self.config.platform == "CPU":
                # Finding number of CPUs per worker is done here rather than in config due to platform dependency
                if self.config.num_lambda > self.config.max_CPU_cores:
                    self.max_workers = self.config.max_CPU_cores
                    cpu_per_worker = 1
                else:
                    self.max_workers = self.config.num_lambda
                    cpu_per_worker = self.config.max_CPU_cores // self.config.num_lambda
                if self.config.extra_args is None:
                    self.config.extra_args = {"threads": cpu_per_worker}
                else:
                    self.config.extra_args["threads"] = cpu_per_worker
            else:
                self.max_workers = len(self._gpu_pool)

            with _futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for lambda_value in self._lambda_values:
                    kwargs = {"lambda_value": lambda_value}
                    jobs = {
                        executor.submit(
                            self.run_single_simulation, **kwargs
                        ): lambda_value
                    }
                for job in _futures.as_completed(jobs):
                    lam = jobs[job]
                    try:
                        result = job.result()
                    except Exception as e:
                        print(e)
                        pass
                    else:
                        results.append(result)

        elif self.config.num_lambda is not None:
            if self.config.platform == "CPU":
                self.config.extra_args = {"threads": self.config.max_CPU_cores}
            self._lambda_values = [
                round(i / (self.config.num_lambda - 1), 5)
                for i in range(0, self.config.num_lambda)
            ]
            for lambda_value in self._lambda_values:
                result = self.run_single_simulation(lambda_value)
                results.append(result)

        else:
            raise ValueError(
                "Vanilla MD not currently supported. Please set num_lambda > 1."
            )

        return results

    def run_single_simulation(self, lambda_value):
        """
        Run a single simulation.

        Parameters:
        -----------
        lambda_value: float
            The lambda value for the simulation.

        temperature: float
            The temperature for the simulation.

        Returns:
        --------
        result: str
            The result of the simulation.
        """
        from loguru import logger as _logger

        _logger.info(f"Running lambda = {lambda_value}")

        def _run(sim):
            # This function is complex due to the mixture of options for minimisation and dynamics
            if self.config.minimise:
                try:
                    df = sim._run()
                    lambda_grad = sim._lambda_grad
                    speed = sim.get_timing()
                    return df, lambda_grad, speed
                except Exception as e:
                    _logger.warning(
                        f"Minimisation/dynamics at Lambda = {lambda_value} failed with the following exception {e}, trying again with minimsation at Lambda = 0."
                    )
                    try:
                        df = sim._run(lambda_minimisation=0.0)
                        lambda_grad = sim._lambda_grad
                        speed = sim.get_timing()
                        return df, lambda_grad, speed
                    except Exception as e:
                        _logger.error(
                            f"Minimisation/dynamics at Lambda = {lambda_value} failed, even after minimisation at Lambda = 0. The following warning was raised: {e}."
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
                        f"Dynamics at Lambda = {lambda_value} failed. The following warning was raised: {e}. This may be due to a lack of minimisation."
                    )

        system = self._system.clone()

        if self.config.platform == "CPU":
            self._initialise_simulation(system, lambda_value)
            try:
                df, lambda_grad, speed = _run(self._sim)
            except Exception:
                raise
            self._sim._cleanup()

        elif self.config.platform == "CUDA":
            if self.config.run_parallel:
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
            except Exception:
                if self.config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
                raise
            else:
                if self.config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
            self._sim._cleanup()

        _ = dataframe_to_parquet(
            df,
            metadata={
                "attrs": df.attrs,
                "lambda": str(lambda_value),
                "lambda_array": self._lambda_values,
                "lambda_grad": lambda_grad,
                "speed": speed,
                "temperature": str(self.config.temperature.value()),
            },
            filepath=self.config.output_directory,
        )
        del system
        _logger.success("Lambda = {} complete".format(lambda_value))
        return f"Lambda = {lambda_value} complete"
