__all__ = ["Controller"]
from ..config import Config as _Config


class Controller:
    """
    Controls the initiation of simulations as well as the assigning of
    resources.
    """

    def __init__(self, system):
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
        self._configured = (
            False  # Will track if simulation options have been set by the user
        )
        # query status of processes
        # checkpointing
        # simulation settings in metadata or to yaml?
        from sire.system import System as sire_system

        if not isinstance(self._system, (sire_system)):
            raise TypeError("System must be Sire system")

        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

    def configure(self, input):
        """
        Configure simulation options.

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
        self._configured = True

    def _create_shared_resources(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        if self.config.platform == "CUDA":
            from multiprocessing import Manager

            # Storing the lock but not the manager - make self.manager
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
        if self._configured:
            return self.config.as_dict()
        else:
            self.config = _Config()
            print(f"No simulation options set, using defaults: {self.config.as_dict()}")
            self._configured = True

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
        if self._configured:
            self.config.lambda_schedule = schedule
        else:
            self.config = _Config(lambda_schedule=schedule)
            self._configured = True

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
        if not self._configured:
            print(
                "No configuration set... using defaults. To set configuration use configure() method."
            )
            self.configure({})
            self.configured = True
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
        if self.config.config_to_file:
            self.dict_to_yaml(self.config.as_dict(), self.config.output_directory)

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

        if not self._configured:
            print(
                "No configuration set... using defaults. To set configuration use configure() method."
            )
            self.configure({})
            self.configured = True

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

        self.dataframe_to_parquet(
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
        return f"Lambda = {lambda_value} complete"

    @staticmethod
    def dataframe_to_parquet(df, metadata, filepath=None):
        """
        Save a dataframe to parquet format with custom metadata.

        Parameters:
        -----------
        df: pandas.DataFrame
            The dataframe to be saved. In this case containing info required for FEP calculation

        metadata: dict
            Dictionary containing metadata to be saved with the dataframe.
            Currently just temperature and lambda value.

        filepath: str or pathlib.PosixPath
            The of the parent directory in to which the parquet file will be saved.
            If None, save to current working directory.
        """

        import pyarrow as pa
        import pyarrow.parquet as pq
        import json
        from pathlib import Path as _Path

        if filepath is None:
            filepath = _Path.cwd()
        elif isinstance(filepath, str):
            filepath = _Path(filepath)

        custom_meta_key = "somd2"

        table = pa.Table.from_pandas(df)
        custom_meta_json = json.dumps(metadata)
        existing_meta = table.schema.metadata

        combined_meta = {
            custom_meta_key.encode(): custom_meta_json.encode(),
            **existing_meta,
        }
        table = table.replace_schema_metadata(combined_meta)
        filename = f"Lam_{metadata['lambda'].replace('.','')[:5]}_T_{metadata['temperature']}.parquet"
        pq.write_table(table, filepath / filename)

    @staticmethod
    def dict_to_yaml(data_dict, file_path, filename="config.yaml"):
        """
        Write a dictionary to a YAML file.

        Parameters:
        -----------
        data_dict: dict
            The dictionary to be written to a YAML file.

        file_path: str or pathlib.PosixPath
            The path to the YAML file to be written.

        filename: str
            The name of the YAML file to be written (default 'config.yaml').
        """
        from pathlib import Path as _Path
        import yaml as _yaml

        try:
            file_path = _Path(file_path) / filename

            # Ensure the parent directory for the file exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open the file in write mode and write the dictionary as YAML
            with file_path.open("w") as yaml_file:
                _yaml.dump(
                    data_dict,
                    yaml_file,
                )
            print("config written")
        except Exception as e:
            print(f"Error writing the dictionary to {file_path}: {e}")
