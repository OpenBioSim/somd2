__all__ = ["controller"]


class controller:
    """
    Controls the initiation of simulations as well as the assigning of
    resources.
    """

    def __init__(self, system, num_lambda, platform=None):
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
        self._num_lambda = num_lambda
        self._options_set = (
            False  # Will track if simulation options have been set by the user
        )
        # import BioSimSpace as _BSS
        from sire.system import System as sire_system

        # if not isinstance(self._system, (_BSS._SireWrappers.system, sire_system)):
        #    raise TypeError("System must be BioSimSpace or Sire system")

        if not isinstance(self._system, (sire_system)):
            raise TypeError("System must be BioSimSpace or Sire system")

        self._set_platform(platform)

        self._platform_options = self._set_platform_options()

        self._schedule = self.create_lambda_schedule()

    def _set_platform(self, platform=None):
        """
        Sets the platform to be used for simulations.

        Parameters:
        -----------

        platform: str
            The platform to be used for simulations. If None then check for
            CUDA_VISIBLE_DEVICES, otherwise use CPU.
        """
        import os as _os

        if platform is not None:
            if platform not in ["CPU", "CUDA"]:
                raise ValueError("Platform must be CPU or CUDA")
            self._platform = platform
        else:
            if "CUDA_VISIBLE_DEVICES" in _os.environ:
                self._platform = "CUDA"
            else:
                self._platform = "CPU"

    def _set_platform_options(self):
        """
        Sets options for the current platform.

        Returns:
        --------
        platform_options: dict
            Dictionary of platform options.
        """
        import os as _os

        if self._platform == "CPU":
            num_cpu = _os.cpu_count()
            num_workers = 1
            if self._num_lambda > num_cpu:
                num_workers = num_cpu
                cpu_per_worker = 1
            else:
                num_workers = self._num_lambda
                cpu_per_worker = num_cpu // self._num_lambda
            platform_options = {
                "num_workers": num_workers,
                "cpu_per_worker": cpu_per_worker,
            }
        elif self._platform == "CUDA":
            devices = self.zero_CUDA_devices(self.get_CUDA_devices())
            platform_options = {"num_workers": len(devices)}

        else:
            raise ValueError("Platform not recognised")

        self._create_shared_resources()

        return platform_options

    def _create_shared_resources(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        if self._platform == "CUDA":
            from multiprocessing import Manager

            manager = Manager()
            self._lock = manager.Lock()
            self._gpu_pool = manager.list(
                self.zero_CUDA_devices(self.get_CUDA_devices())
            )

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

    def get_platform(self):
        """
        Returns the platform to be used for simulations.
        """
        return self._platform

    @staticmethod
    def create_lambda_schedule():
        """
        Creates a lambda schedule for the simulation.
        Currently just creates a basic morph

        Returns:
        --------
        schedule: :class: `LambdaSchedule <Sire.cas.LambdaSchedule>`
            A sire lambda schedule.
        """
        from sire import cas

        schedule = cas.LambdaSchedule()
        schedule.add_stage(
            "morphing",
            (1 - schedule.lam()) * schedule.initial()
            + schedule.lam() * schedule.final(),
        )
        return schedule

    def _set_lambda_schedule(self):
        """
        Sets the lambda schedule for the simulation.
        """
        self._schedule = self.create_lambda_schedule()

    @staticmethod
    def get_CUDA_devices():
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
    def zero_CUDA_devices(devices):
        """
        Set all device numbers relative to the lowest
        (the device number becomes equal to its index in the list).

        Returns:
        --------
        devices (list): List of zeroed available device numbers.
        """
        return [str(devices.index(value)) for value in devices]

    def run_simulations(self):
        """
        Use concurrent.futures to run lambda windows in paralell

        Returns:
        --------
        results (list): List of simulation results.
        """
        import concurrent.futures as _futures

        results = []
        with _futures.ProcessPoolExecutor(
            max_workers=self._platform_options["num_workers"]
        ) as executor:
            self._lambda_values = [
                round(i / (self._num_lambda - 1), 5) for i in range(0, self._num_lambda)
            ]
            for lambda_value in self._lambda_values:
                kwargs = {"lambda_value": lambda_value}
                jobs = {
                    executor.submit(self.run_single_simulation, **kwargs): lambda_value
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

        return results

    def run_single_simulation(self, lambda_value, temperature=300):
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
        from sire.units import kelvin, atm
        from ._sire_merge_runsim import MergedSimulation
        from loguru import logger as _logger

        def _run(system, map, lambda_value, lam_minimisation=None):
            if lam_minimisation is None:
                try:
                    sim = MergedSimulation(
                        system,
                        map,
                        lambda_val=lambda_value,
                        lambda_array=self._lambda_values,
                        minimise=True,
                        no_bookkeeping_time="2ps",
                    )
                except Exception:
                    _logger.warning(f"System creation at {lambda_value} failed")
                    raise
                try:
                    sim._setup_dynamics(lam_val_min=lam_minimisation)
                    df = sim._run_with_bookkeeping(runtime="10ps")
                except Exception:
                    _logger.warning(
                        f"Minimisation/dynamics at Lambda = {lambda_value} failed, trying again with minimsation at Lambda = {lam_minimisation}"
                    )
                    df = _run(system, map, lambda_value, lam_minimisation=0.0)
                    return df
                else:
                    return df
            else:
                try:
                    sim = MergedSimulation(
                        system,
                        map,
                        lambda_val=lambda_value,
                        lambda_array=self._lambda_values,
                        minimise=True,
                        no_bookkeeping_time="2ps",
                    )
                except Exception:
                    _logger.warning(f"System creation at {lambda_value} failed")
                    raise
                try:
                    sim._setup_dynamics(lam_val_min=lam_minimisation)
                    df = sim._run_with_bookkeeping(runtime="10ps")
                except Exception:
                    _logger.error(
                        f"Minimisation/dynamics at Lambda = {lambda_value} failed, even after minimisation at Lambda = {lam_minimisation}"
                    )
                    raise
                else:
                    return df

        # set all properties not specific to platform
        map = {
            "integrator": "langevin_middle",
            "temperature": temperature * kelvin,
        }
        # Pressure control. Only set if the system has a periodic space.
        if (
            self._system.has_property("space")
            and self._system.property("space").is_periodic()
        ):
            map["pressure"] = 1.0 * atm
        system = self._system.clone()

        if self._platform == "CPU":
            if lambda_value is not None:
                print(
                    f"Running lambda = {lambda_value} using {self._platform_options['cpu_per_worker']} CPUs"
                )
            map["platform"] = self._platform
            map["threads"] = self._platform_options["cpu_per_worker"]
            try:
                df = _run(system, map, lambda_value=lambda_value)
            except Exception:
                raise

        elif self._platform == "CUDA":
            with self._lock:
                gpu_num = self._gpu_pool[0]
                self._remove_gpu_from_pool(gpu_num)
                if lambda_value is not None:
                    print(f"Running lambda = {lambda_value} on GPU {gpu_num}")
            map["platform"] = (self._platform,)
            map["device"] = (gpu_num,)

            try:
                df = _run(system, map, lambda_value=lambda_value)
            except Exception:
                with self._lock:
                    self._update_gpu_pool(gpu_num)
                    print(f"Lambda = {lambda_value} failed", flush=True)
                raise
            else:
                with self._lock:
                    self._update_gpu_pool(gpu_num)
                    print(f"Lambda = {lambda_value} complete", flush=True)

        self.dataframe_to_parquet(
            df,
            metadata={
                "lambda": str(lambda_value),
                "temperature": str(map["temperature"].value()),
                "lambda_array": self._lambda_values,
            },
        )
        return f"Lambda = {lambda_value} complete"

    @staticmethod
    def dataframe_to_parquet(df, metadata):
        """
        Save a dataframe to parquet format with custom metadata.

        Parameters:
        -----------
        df: pandas.DataFrame`
            The dataframe to be saved. In this case containing info required for FEP calculation

        metadata: dict
            Dictionary containing metadata to be saved with the dataframe.
            Currently just temperature and lambda value.
        """

        import pyarrow as pa
        import pyarrow.parquet as pq
        import json

        custom_meta_key = "SOMD2.iot"

        table = pa.Table.from_pandas(df)
        custom_meta_json = json.dumps(metadata)
        existing_meta = table.schema.metadata

        combined_meta = {
            custom_meta_key.encode(): custom_meta_json.encode(),
            **existing_meta,
        }
        table = table.replace_schema_metadata(combined_meta)
        filename = f"Lam_{metadata['lambda'].replace('.','')[:5]}_T_{metadata['temperature']}.parquet"
        pq.write_table(table, filename)


if __name__ == "__main__":
    import sire as sr

    mols = sr.stream.load("Methane_Ethane_direct.bss")
    platform = "CUDA"
    r = controller(mols, platform=platform, num_lambda=10)
    results = r.run_simulations()
