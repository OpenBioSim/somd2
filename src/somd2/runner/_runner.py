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
                self._platform = "GPU"
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
        Set all device numbers relative to the lowest (the device number becomes equal to its index in the list).

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

        results = {}
        with _futures.ProcessPoolExecutor(
            max_workers=self._platform_options["num_workers"]
        ) as executor:
            jobs = []

            self._lambda_values = [
                i / (self._num_lambda - 1) for i in range(0, self._num_lambda)
            ]
            for lambda_value in self._lambda_values:
                kwargs = {"lambda_value": lambda_value}
                jobs.append(executor.submit(self.run_single_simulation, **kwargs))
            for job in _futures.as_completed(jobs):
                lambda_val, result = job.result()
                results[str(lambda_val)] = result

        return results

    def run_single_simulation(self, lambda_value, temperature=300):
        """
        Run a single simulation.

        Parameters:
        -----------
        lambda_value: float
            The lambda value for the simulation.

        Returns:
        --------
        result: str
            The result of the simulation.
        """
        from sire.units import kelvin, atm
        from _sire_merge_runsim import MergedSimulation

        # set all properties not specific to platform
        map = {
            "Integrator": "langevin_middle",
            "Temperature": temperature * kelvin,
            "Pressure": 1.0 * atm,
        }
        if self._platform == "CPU":
            if lambda_value is not None:
                print(
                    f"Running lambda = {lambda_value} using {self._platform_options['cpu_per_worker']} CPUs"
                )
            map["Platform"] = self._platform
            map["Threads"] = self._platform_options["cpu_per_worker"]
            # run_merged(self._system, lambda_value, map, minimise=False)
            sim = MergedSimulation(
                self._system,
                map,
                lambda_val=lambda_value,
                minimise=True,
                no_bookkeeping_time="2ps",
            )
            df = sim._run_with_bookkeeping(runtime="10ps")
            if lambda_value is not None:
                return f"Lambda = {lambda_value} complete"
            else:
                return f"Temperature = {temperature} complete"

        elif self._platform == "CUDA":
            with self._lock:
                gpu_num = self._gpu_pool[0]
                self._remove_gpu_from_pool(gpu_num)
                if lambda_value is not None:
                    print(f"Running lambda = {lambda_value} on GPU {gpu_num}")
            map["Platform"] = (self._platform,)
            map["device"] = (gpu_num,)
            sim = MergedSimulation(
                self._system,
                map,
                lambda_val=lambda_value,
                minimise=True,
                no_bookkeeping_time="2ps",
                lambda_array=self._lambda_values,
            )
            df = sim._run_with_bookkeeping(runtime="10ps")
            with self._lock:
                self._update_gpu_pool(gpu_num)
            return lambda_value, df


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

    mols = sr.stream.load("merged_molecule.s3")
    platform = "CPU"
    r = controller(mols, platform=platform, num_lambda=10)
    results = r.run_simulations()
    for key, dataframe in results.items():
        dataframe_to_parquet(dataframe, metadata={"lambda": key, "temperature": 300})
