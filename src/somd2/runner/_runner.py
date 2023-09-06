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

        self._check_space_options()

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

    def create_sim_options(self, options):
        """
        Creates a dictionary of simulation options.

        Parameters:
        -----------
        options: dict
            Dictionary of simulation options.
        """
        self._sim_options = self._verify_sim_options(options)

        self._verify_output_directory()

        self._options_set = True

    def _verify_sim_options(self, options):
        """
        Verify user-input simulation options, checking whether options are valid and verifyiong units.
        Default values are used for any options not specified by the user.

        Parameters:
        -----------
        options: dict
            Dictionary of simulation options. Given by user.

        Returns:
        --------
        options_ver: dict
            Dictionary of verified simulation options.
        """

        from sire import u as _u

        inputs = list(options.keys())
        option_list_withunits = [
            "temperature",
            "timestep",
            "no bookkeeping time",
            "runtime",
            "energy frequency",
            "frame frequency",
        ]
        # Only run NPT if pressure is specified
        if self._has_space and "pressure" in inputs:
            option_list_withunits.append("pressure")
        options_list_bool = ["save velocities", "minimise"]
        options_list_other = ["integrator", "output directory"]
        options_ver = {}
        for key in options:
            if key not in (
                option_list_withunits + options_list_bool + options_list_other
            ):
                raise ValueError(
                    f"Option {key} not recognised, allowed options are {option_list_withunits + options_list_bool + options_list_other}"
                )
            else:
                for option in option_list_withunits:
                    if key == option:
                        try:
                            options_ver[key] = _u(options[key])
                            continue
                        except Exception:
                            raise ValueError(
                                f"Option {key} has invalid units, please see list of valid sire units"
                            )
                for option in options_list_bool:
                    if key == option:
                        if options[key] not in [True, False]:
                            raise ValueError(
                                f"Option {key} must be True or False, not {options[key]}"
                            )
                        else:
                            options_ver[key] = options[key]
                        continue
                for option in options_list_other:
                    if key == option:
                        # This would be a good place to have a list of supported integrators
                        options_ver[key] = options[key]
                        continue

        defaults = self.get_defaults()
        opt_master = {**defaults, **options_ver}
        return opt_master

    def _verify_output_directory(self):
        """
        Verify that the output directory exists and is writeable.
        If it does not yet exist, create it.
        """
        from pathlib import Path as _Path

        output_dir = self._sim_options["output directory"]
        if not _Path(output_dir).exists() or not _Path(output_dir).is_dir():
            try:
                _Path(output_dir).mkdir(parents=True, exist_ok=True)
            except:
                raise ValueError(
                    f"Output directory {output_dir} does not exist and cannot be created"
                )

    @staticmethod
    def get_defaults():
        """
        Returns a dictionary of default simulation options.

        returns:
        --------
        defaults: dict
            Dictionary of default simulation options.
        """
        from sire import u as _u
        from pathlib import Path as _Path

        defaults = {
            "temperature": _u("300 K"),
            "timestep": _u("2 fs"),
            "no bookkeeping time": _u("2 ps"),
            "runtime": _u("10 ps"),
            "energy frequency": _u("0.05 ps"),
            "frame frequency": _u("1 ps"),
            "save velocities": False,
            "minimise": True,
            "integrator": "langevin_middle",
            "output directory": _Path.cwd() / "outputs",
        }
        return defaults

    def get_options(self):
        """
        Returns a dictionary of simulation options.

        returns:
        --------
        options: dict
            Dictionary of simulation options.
        """
        if self._options_set:
            return self._sim_options
        else:
            print("No simulation options set, using defaults")
            return self.get_defaults()

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
        if not self._options_set:
            self._sim_options = self.get_defaults()
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
        from ._sire_merge_runsim import MergedSimulation
        from loguru import logger as _logger

        def _run(system, map, lambda_value, lam_minimisation=None):
            try:
                sim = MergedSimulation(
                    system,
                    map,
                    lambda_val=lambda_value,
                    lambda_array=self._lambda_values,
                    minimise=self._sim_options["minimise"],
                    no_bookkeeping_time=self._sim_options["no bookkeeping time"],
                )
            except Exception:
                _logger.warning(f"System creation at {lambda_value} failed")
                raise
            if lam_minimisation is None:
                try:
                    sim._setup_dynamics(
                        lam_val_min=lam_minimisation,
                        timestep=self._sim_options["timestep"],
                    )
                    df = sim._run_with_bookkeeping(
                        runtime=self._sim_options["runtime"],
                        energy_frequency=self._sim_options["energy frequency"],
                        frame_frequency=self._sim_options["frame frequency"],
                        save_velocities=self._sim_options["save velocities"],
                        traj_directory=self._sim_options["output directory"],
                    )
                except Exception as e:
                    _logger.warning(
                        f"Minimisation/dynamics at Lambda = {lambda_value} failed, trying again with minimsation at Lambda = {lam_minimisation}. The following warning was raised: {e}"
                    )
                    df = _run(system, map, lambda_value, lam_minimisation=0.0)
                    sim._cleanup()
                    return df
                else:
                    sim._cleanup()
                    return df
            else:
                try:
                    sim._setup_dynamics(lam_val_min=lam_minimisation)
                    df = sim._run_with_bookkeeping(
                        runtime=self._sim_options["runtime"],
                        energy_frequency=self._sim_options["energy frequency"],
                        frame_frequency=self._sim_options["frame frequency"],
                        save_velocities=self._sim_options["save velocities"],
                        traj_directory=self._sim_options["output directory"],
                    )
                except Exception as e:
                    _logger.error(
                        f"Minimisation/dynamics at Lambda = {lambda_value} failed, even after minimisation at Lambda = {lam_minimisation}. The following warning was raised: {e}."
                    )
                    raise
                else:
                    return df

        if not self._options_set:
            self._sim_options = self.get_defaults()
        map_options = ["integrator", "temperature", "pressure"]
        map = {}
        for option in map_options:
            try:
                map[option] = self._sim_options[option]
            except KeyError:
                continue
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
            map["platform"] = self._platform
            map["device"] = gpu_num

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
            filepath=self._sim_options["output directory"],
        )
        del system
        return f"Lambda = {lambda_value} complete"

    @staticmethod
    def dataframe_to_parquet(df, metadata, filepath=None):
        """
        Save a dataframe to parquet format with custom metadata.

        Parameters:
        -----------
        df: pandas.DataFrame`
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
