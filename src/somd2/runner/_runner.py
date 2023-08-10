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
            self.create_gpu_pool()
            platform_options = {"num_workers": len(devices)}

        else:
            raise ValueError("Platform not recognised")

        return platform_options

    def create_gpu_pool(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        from multiprocessing import Manager

        manager = Manager()
        self._gpu_pool = manager.list(self.zero_CUDA_devices(self.get_CUDA_devices()))
        self._lock = manager.Lock()

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

        results = []
        with _futures.ProcessPoolExecutor(
            max_workers=self._platform_options["num_workers"]
        ) as executor:
            jobs = []

            lambda_values = [
                i / (self._num_lambda - 1) for i in range(0, self._num_lambda)
            ]
            for lambda_value in lambda_values:
                kwargs = {"lambda_value": lambda_value}
                jobs.append(executor.submit(self.run_single_simulation, **kwargs))
            for job in _futures.as_completed(jobs):
                result = job.result()
                results.append(result)
                print(result)

        return list(results)

    def run_single_simulation(self, lambda_value):
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
        from sire.convert import to
        from openmm import LocalEnergyMinimizer
        from sire.units import kelvin

        def run(system, schedule, map, steps, minimise=True):
            omm = to(system, "openmm", map=map)
            omm.set_lambda_schedule(schedule)
            omm.set_lambda(lambda_value)
            if minimise:
                LocalEnergyMinimizer.minimize(omm)

            omm.getIntegrator().step(steps)

        if self._platform == "CPU":
            print(
                f"Running lambda = {lambda_value} using {self._platform_options['cpu_per_worker']} CPUs"
            )
            map = {
                "integrator": "langevin_middle",
                "temperature": 300 * kelvin,
                "platform": self._platform,
                "threads": self._platform_options["cpu_per_worker"],
            }
            steps = 10
            run(self._system, self._schedule, map, steps, minimise=False)
            return f"Lambda = {lambda_value} complete"

        elif self._platform == "CUDA":
            with self._lock:
                gpu_num = self._gpu_pool[0]
                self._remove_gpu_from_pool(gpu_num)
                print(f"Running lambda = {lambda_value} on GPU {gpu_num}")
            map = {
                "integrator": "langevin_middle",
                "temperature": 300 * kelvin,
                "platform": self._platform,
                "device": gpu_num,
            }
            steps = 10
            run(self._system, self._schedule, map, steps, minimise=False)
            with self._lock:
                self._update_gpu_pool(gpu_num)
            return f"Lambda = {lambda_value} complete"


if __name__ == "__main__":
    import sire as sr

    mols = sr.stream.load("merged_molecule.s3")
    platform = "CPU"
    r = controller(mols, platform=platform, num_lambda=10)
    results = r.run_simulations()
    print(results)
