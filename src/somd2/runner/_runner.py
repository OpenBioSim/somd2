import concurrent.futures
import os
import sire as sr
from openmm import LocalEnergyMinimizer
import multiprocessing

__all__ = ["controller_CPU", "controller_GPU"]


class controller_CPU:
    """Controls the initiation of simulations as well as the assigning of CPUs.
    Fairly rudiementary at the moment, just trys to run all simulations in parallel on all available CPUs.

    Args:
        mols (sire.mol.MoleculeSet): Merged molecules to be simulated.
        num_lambda (int): Number of lambda windows to be simulated."""

    def __init__(self, mols, num_lambda):
        self.mols = mols
        self.num_lambda = num_lambda
        self.set_lambda_schedule()
        self.num_workers, self.cpu_per_worker = self.calculate_workers()

    def _set_lambda_schedule(self):
        """Create Lambda schedule that will be used for all simulations

        Args:
            mols (sire.mol.MoleculeSet): Molecules to be simulated."""
        self.lambda_schedule = self.create_lambda_schedule()

    @staticmethod
    def create_lambda_schedule():
        """Create a lambda schedule, this is where more commplex morphing
        etc. can be added in future.

        Args:
            omm (sire.openmm.OpenMM): OpenMM object for which the schedule is created.
        """
        schedule = sr.cas.LambdaSchedule()
        schedule.add_stage(
            "morphing",
            (1 - schedule.lam()) * schedule.initial()
            + schedule.lam() * schedule.final(),
        )
        return schedule

    def _calculate_workers(self):
        """Calculate number of workers to be used in parallel.
        Also calculate number of CPUs to be used by each openmm simulation."""
        num_cpu = os.cpu_count()
        num_workers = 1
        if self.num_lambda > num_cpu:
            num_workers = num_cpu
            cpu_per_worker = 1
        else:
            num_workers = self.num_lambda
            cpu_per_worker = num_cpu // self.num_lambda
        print(f"Number of CPUs available: {os.cpu_count()}")
        return num_workers, cpu_per_worker

    def run_simulations(self):
        """Run simulations in parallel."""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            jobs = []
            results_done = []
            lambda_values = [
                i / (self.num_lambda - 1) for i in range(0, self.num_lambda)
            ]
            for lambda_value in lambda_values:
                kw = {"lambda_value": lambda_value}
                jobs.append(executor.submit(self._run_single_simulation, **kw))
            for job in concurrent.futures.as_completed(jobs):
                result = job.result()
                results_done.append(result)
                print(result)

    def _run_single_simulation(self, lambda_value):
        """Run a single simulation.

        Args:
            lambda_value (float): Lambda value for simulation."""
        ############################################################################################################################################
        ### Very likely that this whole function will need to be replaced by a dedicated runner - this acts as a guide to CPU queue functionality###
        ############################################################################################################################################
        print(f"Running lambda = {lambda_value} uisng {self.cpu_per_worker} threads")
        map = {
            "integrator": "langevin_middle",
            "temperature": 300 * sr.units.kelvin,
            "platform": "CPU",
            "threads": self.cpu_per_worker,
        }
        self.omm = sr.convert.to(
            self.mols,
            "openmm",
            map=map,
        )
        self.omm.set_lambda_schedule(self.lambda_schedule)
        self.omm.set_lambda(lambda_value)
        s = self.omm.getState(getEnergy=True)
        print(
            f"Energy of lambda = {lambda_value} before minimisation = {s.getPotentialEnergy()}"
        )
        LocalEnergyMinimizer.minimize(self.omm)
        s = self.omm.getState(getEnergy=True)
        print(
            f"Energy of lambda = {lambda_value} after minimisation = {s.getPotentialEnergy()}"
        )
        self.omm.getIntegrator().step(100)
        s = self.omm.getState(getEnergy=True)
        return lambda_value, s.getPotentialEnergy()


class controller_GPU:
    """Controls the initiation of simulations as well as the assigning of GPUS.

    Args:
        mols (sire.mol.MoleculeSet): Merged molecules to be simulated.
        num_lambda (int): Number of lambda windows to be simulated."""

    def __init__(self, mols, num_lambda):
        self.mols = mols
        init_gpu_pool = self.zero_CUDA_devices(self.get_CUDA_devices())
        manager = multiprocessing.Manager()
        self._lock = manager.Lock()
        self._gpu_pool = manager.list(init_gpu_pool)
        self._set_lambda_schedule()
        self.num_lambda = num_lambda

    @staticmethod
    def get_CUDA_devices():
        """Get list of available GPUs from CUDA_VISIBLE_DEVICES."""
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            raise ValueError("CUDA_VISIBLE_DEVICES not set")
        else:
            available_devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
            print("CUDA_VISIBLE_DEVICES set to", available_devices)
            num_gpus = len(available_devices)
            print("Number of GPUs available:", num_gpus)
            return available_devices

    @staticmethod
    def zero_CUDA_devices(devices):
        """Set all device numbers relative to the lowest (the device number becomes equal to its index in the list).

        Args:
            devices (list): List of available device numbers."""
        return [str(devices.index(value)) for value in devices]

    def _set_lambda_schedule(self):
        """Create Lambda schedule that will be used for all simulations

        Args:
            mols (sire.mol.MoleculeSet): Molecules to be simulated."""
        self.lambda_schedule = self.create_lambda_schedule()

    def _update_gpu_pool(self, gpu_num):
        """Used to update GPU pool once GPU becomes free

        Args:
            gpu_num (str): GPU number to be added to pool."""
        print(f"returning GPU {gpu_num} to pool")
        self._gpu_pool.append(gpu_num)
        print(self._gpu_pool)

    def _remove_from_gpu_pool(self, gpu_num):
        """Used to remove GPU from pool once it is in use.

        Args:
            gpu_num (str): GPU number to be removed from pool."""
        print(f"Removing GPU {gpu_num} from pool")
        self._gpu_pool.remove(gpu_num)
        print("after removal")
        print(self._gpu_pool)

    @staticmethod
    def create_lambda_schedule():
        """Create a lambda schedule, this is where more commplex morphing
        etc. can be added in future.

        Args:
            omm (sire.openmm.OpenMM): OpenMM object for which the schedule is created.
        """
        schedule = sr.cas.LambdaSchedule()
        schedule.add_stage(
            "morphing",
            (1 - schedule.lam()) * schedule.initial()
            + schedule.lam() * schedule.final(),
        )
        return schedule

    def run_simulations(self):
        """Run simulations in parallel."""
        max_wrkrs = len(self._gpu_pool)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_wrkrs) as executor:
            jobs = []
            results_done = []
            lambda_values = [
                i / (self.num_lambda - 1) for i in range(0, self.num_lambda)
            ]
            for lambda_value in lambda_values:
                kw = {"lambda_value": lambda_value}
                jobs.append(executor.submit(self.run_single_simulation, **kw))
            for job in concurrent.futures.as_completed(jobs):
                result = job.result()
                results_done.append(result)
                print(result)

    def run_single_simulation(self, lambda_value):
        """Run a single simulation.

        Args:
            lambda_value (float): Lambda value for simulation."""
        ############################################################################################################################################
        ### Very likely that this whole function will need to be replaced by a dedicated runner - this acts as a guide to GPU queue functionality###
        ############################################################################################################################################
        if len(self._gpu_pool) == 0:
            raise ValueError("No GPUs available")
        else:
            with self._lock:
                gpu_num = self._gpu_pool[0]
                self._remove_from_gpu_pool(gpu_num)
                print(self._gpu_pool)
            print(f"Running lambda = {lambda_value} on GPU {gpu_num}")
            map = {
                "integrator": "langevin_middle",
                "temperature": 300 * sr.units.kelvin,
                "platform": "CUDA",
                "device": gpu_num,
            }
            # Need to double check that no copying needs to be done
            self.omm = sr.convert.to(
                self.mols,
                "openmm",
                map=map,
            )
            self.omm.set_lambda_schedule(self.lambda_schedule)
            self.omm.set_lambda(lambda_value)

            s = self.omm.getState(getEnergy=True)
            print(
                f"Energy of lambda = {lambda_value} before minimisation = {s.getPotentialEnergy()}"
            )
            LocalEnergyMinimizer.minimize(self.omm)
            s = self.omm.getState(getEnergy=True)
            print(
                f"Energy of lambda = {lambda_value} after minimisation = {s.getPotentialEnergy()}"
            )
            self.omm.getIntegrator().step(100)
            s = self.omm.getState(getEnergy=True)
            with self._lock:
                self._update_gpu_pool(gpu_num)
            return lambda_value, s.getPotentialEnergy()


if __name__ == "__main__":
    mols = sr.stream.load("merged_molecule.s3")
    platform = "CPU"
    if platform == "CPU":
        r = controller_CPU(mols, 11)
        r.run_simulations()
    elif platform == "GPU":
        r = controller_GPU(mols, 11)
        r.run_simulations()
    else:
        raise ValueError("Platform not recognised")
