"""Configuration class for SOMD2 runner.
Takes in a list of simulation options (currently in the form of a dictionary), checks the valdiity of them,
and outputs a config class object in which options are converted to SI units"""
__all__ = ["Config"]
from sire import u as _u
from pathlib import Path as _Path
from loguru import logger as _logger


class Config:
    """Class for storing simulation options in SI units"""

    def __init__(
        self,
        runtime="10ps",
        timestep="2fs",
        temperature="300K",
        pressure="1 atm",
        integrator="langevin_middle",
        cutoff_type="PME",
        num_lambda=11,
        lambda_schedule=None,
        minimise=True,
        equilibrate=False,
        equilibration_time="2ps",
        equilibration_timestep="1fs",
        energy_frequency="1ps",
        save_trajectories=True,
        frame_frequency="2ps",
        save_velocities=False,
        checkpoint=True,
        checkpoint_frequency="5ps",
        platform=None,
        max_CPU_cores=None,
        max_GPUS=None,
        run_parallel=False,
        vacuum=False,
        output_directory=_Path.cwd() / "output",
        config_to_file=False,
        extra_args=None,
    ):
        """Initialise the config class object

        Parameters:
        -----------
        runtime: str
            Length of simulation

        timestep: str
            Simulation timestep

        temperature: str
            Simulation temperature

        pressure: str
            Simulation pressure - simulations will run in the NVT ensemble unless presure is specified

        integrator: str
            Integrator to use for simulation

        cutoff_type: str
            Cutoff type to use for simulation (currently only PMF and RF are supported)

        num_lambda: int
            Number of lambda windows to use for alchemical free energy simulations
            (default None, runs vanilla MD)

        lambda_schedule: str
            Lambda schedule to use for alchemical free energy simulations.
            Has no function unless num_lambda is specified, defaults to simple morph
            if not set.

        minimise: bool
            Whether to minimise the system before simulation

        equilibrate: bool
            Whether to equilibrate the system before simulation

        equilibration_time: str
            Length of equilibration

        equilibration_timestep: str
            Equilibration timestep (can be different to simulation timestep, default 2fs)

        energy_frequency: str
            Frequency at which to output energy data

        save_trajectories: bool
            Whether to save trajectory frames to a separate file

        frame_frequency: str
            Frequency at which to output trajectory frames

        save_velocities: bool
            Whether to save velocities in trajectory frames

        checkpoint: bool
            Whether to save checkpoint files - turning this off may increase performance,
            but will make the simulations unrecoverable if they crash

        checkpoint_frequency: str
            Frequency at which to save checkpoint files, should be larger than
            min(energy_frequency, frame_frequency)

        platform: str
            Platform to run simulation on (CPU or CUDA)

        max_CPU_cores: int
            Maximum number of CPU cores to use for simulation (default None, uses all available)
            Does nothing if platfrom is set to CUDA

        max_GPUS: int
            Maximum number of GPUs to use for simulation (default None, uses all available)
            does nothing if platform is set to CPU

        run_parallel: bool
            Whether to run simulation in parallel (default False)

        vacuum: bool
            Set True if running in vacuum (default False)

        output_directory: str
            Directory in which energy and trajectory info is stored

        config_to_file: bool
            Dump configuration options to a yaml file in the output directory

        extra_args: dict
            Dictionary passed to sire as a map - only use if there is no other way,
            inputs for this dictionary are not checked for validity.
            !!Arguments set here here may overwrite config options!!
        """

        self.runtime = runtime
        self.timestep = timestep
        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.cutoff_type = cutoff_type
        self.num_lambda = num_lambda
        self.lambda_schedule = lambda_schedule
        self.minimise = minimise
        self.equilibrate = equilibrate
        self.equilibration_time = equilibration_time
        self.equilibration_timestep = equilibration_timestep
        self.energy_frequency = energy_frequency
        self.save_trajectories = save_trajectories
        self.frame_frequency = frame_frequency
        self.save_velocities = save_velocities
        self.checkpoint = checkpoint
        self.checkpoint_frequency = checkpoint_frequency
        self.platform = platform
        self.max_CPU_cores = max_CPU_cores
        self.max_GPUS = max_GPUS
        self.run_parallel = run_parallel
        self.vacuum = vacuum
        self.output_directory = output_directory
        self.config_to_file = config_to_file
        self.extra_args = extra_args

    def as_dict(self):
        """Convert config object to dictionary"""
        from pathlib import PosixPath as _PosixPath

        d = {}
        for attr, value in self.__dict__.items():
            attr_l = attr[1:]
            if isinstance(value, _PosixPath):
                d[attr_l] = str(value)
            else:
                try:
                    d[attr_l] = value.to_string()
                except AttributeError:
                    d[attr_l] = value
        return d

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        from sire.units import picosecond

        try:
            t = _u(runtime)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(picosecond):
            raise ValueError("Runtime units are invalid.")
        self._runtime = t

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, timestep):
        from sire.units import femtosecond

        try:
            t = _u(timestep)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(femtosecond):
            raise ValueError("Timestep units are invalid.")
        self._timestep = t

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        from sire.units import kelvin

        try:
            t = _u(temperature)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(kelvin):
            raise ValueError("Temperature units are invalid.")
        self._temperature = t

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        from sire.units import atm

        if pressure is not None:
            try:
                p = _u(pressure)
            except Exception as e:
                print(e.message)
            if not p.has_same_units(atm):
                raise ValueError("Pressure units are invalid.")
            self._pressure = p

        else:
            self._pressure = pressure

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        valid_integrators = [
            "verlet",
            "leapfrog",
            "langevin_middle",
            "langevin",
            "noose_hoover",
            "brownian",
        ]
        if str(integrator) not in valid_integrators:
            raise ValueError(
                f"Integrator not recognised. Valid integrators are: {valid_integrators}"
            )
        self._integrator = str(integrator)

    @property
    def cutoff_type(self):
        return self._cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, cutoff_type):
        valid_cutoff_types = ["PME", "RF"]
        if str(cutoff_type) not in valid_cutoff_types:
            raise ValueError(
                f"Cutoff type not recognised. Valid cutoff types are: {valid_cutoff_types}"
            )
        self._cutoff_type = str(cutoff_type)

    @property
    def num_lambda(self):
        return self._num_lambda

    @num_lambda.setter
    def num_lambda(self, num_lambda):
        if num_lambda is not None:
            if not isinstance(num_lambda, int):
                raise ValueError("'num_lambda' must be an integer")
        self._num_lambda = num_lambda

    @property
    def lambda_schedule(self):
        return self._lambda_schedule

    @lambda_schedule.setter
    def lambda_schedule(self, lambda_schedule):
        from sire.cas import LambdaSchedule as _LambdaSchedule

        if lambda_schedule is not None:
            if not isinstance(lambda_schedule, _LambdaSchedule):
                raise ValueError("'lambda_schedule' must be a LambdaSchedule object")
            self._lambda_schedule = lambda_schedule
        else:
            self._lambda_schedule = _LambdaSchedule.standard_morph()

    @property
    def minimise(self):
        return self._minimise

    @minimise.setter
    def minimise(self, minimise):
        if not isinstance(minimise, bool):
            raise ValueError("'Minimise' must be a boolean")
        if not minimise:
            _logger.warning(
                "Minimisation is highly recommended for increased stability in SOMD2"
            )
        self._minimise = minimise

    @property
    def equilibrate(self):
        return self._equilibrate

    @equilibrate.setter
    def equilibrate(self, equilibrate):
        if not isinstance(equilibrate, bool):
            raise ValueError("'Equilibrate' must be a boolean")
        self._equilibrate = equilibrate

    @property
    def equilibration_time(self):
        return self._equilibration_time

    @equilibration_time.setter
    def equilibration_time(self, equilibration_time):
        from sire.units import picosecond

        try:
            t = _u(equilibration_time)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(picosecond):
            raise ValueError("Equilibration time units are invalid.")
        self._equilibration_time = t

    @property
    def equilibration_timestep(self):
        return self._equilibration_timestep

    @equilibration_timestep.setter
    def equilibration_timestep(self, equilibration_timestep):
        from sire.units import femtosecond

        try:
            t = _u(equilibration_timestep)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(femtosecond):
            raise ValueError("Equilibration timestep units are invalid.")
        self._equilibration_timestep = t

    @property
    def energy_frequency(self):
        return self._energy_frequency

    @energy_frequency.setter
    def energy_frequency(self, energy_frequency):
        from sire.units import picosecond

        try:
            t = _u(energy_frequency)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(picosecond):
            raise ValueError("Energy frequency units are invalid.")
        self._energy_frequency = t

    @property
    def save_trajectories(self):
        return self._save_trajectories

    @save_trajectories.setter
    def save_trajectories(self, save_trajectories):
        if not isinstance(save_trajectories, bool):
            raise ValueError("'save_trajectories' must be a boolean")
        self._save_trajectories = save_trajectories

    @property
    def frame_frequency(self):
        return self._frame_frequency

    @frame_frequency.setter
    def frame_frequency(self, frame_frequency):
        from sire.units import picosecond

        try:
            t = _u(frame_frequency)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(picosecond):
            raise ValueError("Frame frequency units are invalid.")
        self._frame_frequency = t

    @property
    def save_velocities(self):
        return self._save_velocities

    @save_velocities.setter
    def save_velocities(self, save_velocities):
        if not isinstance(save_velocities, bool):
            raise ValueError("'save_velocities' must be a boolean")
        self._save_velocities = save_velocities

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint):
        if not isinstance(checkpoint, bool):
            raise ValueError("'checkpoint' must be a boolean")
        self._checkpoint = checkpoint

    @property
    def checkpoint_frequency(self):
        return self._checkpoint_frequency

    @checkpoint_frequency.setter
    def checkpoint_frequency(self, checkpoint_frequency):
        from sire.units import picosecond

        try:
            t = _u(checkpoint_frequency)
        except Exception as e:
            print(e.message)
        if not t.has_same_units(picosecond):
            raise ValueError("Checkpoint frequency units are invalid.")
        if t < self._energy_frequency and t < self._frame_frequency:
            _logger.warning(
                "Checkpoint frequency is low - should be greater min(energy_frequency, frame_frequency)"
            )
        self._checkpoint_frequency = t

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, platform):
        import os as _os

        valid_platforms = ["CPU", "CUDA"]
        if platform is not None:
            if str(platform) not in valid_platforms:
                raise ValueError(
                    f"Platform not recognised. Valid platforms are: {valid_platforms}"
                )
            elif (
                str(platform) == "CUDA"
                and _os.environ.get("CUDA_VISIBLE_DEVICES") is None
            ):
                raise ValueError(
                    "CUDA platform requested but CUDA_VISIBLE_DEVICES not set"
                )
            self._platform = str(platform)
        else:
            if "CUDA_VISIBLE_DEVICES" in _os.environ:
                self._platform = "CUDA"
            else:
                self._platform = "CPU"

    @property
    def max_CPU_cores(self):
        return self._max_CPU_cores

    @max_CPU_cores.setter
    def max_CPU_cores(self, max_CPU_cores):
        import os as _os

        if max_CPU_cores is not None:
            self._max_CPU_cores = max_CPU_cores
            if self._platform == "CUDA":
                _logger.warning(
                    "CUDA platform requested but max_CPU_cores set - ignoring max_CPU_cores"
                )
        else:
            self._max_CPU_cores = _os.cpu_count()

    @property
    def max_GPUS(self):
        return self._max_GPUS

    @max_GPUS.setter
    def max_GPUS(self, max_GPUS):
        import os as _os

        if max_GPUS is not None:
            self._max_GPUS = max_GPUS
            if self._platform == "CPU":
                _logger.warning(
                    "CPU platform requested but max_GPUS set - ignoring max_GPUS"
                )
        else:
            if "CUDA_VISIBLE_DEVICES" in _os.environ:
                self._max_GPUS = len(_os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            else:
                self._max_GPUS = 0

    @property
    def run_parallel(self):
        return self._run_parallel

    @run_parallel.setter
    def run_parallel(self, run_parallel):
        if not isinstance(run_parallel, bool):
            raise ValueError("'run_parallel' must be a boolean")
        self._run_parallel = run_parallel

    @property
    def vacuum(self):
        return self._vacuum

    @vacuum.setter
    def vacuum(self, vacuum):
        if not isinstance(vacuum, bool):
            raise ValueError("'vacuum' must be a boolean")
        if vacuum and self._pressure is not None:
            _logger.warning(
                "Vacuum requested but pressure specified - ignoring pressure"
            )
        self._vacuum = vacuum

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, output_directory):
        if not isinstance(output_directory, _Path):
            try:
                output_directory = _Path(output_directory)
            except Exception as e:
                raise ValueError(f"Could not convert output path. {e}")
        if not _Path(output_directory).exists() or not _Path(output_directory).is_dir():
            try:
                _Path(output_directory).mkdir(parents=True, exist_ok=True)
            except:
                raise ValueError(
                    f"Output directory {output_directory} does not exist and cannot be created"
                )
        self._output_directory = output_directory

    @property
    def config_to_file(self):
        return self._config_to_file

    @config_to_file.setter
    def config_to_file(self, config_to_file):
        if not isinstance(config_to_file, bool):
            raise ValueError("'config_to_file' must be a boolean")
        self._config_to_file = config_to_file

    @property
    def extra_args(self):
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args):
        if extra_args is not None:
            if not isinstance(extra_args, dict):
                raise ValueError("'extra_args' must be a dictionary")
        self._extra_args = extra_args
