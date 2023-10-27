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

"""
Configuration class for SOMD2 runner.
"""

__all__ = ["Config"]

from pathlib import Path as _Path
from loguru import logger as _logger

import sire as _sr


class Config:
    """
    Class for storing a SOMD2 simulation configuration.
    """

    # A dictionary of choices for options that support them. Here we inspect
    # the Sire options module to get the valid choices. This allows us to be
    # forwards compatible with new options.
    _choices = {
        "constraint": _sr.options.Constraint.options(),
        "perturbable_constraint": _sr.options.PerturbableConstraint.options(),
        "integrator": _sr.options.Integrator.options(),
        "cutoff_type": _sr.options.Cutoff.options(),
        "platform": _sr.options.Platform.options(),
        "lambda_schedule": [
            "standard_morph",
            "charge_scaled_morph",
        ],
    }

    def __init__(
        self,
        runtime="1ns",
        timestep="4fs",
        temperature="300K",
        pressure="1 atm",
        integrator="langevin_middle",
        cutoff_type="pme",
        h_mass_factor=1.5,
        num_lambda=11,
        lambda_schedule="standard_morph",
        charge_scale_factor=0.2,
        swap_end_states=False,
        coulomb_power=0.0,
        shift_delta="2A",
        constraint="h_bonds",
        perturbable_constraint=None,
        minimise=True,
        equilibration_time="0ps",
        equilibration_timestep="1fs",
        energy_frequency="1ps",
        save_trajectories=True,
        frame_frequency="20ps",
        save_velocities=False,
        checkpoint=True,
        checkpoint_frequency="100ps",
        platform="auto",
        max_threads=None,
        max_gpus=None,
        run_parallel=True,
        output_directory="output",
        write_config=True,
    ):
        """
        Constructor.

        Parameters
        ----------

        runtime: str
            Simulation length for each lambda window.

        timestep: str
            Integration timestep.

        temperature: str
            Simulation temperature.

        pressure: str
            Simulation pressure. (Simulations will run in the NVT ensemble unless a pressure is specified.)

        integrator: str
            Integrator to use for simulation.

        cutoff_type: str
            Cutoff type to use for simulation.

        h_mass_factor: float
            Factor by which to scale hydrogen masses.

        num_lambda: int
            Number of lambda windows to use.

        lambda_schedule: str
            Lambda schedule to use for alchemical free energy simulations.

        charge_scale_factor: float
            Factor by which to scale charges for charge scaled morph.

        swap_end_states: bool
            Whether to perform the perturbation in the reverse direction.

        couloumb_power : float
            Power to use for the soft-core Coulomb interaction. This is used
            to soften the electrostatic interaction.

        shift_delta : str
            The soft-core shift-delta parameter. This is used to soften the
            Lennard-Jones interaction.

        constraint: str
            Constraint type to use for non-perturbable molecules.

        perturbable_constraint: str
            Constraint type to use for perturbable molecules.

        minimise: bool
            Whether to minimise the system before simulation.

        equilibration_time: str
            Time interval for equilibration.

        equilibration_timestep: str
            Equilibration timestep. (Can be different to simulation timestep.)

        energy_frequency: str
            Frequency at which to output energy data.

        save_trajectories: bool
            Whether to save trajectory files

        frame_frequency: str
            Frequency at which to output trajectory frames.

        save_velocities: bool
            Whether to save velocities in trajectory frames.

        checkpoint: bool
            Whether to checkpoint.

        checkpoint_frequency: str
            Frequency at which to save checkpoint files, should be larger than min(energy_frequency, frame_frequency).

        platform: str
            Platform to run simulation on.

        max_threads: int
            Maximum number of CPU threads to use for simulation. (Default None, uses all available)
            Does nothing if platform is set to CUDA.

        max_gpus: int
            Maximum number of GPUs to use for simulation (Default None, uses all available.)
            Does nothing if platform is set to CPU.

        run_parallel: bool
            Whether to run simulation in parallel.

        output_directory: str
            Path to a directory to store output files.

        write_config: bool
            Whether to write the configuration options to a YAML file in the output directory.
        """

        self.runtime = runtime
        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.cutoff_type = cutoff_type
        self.h_mass_factor = h_mass_factor
        self.timestep = timestep
        self.num_lambda = num_lambda
        self.lambda_schedule = lambda_schedule
        self.charge_scale_factor = charge_scale_factor
        self.swap_end_states = swap_end_states
        self.coulomb_power = coulomb_power
        self.shift_delta = shift_delta
        self.constraint = constraint
        self.perturbable_constraint = perturbable_constraint
        self.minimise = minimise
        self.equilibration_time = equilibration_time
        self.equilibration_timestep = equilibration_timestep
        self.energy_frequency = energy_frequency
        self.save_trajectories = save_trajectories
        self.frame_frequency = frame_frequency
        self.save_velocities = save_velocities
        self.checkpoint = checkpoint
        self.checkpoint_frequency = checkpoint_frequency
        self.platform = platform
        self.max_threads = max_threads
        self.max_gpus = max_gpus
        self.run_parallel = run_parallel
        self.output_directory = output_directory
        self.write_config = write_config

    def __str__(self):
        """Return a string representation of this object."""

        # Get a dictionary representation of the object.
        d = self.as_dict()

        # Initialise the string.
        string = "Config("

        for k, v in d.items():
            if isinstance(v, str):
                string += f"{k.replace('', '')}='{v}', "
            else:
                string += f"{k.replace('', '')}={v}, "

        # Remove the trailing comma and space.
        string = string[:-2]

        # Close the string.
        string += ")"

        return string

    def __repr__(self):
        """Return a string representation of this object."""
        return self.__str__()

    def __eq__(self, other):
        """Equality operator."""
        return self.as_dict() == other.as_dict()

    @staticmethod
    def from_yaml(path):
        """
        Create a Config object from a YAML file.

        Parameters
        ----------

        path: str
            Path to YAML file.
        """
        import os as _os
        import yaml as _yaml

        if not isinstance(path, str):
            raise TypeError("'path' must be of type 'str'")
        if not _os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")

        try:
            with open(path, "r") as f:
                d = _yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Could not load YAML file: {e}")

        return Config(**d)

    def as_dict(self):
        """Convert config object to dictionary"""
        from pathlib import PosixPath as _PosixPath
        from sire.cas import LambdaSchedule as _LambdaSchedule

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

        # Handle the lambda schedule separately so that we can use simplified
        # keyword options.
        if self.lambda_schedule == _LambdaSchedule.standard_morph():
            d["lambda_schedule"] = "standard_morph"
        elif self.lambda_schedule == _LambdaSchedule.charge_scaled_morph(
            self._charge_scale_factor
        ):
            d["lambda_schedule"] = "charge_scaled_morph"
        return d

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        if not isinstance(runtime, str):
            raise TypeError("'runtime' must be of type 'str'")

        from sire.units import picosecond

        try:
            t = _sr.u(runtime)
        except:
            raise ValueError(
                f"Unable to parse 'runtime' as a Sire GeneralUnit: {runtime}"
            )

        if not t.has_same_units(picosecond):
            raise ValueError("'runtime' units are invalid.")

        self._runtime = t

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if not isinstance(temperature, str):
            raise TypeError("'temperature' must be of type 'str'")

        from sire.units import kelvin

        try:
            t = _sr.u(temperature)
        except:
            raise ValueError(
                f"Unable to parse 'temperature' as a Sire GeneralUnit: {temperature}"
            )

        if not t.has_same_units(kelvin):
            raise ValueError("'temperature' units are invalid.")

        self._temperature = t

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        if pressure is not None and not isinstance(pressure, str):
            raise TypeError("'pressure' must be of type 'str'")

        from sire.units import atm

        if pressure is not None:
            try:
                p = _sr.u(pressure)
            except:
                raise ValueError(
                    f"Unable to parse 'pressure' as a Sire GeneralUnit: {pressure}"
                )
            if not p.has_same_units(atm):
                raise ValueError("'pressure' units are invalid.")

            self._pressure = p

        else:
            self._pressure = pressure

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        if not isinstance(integrator, str):
            raise TypeError("'integrator' must be of type 'str'")
        integrator = integrator.lower().replace(" ", "")
        if integrator not in self._choices["integrator"]:
            raise ValueError(
                f"Integrator not recognised. Valid integrators are: {', '.join(self._choices['integrator'])}"
            )
        self._integrator = integrator

    @property
    def cutoff_type(self):
        return self._cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, cutoff_type):
        if not isinstance(cutoff_type, str):
            raise TypeError("'cutoff_type' must be of type 'str'")
        cutoff = cutoff_type.lower().replace(" ", "")
        if cutoff_type not in self._choices["cutoff_type"]:
            raise ValueError(
                f"Cutoff type not recognised. Valid cutoff types are: {', '.join(self._choices['cutoff_type'])}"
            )
        self._cutoff_type = cutoff_type

    @property
    def h_mass_factor(self):
        return self._h_mass_factor

    @h_mass_factor.setter
    def h_mass_factor(self, h_mass_factor):
        if not isinstance(h_mass_factor, float):
            try:
                h_mass_factor = float(h_mass_factor)
            except Exception:
                raise ValueError("'h_mass_factor' must be a float")
        self._h_mass_factor = h_mass_factor

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, timestep):
        if not isinstance(timestep, str):
            raise TypeError("'timestep' must be of type 'str'")

        from sire.units import femtosecond

        try:
            t = _sr.u(timestep)
        except:
            raise ValueError(
                f"Unable to parse 'timestep' as a Sire GeneralUnit: {timestep}"
            )
        if not t.has_same_units(femtosecond):
            raise ValueError("'timestep' units are invalid.")

        if t > _sr.u("2fs") and self.h_mass_factor <= 1.0:
            _logger.warning("Timestep is large - consider repartitioning hydrogen mass")
        self._timestep = t

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
            if not isinstance(lambda_schedule, (str, _LambdaSchedule)):
                raise ValueError(
                    "'lambda_schedule' must be of type 'str' or 'LambdaSchedule' object"
                )
            if isinstance(lambda_schedule, str):
                # Strip whitespace and convert to lower case.
                lambda_schedule = lambda_schedule.strip().lower()
                if lambda_schedule not in self._choices["lambda_schedule"]:
                    raise ValueError(
                        f"Lambda schedule not recognised. Valid lambda schedules are: {self._choices['lambda_schedule']}"
                    )
                if lambda_schedule == "standard_morph":
                    self._lambda_schedule = _LambdaSchedule.standard_morph()
                elif lambda_schedule == "charge_scaled_morph":
                    self._lambda_schedule = _LambdaSchedule.charge_scaled_morph(
                        self._charge_scale_factor
                    )
            else:
                self._lambda_schedule = lambda_schedule
        else:
            self._lambda_schedule = _LambdaSchedule.standard_morph()

    @property
    def charge_scale_factor(self):
        return self._charge_scale_factor

    @charge_scale_factor.setter
    def charge_scale_factor(self, charge_scale_factor):
        if not isinstance(charge_scale_factor, float):
            try:
                charge_scale_factor = float(charge_scale_factor)
            except Exception:
                raise ValueError("'charge_scale_factor' must be a float")
        self._charge_scale_factor = charge_scale_factor
        # Update the lambda schedule if it is charge scaled morph.
        if self._lambda_schedule == "charge_scaled_morph":
            self._lambda_schedule = _LambdaSchedule.charge_scaled_morph(
                self._charge_scale_factor
            )

    @property
    def swap_end_states(self):
        return self._swap_end_states

    @swap_end_states.setter
    def swap_end_states(self, swap_end_states):
        if not isinstance(swap_end_states, bool):
            raise ValueError("'swap_end_states' must be of type 'bool'")
        self._swap_end_states = swap_end_states

    @property
    def coulomb_power(self):
        return self._coulomb_power

    @coulomb_power.setter
    def coulomb_power(self, coulomb_power):
        if not isinstance(coulomb_power, float):
            try:
                coulomb_power = float(coulomb_power)
            except Exception:
                raise ValueError("'coulomb_power' must be a of type 'float'")
        self._coulomb_power = coulomb_power

    @property
    def shift_delta(self):
        return self._shift_delta

    @shift_delta.setter
    def shift_delta(self, shift_delta):
        if not isinstance(shift_delta, str):
            raise TypeError("'shift_delta' must be of type 'str'")

        from sire.units import angstrom

        try:
            sd = _sr.u(shift_delta)
        except:
            raise ValueError(
                f"Unable to parse 'shift_delta' as a Sire GeneralUnit: {shift_delta}"
            )
        if not sd.has_same_units(angstrom):
            raise ValueError("'shift_delta' units are invalid.")

        self._shift_delta = sd

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, constraint):
        if constraint is not None:
            if not isinstance(constraint, str):
                raise TypeError("'constraint' must be of type 'str'")
            constraint = constraint.lower().replace(" ", "")
            if constraint not in self._choices["constraint"]:
                raise ValueError(
                    f"'constraint' not recognised. Valid constraints are: {', '.join(self._choices['constraint'])}"
                )
            if constraint == "none":
                self._constraint = None
            else:
                self._constraint = constraint
        else:
            self._constraint = None

    @property
    def perturbable_constraint(self):
        return self._perturbable_constraint

    @perturbable_constraint.setter
    def perturbable_constraint(self, perturbable_constraint):
        if perturbable_constraint is not None:
            if not isinstance(perturbable_constraint, str):
                raise TypeError("'perturbable_constraint' must be of type 'str'")
            perturbable_constraint = perturbable_constraint.lower().replace(" ", "")
            if perturbable_constraint not in self._choices["perturbable_constraint"]:
                raise ValueError(
                    f"'perturbable_constrant' not recognised. Valid constraints are: {', '.join(self._choices['constraint'])}"
                )
            if perturbable_constraint == "none":
                self._perturbable_constraint = None
            else:
                self._perturbable_constraint = perturbable_constraint
        else:
            self._perturbable_constraint = None

    @property
    def minimise(self):
        return self._minimise

    @minimise.setter
    def minimise(self, minimise):
        if not isinstance(minimise, bool):
            raise ValueError("'minimise' must be of type 'bool'")
        if not minimise:
            _logger.warning(
                "Minimisation is highly recommended for increased stability in SOMD2"
            )
        self._minimise = minimise

    @property
    def equilibration_time(self):
        return self._equilibration_time

    @equilibration_time.setter
    def equilibration_time(self, equilibration_time):
        if not isinstance(equilibration_time, str):
            raise TypeError("'equilibration_time' must be of type 'str'")

        from sire.units import picosecond

        try:
            t = _sr.u(equilibration_time)
        except:
            raise ValueError(
                f"Unable to parse 'equilibration_time' as a Sire GeneralUnit: {equilibration_time}"
            )

        if t.value() != 0 and not t.has_same_units(picosecond):
            raise ValueError("'equilibration_time' units are invalid.")

        self._equilibration_time = t

    @property
    def equilibration_timestep(self):
        return self._equilibration_timestep

    @equilibration_timestep.setter
    def equilibration_timestep(self, equilibration_timestep):
        if not isinstance(equilibration_timestep, str):
            raise TypeError("'equilibration_timestep' must be of type 'str'")

        from sire.units import femtosecond

        try:
            t = _sr.u(equilibration_timestep)
        except:
            raise valueError(
                f"Unable to parse 'equilibration_timestep' as a Sire GeneralUnit: {equilibration_timestep}"
            )

        if not t.has_same_units(femtosecond):
            raise ValueError("'equilibration_timestep' units are invalid.")

        self._equilibration_timestep = t

    @property
    def energy_frequency(self):
        return self._energy_frequency

    @energy_frequency.setter
    def energy_frequency(self, energy_frequency):
        if not isinstance(energy_frequency, str):
            raise TypeError("'energy_frequency' must be of type 'str'")

        from sire.units import picosecond

        try:
            t = _sr.u(energy_frequency)
        except:
            raise ValueError(
                f"Unable to parse 'energy_frequency' as a Sire GeneralUnit: {energy_frequency}"
            )

        if not t.has_same_units(picosecond):
            raise ValueError("'energy_frequency' units are invalid.")

        self._energy_frequency = t

    @property
    def save_trajectories(self):
        return self._save_trajectories

    @save_trajectories.setter
    def save_trajectories(self, save_trajectories):
        if not isinstance(save_trajectories, bool):
            raise ValueError("'save_trajectories' must be of type 'bool'")
        self._save_trajectories = save_trajectories

    @property
    def frame_frequency(self):
        return self._frame_frequency

    @frame_frequency.setter
    def frame_frequency(self, frame_frequency):
        if not isinstance(frame_frequency, str):
            raise TypeError("'frame_frequency' must be of type 'str'")

        from sire.units import picosecond

        try:
            t = _sr.u(frame_frequency)
        except:
            raise ValueError(
                f"Unable to parse 'frame_frequency' as a Sire GeneralUnit: {frame_frequency}"
            )

        if not t.has_same_units(picosecond):
            raise ValueError("'frame_frequency' units are invalid.")

        self._frame_frequency = t

    @property
    def save_velocities(self):
        return self._save_velocities

    @save_velocities.setter
    def save_velocities(self, save_velocities):
        if not isinstance(save_velocities, bool):
            raise ValueError("'save_velocities' must be of type 'bool'")
        self._save_velocities = save_velocities

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint):
        if not isinstance(checkpoint, bool):
            raise ValueError("'checkpoint' must be of type 'bool'")
        self._checkpoint = checkpoint

    @property
    def checkpoint_frequency(self):
        return self._checkpoint_frequency

    @checkpoint_frequency.setter
    def checkpoint_frequency(self, checkpoint_frequency):
        if not isinstance(checkpoint_frequency, str):
            raise TypeError("'checkpoint_frequency' must be of type 'str'")

        from sire.units import picosecond

        try:
            t = _sr.u(checkpoint_frequency)
        except:
            raise ValueError(
                f"Unable to parse 'checkpoint_frequency' as a Sire GeneralUnit: {checkpoint_frequency}"
            )

        if not t.has_same_units(picosecond):
            raise ValueError("'checkpoint_frequency' units are invalid.")
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
        import sys as _sys

        if not isinstance(platform, str):
            raise TypeError("'platform' must be of type 'str'")
        platform = platform.lower().replace(" ", "")
        if platform not in self._choices["platform"]:
            raise ValueError(
                f"Platform not recognised. Valid platforms are: {', '.join(self._choices['platform'])}"
            )
        if platform == "cuda" and _os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            raise ValueError("CUDA platform requested but CUDA_VISIBLE_DEVICES not set")
        elif platform == "opencl" and _os.environ.get("OPENCL_VISIBLE_DEVICES") is None:
            raise ValueError(
                "OpenCL platform requested but OPENCL_VISIBLE_DEVICES not set"
            )
        else:
            if platform in ["cuda", "auto"] and "CUDA_VISIBLE_DEVICES" in _os.environ:
                self._platform = "cuda"
            elif (
                platform in ["opencl", "auto"]
                and "OPENCL_VISIBLE_DEVICES" in _os.environ
            ):
                self._platform = "cuda"
            elif platform in ["auto", "metal"]:
                if _sys.platform == "darwin":
                    self._platform = "metal"
                else:
                    if platform == "metal":
                        raise ValueError("Metal platform is only available on macOS")
            else:
                self._platform = "cpu"

    @property
    def max_threads(self):
        return self._max_threads

    @max_threads.setter
    def max_threads(self, max_threads):
        import os as _os

        if max_threads is None or (
            isinstance(max_threads, str)
            and max_threads.lower().replace(" ", "") == "none"
        ):
            self._max_threads = _os.cpu_count()

        else:
            try:
                self._max_threads = int(max_threads)
            except:
                raise ValueError("'max_threads' must be of type 'int'")
            if self._platform == "CUDA":
                _logger.warning(
                    "CUDA platform requested but max_threads set - ignoring max_threads"
                )

    @property
    def max_gpus(self):
        return self._max_gpus

    @max_gpus.setter
    def max_gpus(self, max_gpus):
        import os as _os

        if max_gpus is None or (
            isinstance(max_gpus, str) and max_gpus.lower().replace(" ", "") == "none"
        ):
            if "CUDA_VISIBLE_DEVICES" in _os.environ:
                self._max_gpus = len(_os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            else:
                self._max_gpus = 0

        else:
            try:
                self._max_gpus = int(max_gpus)
            except:
                raise ValueError("'max_gpus' must be of type 'int'")
            if self._platform == "CPU":
                _logger.warning(
                    "CPU platform requested but max_gpus set - ignoring max_gpus"
                )

    @property
    def run_parallel(self):
        return self._run_parallel

    @run_parallel.setter
    def run_parallel(self, run_parallel):
        if not isinstance(run_parallel, bool):
            raise ValueError("'run_parallel' must be of type 'bool'")
        self._run_parallel = run_parallel

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
    def write_config(self):
        return self._write_config

    @write_config.setter
    def write_config(self, write_config):
        if not isinstance(write_config, bool):
            raise ValueError("'write_config' must be of type 'bool'")
        self._write_config = write_config

    @classmethod
    def _create_parser(cls):
        """
        Internal method to create a argparse parser for the config object.
        """

        import configargparse
        import inspect

        # Inspect this object to get the default parameters and docstrings.
        sig = inspect.signature(Config.__init__)
        params = sig.parameters
        params = {
            key: value
            for key, value in params.items()
            if key not in ["self", "args", "kwargs"]
        }

        # Get the docstrings for each parameter.
        doc = inspect.getdoc(Config.__init__).split("\n")

        # Create a dictionary to map the parameter name to the help string.
        help = {}

        # Loop over all parameters.
        for param in params:
            found_param = False
            string = ""

            # Loop over all lines in the docstring until we find the parameter.
            for line in doc:
                if param in line:
                    found_param = True
                elif found_param:
                    if line == "":
                        found_param = False
                        break
                    else:
                        string += line

            # Store the help string for this parameter.
            help[param] = string

        # Initialise the parser.
        parser = configargparse.ArgParser(
            description="SOMD2: GPU accelerated alchemical free-energy engine.",
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            add_config_file_help=True,
        )

        # Add a YAML config file option.
        parser.add(
            "--config",
            required=False,
            is_config_file=True,
            help="YAML config file path",
        )

        # Add the parameters.
        for param in params:
            # Convert underscores to hyphens for the command line.
            cli_param = param.replace("_", "-")

            # Get the type of the parameter. If None, then use str.
            typ = str if params[param].default is None else type(params[param].default)

            # This parameter has choices.
            if param in cls._choices:
                parser.add_argument(
                    f"--{cli_param}",
                    type=typ,
                    default=params[param].default,
                    choices=cls._choices[param],
                    help=help[param],
                    required=False,
                )
            # This is a standard parameter.
            else:
                parser.add_argument(
                    f"--{cli_param}",
                    type=typ,
                    default=params[param].default,
                    help=help[param],
                    required=False,
                )

        return parser
