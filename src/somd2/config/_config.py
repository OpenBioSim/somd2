######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2024
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


from typing import Iterable as _Iterable
from openmm import Platform as _Platform
from pathlib import Path as _Path

import sire as _sr

from somd2 import _logger

# List of supported Sire platforms.
_sire_platforms = _sr.options.Platform.options()

# List of registered OpenMM platforms.
_omm_platforms = [
    _Platform.getPlatform(x).getName().lower()
    for x in range(0, _Platform.getNumPlatforms())
]

# List of available and supported platforms.
_platforms = ["auto"] + [x for x in _sire_platforms if x in _omm_platforms]


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
        "integrator": [
            x
            for x in _sr.options.Integrator.options()
            if x not in ["auto", "verlet", "leapfrog"]
        ],
        "cutoff_type": _sr.options.Cutoff.options(),
        "platform": _platforms,
        "lambda_schedule": [
            "standard_morph",
            "charge_scaled_morph",
        ],
        "log_level": [level.lower() for level in _logger._core.levels],
    }

    # A dictionary of nargs for the various options.
    _nargs = {
        "lambda_values": "+",
    }

    def __init__(
        self,
        log_level="info",
        log_file=None,
        runtime="1ns",
        timestep="4fs",
        temperature="300K",
        pressure="1 atm",
        barostat_frequency=25,
        integrator="langevin_middle",
        cutoff_type="pme",
        cutoff="7.5A",
        h_mass_factor=1.5,
        num_lambda=11,
        lambda_values=None,
        lambda_schedule="standard_morph",
        charge_scale_factor=0.2,
        swap_end_states=False,
        coulomb_power=0.0,
        shift_delta="2A",
        restraints=None,
        constraint="h_bonds",
        perturbable_constraint="h_bonds_not_heavy_perturbed",
        include_constrained_energies=False,
        dynamic_constraints=True,
        com_reset_frequency=10,
        minimise=True,
        equilibration_time="0ps",
        equilibration_timestep="1fs",
        equilibration_constraints=False,
        energy_frequency="1ps",
        save_trajectories=True,
        frame_frequency="20ps",
        save_velocities=False,
        checkpoint_frequency="100ps",
        platform="auto",
        max_threads=None,
        max_gpus=None,
        run_parallel=True,
        output_directory="output",
        restart=False,
        write_config=True,
        overwrite=False,
        somd1_compatibility=False,
        pert_file=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        runtime: str
            Simulation length for each lambda window.

        timestep: str
            Integration time step.

        temperature: str
            Simulation temperature.

        pressure: str
            Simulation pressure. (Simulations will run in the NVT ensemble unless a pressure is specified.)

        barostat_frequency: int
            The number of integration steps between barostat updates.

        integrator: str
            Integrator to use for simulation.

        cutoff_type: str
            Cutoff type to use for simulation.

        cutoff: str
            Non-bonded cutoff distance. Use "infinite" for no cutoff.

        h_mass_factor: float
            Factor by which to scale hydrogen masses.

        num_lambda: int
            Number of lambda windows to use.

        lambda_values: [float]
            A list of lambda values. When specified, this takes precedence over
            the 'num_lambda' option.

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

        restraints: sire.mm._MM.Restraints
            A single set of restraints, or a list of
            sets of restraints that will be applied to
            the atoms during the simulation.

        constraint: str
            Constraint type to use for non-perturbable molecules.

        perturbable_constraint: str
            Constraint type to use for perturbable molecules. If None, then
            this will be set according to what is chosen for the
            non-perturbable constraint.

        include_constrained_energies: bool
            Whether to include constrained energies in the potential.

        dynamic_constraints: bool
            Whether or not to update the length of constraints of perturbable
            bonds with lambda. This defaults to True, meaning that changing
            lambda will change any constraint on a perturbable bond to equal
            to the value of r0 at that lambda value. If this is False, then
            the constraint is set based on the current length.

        com_reset_frequency: int
            Frequency at which to reset the centre of mass of the system.

        minimise: bool
            Whether to minimise the system before simulation.

        equilibration_time: str
            Time interval for equilibration. Only simulations starting from
            scratch will be equilibrated.

        equilibration_timestep: str
            Equilibration timestep. (Can be different to simulation timestep.)

        equilibration_constraints: bool
            Whether to use constraints during equilibration.

        energy_frequency: str
            Frequency at which to output energy data.

        save_trajectories: bool
            Whether to save trajectory files

        frame_frequency: str
            Frequency at which to output trajectory frames.

        save_velocities: bool
            Whether to save velocities in trajectory frames.

        checkpoint_frequency: str
            Frequency at which to save checkpoint files, should be larger than min(energy_frequency, frame_frequency).
            If zero, then no checkpointing will be performed.

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

        restart: bool
            Whether to restart from a previous simulation using files found in 'output-directory'.

        write_config: bool
            Whether to write the configuration options to a YAML file in the output directory.

        log_level: str
            Log level to use.

        log_file: str
            Name of log file, will be saved in output directory.

        overwrite: bool
            Whether to overwrite files in the output directory, if files are detected and
            this is false, SOMD2 will exit without overwriting.

        somd1_compatibility: bool
            Whether to run using a SOMD1 compatible perturbation.

        pert_file: str
            The path to a SOMD1 perturbation file to apply to the reference system.
            When set, this will automatically set 'somd1_compatibility' to True.
        """

        # Setup logger before doing anything else
        self.log_level = log_level
        self.log_file = log_file
        self.output_directory = output_directory

        self.runtime = runtime
        self.temperature = temperature
        self.pressure = pressure
        self.barostat_frequency = barostat_frequency
        self.integrator = integrator
        self.cutoff_type = cutoff_type
        self.cutoff = cutoff
        self.h_mass_factor = h_mass_factor
        self.timestep = timestep
        self.num_lambda = num_lambda
        self.lambda_values = lambda_values
        self.lambda_schedule = lambda_schedule
        self.charge_scale_factor = charge_scale_factor
        self.swap_end_states = swap_end_states
        self.coulomb_power = coulomb_power
        self.shift_delta = shift_delta
        self.restraints = restraints
        self.constraint = constraint
        self.perturbable_constraint = perturbable_constraint
        self.include_constrained_energies = include_constrained_energies
        self.dynamic_constraints = dynamic_constraints
        self.com_reset_frequency = com_reset_frequency
        self.minimise = minimise
        self.equilibration_time = equilibration_time
        self.equilibration_timestep = equilibration_timestep
        self.equilibration_constraints = equilibration_constraints
        self.energy_frequency = energy_frequency
        self.save_trajectories = save_trajectories
        self.frame_frequency = frame_frequency
        self.save_velocities = save_velocities
        self.checkpoint_frequency = checkpoint_frequency
        self.platform = platform
        self.max_threads = max_threads
        self.max_gpus = max_gpus
        self.run_parallel = run_parallel
        self.restart = restart
        self.somd1_compatibility = somd1_compatibility
        self.pert_file = pert_file

        self.write_config = write_config

        self.overwrite = overwrite

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

        from ..io import yaml_to_dict as _yaml_to_dict

        d = _yaml_to_dict(path)

        return Config(**d)

    def as_dict(self, sire_compatible=False):
        """Convert config object to dictionary

        Parameters
        ----------
        sire_compatible: bool
            Whether to convert to a dictionary compatible with Sire,
            this simply converts any options with a value of None to a
            boolean with the value False.
        """
        from pathlib import Path as _Path

        from sire.cas import LambdaSchedule as _LambdaSchedule

        d = {}
        for attr, value in self.__dict__.items():
            if attr.startswith("_extra") or attr.startswith("extra"):
                continue
            attr_l = attr[1:]
            if isinstance(value, _Path):
                d[attr_l] = str(value)
            else:
                try:
                    d[attr_l] = value.to_string()
                except AttributeError:
                    d[attr_l] = value
            if value is None and sire_compatible:
                d[attr_l] = False

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

        if t.value() != 0 and not t.has_same_units(picosecond):
            raise ValueError("'runtime' units are invalid.")

        if t.value() == 0:
            _logger.warning(
                "Runtime is zero - simulation will not run. Set 'runtime' to a non-zero value."
            )

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
                # Handle special case of pressure = "none"
                if pressure.lower().replace(" ", "") == "none":
                    self._pressure = None
                    return
                raise ValueError(
                    f"Unable to parse 'pressure' as a Sire GeneralUnit: {pressure}"
                )
            if not p.has_same_units(atm):
                raise ValueError("'pressure' units are invalid.")

            self._pressure = p

        else:
            self._pressure = pressure

    @property
    def barostat_frequency(self):
        return self._barostat_frequency

    @barostat_frequency.setter
    def barostat_frequency(self, barostat_frequency):
        if not isinstance(barostat_frequency, int):
            raise TypeError("'barostat_frequency' must be of type 'int'")

        if barostat_frequency <= 0:
            raise ValueError("'barostat_frequency' must be a positive integer")

        self._barostat_frequency = barostat_frequency

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
        cutoff_type = cutoff_type.lower().replace(" ", "")
        if cutoff_type not in self._choices["cutoff_type"]:
            raise ValueError(
                f"Cutoff type not recognised. Valid cutoff types are: {', '.join(self._choices['cutoff_type'])}"
            )
        self._cutoff_type = cutoff_type

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff):
        if not isinstance(cutoff, str):
            raise TypeError("'cutoff' must be of type 'str'")

        from sire.units import angstrom

        if cutoff is not None:
            # Handle special case of cutoff = "infinite"
            if cutoff.lower().replace(" ", "") == "infinite":
                self._cutoff = "infinite"
            else:
                try:
                    c = _sr.u(cutoff)
                except:
                    raise ValueError(
                        f"Unable to parse 'cutoff' as a Sire GeneralUnit: {cutoff}"
                    )
                if not c.has_same_units(angstrom):
                    raise ValueError("'cutoff' units are invalid.")

                self._cutoff = c

        else:
            self._cutoff = cutoff

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
        if h_mass_factor < 1.0:
            _logger.warning(
                "Requested hydrogen mass repartitioning factor is less than 1.0. "
                "This will result in a reduction of the mass of hydrogen atoms, "
                "and will likely lead to undesired simulation behaviour."
            )
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

        if t.value() != 0 and not t.has_same_units(femtosecond):
            raise ValueError("'timestep' units are invalid.")

        if t.value() == 0:
            _logger.warning(
                "Timestep is zero - simulation will not run. Set 'timestep' to a non-zero value."
            )

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
    def lambda_values(self):
        return self._lambda_values

    @lambda_values.setter
    def lambda_values(self, lambda_values):
        if lambda_values is not None:
            if not isinstance(lambda_values, _Iterable):
                raise ValueError("'lambda_values' must be an iterable")
            try:
                lambda_values = [float(x) for x in lambda_values]
            except:
                raise ValueError("'lambda_values' must be an iterable of floats")

            if not all(0 <= x <= 1 for x in lambda_values):
                raise ValueError(
                    "All entries in 'lambda_values' must be between 0 and 1"
                )

            self._num_lambda = len(lambda_values)

        self._lambda_values = lambda_values

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
                    self._lambda_schedule = _LambdaSchedule.charge_scaled_morph(0.2)
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
    def restraints(self):
        return self._restraints

    @restraints.setter
    def restraints(self, restraints):
        # If not supplied as a list, convert to a list.
        if restraints is not None:
            if not isinstance(restraints, _Iterable):
                restraints = [restraints]

            # Check that all restraints are of the correct type.
            for restraint in restraints:
                if not isinstance(restraint, _sr.mm._MM.Restraints):
                    raise ValueError(
                        "'restraints' must be a sire.mm._MM.Restraints object, or a list of these objects."
                    )

        self._restraints = restraints

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
            else:
                self._constraint = constraint
        else:
            self._constraint = "none"

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
                    f"'perturbable_constraint' not recognised. Valid constraints are: {', '.join(self._choices['perturbable_constraint'])}"
                )
            else:
                self._perturbable_constraint = perturbable_constraint
        else:
            self._perturbable_constraint = None

    @property
    def include_constrained_energies(self):
        return self._include_constrained_energies

    @include_constrained_energies.setter
    def include_constrained_energies(self, include_constrained_energies):
        if not isinstance(include_constrained_energies, bool):
            raise ValueError("'include_constrained_energies' must be of type 'bool'")
        self._include_constrained_energies = include_constrained_energies

    @property
    def dynamic_constraints(self):
        return self._dynamic_constraints

    @dynamic_constraints.setter
    def dynamic_constraints(self, dynamic_constraints):
        if not isinstance(dynamic_constraints, bool):
            raise ValueError("'dynamic_constraints' must be of type 'bool'")
        self._dynamic_constraints = dynamic_constraints

    @property
    def com_reset_frequency(self):
        return self._com_reset_frequency

    @com_reset_frequency.setter
    def com_reset_frequency(self, com_reset_frequency):
        if not isinstance(com_reset_frequency, int):
            try:
                com_reset_frequency = int(com_reset_frequency)
            except Exception:
                raise ValueError("'com_reset_frequency' must of type 'int'")
        self._com_reset_frequency = com_reset_frequency

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

        if t.value() != 0 and not t.has_same_units(femtosecond):
            raise ValueError("'equilibration_timestep' units are invalid.")

        if t.value() == 0:
            _logger.warning(
                "Equilibration timestep is zero - simulation will not run. Set 'equilibration_timestep' to a non-zero value."
            )

        self._equilibration_timestep = t

    @property
    def equilibration_constraints(self):
        return self._equilibration_constraints

    @equilibration_constraints.setter
    def equilibration_constraints(self, equilibration_constraints):
        if not isinstance(equilibration_constraints, bool):
            raise ValueError("'equilibration_constraints' must be of type 'bool'")
        self._equilibration_constraints = equilibration_constraints

        if not equilibration_constraints and self.equilibration_timestep > _sr.u("1fs"):
            _logger.warning(
                "Equilibration constraints are recommeded for stability when "
                "using a timestep greater than 1fs."
            )

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

        if t.value() != 0 and not t.has_same_units(picosecond):
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

        if t.value() != 0 and not t.has_same_units(picosecond):
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

        if t.value() != 0 and not t.has_same_units(picosecond):
            raise ValueError("'checkpoint_frequency' units are invalid.")

        if (
            t.value() < self._energy_frequency.value()
            and t.value() < self._frame_frequency.value()
        ):
            _logger.warning(
                "Checkpoint frequency is low. Should be greater min(energy_frequency, frame_frequency)"
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
        elif platform == "hip" and _os.environ.get("HIP_VISIBLE_DEVICES") is None:
            raise ValueError("HIP platform requested but HIP_VISIBLE_DEVICES not set")
        else:
            # Set platform in order of priority.

            # CUDA.
            if "cuda" in self._choices["platform"] and platform in ["cuda", "auto"]:
                self._platform = "cuda"

            # OpenCL.
            elif "opencl" in self._choices["platform"] and platform in [
                "opencl",
                "auto",
            ]:
                self._platform = "opencl"

            # HIP.
            elif "hip" in self._choices["platform"] and platform in ["hip", "auto"]:
                self._platform = "hip"

            # Metal.
            elif "metal" in self._choices["platform"] and platform in ["auto", "metal"]:
                self._platform = "metal"

            # Reference.
            elif "reference" in self._choices["platform"] and platform in [
                "auto",
                "reference",
            ]:
                self._platform = "reference"

            # CPU. (Fallback.)
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
    def restart(self):
        return self._restart

    @restart.setter
    def restart(self, restart):
        if not isinstance(restart, bool):
            raise ValueError("'restart' must be of type 'bool'")
        self._restart = restart

    @property
    def somd1_compatibility(self):
        return self._somd1_compatibility

    @somd1_compatibility.setter
    def somd1_compatibility(self, somd1_compatibility):
        if not isinstance(somd1_compatibility, bool):
            raise ValueError("'somd1_compatibility' must be of type 'bool'")
        self._somd1_compatibility = somd1_compatibility

    @property
    def pert_file(self):
        return self._pert_file

    @pert_file.setter
    def pert_file(self, pert_file):
        import os

        if pert_file is not None and not isinstance(pert_file, str):
            raise TypeError("'pert_file' must be of type 'str'")

        if pert_file is not None and not os.path.exists(pert_file):
            raise ValueError(f"Perturbation file does not exist: {pert_file}")

        self._pert_file = pert_file

        if pert_file is not None:
            self._somd1_compatibility = True

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
        if self.log_file is not None:
            # Can now add the log file
            _logger.add(output_directory / self.log_file, level=self.log_level.upper())
            _logger.debug(f"Logging to {output_directory / self.log_file}")
        self._output_directory = output_directory

    @property
    def write_config(self):
        return self._write_config

    @write_config.setter
    def write_config(self, write_config):
        if not isinstance(write_config, bool):
            raise ValueError("'write_config' must be of type 'bool'")
        self._write_config = write_config

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, log_level):
        if not isinstance(log_level, str):
            raise TypeError("'log_level' must be of type 'str'")
        log_level = log_level.lower().replace(" ", "")
        if log_level not in self._choices["log_level"]:
            raise ValueError(
                f"Log level not recognised. Valid log levels are: {', '.join(self._choices['log_level'])}"
            )
        # Do logging setup here for use in the rest of the ocnfig and all other modules.
        import sys

        _logger.remove()
        _logger.add(sys.stderr, level=log_level.upper(), enqueue=True)
        self._log_level = log_level

    @property
    def log_file(self):
        return self._log_file

    @log_file.setter
    def log_file(self, log_file):
        if log_file is not None and not isinstance(log_file, str):
            raise TypeError("'log_file' must be of type 'str'")
        # Can't add the logfile to the logger here as we don't know the output directory yet.
        self._log_file = log_file

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, overwrite):
        if not isinstance(overwrite, bool):
            raise ValueError("'overwrite' must be of type 'bool'")
        self._overwrite = overwrite

    @classmethod
    def _create_parser(cls):
        """
        Internal method to create a argparse parser for the config object.
        """

        import argparse
        import inspect

        # Inspect the signature to get the parameters.
        sig = inspect.signature(Config.__init__)
        params = sig.parameters
        params = {
            key: value
            for key, value in params.items()
            if key not in ["self", "args", "kwargs", "restraints"]
        }

        # Get the docstring.
        doc = inspect.getdoc(Config.__init__).split("\n")

        # Create a dictionary to map the parameter name to the help string.
        help = {}

        # Loop to find the docstring for each parameter.
        for param in params:
            found_param = False
            string = ""

            # Loop over all lines in the docstring until we find the parameter.
            for line in doc:
                line = line.strip()
                if line.startswith(param):
                    found_param = True
                elif found_param:
                    if line == "":
                        found_param = False
                        break
                    else:
                        string += f" {line}"

            # Store the help string for this parameter.
            help[param] = string

        # Initialise the parser.
        parser = argparse.ArgumentParser(
            description="SOMD2: GPU accelerated alchemical free-energy engine.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Add a YAML config file option.
        parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="YAML config file path. Other command-line options will override the config file.",
        )

        # Add the parameters.
        for param in params:
            # Convert underscores to hyphens for the command line.
            cli_param = param.replace("_", "-")

            # Get the type of the parameter. If None, then use str.
            typ = str if params[param].default is None else type(params[param].default)

            # Get the nargs for the parameter.
            if param in cls._nargs:
                nargs = cls._nargs[param]
            else:
                nargs = None

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
                if typ == bool:
                    parser.add_argument(
                        f"--{cli_param}",
                        action=argparse.BooleanOptionalAction,
                        default=params[param].default,
                        help=help[param],
                        required=False,
                    )
                else:
                    parser.add_argument(
                        f"--{cli_param}",
                        type=typ,
                        nargs=nargs,
                        default=params[param].default,
                        help=help[param],
                        required=False,
                    )

        return parser
