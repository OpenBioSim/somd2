######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2025
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
        "lambda_energy": "+",
        "rest2_scale": "+",
    }

    def __init__(
        self,
        log_level="info",
        log_file="log.txt",
        runtime="1 ns",
        timestep="4 fs",
        temperature="300 K",
        pressure="1 atm",
        surface_tension=None,
        barostat_frequency=25,
        integrator="langevin_middle",
        cutoff_type="pme",
        cutoff="7.5 A",
        h_mass_factor=1.5,
        hmr=True,
        num_lambda=11,
        lambda_values=None,
        lambda_energy=None,
        lambda_schedule="standard_morph",
        charge_scale_factor=0.2,
        swap_end_states=False,
        coulomb_power=0.0,
        shift_coulomb="1 A",
        shift_delta="1.5 A",
        restraints=None,
        constraint="h_bonds",
        perturbable_constraint="h_bonds_not_heavy_perturbed",
        include_constrained_energies=False,
        dynamic_constraints=True,
        ghost_modifications=True,
        charge_difference=None,
        coalchemical_restraint_dist=None,
        com_reset_frequency=10,
        minimise=True,
        equilibration_time="0 ps",
        equilibration_timestep="1 fs",
        equilibration_constraints=False,
        energy_frequency="1 ps",
        save_trajectories=True,
        frame_frequency="100 ps",
        save_velocities=False,
        checkpoint_frequency="100 ps",
        num_checkpoint_workers=None,
        num_energy_neighbours=None,
        null_energy="10000 kcal/mol",
        platform="auto",
        max_threads=None,
        max_gpus=None,
        opencl_platform_index=0,
        oversubscription_factor=1,
        replica_exchange=False,
        perturbed_system=None,
        gcmc=False,
        gcmc_selection=None,
        gcmc_excess_chemical_potential="-6.09 kcal/mol",
        gcmc_standard_volume="30.543 A^3",
        gcmc_num_waters=20,
        gcmc_radius="4 A",
        gcmc_bulk_sampling_probability=0.1,
        gcmc_tolerance=0.0,
        rest2_scale=1.0,
        rest2_selection=None,
        output_directory="output",
        restart=False,
        use_backup=False,
        write_config=True,
        overwrite=False,
        somd1_compatibility=False,
        pert_file=None,
        save_energy_components=False,
        page_size=None,
        timeout="300 s",
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
            Simulation pressure. (Simulations will run in the NVT ensemble unless
            a pressure is specified.)

        surface_tension: str
            Surface tension to use for NPT simulations with a membrane barostat.

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

        hmr: bool
            Whether to use hydrogen mass repartitioning. If False, then the masses
            of the input system will be used. This can be useful if you have
            already repartitioned the masses, or use a different repartitioning
            scheme.

        num_lambda: int
            Number of lambda windows to use.

        lambda_values: [float]
            A list of lambda values. When specified, this takes precedence over
            the 'num_lambda' option.

        lambda_energy: [float]
            A list of lambda values at which to output energy data. If not set,
            then this will be set to the same as 'lambda_values', or the values
            defined by 'num_lambda' if 'lambda_values' is not set.

        lambda_schedule: str
            Lambda schedule to use for alchemical free energy simulations.

        charge_scale_factor: float
            Factor by which to scale charges for charge scaled morph.

        swap_end_states: bool
            Whether to swap the end states of the alchemical system.

        coulomb_power : float
            Power to use for the soft-core Coulomb interaction. This is used
            to soften the electrostatic interaction.

        shift_coulomb : str
            The soft-core shift-coulomb parameter. This is used to soften the
            Coulomb interaction.

        shift_delta : str
            The soft-core shift-delta parameter. This is used to soften the
            Lennard-Jones interaction.

        restraints: sire.mm._MM.Restraints
            A single set of restraints, or a list of sets of restraints that
            will be applied to the atoms during the simulation.

        constraint: str
            Constraint type to use for non-perturbable molecules.

        perturbable_constraint: str
            Constraint type to use for perturbable molecules. If None, then
            this will be set according to what is chosen for the non-perturbable
            constraint.

        include_constrained_energies: bool
            Whether to include constrained energies in the potential.

        dynamic_constraints: bool
            Whether or not to update the length of constraints of perturbable
            bonds with lambda. This defaults to True, meaning that changing
            lambda will change any constraint on a perturbable bond to equal
            to the value of r0 at that lambda value. If this is False, then
            the constraint is set based on the current length.

        ghost_modifications: bool
            Whether to modify bonded terms between ghost atoms and the physical
            system to avoid spurious coupling between the two, which can lead to
            sampling of non-physical conformations. We implement the recommended
            modifcations from https://pubs.acs.org/doi/10.1021/acs.jctc.0c01328

        charge_difference: int
            The charge difference between the two end states. (Perturbed minus
            reference.) If None, then alchemical ions will automatically be
            added to keep the charge constant throughout the perturbation. If
            specified, then the user defined value will take precedence. Note
            the reference used for the charge difference is the same, regardless
            of whether swap-end-states is set, i.e. the states are swapped after
            the charge difference is calculated and alchemical ions are added.

        coalchemical_restraint_dist: str
            The minimum distance at which co-alchemical ions will be kept relative
            to the centre of mass of the perturbable molecule in the system. This is
            used to keep the co-alchemical ion in the bulk, preventing it from interacting
            with the protein or ligand. If None, then no restraint will be applied.
            Only functions for charge change perturbations.

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
            Frequency at which to output energy data. If running using 'replica_exchange',
            then this will also be the frequency at which replica swaps are attempted.
            When performing Grand Canonical Monte Carlo (GCMC) water insertions/deletions
            via 'gcmc=True', this will also be the frequency at which GCMC moves are
            attempted.

        save_trajectories: bool
            Whether to save trajectory files

        frame_frequency: str
            Frequency at which to output trajectory frames.

        save_velocities: bool
            Whether to save velocities in trajectory frames.

        checkpoint_frequency: str
            Frequency at which to save checkpoint files, should be larger than
            min(energy_frequency, frame_frequency). If zero, then no checkpointing
            will be performed.

        num_checkpoint_workers: int
            The number of parallel workers to use when checkpointing during a replica
            exchange simulation. By default, this is set to the number of concurrent
            GPU contexts, i.e. the number of GPUs multiplied by the oversubscription
            factor. The option can be used to reduce the number of workers, which
            can be useful when the system size is large, i.e. when many large
            trajectory files could be written simultaneously.

        platform: str
            Platform to run simulation on.

        max_threads: int
            Maximum number of CPU threads to use for simulation. (Default None, uses all available)
            Does nothing if platform is set to CUDA.

        max_gpus: int
            Maximum number of GPUs to use for simulation (Default None, uses all available.)
            Does nothing if platform is set to CPU.

        opencl_platform_index: int
            The OpenCL platform index to use when multiple OpenCL implementations are
            available on the system.

        oversubscription_factor: int
            The number of OpenMM contexts that can be run on a single GPU at the same time.

        replica_exchange: bool
            Whether to run replica exchange simulation. Currently this can only be used when
            GPU resources are available.

        perturbed_system: str
            The path to a stream file containing a Sire system for the equilibrated perturbed
            end state (lambda = 1). This will be used as the starting conformation all lambda
            windows > 0.5 when performing a replica exchange simulation.

        gcmc: bool
            Whether to perform Grand Canonical Monte Carlo (GCMC) water insertions/deletions.

        gcmc_selection: str
            A sire sslection string specifying the atoms that define the centre of geometry
            of the GCMC sphere. If None, then GCMC moves will be attempted within the entire
            simulation volume.

        gcmc_excess_chemical_potential: str
            The excess chemical potential of water in kcal/mol. The default value is calibrated
            for the TIP3P water model. This can be calculated from the free energy of decoupling
            a single water molecule from bulk.

        gcmc_standard_volume: str
            The standard volume of a water molecule in A^3. The default value is calibrated
            from NPT simulation of TIP3P water.

        gcmc_num_waters: int
            The additional number of ghost water molecules to add to the system. These are
            used as placeholders for GCMC insertion moves.

        gcmc_radius: str
            The radius of the GCMC sphere.

        gcmc_bulk_sampling_probability: float
            The probability of performing bulk GCMC moves, i.e. within the entire simulation
            box rather than the GCMC sphere. These can be used to maintain a constant bulk
            density, i.e. acting as a barostat. (This option has no affect when
            'gcmc_selection=None'.)

        gcmc_tolerance: float
            The tolerance for the GCMC acceptance probability, i.e. the minimum probability
            of acceptance for a move. This can be used to exclude low probability candidates
            that can cause instabilities or crashes for the MD engine.

        rest2_scale: float, list(float)
            The scaling factor for Replica Exchange with Solute Tempering (REST) simulations.
            This is the factor by which the temperature of the solute is scaled with respect to
            the rest of the system. This can either be a single scaling factor, or a list of
            scale factors for each lambda window. When a single scaling factor is used, then
            the scale factor will be interpolated between a value of 1.0 in the end states,
            and the value of 'rest2_scale' in intermediate lambda = 0.5 state. When multiple
            values are used, then the number should match the number of lambda windows at which
            energies are sampled.

        rest2_selection: str
            A sire selection string for atoms to include in the REST2 region in
            addition to any perturbable molecules. For example, "molidx 0 and residx 0,1,2"
            would select atoms from the first three residues of the first molecule. If None,
            then all atoms within perturbable molecules will be included in the REST2 region.
            When atoms within a perturbable molecule are included in the selection, then only
            those atoms will be considered as part of the REST2 region. This allows REST2 to
            be applied to protein mutations.

        output_directory: str
            Path to a directory to store output files.

        restart: bool
            Whether to restart from a previous simulation using files found in 'output-directory'.

        use_backup: bool
            Whether to use backup files when restarting a simulation. If True, then
            files from the last but one checkpoint will be used, rather than the most
            recent checkpoint files. This can be useful if the most recent checkpoint
            files are corrupted, or incomplete, e.g. you are recovering from a crash.

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

        save_energy_components: bool
            Whether to save the energy contribution for each force when checkpointing.
            This is useful when debugging crashes.

        page_size: int
            The page size for trajectory handling in megabytes. If None, then Sire
            will automatically set the page size.

        timeout: str
            Timeout for the minimiser and file lock.

        num_energy_neighbours: int
            The number of neighbouring windows to use when computing the energy
            trajectory for the a given simulation lambda value. This can be
            used to compute energies over a subset of windows, hence reducing
            the cost of computing the energy trajectory. A value of 'null_energy'
            will be added to the energy trajectory for the windows that are
            omitted. If None, then all windows will be used.

        null_energy: str
            The energy value to use for lambda windows that are not
            being computed as part of the energy trajectory.
        """

        # Setup logger before doing anything else
        self.log_level = log_level
        self.log_file = log_file
        self.output_directory = output_directory

        self.runtime = runtime
        self.temperature = temperature
        self.pressure = pressure
        self.surface_tension = surface_tension
        self.barostat_frequency = barostat_frequency
        self.integrator = integrator
        self.cutoff_type = cutoff_type
        self.cutoff = cutoff
        self.h_mass_factor = h_mass_factor
        self.hmr = hmr
        self.timestep = timestep
        self.num_lambda = num_lambda
        self.lambda_values = lambda_values
        self.lambda_energy = lambda_energy
        self.lambda_schedule = lambda_schedule
        self.charge_scale_factor = charge_scale_factor
        self.swap_end_states = swap_end_states
        self.coulomb_power = coulomb_power
        self.shift_coulomb = shift_coulomb
        self.shift_delta = shift_delta
        self.restraints = restraints
        self.constraint = constraint
        self.perturbable_constraint = perturbable_constraint
        self.include_constrained_energies = include_constrained_energies
        self.dynamic_constraints = dynamic_constraints
        self.ghost_modifications = ghost_modifications
        self.charge_difference = charge_difference
        self.coalchemical_restraint_dist = coalchemical_restraint_dist
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
        self.num_checkpoint_workers = num_checkpoint_workers
        self.platform = platform
        self.max_threads = max_threads
        self.max_gpus = max_gpus
        self.opencl_platform_index = opencl_platform_index
        self.oversubscription_factor = oversubscription_factor
        self.replica_exchange = replica_exchange
        self.perturbed_system = perturbed_system
        self.gcmc = gcmc
        self.gcmc_selection = gcmc_selection
        self.gcmc_excess_chemical_potential = gcmc_excess_chemical_potential
        self.gcmc_standard_volume = gcmc_standard_volume
        self.gcmc_num_waters = gcmc_num_waters
        self.gcmc_radius = gcmc_radius
        self.gcmc_bulk_sampling_probability = gcmc_bulk_sampling_probability
        self.gcmc_tolerance = gcmc_tolerance
        self.rest2_scale = rest2_scale
        self.rest2_selection = rest2_selection
        self.restart = restart
        self.use_backup = use_backup
        self.somd1_compatibility = somd1_compatibility
        self.pert_file = pert_file
        self.save_energy_components = save_energy_components
        self.timeout = timeout
        self.num_energy_neighbours = num_energy_neighbours
        self.null_energy = null_energy
        self.page_size = page_size

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

        # Use the path for the perturbed_system option, since the system
        # isn't serializable.
        if self.perturbed_system is not None:
            d["perturbed_system"] = str(self._perturbed_system_file)
            d.pop("perturbed_system_file", None)

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
    def surface_tension(self):
        return self._surface_tension

    @surface_tension.setter
    def surface_tension(self, surface_tension):
        if surface_tension is not None and not isinstance(surface_tension, str):
            raise TypeError("'surface_tension' must be of type 'str'")

        from sire.units import atm, angstrom

        if surface_tension is not None:
            try:
                st = _sr.u(surface_tension)
            except:
                raise ValueError(
                    f"Unable to parse 'surface_tension' as a Sire GeneralUnit: {surface_tension}"
                )
            # Make sure we can handle a value of zero.
            if st == 0:
                st = 0 * atm * angstrom
            elif not st.has_same_units(atm * angstrom):
                raise ValueError("'surface_tension' units are invalid.")

            self._surface_tension = st

        else:
            self._surface_tension = surface_tension

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
    def hmr(self):
        return self._hmr

    @hmr.setter
    def hmr(self, hmr):
        if not isinstance(hmr, bool):
            raise ValueError("'hmr' must be of type 'bool'")
        self._hmr = hmr

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

            # Round to 5dp.
            lambda_values = [round(x, 5) for x in lambda_values]

            self._num_lambda = len(lambda_values)

        self._lambda_values = lambda_values

    @property
    def lambda_energy(self):
        return self._lambda_energy

    @lambda_energy.setter
    def lambda_energy(self, lambda_energy):
        if lambda_energy is not None:
            if not isinstance(lambda_energy, _Iterable):
                raise ValueError("'lambda_energy' must be an iterable")
            try:
                lambda_energy = [float(x) for x in lambda_energy]
            except:
                raise ValueError("'lambda_energy' must be an iterable of floats")

            if not all(0 <= x <= 1 for x in lambda_energy):
                raise ValueError(
                    "All entries in 'lambda_energy' must be between 0 and 1"
                )

            # Round to 5dp.
            lambda_energy = [round(x, 5) for x in lambda_energy]

        self._lambda_energy = lambda_energy

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
    def shift_coulomb(self):
        return self._shift_coulomb

    @shift_coulomb.setter
    def shift_coulomb(self, shift_coulomb):
        if not isinstance(shift_coulomb, str):
            raise TypeError("'shift_coulomb' must be of type 'str'")

        from sire.units import angstrom

        try:
            sc = _sr.u(shift_coulomb)
        except:
            raise ValueError(
                f"Unable to parse 'shift_coulomb' as a Sire GeneralUnit: {shift_coulomb}"
            )
        if not sc.has_same_units(angstrom):
            raise ValueError("'shift_coulomb' units are invalid.")

        self._shift_coulomb = sc

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
    def ghost_modifications(self):
        return self._ghost_modifications

    @ghost_modifications.setter
    def ghost_modifications(self, ghost_modifications):
        if not isinstance(ghost_modifications, bool):
            raise ValueError("'ghost_modifications' must be of type 'bool'")
        self._ghost_modifications = ghost_modifications

    @property
    def charge_difference(self):
        return self._charge_difference

    @charge_difference.setter
    def charge_difference(self, charge_difference):
        if charge_difference is not None:
            if not isinstance(charge_difference, int):
                try:
                    charge_difference = int(charge_difference)
                except:
                    raise ValueError("'charge_difference' must be an integer")
        self._charge_difference = charge_difference

    @property
    def coalchemical_restraint_dist(self):
        return self._coalchemical_restraint_dist

    @coalchemical_restraint_dist.setter
    def coalchemical_restraint_dist(self, coalchemical_restraint_dist):
        if coalchemical_restraint_dist is not None:
            if not isinstance(coalchemical_restraint_dist, str):
                raise TypeError("'coalchemical_restraint_dist' must be of type 'str'")

            from sire.units import angstrom

            try:
                c = _sr.u(coalchemical_restraint_dist)
            except:
                raise ValueError(
                    "Unable to parse 'coalchemical_restraint_dist' as a "
                    f"Sire GeneralUnit: {coalchemical_restraint_dist}"
                )
            if not c.has_same_units(angstrom):
                raise ValueError("'coalchemical_restraint_dist' units are invalid.")

            self._coalchemical_restraint_dist = c
        else:
            self._coalchemical_restraint_dist = None

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
                "Minimisation is highly recommended for increased stability."
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
            and t.value() > 0
        ):
            _logger.warning(
                "Checkpoint frequency is low. Should be greater min(energy_frequency, frame_frequency)"
            )
        if t.value() > self._runtime.value():
            _logger.warning(
                "Checkpoint frequency < runtime, checkpointing will not occur before runtime is reached."
            )
            t = _sr.u("0ps")
        self._checkpoint_frequency = t

    @property
    def num_checkpoint_workers(self):
        return self._num_checkpoint_workers

    @num_checkpoint_workers.setter
    def num_checkpoint_workers(self, num_checkpoint_workers):
        if num_checkpoint_workers is not None:
            if not isinstance(num_checkpoint_workers, int):
                try:
                    num_checkpoint_workers = int(num_checkpoint_workers)
                except:
                    raise ValueError("'num_checkpoint_workers' must be of type 'int'")
            if num_checkpoint_workers < 1:
                raise ValueError("'num_checkpoint_workers' must be greater than 0")
        self._num_checkpoint_workers = num_checkpoint_workers

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
            elif "OPENCL_VISIBLE_DEVICES" in _os.environ:
                self._max_gpus = len(_os.environ["OPENCL_VISIBLE_DEVICES"].split(","))
            elif "HIP_VISIBLE_DEVICES" in _os.environ:
                self._max_gpus = len(_os.environ["HIP_VISIBLE_DEVICES"].split(","))
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
    def opencl_platform_index(self):
        return self._opencl_platform_index

    @opencl_platform_index.setter
    def opencl_platform_index(self, opencl_platform_index):
        if not isinstance(opencl_platform_index, int):
            try:
                opencl_platform_index = int(opencl_platform_index)
            except:
                raise ValueError("'opencl_platform_index' must be of type 'int'")
        if opencl_platform_index < 0:
            raise ValueError(
                "'opencl_platform_index' must be greater than or equal to 0"
            )
        self._opencl_platform_index = opencl_platform_index

    @property
    def oversubscription_factor(self):
        return self._oversubscription_factor

    @oversubscription_factor.setter
    def oversubscription_factor(self, oversubscription_factor):
        if not isinstance(oversubscription_factor, int):
            try:
                oversubscription_factor = int(oversubscription_factor)
            except:
                raise ValueError("'oversubscription_factor' must be of type 'int'")

        if oversubscription_factor < 1:
            raise ValueError("'oversubscription_factor' must be greater than 1")

        self._oversubscription_factor = oversubscription_factor

    @property
    def replica_exchange(self):
        return self._replica_exchange

    @replica_exchange.setter
    def replica_exchange(self, replica_exchange):
        if not isinstance(replica_exchange, bool):
            raise ValueError("'replica_exchange' must be of type 'bool'")
        self._replica_exchange = replica_exchange

    @property
    def perturbed_system(self):
        return self._perturbed_system

    @perturbed_system.setter
    def perturbed_system(self, perturbed_system):
        if perturbed_system is not None:
            if isinstance(perturbed_system, str):
                import os

                if not os.path.exists(perturbed_system):
                    raise ValueError(
                        f"'perturbed_system' stream file does not exist: {perturbed_system}"
                    )

                try:
                    self._perturbed_system = _sr.stream.load(perturbed_system)
                    self._perturbed_system_file = perturbed_system
                except Exception as e:
                    raise ValueError(
                        f"Unable to load 'perturbed_system' stream file: {e}"
                    )
            else:
                raise TypeError("'perturbed_system' must be of type 'str'")
        else:
            self._perturbed_system = None
            self._perturbed_system_file = None

    @property
    def gcmc(self):
        return self._gcmc

    @gcmc.setter
    def gcmc(self, gcmc):
        if not isinstance(gcmc, bool):
            raise ValueError("'gcmc' must be of type 'bool'")

        # GCMC isn't supported on macOS.
        if gcmc:
            import platform as _platform

            if _platform.system() == "Darwin":
                raise ValueError("GCMC is not supported on macOS systems.")

        self._gcmc = gcmc

    @property
    def gcmc_selection(self):
        return self._gcmc_selection

    @gcmc_selection.setter
    def gcmc_selection(self, gcmc_selection):
        if gcmc_selection is not None:
            if not isinstance(gcmc_selection, str):
                raise TypeError("'gcmc_selection' must be of type 'str'")
        self._gcmc_selection = gcmc_selection

    @property
    def gcmc_excess_chemical_potential(self):
        return self._gcmc_excess_chemical_potential

    @gcmc_excess_chemical_potential.setter
    def gcmc_excess_chemical_potential(self, gcmc_excess_chemical_potential):
        if not isinstance(gcmc_excess_chemical_potential, str):
            raise TypeError("'gcmc_excess_chemical_potential' must be of type 'str'")

        from sire.units import kcal_per_mol

        try:
            gcmc_e = _sr.u(gcmc_excess_chemical_potential)
        except:
            raise ValueError(
                "Unable to parse 'gcmc_excess_chemical_potential' "
                f"as a Sire GeneralUnit: {gcmc_excess_chemical_potential}"
            )

        if not gcmc_e.has_same_units(kcal_per_mol):
            raise ValueError("'gcmc_excess_chemical_potential' units are invalid.")

        self._gcmc_excess_chemical_potential = gcmc_e

    @property
    def gcmc_standard_volume(self):
        return self._gcmc_standard_volume

    @gcmc_standard_volume.setter
    def gcmc_standard_volume(self, gcmc_standard_volume):
        if not isinstance(gcmc_standard_volume, str):
            raise TypeError("'gcmc_standard_volume' must be of type 'str'")

        from sire.units import angstrom3

        try:
            gcmc_v = _sr.u(gcmc_standard_volume)
        except:
            raise ValueError(
                "Unable to parse 'gcmc_standard_volume' "
                f"as a Sire GeneralUnit: {gcmc_standard_volume}"
            )

        if not gcmc_v.has_same_units(angstrom3):
            raise ValueError("'gcmc_standard_volume' units are invalid.")

        self._gcmc_standard_volume = gcmc_v

    @property
    def gcmc_num_waters(self):
        return self._gcmc_num_waters

    @gcmc_num_waters.setter
    def gcmc_num_waters(self, gcmc_num_waters):
        if gcmc_num_waters is not None:
            if not isinstance(gcmc_num_waters, int):
                try:
                    gcmc_num_waters = int(gcmc_num_waters)
                except:
                    raise ValueError("'gcmc_num_waters' must be an integer")

            if gcmc_num_waters < 0:
                raise ValueError("'gcmc_num_waters' must be greater than or equal to 0")
        self._gcmc_num_waters = gcmc_num_waters

    @property
    def gcmc_radius(self):
        return self._gcmc_radius

    @gcmc_radius.setter
    def gcmc_radius(self, gcmc_radius):
        if not isinstance(gcmc_radius, str):
            raise TypeError("'gcmc_radius' must be of type 'str'")

        from sire.units import angstrom

        try:
            gcmc_r = _sr.u(gcmc_radius)
        except:
            raise ValueError(
                "Unable to parse 'gcmc_radius' " f"as a Sire GeneralUnit: {gcmc_radius}"
            )

        if not gcmc_r.has_same_units(angstrom):
            raise ValueError("'gcmc_radius' units are invalid.")

        self._gcmc_radius = gcmc_r

    @property
    def gcmc_bulk_sampling_probability(self):
        return self._gcmc_bulk_sampling_probability

    @gcmc_bulk_sampling_probability.setter
    def gcmc_bulk_sampling_probability(self, gcmc_bulk_sampling_probability):
        if not isinstance(gcmc_bulk_sampling_probability, float):
            try:
                gcmc_bulk_sampling_probability = float(gcmc_bulk_sampling_probability)
            except Exception:
                raise ValueError("'gcmc_bulk_sampling_probability' must be a float")
        if gcmc_bulk_sampling_probability < 0.0 or gcmc_bulk_sampling_probability > 1.0:
            raise ValueError(
                "'gcmc_bulk_sampling_probability' must be between 0.0 and 1.0"
            )
        self._gcmc_bulk_sampling_probability = gcmc_bulk_sampling_probability

    @property
    def gcmc_tolerance(self):
        return self._gcmc_tolerance

    @gcmc_tolerance.setter
    def gcmc_tolerance(self, gcmc_tolerance):
        if not isinstance(gcmc_tolerance, float):
            try:
                gcmc_tolerance = float(gcmc_tolerance)
            except Exception:
                raise ValueError("'gcmc_tolerance' must be a float")
        if gcmc_tolerance < 0.0:
            raise ValueError("'gcmc_tolerance' must be greater than or equal to 0.0")
        self._gcmc_tolerance = gcmc_tolerance

    @property
    def rest2_scale(self):
        return self._rest2_scale

    @rest2_scale.setter
    def rest2_scale(self, rest2_scale):
        # Convert to an iterable.
        if not isinstance(rest2_scale, _Iterable):
            rest2_scale = [rest2_scale]

        # Convert to floats.
        try:
            rest2_scale = [float(x) for x in rest2_scale]
        except:
            raise ValueError("'rest2_scale' must be a float, or iterable of floats")

        # Check that all values are greater than 1.0.
        for scale in rest2_scale:
            if scale < 1.0:
                raise ValueError("'rest2_scale' must be greater than or equal to 1.0")

        if len(rest2_scale) == 1:
            rest2_scale = rest2_scale[0]
        self._rest2_scale = rest2_scale

    @property
    def rest2_selection(self):
        return self._rest2_selection

    @rest2_selection.setter
    def rest2_selection(self, rest2_selection):
        if rest2_selection is not None:
            if not isinstance(rest2_selection, str):
                raise TypeError("'rest2_selection' must be of type 'str'")
        self._rest2_selection = rest2_selection

    @property
    def restart(self):
        return self._restart

    @restart.setter
    def restart(self, restart):
        if not isinstance(restart, bool):
            raise ValueError("'restart' must be of type 'bool'")
        self._restart = restart

    @property
    def use_backup(self):
        return self._use_backup

    @use_backup.setter
    def use_backup(self, use_backup):
        if not isinstance(use_backup, bool):
            raise ValueError("'use_backup' must be of type 'bool'")
        self._use_backup = use_backup

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

    @property
    def save_energy_components(self):
        return self._save_energy_components

    @save_energy_components.setter
    def save_energy_components(self, save_energy_components):
        if not isinstance(save_energy_components, bool):
            raise ValueError("'save_energy_components' must be of type 'bool'")
        self._save_energy_components = save_energy_components

    @property
    def page_size(self):
        return self._page_size

    @page_size.setter
    def page_size(self, page_size):
        if page_size is not None:
            if not isinstance(page_size, int):
                try:
                    page_size = int(page_size)
                except:
                    raise ValueError("'page_size' must be of type 'int'")

            if page_size < 1:
                raise ValueError("'page_size' must be greater than 0")

        self._page_size = page_size

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        if not isinstance(timeout, str):
            raise TypeError("'timeout' must be of type 'str'")

        from sire.units import second

        try:
            t = _sr.u(timeout)
        except:
            raise ValueError(
                f"Unable to parse 'timeout' as a Sire GeneralUnit: {timeout}"
            )

        if t.value() != 0 and not t.has_same_units(second):
            raise ValueError("'timeout' units are invalid.")

        self._timeout = t

    @property
    def num_energy_neighbours(self):
        return self._num_energy_neighbours

    @num_energy_neighbours.setter
    def num_energy_neighbours(self, num_energy_neighbours):
        if num_energy_neighbours is not None:
            if not isinstance(num_energy_neighbours, int):
                try:
                    num_energy_neighbours = int(num_energy_neighbours)
                except:
                    raise ValueError("'num_energy_neighbours' must be of type 'int'")
        self._num_energy_neighbours = num_energy_neighbours

    @property
    def null_energy(self):
        return self._null_energy

    @null_energy.setter
    def null_energy(self, null_energy):
        if not isinstance(null_energy, str):
            raise TypeError("'null_energy' must be of type 'str'")

        from sire.units import kcal_per_mol

        try:
            e = _sr.u(null_energy)
        except:
            raise ValueError(
                f"Unable to parse 'null_energy' as a Sire GeneralUnit: {null_energy}"
            )

        if e.value() != 0 and not e.has_same_units(kcal_per_mol):
            raise ValueError("'null_energy' units are invalid.")

        self._null_energy = e

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
        # Do logging setup here for use in the rest of the config and all other modules.
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
        for param in sorted(params):
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

    def _reset_logger(self, logger):
        """
        Internal method to reset the logger.

        This can be used when a parallel process is spawned to ensure that
        the logger is correctly configured.
        """

        import sys

        logger.remove()
        logger.add(sys.stderr, level=self.log_level.upper(), enqueue=True)
        if self.log_file is not None and self.output_directory is not None:
            logger.add(
                self.output_directory / self.log_file, level=self.log_level.upper()
            )
