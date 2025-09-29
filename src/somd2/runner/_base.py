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

__all__ = ["RunnerBase"]

from pathlib import Path as _Path

import sire as _sr
from sire.system import System as _System

from somd2 import _logger

from ..config import Config as _Config
from ..io import dataframe_to_parquet as _dataframe_to_parquet
from ..io import dict_to_yaml as _dict_to_yaml
from ..io import parquet_append as _parquet_append
from .._utils import _lam_sym


class RunnerBase:
    """
    Base class for the SOMD2 simulation runner.
    """

    def __init__(self, system, config):
        """
        Constructor.

        Parameters
        ----------

        system: str, :class: `System <sire.system.System>`
            The perturbable system to be simulated. This can be either a path
            to a stream file, or a Sire system object.

        config: :class: `Config <somd2.config.Config>`
            The configuration options for the simulation.
        """

        if not isinstance(system, (str, _System)):
            msg = "'system' must be of type 'str' or 'sire.system.System'"
            _logger.error(msg)
            raise TypeError(msg)

        if isinstance(system, str):
            # Try to load the stream file.
            try:
                self._system = _sr.stream.load(system)
            except:
                msg = f"Unable to load system from stream file: '{system}'"
                _logger.error(msg)
                raise IOError(msg)
        else:
            self._system = system.clone()

        # Validate the configuration.
        if not isinstance(config, _Config):
            msg = "'config' must be of type 'somd2.config.Config'"
            _logger.error(msg)
            raise TypeError(msg)
        self._config = config
        self._config._extra_args = {}

        # Log the versions of somd2 and sire.
        from somd2 import __version__, _sire_version, _sire_revisionid

        _logger.info(f"somd2 version: {__version__}")
        _logger.info(f"sire version: {_sire_version}+{_sire_revisionid}")

        # Check whether we need to apply a perturbation to the reference system.
        if self._config.pert_file is not None:
            _logger.info(
                f"Applying perturbation to reference system: {self._config.pert_file}"
            )
            try:
                from .._utils._somd1 import apply_pert

                self._system = apply_pert(self._system, self._config.pert_file)
            except Exception as e:
                msg = f"Unable to apply perturbation to reference system: {e}"
                _logger.error(msg)
                raise IOError(msg)

            # If we're not using SOMD1 compatibility, then reconstruct the original
            # perturbable system. We only need to do this if applying modifications
            # to ghost atom bonded terms.
            if (
                not self._config.somd1_compatibility
                and self._config.ghost_modifications
            ):
                from .._utils._somd1 import reconstruct_system

                self._system = reconstruct_system(self._system)

        # Make sure the system contains perturbable molecules.
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            msg = "No perturbable molecules in the system"
            _logger.error(msg)
            raise KeyError(msg)

        # Link properties to the lambda = 0 end state.
        self._system = _sr.morph.link_to_reference(self._system)

        # Set the default configuration options.

        # Restrict the atomic properties used to define light atoms when
        # applying constraints.
        self._config._extra_args["check_for_h_by_max_mass"] = True
        self._config._extra_args["check_for_h_by_mass"] = False
        self._config._extra_args["check_for_h_by_element"] = False
        self._config._extra_args["check_for_h_by_ambertype"] = False

        # Make sure that perturbable LJ sigmas aren't scaled to zero.
        self._config._extra_args["fix_perturbable_zero_sigmas"] = True

        # We're running in SOMD1 compatibility mode.
        if self._config.somd1_compatibility:
            from .._utils._somd1 import make_compatible

            # First, try to make the perturbation SOMD1 compatible.

            _logger.info("Applying SOMD1 perturbation compatibility.")
            self._system = make_compatible(self._system)
            self._system = _sr.morph.link_to_reference(self._system)

            # Next, swap the water topology so that it is in AMBER format.

            try:
                waters = self._system["water"]
            except:
                waters = []

            if len(waters) > 0:
                from sire.legacy.IO import isAmberWater as _isAmberWater
                from sire.legacy.IO import setAmberWater as _setAmberWater

                if not _isAmberWater(waters[0]):
                    num_atoms = waters[0].num_atoms()

                    if num_atoms == 3:
                        # Here we assume TIP3P for any 3-point water model.
                        model = "tip3p"
                    elif num_atoms == 4:
                        # Check for OPC water.
                        try:
                            if (
                                waters[0]
                                .search("element Xx")
                                .atoms()[0]
                                .charge()
                                .value()
                                < -1.1
                            ):
                                model = "opc"
                            else:
                                model = "tip4p"
                        except:
                            model = "tip4p"
                    elif num_atoms == 5:
                        model = "tip5p"
                    try:
                        self._system = _System(
                            _setAmberWater(self._system._system, model)
                        )
                        _logger.info(
                            "Converting water topology to AMBER format for SOMD1 compatibility."
                        )
                    except Exception as e:
                        _logger.error(
                            "Unable to convert water topology to AMBER format for SOMD1 compatibility."
                        )
                        raise e

            # Ghost atoms are considered light when adding bond constraints.
            self._config._extra_args["ghosts_are_light"] = True

        # Apply Boresch modifications to bonded terms involving ghost atoms to
        # avoid spurious couplings to the physical system at the end states.
        elif self._config.ghost_modifications:
            from ghostly import modify

            _logger.info("Applying Boresch modifications to ghost atom bonded terms")
            self._system = modify(self._system)

        # Check for a periodic space.
        self._has_space = self._check_space()

        # Check for water.
        try:
            # The search will fail if there are no water molecules.
            water = self._system["water"].molecules()
            self._has_water = True
        except:
            self._has_water = False

        # Check the end state contraints.
        self._check_end_state_constraints()

        # Get the charge difference between the two end states.
        charge_diff = self._get_charge_difference(self._system)

        # Make sure the difference is integer valued to 2 decimal places.
        if not round(charge_diff, 2).is_integer():
            _logger.warning("Charge difference between end states is not an integer.")
        charge_diff = round(charge_diff)

        # Make sure the charge difference matches the expected value
        # from the config.
        if (
            self._config.charge_difference is not None
            and self._config.charge_difference != charge_diff
        ):
            _logger.warning(
                f"The charge difference of {charge_diff} between the end states "
                f"does not match the specified value of {self._config.charge_difference}"
            )
            # The user value takes precedence.
            charge_diff = self._config.charge_difference
            _logger.info(
                f"Using user-specified value of {self._config.charge_difference}"
            )
        else:
            # Report that the charge will automatically be held constant.
            if charge_diff != 0 and self._config.charge_difference is None:
                _logger.info(
                    f"There is a charge difference of {charge_diff} between the end states. "
                    f"Adding alchemical ions to keep the charge constant."
                )

        # Create alchemical ions.
        if charge_diff != 0:
            self._system = self._create_alchemical_ions(self._system, charge_diff)

        # Set the lambda values.
        if self._config.lambda_values:
            self._lambda_values = self._config.lambda_values
        else:
            self._lambda_values = [
                round(i / (self._config.num_lambda - 1), 5)
                for i in range(0, self._config.num_lambda)
            ]

        # Set the lambda energy list.
        if self._config.lambda_energy is not None:
            self._lambda_energy = self._config.lambda_energy
        else:
            self._lambda_energy = self._lambda_values

        # Make sure the lambda values are in the lambda energy list.
        is_missing = False
        for lambda_value in self._lambda_values:
            if lambda_value not in self._lambda_energy:
                self._lambda_energy.append(lambda_value)
                is_missing = True

        # Make sure the lambda_values entries are unique.
        if not len(self._lambda_values) == len(set(self._lambda_values)):
            msg = "Duplicate entries in 'lambda_values' list."
            _logger.error(msg)
            raise ValueError(msg)

        # Make sure the lambda_energy entries are unique.
        if not len(self._lambda_energy) == len(set(self._lambda_energy)):
            msg = "Duplicate entries in 'lambda_energy' list."
            _logger.error(msg)
            raise ValueError(msg)

        from math import isclose

        # Set the REST2 scale factors.
        if self._config.rest2_scale is not None:
            # Single value. Interpolate between 1.0 at the end states and rest2_scale
            # at lambda = 0.5.
            if isinstance(self._config.rest2_scale, float):
                scale_factors = []
                for lambda_value in self._lambda_energy:
                    scale_factors.append(
                        1.0
                        + (self._config.rest2_scale - 1.0)
                        * (1.0 - 2.0 * abs(lambda_value - 0.5))
                    )
                self._rest2_scale_factors = scale_factors
            else:
                if len(self._config.rest2_scale) != len(self._lambda_energy):
                    msg = f"Length of 'rest2_scale' must match the number of {_lam_sym} values."
                    if is_missing:
                        msg += f"If you have omitted some 'lambda_values` from `lambda_energy`, please "
                        f"add them to `lambda_energy`, along with the corresponding `rest2_scale` values."
                    _logger.error(msg)
                    raise ValueError(msg)
                # Make sure the end states are close to 1.0.
                if isclose(self._lambda_energy[0], 0.0, abs_tol=1e-4):
                    if not isclose(self._config.rest2_scale[0], 1.0, abs_tol=1e-4):
                        msg = f"'rest2_scale' must be 1.0 at {_lam_sym}=0."
                        _logger.error(msg)
                        raise ValueError(msg)
                if isclose(self._lambda_energy[-1], 1.0, abs_tol=1e-4):
                    if not isclose(self._config.rest2_scale[-1], 1.0, abs_tol=1e-4):
                        msg = f"'rest2_scale' must be 1.0 at {_lam_sym}=1."
                        _logger.error(msg)
                        raise ValueError(msg)
                self._rest2_scale_factors = self._config.rest2_scale

        # Apply hydrogen mass repartitioning.
        if self._config.hmr:
            # Work out the current hydrogen mass factor.
            factor_non_water, factor_water = self._get_h_mass_factor(self._system)

            # If using SOMD1 compatibility, then adjust the default value.
            if self._config.somd1_compatibility and self._config.h_mass_factor == 3.0:
                self._config.h_mass_factor = 1.5
                _logger.info(
                    "Using hydrogen mass repartitioning factor of 1.5 for SOMD1 compatibility."
                )

            # We don't support repartiioning water molecules, so check those first.
            if factor_water is not None:
                if not isclose(factor_water, 1.0, abs_tol=1e-4):
                    msg = (
                        "Water molecules have already been repartitioned with "
                        f"a factor of {factor_water:.3f}. We only support "
                        "repartitioning of non-water molecules."
                    )
                    _logger.error(msg)
                    raise ValueError(msg)

            # HMR has already been applied.
            if factor_non_water is not None:
                if not isclose(factor_non_water, 1.0, abs_tol=1e-4):
                    _logger.info(
                        f"Detected existing hydrogen mass repartioning factor of {factor_non_water:.3f}"
                    )

                    if not isclose(
                        factor_non_water, self._config.h_mass_factor, abs_tol=1e-4
                    ):
                        new_factor = self._config.h_mass_factor / factor_non_water
                        _logger.warning(
                            f"Existing hydrogen mass repartitioning factor of {factor_non_water:.3f} "
                            f"does not match the requested value of {self._config.h_mass_factor:.3f}. "
                            f"Applying new factor of {new_factor:.3f}."
                        )
                        self._system = self._repartition_h_mass(
                            self._system, new_factor
                        )

                else:
                    self._system = self._repartition_h_mass(
                        self._system, self._config.h_mass_factor
                    )

        # Make sure the REST2 selection is valid.
        if self._config.rest2_selection is not None:
            from sire.mol import selection_to_atoms

            try:
                atoms = selection_to_atoms(self._system, self._config.rest2_selection)
            except:
                msg = "Invalid 'rest2_selection' value."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the user hasn't selected all atoms.
            if len(atoms) == self._system.num_atoms():
                msg = "REST2 selection cannot contain all atoms in the system."
                _logger.error(msg)
                raise ValueError(msg)

        # Flag whether this is a GPU simulation.
        self._is_gpu = self._config.platform in ["cuda", "opencl", "hip"]

        # Need to verify before doing any directory checks.
        if self._config.restart:
            self._verify_restart_config()

        # Check the output directories and create names of output files.
        self._filenames = self._prepare_output()

        # Store the current system as a reference.
        self._reference_system = self._system.clone()

        # Check for a valid restart.
        if self._config.restart:
            if self._config.use_backup:
                self._restore_backup_files()
            self._is_restart, self._system = self._check_restart()
        else:
            self._is_restart = False
            self._cleanup()

        # Save config whenever 'configure' is called to keep it up to date.
        if self._config.write_config:
            _dict_to_yaml(
                self._config.as_dict(),
                self._filenames[0]["config"],
            )

        # Save the end state topologies to the output directory.
        if isinstance(self._system, list):
            mols = self._system[0]
        else:
            mols = self._system
        # Add ghost waters to the system.
        if self._config.gcmc and self._has_space:
            # Make sure that a pressure has not been set.
            if self._config.pressure is not None:
                msg = "GCMC simulations must be run in the NVT ensemble."
                _logger.error(msg)
                raise ValueError(msg)

            from loch import GCMCSampler
            from numpy.random import default_rng

            # Create a random number generator.
            rng = default_rng()

            # Check that the system is solvated with water molecules. This
            # is required for GCMC simulations since the existing waters
            # provide a template for the ghost waters.
            try:
                water = mols["water"].molecules()[0]
            except:
                msg = "No water molecules in the system. Cannot perform GCMC."
                _logger.error(msg)
                raise ValueError(msg)

            # Create the GCMC system.
            mols = GCMCSampler._prepare_system(
                mols, water, rng, self._config.gcmc_num_waters
            )

            # Store the excess chemical potential.
            self._mu_ex = self._config.gcmc_excess_chemical_potential.value()

        # Append only this number of lines from the end of the dataframe during checkpointing.
        self._energy_per_block = int(
            self._config.checkpoint_frequency / self._config.energy_frequency
        )

        # Zero the energy sample.
        self._nrg_sample = 0

        # GCMC specific validation.
        if self._config.gcmc:
            if self._config.platform != "cuda":
                msg = "GCMC simulations require the CUDA platform."
                _logger.error(msg)
                raise ValueError(msg)

            if not self._has_space:
                msg = "GCMC simulations require a periodic space."
                _logger.error(msg)
                raise ValueError(msg)

            if self._config.pressure != None:
                msg = "GCMC simulations must be run in the NVT ensemble."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the frame frequency is a multiple of the energy frequency.

            # Get the ratio.
            ratio = (
                self._config.frame_frequency / self._config.energy_frequency
            ).value()

            # Make sure it's an integer.
            if not isclose(ratio, round(ratio), abs_tol=1e-4):
                msg = "'frame_frequency' must be a multiple of 'energy_frequency'."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the checkpoint frequency is a multiple of the frame frequency.

            # Get the ratio.
            ratio = (
                self._config.checkpoint_frequency / self._config.frame_frequency
            ).value()

            # Make sure it's an integer.
            if not isclose(ratio, round(ratio), abs_tol=1e-4):
                msg = "'checkpoint_frequency' must be a multiple of 'frame_frequency'."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the runtime is a multiple of the frame frequency.

            # Get the ratio.
            ratio = (self._config.runtime / self._config.frame_frequency).value()

            # Make sure it's an integer.
            if not isclose(ratio, round(ratio), abs_tol=1e-4):
                msg = "'runtime' must be a multiple of 'frame_frequency'."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the selection is valid.
            if self._config.gcmc_selection is not None:
                try:
                    atoms = _sr.mol.selection_to_atoms(
                        self._system, self._config.gcmc_selection
                    )
                except:
                    msg = "Invalid 'gcmc_selection' value."
                    _logger.error(msg)
                    raise ValueError(msg)

        # Store the initial system time.
        if isinstance(self._system, list):
            self._initial_time = []
            for system in self._system:
                if system is None:
                    self._initial_time.append(_sr.u("0 ps"))
                else:
                    self._initial_time.append(system.time())
        else:
            self._initial_time = [self._system.time()] * len(self._lambda_values)

        # Check for missing systems in a multi-system simulation.
        if isinstance(self._system, list):
            ref_system = None
            missing_systems = []
            for i, system in enumerate(self._system):
                if system is not None:
                    ref_system = None
                else:
                    missing_systems.append(i)
            if ref_system is None:
                ref_system = self._reference_system

            # Fill in any missing systems.
            for i in missing_systems:
                self._system[i] = ref_system.clone()

        # Create the lock file name.
        self._lock_file = str(self._config.output_directory / "somd2.lock")

        # Write the end-state topologies to the output directory.
        if isinstance(self._system, list):
            mols = self._system[0]
        else:
            mols = self._system
        mols0 = _sr.morph.link_to_reference(mols)
        mols1 = _sr.morph.link_to_perturbed(mols)
        _sr.save(mols0, self._filenames["topology0"])
        _sr.save(mols1, self._filenames["topology1"])

        # Create the default dynamics kwargs dictionary. These can be overloaded
        # as needed.
        self._dynamics_kwargs = {
            "integrator": config.integrator,
            "temperature": config.temperature,
            "pressure": config.pressure if self._has_water else None,
            "barostat_frequency": config.barostat_frequency,
            "timestep": config.timestep,
            "restraints": config.restraints,
            "cutoff_type": config.cutoff_type,
            "cutoff": config.cutoff,
            "schedule": config.lambda_schedule,
            "platform": config.platform,
            "constraint": config.constraint,
            "perturbable_constraint": config.perturbable_constraint,
            "include_constrained_energies": config.include_constrained_energies,
            "dynamic_constraints": config.dynamic_constraints,
            "swap_end_states": config.swap_end_states,
            "com_reset_frequency": config.com_reset_frequency,
            "vacuum": not self._has_space,
            "coulomb_power": config.coulomb_power,
            "shift_coulomb": config.shift_coulomb,
            "shift_delta": config.shift_delta,
            "rest2_selection": config.rest2_selection,
            "map": config._extra_args,
        }

        # Create the GCMC specific kwargs dictionary.
        if self._config.gcmc:
            self._gcmc_kwargs = {
                "reference": self._config.gcmc_selection,
                "excess_chemical_potential": str(
                    self._config.gcmc_excess_chemical_potential
                ),
                "standard_volume": str(self._config.gcmc_standard_volume),
                "radius": str(self._config.gcmc_radius),
                "num_ghost_waters": self._config.gcmc_num_waters,
                "bulk_sampling_probability": self._config.gcmc_bulk_sampling_probability,
                "cutoff_type": self._config.cutoff_type,
                "cutoff": str(self._config.cutoff),
                "temperature": str(self._config.temperature),
                "lambda_schedule": self._config.lambda_schedule,
                "coulomb_power": self._config.coulomb_power,
                "shift_coulomb": str(self._config.shift_coulomb),
                "shift_delta": str(self._config.shift_delta),
                "overwrite": self._config.overwrite,
                "no_logger": True,
            }
        else:
            self._gcmc_kwargs = None

    def _check_space(self):
        """
        Check if the system has a periodic space.

        Returns
        -------

        has_space: bool
            Whether the system has a periodic space.
        """
        if (
            self._system.has_property("space")
            and self._system.property("space").is_periodic()
        ):
            return True
        else:
            _logger.info("No periodic space detected. Assuming vacuum simulation.")
            if self._config.cutoff_type != "none":
                _logger.info(
                    "Cannot use PME for non-periodic simulations. Using no cutoff instead."
                )
                self._config.cutoff_type = "none"
            return False

    def _check_end_state_constraints(self):
        """
        Internal function to check whether the constraints are the same at the two
        end states.
        """

        # Find all perturbable molecules in the system..
        pert_mols = self._system.molecules("property is_perturbable")

        # Check constraints at lambda = 0 and lambda = 1 for each perturbable molecule.
        for mol in pert_mols:
            # Create a dynamics object.
            d = mol.dynamics(
                constraint=self._config.constraint,
                perturbable_constraint=self._config.perturbable_constraint,
                platform="cpu",
                map=self._config._extra_args,
            )

            # Get the constraints at lambda = 0.
            constraints0 = d.get_constraints()

            # Update to lambda = 1.
            d.set_lambda(1)

            # Get the constraints at lambda = 1.
            constraints1 = d.get_constraints()

            # Check for equivalence.
            if len(constraints0) != len(constraints1):
                _logger.info(
                    f"Constraints are at not the same at {_lam_sym} = 0 and {_lam_sym} = 1."
                )
            else:
                for c0, c1 in zip(constraints0, constraints1):
                    if c0 != c1:
                        _logger.info(
                            f"Constraints are at not the same at {_lam_sym} = 0 and {_lam_sym} = 1."
                        )
                        break

    @staticmethod
    def _get_charge_difference(system):
        """
        Internal function to check the charge difference between the two end states.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be perturbed.

        Returns
        -------

        charge_diff: int
            The charge difference between the perturbed and reference states.
        """

        reference = _sr.morph.link_to_reference(system).charge().value()
        perturbed = _sr.morph.link_to_perturbed(system).charge().value()

        return perturbed - reference

    @staticmethod
    def _create_alchemical_ions(system, charge_diff):
        """
        Internal function to create alchemical ions to maintain a constant charge.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be perturbed.

        charge_diff: int
            The charge difference between perturbed and reference states.

        Returns
        -------

        system: :class: `System <sire.system.System>`
            The perturbed system with alchemical ions added.
        """

        from sire.legacy.IO import createChlorineIon as _createChlorineIon
        from sire.legacy.IO import createSodiumIon as _createSodiumIon

        # Clone the system.
        system = system.clone()

        # The number of waters to convert is the absolute charge difference.
        num_waters = abs(charge_diff)

        # Make sure there are enough waters to convert. The charge difference should
        # never be this large, but it prevents a crash if it is.
        if num_waters > len(system["water"].molecules()):
            raise ValueError(
                f"Insufficient waters to convert to ions. {num_waters} required, "
                f"{len(system['water'].molecules())} available."
            )

        # Reference coordinates.
        coords = system.molecules("property is_perturbable").coordinates()
        coord_string = f"{coords[0].value()}, {coords[1].value()}, {coords[2].value()}"

        # Find the furthest N waters from the perturbable molecule.
        waters = system[f"furthest {num_waters} waters from {coord_string}"].molecules()

        # Determine the water model.
        if waters[0].num_atoms() == 3:
            model = "tip3p"
        elif waters[0].num_atoms() == 4:
            model = "tip4p"
        elif waters[0].num_atoms() == 5:
            # Note that AMBER has no ion model for tip5p.
            model = "tip4p"

        # Store the molecule numbers for the system.
        numbers = system.numbers()

        # Create the ions.
        for water in waters:
            # Flag to indicate whether we need to reverse the alchemical ion
            # perturbation, i.e. ion to water, rather than water to ion.
            is_reverse = False

            # Create an ion to keep the charge constant throughout the
            # perturbation.
            if charge_diff > 0:
                # Try to find a free chlorine ion so that we match parameters.
                try:
                    has_ion = False
                    ions = system["element Cl"].molecules()
                    for ion in ions:
                        if ion.num_atoms() == 1:
                            has_ion = True
                            _logger.debug("Found Cl- ion in system.")
                            break

                    # If there isn't an ion, then try searching for a free sodium ion.
                    if not has_ion:
                        ions = system["element Na"].molecules()
                        for ion in ions:
                            if ion.num_atoms() == 1:
                                has_ion = True
                                is_reverse = True
                                _logger.debug("Found Na+ ion in system.")
                                break

                    # If not found, create one using a template.
                    if not has_ion:
                        _logger.debug(f"Creating Cl- ion from {model} water template.")
                        ion = _createChlorineIon(
                            water["element O"].coordinates(), model
                        )

                # If not found, create one using a template.
                except:
                    _logger.debug(f"Creating Cl- ion from {model} water template.")
                    ion = _createChlorineIon(water["element O"].coordinates(), model)

                # Create the ion string.
                if is_reverse:
                    ion_str = "Na+"
                else:
                    ion_str = "Cl-"

            else:
                # Try to find a free sodium ion so that we match parameters.
                try:
                    has_ion = False
                    ions = system["element Na"].molecules()
                    for ion in ions:
                        if ion.num_atoms() == 1:
                            has_ion = True
                            _logger.debug("Found Na+ ion in system.")
                            break

                    # If there isn't an ion, then try searching for a free chlorine ion.
                    if not has_ion:
                        ions = system["element Cl"].molecules()
                        for ion in ions:
                            if ion.num_atoms() == 1:
                                has_ion = True
                                is_reverse = True
                                _logger.debug("Found Cl- ion in system.")
                                break

                    # If not found, create one using a template.
                    if not has_ion:
                        _logger.debug(f"Creating Na+ ion from {model} water template.")
                        ion = _createSodiumIon(water["element O"].coordinates(), model)

                # If not found, create one using a template.
                except:
                    _logger.debug(f"Creating Na+ ion from {model} water template.")
                    ion = _createSodiumIon(water["element O"].coordinates(), model)

                # Create the ion string.
                if is_reverse:
                    ion_str = "Cl-"
                else:
                    ion_str = "Na+"

            # Create an alchemical ion: ion --> water.
            if is_reverse:
                merged = _sr.morph.merge(ion, water, map={"as_new_molecule": False})
            # Create an alchemical ion: water --> ion.
            else:
                merged = _sr.morph.merge(water, ion, map={"as_new_molecule": False})

            # Flag that this an alchemical ion.
            merged = merged.edit().set_property("is_alchemical_ion", True).commit()

            # Update the system.
            system.update(merged)

            # Get the index of the perturbed water.
            index = numbers.index(water.number())

            # Log that we are adding an alchemical ion.
            if is_reverse:
                _logger.info(
                    f"Water at molecule index {index} will be perturbed from a "
                    f"{ion_str} ion to keep charge constant."
                )
            else:
                _logger.info(
                    f"Water at molecule index {index} will be perturbed to a "
                    f"{ion_str} ion to keep charge constant."
                )

        return system

    @staticmethod
    def _create_filenames(lambda_array, lambda_value, output_directory, restart=False):
        # Create incremental file name for current restart.
        def increment_filename(base_filename, suffix):
            file_number = 0
            file_path = _Path(output_directory)
            while True:
                filename = (
                    f"{base_filename}_{file_number}.{suffix}"
                    if file_number > 0
                    else f"{base_filename}.{suffix}"
                )
                full_path = file_path / filename
                if not full_path.exists():
                    return filename
                file_number += 1

        if lambda_value not in lambda_array:
            raise ValueError("lambda_value not in lambda_array")
        lam = f"{lambda_value:.5f}"
        filenames = {}
        filenames["checkpoint"] = str(output_directory / f"checkpoint_{lam}.s3")
        filenames["energy_traj"] = str(output_directory / f"energy_traj_{lam}.parquet")
        filenames["trajectory"] = str(output_directory / f"traj_{lam}.dcd")
        filenames["trajectory_chunk"] = str(output_directory / f"traj_{lam}_")
        filenames["energy_components"] = str(
            output_directory / f"energy_components_{lam}.txt"
        )
        filenames["gcmc_ghosts"] = str(output_directory / f"gcmc_ghosts_{lam}.txt")
        if restart:
            filenames["config"] = str(
                output_directory / increment_filename("config", "yaml")
            )
        else:
            filenames["config"] = str(output_directory / "config.yaml")
        return filenames

    def _prepare_output(self):
        """
        Prepare the output directory and create for simulation, creating the
        file names for each lambda value. The directory is checked for existing
        output and files are deleted if necessary. Incremental file names are
        created for restarts.

        Returns
        -------

        filenames: dict
            Dictionary of file names for each lambda value.
        """
        from pathlib import Path as _Path
        from sys import exit as _exit

        filenames = {}
        deleted = []
        for i, lambda_value in enumerate(self._lambda_values):
            files = self._create_filenames(
                self._lambda_values,
                lambda_value,
                self._config.output_directory,
                self._config.restart,
            )
            filenames[i] = files
            if not self._config.restart:
                for file in files.values():
                    if _Path.exists(_Path(file)):
                        deleted.append(_Path(file))
        if len(deleted) > 0:
            if not self._config.overwrite:
                deleted_str = [str(file) for file in deleted]
                _logger.error(
                    f"The following files already exist, use --overwrite to overwrite them: {list(set((deleted_str)))} \n"
                )
                _exit(1)
            # Loop over files to be deleted, ignoring duplicates.
            for file in list(set(deleted)):
                file.unlink()

        # File names for end-state topologies. This can be used for trajectory
        # visulation and analysis.
        filenames["topology0"] = str(self._config.output_directory / "system0.prm7")
        filenames["topology1"] = str(self._config.output_directory / "system1.prm7")

        return filenames

    def _check_restart(self):
        """
        Check the output directory for a valid restart state.

        Returns
        -------

        is_restart: bool
            Whether the simulation is a restart.

        system: :class: `System <sire.system.System>`, List[:class: `System <sire.system.System>`]
            The system or list of systems to be restarted.
        """

        # List to store systems for each lambda value.
        systems = [None] * len(self._lambda_values)

        for i, lambda_value in enumerate(self._lambda_values):
            # Try to load the checkpoint file.
            try:
                system = _sr.stream.load(self._filenames[i]["checkpoint"])
            except:
                if not self._config.replica_exchange:
                    _logger.warning(
                        f"Unable to load checkpoint file for {_lam_sym}={lambda_value:.5f}, starting from scratch."
                    )
                # Repex requires all files to be present.
                else:
                    msg = f"Unable to load checkpoint file for {_lam_sym}={lambda_value:.5f}."
                    _logger.error(msg)
                    raise ValueError(msg)
            else:
                # Check the system is the same as the reference system.
                are_same, reason = self._systems_are_same(self._system, system)
                if not are_same:
                    raise ValueError(
                        f"Checkpoint file does not match system for the following reason: {reason}."
                    )
                # Make sure the configuration is consistent.
                try:
                    self._compare_configs(
                        self._last_config, dict(system.property("config"))
                    )
                except Exception as e:
                    config = dict(system.property("config"))
                    _logger.debug(
                        f"last config: {self._last_config}, current config: {config}"
                    )
                    msg = f"Config for {_lam_sym}={lambda_value} does not match previous config: {str(e)}"
                    _logger.error(msg)
                    raise ValueError(msg)
                # Make sure the lambda value is consistent.
                else:
                    lambda_restart = system.property("lambda")
                    try:
                        lambda_restart == lambda_value
                    except:
                        filename = self._filenames[i]["checkpoint"]
                        msg = (
                            f"Lambda value from checkpoint file {filename} for {_lam_sym}={lambda_restart} "
                            f"does not match expected value {_lam_sym}={lambda_value}."
                        )
                        _logger.error(msg)
                        raise ValueError(msg)

                # Store the system to the list.
                systems[i] = _sr.morph.link_to_perturbed(system)

        # If this is a GCMC simulation, then remove all ghost waters from each of the systems.
        if self._config.gcmc:
            _logger.info("Removing existing ghost waters from GCMC checkpoint systems")
            for i, system in enumerate(systems):
                if system is not None:
                    # Remove the ghost waters from the system.
                    try:
                        for mol in system["property is_ghost_water"].molecules():
                            _logger.debug(
                                f"Removing ghost water molecule {mol.number()} for {_lam_sym}={self._lambda_values[i]:.5f}"
                            )
                            system.remove(mol)
                    except:
                        pass

        return True, systems

    @staticmethod
    def _compare_configs(config1, config2):
        """
        Internal function to check compatibility between two configuration files.
        """

        if not isinstance(config1, dict):
            raise TypeError("'config1' must be of type 'dict'")
        if not isinstance(config2, dict):
            raise TypeError("'config2' must be of type 'dict'")

        from sire.units import GeneralUnit as _GeneralUnit

        # Define the subset of settings that are allowed to change after restart.
        allowed_diffs = [
            "runtime",
            "restart",
            "minimise",
            "max_threads",
            "equilibration_time",
            "equilibration_timestep",
            "equilibration_constraints",
            "energy_frequency",
            "save_trajectory",
            "frame_frequency",
            "save_velocities",
            "platform",
            "max_threads",
            "max_gpus",
            "restart",
            "save_trajectories",
            "write_config",
            "log_level",
            "log_file",
            "overwrite",
            "timeout",
        ]
        for key in config1.keys():
            if key not in allowed_diffs:
                # Extract the config values.
                v1 = config1[key]
                v2 = config2[key]

                # Convert GeneralUnits to strings for comparison.
                if isinstance(v1, _GeneralUnit):
                    v1 = str(v1)
                if isinstance(v2, _GeneralUnit):
                    v2 = str(v2)

                # Convert Sire containers to lists for comparison.
                try:
                    v1 = v1.to_list()
                except:
                    pass
                try:
                    v2 = v2.to_list()
                except:
                    pass

                if (v1 == None and v2 == False) or (v2 == None and v1 == False):
                    continue
                elif v1 != v2:
                    raise ValueError(
                        f"{key} has changed since the last run. This is not "
                        "allowed when using the restart option."
                    )

    def _verify_restart_config(self):
        """
        Verify that the config file matches the config file used to create the
        checkpoint file.
        """
        import yaml as _yaml

        def get_last_config(output_directory):
            """
            Returns the last config file in the output directory.
            """
            import os as _os

            config_files = [
                file
                for file in _os.listdir(output_directory)
                if file.endswith(".yaml") and file.startswith("config")
            ]
            config_files.sort()
            return config_files[-1]

        try:
            last_config = get_last_config(self._config.output_directory)
            with open(self._config.output_directory / last_config) as file:
                _logger.debug(f"Opening config file {last_config}")
                self._last_config = _yaml.safe_load(file)
            config = self._config.as_dict()
        except:
            _logger.info(
                f"No config files found in {self._config.output_directory}, "
                "attempting to retrieve config from lambda = 0 checkpoint file."
            )
            try:
                system_temp = _sr.stream.load(
                    str(self._config.output_directory / "checkpoint_0.00000.s3")
                )
            except:
                expdir = self._config.output_directory / "checkpoint_0.00000.s3"
                _logger.error(f"Unable to load checkpoint file from {expdir}.")
                raise
            else:
                self._last_config = dict(system_temp.property("config"))
                config = self._config.as_dict(sire_compatible=True)
                del system_temp

        self._compare_configs(self._last_config, config)

    @staticmethod
    def _systems_are_same(system0, system1):
        """
        Check for equivalence between a pair of sire systems.

        Parameters
        ----------

        system0: sire.system.System
            The first system to be compared.

        system1: sire.system.System
            The second system to be compared.

        Returns
        -------

        are_same: bool
            Whether the systems are the same.
        """
        if not isinstance(system0, _System):
            raise TypeError("'system0' must be of type 'sire.system.System'")
        if not isinstance(system1, _System):
            raise TypeError("'system1' must be of type 'sire.system.System'")

        # Check for matching number of molecules.
        if not len(system0.molecules()) == len(system1.molecules()):
            reason = "number of molecules do not match"
            return False, reason

        # Check for matching number of residues.
        if not len(system0.residues()) == len(system1.residues()):
            reason = "number of residues do not match"
            return False, reason

        # Check for matching number of atoms.
        if not len(system0.atoms()) == len(system1.atoms()):
            reason = "number of atoms do not match"
            return False, reason

        return True, None

    @staticmethod
    def _get_gpu_devices(platform, oversubscription_factor=1):
        """
        Get list of available GPUs from CUDA_VISIBLE_DEVICES,
        OPENCL_VISIBLE_DEVICES, or HIP_VISIBLE_DEVICES.

        Parameters
        ----------

        platform: str
            The GPU platform to be used for simulations.

        oversubscription_factor: int
            The number of concurrent workers per GPU. Default is 1.

        Returns
        --------

        available_devices : [int]
            List of available device numbers.
        """

        if not isinstance(platform, str):
            raise TypeError("'platform' must be of type 'str'")

        platform = platform.lower().replace(" ", "")

        if platform not in ["cuda", "opencl", "hip"]:
            raise ValueError("'platform' must be one of 'cuda', 'opencl', or 'hip'.")

        import os as _os

        if platform == "cuda":
            if _os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                raise ValueError("CUDA_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                _logger.info(f"CUDA_VISIBLE_DEVICES set to {available_devices}")
        elif platform == "opencl":
            if _os.environ.get("OPENCL_VISIBLE_DEVICES") is None:
                raise ValueError("OPENCL_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("OPENCL_VISIBLE_DEVICES").split(",")
                _logger.info(f"OPENCL_VISIBLE_DEVICES set to {available_devices}")
        elif platform == "hip":
            if _os.environ.get("HIP_VISIBLE_DEVICES") is None:
                raise ValueError("HIP_VISIBLE_DEVICES not set")
            else:
                available_devices = _os.environ.get("HIP_VISIBLE_DEVICES").split(",")
                _logger.info(f"HIP_VISIBLE_DEVICES set to {available_devices}")

        num_gpus = len(available_devices)
        _logger.info(f"Number of GPUs available: {num_gpus}")
        _logger.info(f"Number of concurrent workers per GPU: {oversubscription_factor}")

        return available_devices

    @staticmethod
    def _get_h_mass_factor(system):
        """
        Get the current hydrogen mass factor.

        Parameters
        ----------

        system : :class: `System <sire.system.System>`
            The system of interest.

        Returns

        h_mass_non_water : float
            The mass of the first non-water hydrogen.

        h_mass_water : float
            The mass of the first water hydrogen.
        """

        from sire.mol import Element

        # Store the expected hydrogen mass.
        expected_h_mass = Element("H").mass().value()

        # Get the mass of the first non-water hydrogen.
        try:
            h_mass = system["not water"].molecules()["element H"][0].mass()
            h_mass_non_water = round(h_mass.value() / expected_h_mass, 3)
        except:
            h_mass_non_water = None

        # Get the mass of the first water hydrogen.
        try:
            h_mass = system["water"].molecules()["element H"][0].mass()
            h_mass_water = round(h_mass.value() / expected_h_mass, 3)
        except:
            h_mass_water = None

        return h_mass_non_water, h_mass_water

    @staticmethod
    def _repartition_h_mass(system, factor=1.0):
        """
        Repartition hydrogen masses.

        Parameters
        ----------

        system : :class: `System <sire.system.System>`
            The system to be repartitioned.

        factor :float
            The factor by which hydrogen masses will be scaled.

        Returns
        -------

        system : :class: `System <sire.system.System>`
            The repartitioned system.
        """

        if not isinstance(factor, float):
            raise TypeError("'factor' must be of type 'float'")

        from math import isclose

        # Early exit if no repartitioning is required.
        if isclose(factor, 1.0, abs_tol=1e-4):
            return system

        from sire.morph import (
            repartition_hydrogen_masses as _repartition_hydrogen_masses,
        )

        _logger.info(f"Repartitioning hydrogen masses with factor {factor:.3f}")

        return _repartition_hydrogen_masses(
            system,
            mass_factor=factor,
            ignore_water=True,
        )

    def _checkpoint(
        self,
        system,
        index,
        block,
        speed,
        lambda_energy=None,
        lambda_grad=None,
        is_final_block=False,
    ):
        """
        Save a checkpoint file.

        Parameters
        ----------

        system : :class: `System <sr.system.System>`
            The system to be saved.

        index : int
            The index of the window or replica.

        block : int
            The current block number.

        speed: float
            The speed of the simulation is ns/day.

        lambda_energy : List[float]
            The sampled lambda energy values.

        lambda_grad : List[float]
            Lambda values for finite-difference gradients.

        is_final_block: bool
            Whether this is the final block of the simulation.
        """

        from shutil import copyfile as _copyfile
        from somd2 import __version__, _sire_version, _sire_revisionid

        # Save the end-state GCMC topologies for trajectory analysis and visualisation.
        if self._config.gcmc and block == 0 and index == 0:
            mols0 = _sr.morph.link_to_reference(system)
            mols1 = _sr.morph.link_to_perturbed(system)

            # Save to AMBER format.
            _sr.save(mols0, self._filenames["topology0"])
            _sr.save(mols1, self._filenames["topology1"])

            # Save to PDB format.
            _sr.save(
                mols0,
                self._filenames["topology0"].replace(".prm7", ".pdb"),
            )
            _sr.save(
                mols1,
                self._filenames["topology1"].replace(".prm7", ".pdb"),
            )

        # Get the lambda value.
        lam = self._lambda_values[index]

        # Get the energy trajectory.
        df = system.energy_trajectory(to_alchemlyb=True, energy_unit="kT")

        # Set the lambda values at which energies were sampled.
        if lambda_energy is None:
            lambda_energy = self._lambda_values

        # Create the metadata.
        metadata = {
            "attrs": df.attrs,
            "somd2 version": __version__,
            "sire version": f"{_sire_version}+{_sire_revisionid}",
            "lambda": str(lam),
            "speed": speed,
            "temperature": str(self._config.temperature.value()),
        }

        # Add the lambda gradient if available.
        if lambda_grad is not None:
            metadata["lambda_grad"] = lambda_grad

        if is_final_block:
            # Assemble and save the final trajectory.
            if self._config.save_trajectories:
                # Save the final trajectory chunk to file.
                if system.num_frames() > 0:
                    traj_filename = (
                        self._filenames[index]["trajectory_chunk"] + f"{block}.dcd"
                    )
                    _sr.save(
                        system.trajectory(),
                        traj_filename,
                        format=["DCD"],
                    )

                # Create the final topology file name.
                topology0 = self._filenames["topology0"]

                # Create the final trajectory file name.
                traj_filename = self._filenames[index]["trajectory"]

                # Glob for the trajectory chunks.
                from glob import glob

                traj_chunks = sorted(
                    glob(f"{self._filenames[index]['trajectory_chunk']}*")
                )

                # If this is a restart, then we need to check for an existing
                # trajectory file with the same name. If it exists and is non-empty,
                # then copy it to a backup file and prepend it to the list of chunks.
                if self._config.restart:
                    path = _Path(traj_filename)
                    if path.exists() and path.stat().st_size > 0:
                        from shutil import copyfile

                        copyfile(traj_filename, f"{traj_filename}.bak")
                        traj_chunks = [f"{traj_filename}.bak"] + traj_chunks

                # Load the topology and chunked trajectory files.
                mols = _sr.load([topology0] + traj_chunks)

                # Save the final trajectory to a single file.
                _sr.save(mols.trajectory(), traj_filename, format=["DCD"])

                # Now remove the chunked trajectory files.
                for chunk in traj_chunks:
                    _Path(chunk).unlink()

            # Add config and lambda value to the system properties.
            system.set_property("config", self._config.as_dict(sire_compatible=True))
            system.set_property("lambda", lam)

            # Backup the existing checkpoint file, if it exists.
            path = _Path(self._filenames[index]["checkpoint"])
            if path.exists() and path.stat().st_size > 0:
                _copyfile(
                    self._filenames[index]["checkpoint"],
                    str(self._filenames[index]["checkpoint"]) + ".bak",
                )

            # Stream the final system to file.
            _sr.stream.save(system, self._filenames[index]["checkpoint"])

            # Backup the existing energy trajectory file, if it exists.
            path = _Path(self._filenames[index]["energy_traj"])
            if path.exists() and path.stat().st_size > 0:
                _copyfile(
                    self._filenames[index]["energy_traj"],
                    str(self._filenames[index]["energy_traj"]) + ".bak",
                )

            # Create the final parquet file.
            _dataframe_to_parquet(
                df,
                metadata=metadata,
                filename=self._filenames[index]["energy_traj"],
            )

        else:
            # Update the starting block if necessary.
            if block == 0:
                block = self._start_block

            # Save the current trajectory chunk to file.
            if self._config.save_trajectories:
                if system.num_frames() > 0:
                    traj_filename = (
                        self._filenames[index]["trajectory_chunk"] + f"{block}.dcd"
                    )
                    _sr.save(
                        system.trajectory(),
                        traj_filename,
                        format=["DCD"],
                    )

            # Encode the configuration and lambda value as system properties.
            system.set_property("config", self._config.as_dict(sire_compatible=True))
            system.set_property("lambda", lam)

            # Backup the existing checkpoint file, if it exists.
            path = _Path(self._filenames[index]["checkpoint"])
            if path.exists() and path.stat().st_size > 0:
                _copyfile(
                    self._filenames[index]["checkpoint"],
                    str(self._filenames[index]["checkpoint"]) + ".bak",
                )

            # Stream the checkpoint to file.
            _sr.stream.save(system, self._filenames[index]["checkpoint"])

            # Create the parquet file name.
            filename = self._filenames[index]["energy_traj"]

            # Create the parquet file.
            if block == self._start_block:
                _dataframe_to_parquet(df, metadata=metadata, filename=filename)
            # Append to the parquet file.
            else:
                # Backup the existing energy trajectory file, if it exists.
                path = _Path(self._filenames[index]["energy_traj"])
                if path.exists() and path.stat().st_size > 0:
                    _copyfile(
                        self._filenames[index]["energy_traj"],
                        str(self._filenames[index]["energy_traj"]) + ".bak",
                    )

                _parquet_append(
                    filename,
                    df.iloc[-self._energy_per_block :],
                )

    def _save_energy_components(self, index, context):
        """
        Internal function to save the energy components for each force group to file.

        Parameters
        ----------

        index : int
            The index of the window or replica.

        context : openmm.Context
            The current OpenMM context.
        """

        from copy import deepcopy
        import openmm

        # Get the current context and system.
        system = deepcopy(context.getSystem())

        # Add each force to a unique group.
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)

        # Create a new context.
        new_context = openmm.Context(system, deepcopy(context.getIntegrator()))
        new_context.setPositions(context.getState(getPositions=True).getPositions())

        header = f"{'# Sample':>10}"
        record = f"{self._nrg_sample:>10}"

        # Process the records.
        for i, f in enumerate(system.getForces()):
            state = new_context.getState(getEnergy=True, groups={i})
            name = f.getName()
            name_len = len(name)
            header += f"{f.getName():>{name_len+2}}"
            record += f"{state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalories_per_mole):>{name_len+2}.2f}"

        # Write to file.
        if self._nrg_sample == 0:
            with open(self._filenames[index]["energy_components"], "w") as f:
                f.write(header + "\n")
                f.write(record + "\n")
        else:
            with open(self._filenames[index]["energy_components"], "a") as f:
                f.write(record + "\n")

        # Increment the sample number.
        self._nrg_sample += 1

    def _restore_backup_files(self):
        """
        Restore backup files in the working directory.
        """

        from glob import glob as _glob
        from shutil import copyfile as _copyfile

        # Find all files with a .bak extension in the working directory.
        backup_files = _glob(str(self._config.output_directory / "*.bak"))

        # Strip the .bak extension and copy to the original file name.
        for file in backup_files:
            path = _Path(file)
            new_path = _Path(str(path)[:-4])
            try:
                _copyfile(file, new_path)
            except Exception as e:
                msg = f"Unable to restore backup file {file}: {str(e)}"
                _logger.error(msg)
                raise IOError(msg)

    def _cleanup(self):
        """
        Clean up backup files from the working directory.
        """

        from glob import glob as _glob

        # Find all files with a .bak extension in the working directory.
        backup_files = _glob(str(self._config.output_directory / "*.bak"))

        for file in backup_files:
            path = _Path(file)
            try:
                path.unlink()
            except Exception as e:
                _logger.warning(f"Unable to delete backup file {file}: {str(e)}")
