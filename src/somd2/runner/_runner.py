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

__all__ = ["Runner"]

import platform as _platform

from sire import morph as _morph
from sire import stream as _stream
from sire.system import System as _System

from ..config import Config as _Config
from ..io import dataframe_to_parquet as _dataframe_to_parquet
from ..io import dict_to_yaml as _dict_to_yaml

from somd2 import _logger

if _platform.system() == "Windows":
    _lam_sym = "lambda"
else:
    _lam_sym = "λ"


class Runner:
    """
    Controls the initiation of simulations as well as the assigning of
    resources.
    """

    from multiprocessing import Manager

    _manager = Manager()
    _lock = _manager.Lock()
    _queue = _manager.Queue()

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
            raise TypeError("'system' must be of type 'str' or 'sire.system.System'")

        if isinstance(system, str):
            # Try to load the stream file.
            try:
                self._system = _stream.load(system)
            except:
                raise IOError(f"Unable to load system from stream file: '{system}'")
        else:
            self._system = system

        # Validate the configuration.
        if not isinstance(config, _Config):
            raise TypeError("'config' must be of type 'somd2.config.Config'")
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
                from ._somd1 import _apply_pert

                self._system = _apply_pert(self._system, self._config.pert_file)
                self._config.somd1_compatibility = True
            except Exception as e:
                raise IOError(f"Unable to apply perturbation to reference system: {e}")

        # Make sure the system contains perturbable molecules.
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        # Link properties to the lambda = 0 end state.
        self._system = _morph.link_to_reference(self._system)

        # We're running in SOMD1 compatibility mode.
        if self._config.somd1_compatibility:
            from ._somd1 import _make_compatible

            # First, try to make the perturbation SOMD1 compatible.

            _logger.info("Applying SOMD1 perturbation compatibility.")
            self._system = _make_compatible(self._system)
            self._system = _morph.link_to_reference(self._system)

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
                        # Here we assume tip3p no SPC/E.
                        model = "tip3p"
                    elif num_atoms == 4:
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
                    except:
                        _logger.error(
                            "Unable to convert water topology to AMBER format for SOMD1 compatibility."
                        )

            # Only check for light atoms by the maximum end state mass if running
            # in SOMD1 compatibility mode. Ghost atoms are considered light when
            # adding bond constraints. Also fix the LJ sigma for ghost atoms so
            # it isn't scaled to zero.
            self._config._extra_args["ghosts_are_light"] = True
            self._config._extra_args["check_for_h_by_max_mass"] = True
            self._config._extra_args["check_for_h_by_mass"] = False
            self._config._extra_args["check_for_h_by_element"] = False
            self._config._extra_args["check_for_h_by_ambertype"] = False
            self._config._extra_args["fix_perturbable_zero_sigmas"] = True
            self._config._extra_args["use_taylor_softening"] = True

        # Check for a periodic space.
        self._check_space()

        # Check the end state contraints.
        self._check_end_state_constraints()

        # Get the charge difference between the two end states.
        charge_diff = self._get_charge_difference(self._system)

        # Make sure the difference is integer valued to 5 decimal places.
        if not round(charge_diff, 4).is_integer():
            _logger.warning("Charge difference between end states is not an integer.")
        charge_diff = int(round(charge_diff, 4))

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

        # Work out the current hydrogen mass factor.
        h_mass_factor, has_hydrogen = self._get_h_mass_factor(self._system)

        # HMR has already been applied.
        from math import isclose

        if has_hydrogen:
            if not isclose(h_mass_factor, 1.0, abs_tol=1e-4):
                _logger.info(
                    f"Detected existing hydrogen mass repartioning factor of {h_mass_factor:.3f}."
                )

                if not isclose(h_mass_factor, self._config.h_mass_factor, abs_tol=1e-4):
                    new_factor = self._config.h_mass_factor / h_mass_factor
                    _logger.warning(
                        f"Existing hydrogen mass repartitioning factor of {h_mass_factor:.3f} "
                        f"does not match the requested value of {self._config.h_mass_factor:.3f}. "
                        f"Applying new factor of {new_factor:.3f}."
                    )
                    self._system = self._repartition_h_mass(self._system, new_factor)

            else:
                self._system = self._repartition_h_mass(
                    self._system, self._config.h_mass_factor
                )

        # Flag whether this is a GPU simulation.
        self._is_gpu = self._config.platform in ["cuda", "opencl", "hip"]

        # Need to verify before doing any directory checks
        if self._config.restart:
            self._verify_restart_config()

        # Check the output directories and create names of output files.
        self._check_directory()

        # Save config whenever 'configure' is called to keep it up to date
        if self._config.write_config:
            _dict_to_yaml(
                self._config.as_dict(),
                self._config.output_directory,
                self._fnames[self._lambda_values[0]]["config"],
            )

    def __str__(self):
        """Return a string representation of the object."""
        return f"Runner(system={self._system}, config={self._config})"

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def _create_shared_resources(self):
        """
        Creates shared list that holds currently available GPUs.
        Also intialises the list with all available GPUs.
        """
        if self._is_gpu:
            if self._config.max_gpus is None:
                self._gpu_pool = self._manager.list(
                    self._zero_gpu_devices(self._get_gpu_devices(self._config.platform))
                )
            else:
                self._gpu_pool = self._manager.list(
                    self._zero_gpu_devices(
                        self._get_gpu_devices(self._config.platform)[
                            : self._config.max_gpus
                        ]
                    )
                )

    def _check_space(self):
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
            _logger.info("No periodic space detected. Assuming vacuum simulation.")
            if self._config.cutoff_type == "pme":
                _logger.info(
                    "Cannot use PME for non-periodic simulations. Using RF cutoff instead."
                )
                self._config.cutoff_type = "rf"

    def _check_end_state_constraints(self):
        """
        Internal function to check whether the constrants are the same at the two
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

        reference = _morph.link_to_reference(system).charge().value()
        perturbed = _morph.link_to_perturbed(system).charge().value()

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
                merged = _morph.merge(ion, water, map={"as_new_molecule": False})
            # Create an alchemical ion: water --> ion.
            else:
                merged = _morph.merge(water, ion, map={"as_new_molecule": False})

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

    def _check_directory(self):
        """
        Find the name of the file from which simulations will be started.
        Paired with the 'restart' option.
        """
        from pathlib import Path as __Path
        from ._dynamics import Dynamics

        fnames = {}
        deleted = []
        for lambda_value in self._lambda_values:
            files = Dynamics.create_filenames(
                self._lambda_values,
                lambda_value,
                self._config.output_directory,
                self._config.restart,
            )
            fnames[lambda_value] = files
            if not self._config.restart:
                for file in files.values():
                    fullpath = self._config.output_directory / file
                    if __Path.exists(fullpath):
                        deleted.append(fullpath)
        if len(deleted) > 0:
            if not self._config.overwrite:
                deleted_str = [str(file) for file in deleted]
                _logger.warning(
                    f"The following files already exist, use --overwrite to overwrite them: {list(set((deleted_str)))} \n"
                )
                exit(1)
            # Loop over files to be deleted, ignoring duplicates
            for file in list(set(deleted)):
                file.unlink()
        self._fnames = fnames

    @staticmethod
    def _compare_configs(config1, config2):
        if not isinstance(config1, dict):
            raise TypeError("'config1' must be of type 'dict'")
        if not isinstance(config2, dict):
            raise TypeError("'config2' must be of type 'dict'")

        from sire.units import GeneralUnit as _GeneralUnit

        # Define the subset of settings that are allowed to change after restart
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
            "checkpoint_frequency",
            "platform",
            "max_threads",
            "max_gpus",
            "run_parallel",
            "restart",
            "save_trajectories",
            "write_config",
            "log_level",
            "log_file",
            "overwrite",
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

                # If one is from sire and the other is not, will raise error even though they are the same
                if (v1 == None and v2 == False) or (v2 == None and v1 == False):
                    continue
                elif v1 != v2:
                    raise ValueError(
                        f"{key} has changed since the last run. This is not allowed when using the restart option."
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
            last_config_file = get_last_config(self._config.output_directory)
            with open(self._config.output_directory / last_config_file) as file:
                _logger.debug(f"Opening config file {last_config_file}")
                self.last_config = _yaml.safe_load(file)
            cfg_curr = self._config.as_dict()
        except:
            _logger.info(
                f"""No config files found in {self._config.output_directory},
                attempting to retrieve config from lambda = 0 checkpoint file."""
            )
            try:
                system_temp = _stream.load(
                    str(self._config.output_directory / "checkpoint_0.00000.s3")
                )
            except:
                expdir = self._config.output_directory / "checkpoint_0.00000.s3"
                _logger.error(f"Unable to load checkpoint file from {expdir}.")
                raise
            else:
                self.last_config = dict(system_temp.property("config"))
                cfg_curr = self._config.as_dict(sire_compatible=True)
                del system_temp

        self._compare_configs(self.last_config, cfg_curr)

    @staticmethod
    def _systems_are_same(system0, system1):
        """Check for equivalence between a pair of sire systems.

        Parameters
        ----------
        system0: sire.system.System
            The first system to be compared.

        system1: sire.system.System
            The second system to be compared.
        """
        if not isinstance(system0, _System):
            raise TypeError("'system0' must be of type 'sire.system.System'")
        if not isinstance(system1, _System):
            raise TypeError("'system1' must be of type 'sire.system.System'")

        # Check for matching uids
        if not system0._system.uid() == system1._system.uid():
            reason = "uids do not match"
            return False, reason

        # Check for matching number of molecules
        if not len(system0.molecules()) == len(system1.molecules()):
            reason = "number of molecules do not match"
            return False, reason

        # Check for matching number of residues
        if not len(system0.residues()) == len(system1.residues()):
            reason = "number of residues do not match"
            return False, reason

        # Check for matching number of atoms
        if not len(system0.atoms()) == len(system1.atoms()):
            reason = "number of atoms do not match"
            return False, reason

        return True, None

    def get_config(self):
        """
        Returns a dictionary of configuration options.

        Returns
        -------

        config: dict
            Dictionary of simulation options.
        """
        return self._config.as_dict()

    def _update_gpu_pool(self, gpu_num):
        """
        Updates the GPU pool to remove the GPU that has been assigned to a worker.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be added to the pool.
        """
        self._gpu_pool.append(gpu_num)

    def _remove_gpu_from_pool(self, gpu_num):
        """
        Removes a GPU from the GPU pool.

        Parameters
        ----------

        gpu_num: str
            The GPU number to be removed from the pool.
        """
        self._gpu_pool.remove(gpu_num)

    def _set_lambda_schedule(self, schedule):
        """
        Sets the lambda schedule.

        Parameters
        ----------

        schedule: sr.cas.LambdaSchedule
            Lambda schedule to be set.
        """
        self._config.lambda_schedule = schedule

    @staticmethod
    def _get_gpu_devices(platform):
        """
        Get list of available GPUs from CUDA_VISIBLE_DEVICES,
        OPENCL_VISIBLE_DEVICES, or HIP_VISIBLE_DEVICES.

        Parameters
        ----------

        platform: str
            The GPU platform to be used for simulations.

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

        return available_devices

    @staticmethod
    def _zero_gpu_devices(devices):
        """
        Set all device numbers relative to the lowest
        (the device number becomes equal to its index in the list).

        Returns
        -------

        devices : [int]
            List of zeroed available device numbers.
        """
        return [str(devices.index(value)) for value in devices]

    @staticmethod
    def _get_h_mass_factor(system):
        """
        Get the current hydrogen mass factor.

        Parameters
        ----------

        system : :class: `System <sire.system.System>`
            The system of interest.
        """

        from sire.mol import Element

        # Store the expected hydrogen mass.
        expected_h_mass = Element("H").mass().value()

        # Get the mass of the first hydrogen atom.
        try:
            h_mass = system["element H"][0].mass()
        except:
            return expected_h_mass, False

        # Work out the current hydrogen mass factor. We round to 3dp due to
        # the precision of atomic masses loaded from text files.
        return round(h_mass.value() / expected_h_mass, 3), True

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
        )

    def _initialise_simulation(self, system, lambda_value, device=None):
        """
        Create simulation object.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`
            The system to be simulated.

        lambda_value: float
            The lambda value for the simulation.

        device: int
            The GPU device number to be used for the simulation.
        """
        from ._dynamics import Dynamics

        try:
            self._sim = Dynamics(
                system,
                lambda_val=lambda_value,
                lambda_array=self._lambda_values,
                lambda_energy=self._lambda_energy,
                config=self._config,
                device=device,
                has_space=self._has_space,
            )
        except:
            _logger.warning(f"System creation at {_lam_sym} = {lambda_value} failed")
            raise

    def _cleanup_simulation(self):
        """
        Used to delete simulation objects once the required data has been extracted.
        """
        self._sim._cleanup()

    def run(self):
        """
        Use concurrent.futures to run lambda windows in parallel

        Returns
        --------

        results : [bool]
            List of simulation results. (Currently whether the simulation finished
            successfully or not.)
        """
        results = self._manager.list()
        if self._config.run_parallel and (self._config.num_lambda is not None):
            # Create shared resources.
            self._create_shared_resources()

            # Work out the number of workers and the resources for each.

            # CPU platform.
            if self._config.platform == "cpu":
                # If the number of lambda windows is greater than the number of threads, then
                # the number of threads per worker is set to 1.
                if self._config.num_lambda > self._config.max_threads:
                    self.max_workers = self._config.max_threads
                    threads_per_worker = 1
                # Otherwise, divide the number of threads equally between workers.
                else:
                    self.max_workers = self._config.num_lambda
                    threads_per_worker = (
                        self._config.max_threads // self._config.num_lambda
                    )
                self._config._extra_args["threads"] = threads_per_worker

            # (Multi-)GPU platform.
            elif self._is_gpu:
                self.max_workers = len(self._gpu_pool)

            # All other platforms.
            else:
                self._max_workers = 1

            import concurrent.futures as _futures

            with _futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                jobs = {}
                for lambda_value in self._lambda_values:
                    jobs[executor.submit(self.run_window, lambda_value)] = lambda_value
                try:
                    for job in _futures.as_completed(jobs):
                        lambda_value = jobs[job]
                        try:
                            result = job.result()
                        except Exception as e:
                            result = False

                            _logger.error(
                                f"Exception raised for {_lam_sym} = {lambda_value}: {e}"
                            )
                        with self._lock:
                            results.append(result)
                # Kill all current and future jobs if keyboard interrupt.
                except KeyboardInterrupt:
                    for pid in executor._processes:
                        executor._processes[pid].terminate()

        # Serial configuration.
        elif self._config.num_lambda is not None:
            if self._config.platform == "cpu":
                self._config._extra_args = {"threads": self._config.max_threads}
            self._lambda_values = [
                round(i / (self._config.num_lambda - 1), 5)
                for i in range(0, self._config.num_lambda)
            ]
            for lambda_value in self._lambda_values:
                try:
                    result = self.run_window(lambda_value)
                except Exception as e:
                    result = False

                    _logger.error(
                        f"Exception raised for {_lam_sym} = {lambda_value}: {e}"
                    )
                results.append(result)

        else:
            raise ValueError(
                "Vanilla MD not currently supported. Please set num_lambda > 1."
            )

        return results

    def run_window(self, lambda_value):
        """
        Run a single simulation.

        Parameters
        ----------

        lambda_value: float
            The lambda value for the simulation.

        temperature: float
            The temperature for the simulation.

        Returns
        -------

        result: str
            The result of the simulation.
        """

        def _run(sim, is_restart=False):
            # This function is complex due to the mixture of options for minimisation and dynamics
            if self._config.minimise:
                try:
                    df = sim._run(is_restart=is_restart)
                    lambda_grad = sim._lambda_grad
                    speed = sim.get_timing()
                    return df, lambda_grad, speed
                except Exception as e:
                    _logger.warning(
                        f"Minimisation/dynamics at {_lam_sym} = {lambda_value} failed with the "
                        f"following exception {e}, trying again with minimsation at {_lam_sym} = 0."
                    )
                    try:
                        df = sim._run(lambda_minimisation=0.0, is_restart=is_restart)
                        lambda_grad = sim._lambda_grad
                        speed = sim.get_timing()
                        return df, lambda_grad, speed
                    except Exception as e:
                        _logger.error(
                            f"Minimisation/dynamics at {_lam_sym} = {lambda_value} failed, even after "
                            f"minimisation at {_lam_sym} = 0. The following warning was raised: {e}."
                        )
                        raise
            else:
                try:
                    df = sim._run(is_restart)
                    lambda_grad = sim._lambda_grad
                    speed = sim.get_timing()
                    return df, lambda_grad, speed
                except Exception as e:
                    _logger.error(
                        f"Dynamics at {_lam_sym} = {lambda_value} failed. The following warning was "
                        f"raised: {e}. This may be due to a lack of minimisation."
                    )

        if self._config.restart:
            _logger.debug(f"Restarting {_lam_sym} = {lambda_value} from file")
            try:
                system = _stream.load(
                    str(
                        self._config.output_directory
                        / self._fnames[lambda_value]["checkpoint"]
                    )
                ).clone()
            except:
                _logger.warning(
                    f"Unable to load checkpoint file for {_lam_sym}={lambda_value}, starting from scratch."
                )
                system = self._system.clone()
                is_restart = False
            else:
                aresame, reason = self._systems_are_same(self._system, system)
                if not aresame:
                    raise ValueError(
                        f"Checkpoint file does not match system for the following reason: {reason}."
                    )
                try:
                    self._compare_configs(
                        self.last_config, dict(system.property("config"))
                    )
                except:
                    cfg_here = dict(system.property("config"))
                    _logger.debug(
                        f"last config: {self.last_config}, current config: {cfg_here}"
                    )
                    _logger.error(
                        f"Config for {_lam_sym}={lambda_value} does not match previous config."
                    )
                    raise
                else:
                    lambda_encoded = system.property("lambda")
                    try:
                        lambda_encoded == lambda_value
                    except:
                        fname = self._fnames[lambda_value]["checkpoint"]
                        raise ValueError(
                            f"Lambda value from checkpoint file {fname} ({lambda_encoded}) does not match expected value ({lambda_value})."
                        )
                is_restart = True
        else:
            system = self._system.clone()
            is_restart = False
        if is_restart:
            acc_time = system.time()
            if acc_time > self._config.runtime - self._config.timestep:
                _logger.success(
                    f"{_lam_sym} = {lambda_value} already complete. Skipping."
                )
                return True
            else:
                _logger.info(
                    f"Restarting {_lam_sym} = {lambda_value} at time {acc_time}, time remaining = {self._config.runtime - acc_time}"
                )
        # GPU platform.
        if self._is_gpu:
            if self._config.run_parallel:
                with self._lock:
                    gpu_num = self._gpu_pool[0]
                    self._remove_gpu_from_pool(gpu_num)
                    if lambda_value is not None:
                        _logger.info(
                            f"Running {_lam_sym} = {lambda_value} on GPU {gpu_num}"
                        )
            # Assumes that device for non-parallel GPU jobs is 0
            else:
                gpu_num = 0
                _logger.info(f"Running {_lam_sym} = {lambda_value} on GPU 0")
            self._initialise_simulation(system, lambda_value, device=gpu_num)
            try:
                df, lambda_grad, speed = _run(self._sim, is_restart=is_restart)
            except:
                if self._config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
                raise
            else:
                if self._config.run_parallel:
                    with self._lock:
                        self._update_gpu_pool(gpu_num)
            self._sim._cleanup()

        # All other platforms.
        else:
            _logger.info(f"Running {_lam_sym} = {lambda_value}")

            self._initialise_simulation(system, lambda_value)
            try:
                df, lambda_grad, speed = _run(self._sim, is_restart=is_restart)
            except:
                raise
            self._sim._cleanup()

        from somd2 import __version__, _sire_version, _sire_revisionid

        # Add the current lambda value to the list of lambda values and sort.
        lambda_array = self._lambda_energy.copy()
        if lambda_value not in lambda_array:
            lambda_array.append(lambda_value)
        lambda_array = sorted(lambda_array)

        # Write final dataframe for the system to the energy trajectory file.
        # Note that sire s3 checkpoint files contain energy trajectory data, so this works even for restarts.
        _ = _dataframe_to_parquet(
            df,
            metadata={
                "attrs": df.attrs,
                "somd2 version": __version__,
                "sire version": f"{_sire_version}+{_sire_revisionid}",
                "lambda": str(lambda_value),
                "lambda_array": lambda_array,
                "lambda_grad": lambda_grad,
                "speed": speed,
                "temperature": str(self._config.temperature.value()),
            },
            filepath=self._config.output_directory,
            filename=self._fnames[lambda_value]["energy_traj"],
        )
        del system
        _logger.success(
            f"{_lam_sym} = {lambda_value} complete, speed = {speed:.2f} ns day-1"
        )
        return True
