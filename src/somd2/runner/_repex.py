######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
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

__all__ = ["RepexRunner"]

from filelock import FileLock as _FileLock
from numba import njit as _njit
from shutil import copyfile as _copyfile

import numpy as _np
import pickle as _pickle
import sys as _sys

import sire as _sr

from somd2 import _logger

from .._utils import _lam_sym

from ._base import RunnerBase as _RunnerBase


class DynamicsCache:
    """
    A class for caching dynamics objects.
    """

    def __init__(
        self,
        system,
        lambdas,
        rest2_scale_factors,
        num_gpus,
        dynamics_kwargs,
        gcmc_kwargs=None,
        output_directory=None,
        perturbed_positions=None,
        perturbed_box=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`, List[:class: `System <sire.system.System>`]
            The perturbable system, or systems, to be simulated.

        lambdas: np.ndarray
            The lambda value for each replica.

        rest2_scale_factors: np.ndarray
            The REST2 scaling factor for each replica.

        num_gpus: int
            The number of GPUs to use.

        dynamics_kwargs: dict
            A dictionary of default dynamics keyword arguments.

        gcmc_kwargs: dict
            GCMC specific keyword arguments. If None, then GCMC is not used.

        output_directory: pathlib.Path
            The directory for simulation output.

        perturbed_positions: numpy.ndarray
            The positions for the perturbed state. If None, then the perturbed state
            is not used.

        perturbed_box: numpy.ndarray
            The box vectors for the perturbed state. If None, then the perturbed state
            is not used.
        """

        # Warn if the number of replicas is not a multiple of the number of GPUs.
        if len(lambdas) > num_gpus and len(lambdas) % num_gpus != 0:
            _logger.warning(
                "The number of replicas is not a multiple of the number of GPUs. "
                "This may result in suboptimal performance."
            )

        # Initialise attributes.
        self._lambdas = lambdas
        self._rest2_scale_factors = rest2_scale_factors
        self._states = _np.array(range(len(lambdas)))
        self._old_states = _np.array(range(len(lambdas)))
        self._openmm_states = [None] * len(lambdas)
        self._gcmc_samplers = [None] * len(lambdas)
        self._gcmc_states = [None] * len(lambdas)
        self._num_proposed = _np.matrix(_np.zeros((len(lambdas), len(lambdas))))
        self._num_accepted = _np.matrix(_np.zeros((len(lambdas), len(lambdas))))
        self._num_swaps = _np.matrix(_np.zeros((len(lambdas), len(lambdas))))

        # Create the dynamics objects.
        self._create_dynamics(
            system,
            lambdas,
            rest2_scale_factors,
            num_gpus,
            dynamics_kwargs,
            gcmc_kwargs=gcmc_kwargs,
            output_directory=output_directory,
            perturbed_positions=perturbed_positions,
            perturbed_box=perturbed_box,
        )

    def __setstate__(self, state):
        """
        Set the state of the object.
        """
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        """
        Get the state of the object.
        """

        # Create the state dict.
        d = {
            "_lambdas": self._lambdas,
            "_rest2_scale_factors": self._rest2_scale_factors,
            "_states": self._states,
            "_old_states": self._old_states,
            "_openmm_states": self._openmm_states,
            # Don't pickle the GCMC samplers since they need to be recreated.
            "_gcmc_samplers": len(self._gcmc_samplers) * [None],
            "_gcmc_states": self._gcmc_states,
            "_num_proposed": self._num_proposed,
            "_num_accepted": self._num_accepted,
            "_num_swaps": self._num_swaps,
        }

        return d

    def _create_dynamics(
        self,
        system,
        lambdas,
        rest2_scale_factors,
        num_gpus,
        dynamics_kwargs,
        gcmc_kwargs=None,
        output_directory=None,
        perturbed_positions=None,
        perturbed_box=None,
    ):
        """
        Create the dynamics objects.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`, List[:class: `System <sire.system.System>`]
            The perturbable system, or systems, to be simulated.

        lambdas: np.ndarray
            The lambda value for each replica.

        rest2_scale_factors: np.ndarray
            The REST2 scaling factor for each replica.

        num_gpus: int
            The number of GPUs to use.

        dynamics_kwargs: dict
            A dictionary of default dynamics keyword arguments.

        gcmc_kwargs: dict
            GCMC specific keyword arguments. If None, then GCMC is not used.

        output_directory: pathlib.Path
            The directory for simulation output.

        perturbed_positions: numpy.ndarray
            The positions for the perturbed state. If None, then the perturbed state
            is not used.

        perturbed_box: numpy.ndarray
            The box vectors for the perturbed state. If None, then the perturbed state
            is not used.
        """

        from math import floor

        # Copy the dynamics keyword arguments.
        dynamics_kwargs = dynamics_kwargs.copy()

        # Store the number of replicas.
        num_replicas = len(lambdas)

        # Copy the GCMC keyword arguments.
        if gcmc_kwargs is not None:
            gcmc_kwargs = gcmc_kwargs.copy()

        # Initialise the dynamics object list.
        self._dynamics = []

        # A set of visited device indices.
        devices = set()

        # Determine whether there is a remainder in the number of replicas.
        remainder = num_replicas % num_gpus

        # Store the number of contexts for each device. The last device will
        # have remainder contexts, while all others have
        contexts_per_device = num_replicas * [floor(num_replicas / num_gpus)]

        # Set the last device to have the remainder contexts.
        contexts_per_device[-1] = remainder

        # Create the dynamics objects in serial.
        for i, (lam, scale) in enumerate(zip(lambdas, rest2_scale_factors)):
            # Work out the device index.
            device = i % num_gpus

            # If we've not seen this device before then get the memory statistics
            # prior to creating the dynamics object and GCMC sampler.
            if device not in devices:
                used_mem_before, free_mem_before, total_mem = self._check_device_memory(
                    device
                )

            # This is a restart, get the system for this replica.
            if isinstance(system, list):
                mols = system[i]
            # This is a new simulation.
            else:
                mols = system

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = lam
            dynamics_kwargs["rest2_scale"] = scale

            if gcmc_kwargs is not None:
                try:
                    from loch import GCMCSampler
                except:
                    msg = "loch is not installed. GCMC sampling cannot be performed."
                    _logger.error(msg)

                ghost_file = str(output_directory / f"gcmc_ghosts_{lam:.5f}.txt")

                # Create the GCMC sampler.
                gcmc_sampler = GCMCSampler(
                    mols,
                    device=device,
                    lambda_value=lam,
                    rest2_scale=scale,
                    ghost_file=ghost_file,
                    **gcmc_kwargs,
                )

                # Get the modified GCMC system.
                mols = gcmc_sampler.system()

                # Store the GCMC sampler.
                self._gcmc_samplers[i] = gcmc_sampler

                _logger.info(
                    f"Created GCMC sampler for lambda {lam:.5f} on device {device}"
                )

                # Log the initial position of the GCMC sphere.
                if self._gcmc_samplers[i]._reference is not None:
                    positions = _sr.io.get_coords_array(mols)
                    target = self._gcmc_samplers[i]._get_target_position(positions)
                    _logger.info(
                        f"Initial GCMC sphere centre for lambda {lam:.5f} on device {device}: "
                        f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] A"
                    )

            # Create the dynamics object.
            try:
                dynamics = mols.dynamics(**dynamics_kwargs)
            except Exception as e:
                msg = f"Could not create dynamics object for lambda {lam:.5f} on device {device}: {e}"
                _logger.error(msg)
                raise RuntimeError(msg) from e

            # Update the box vectors and positions if the perturbed state is used.
            if (
                perturbed_positions is not None
                and perturbed_box is not None
                and lam > 0.5
            ):
                from openmm.unit import angstrom

                dynamics.context().setPeriodicBoxVectors(*perturbed_box * angstrom)
                dynamics.context().setPositions(perturbed_positions * angstrom)

            # Bind the GCMC sampler to the dynamics object. This allows the
            # dynamics object to reset the water state in its internal OpenMM
            # context following a crash recovery.
            if gcmc_kwargs is not None:
                gcmc_sampler.bind_dynamics(dynamics)

            # Append the dynamics object.
            self._dynamics.append(dynamics)

            # Check the memory footprint for this device.
            if not device in devices:
                # Add the device to the set of visited devices.
                devices.add(device)

                # Get the current memory usage.
                used_mem, free_mem, total_mem = self._check_device_memory(device)

                # Work out the memory used by this dynamics object and GCMC sampler.
                mem_used = used_mem - used_mem_before

                # Work out the estimated total after all replicas have been created.
                est_total = mem_used * contexts_per_device[device] + used_mem_before

                # If this exceeds the total memory, raise an error.
                if est_total > total_mem:
                    msg = (
                        f"Not enough memory on device {device} for all assigned replicas. "
                        f"Estimated memory usage: {est_total / (1024**3):.2f} GB, "
                        f"Available memory: {total_mem / (1024**3):.2f} GB."
                    )
                    _logger.error(msg)
                    raise MemoryError(msg)

                # If there's less than 20% free memory, raise a warning.
                elif ((total_mem - est_total) / total_mem) < 0.2:
                    _logger.warning(
                        f"Device {device} will have less than 20% free memory "
                        f"after creating all assigned replicas. "
                        f"{est_total / (1024**3):.2f} GB, "
                        f"Available memory: {total_mem / (1024**3):.2f} GB."
                    )

                else:
                    _logger.info(
                        f"Estimated memory usage on device {device} after creating all replicas: "
                        f"{est_total / (1024**3):.2f} GB, "
                        f"Available memory: {total_mem / (1024**3):.2f} GB."
                    )

            _logger.info(
                f"Created dynamics object for lambda {lam:.5f} on device {device}"
            )

    def get(self, index):
        """
        Get the dynamics object (and GCMC sampler) for a given index.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        tuple
            The dynamics object for the replica and its GCMC sampler.
        """
        return self._dynamics[index], self._gcmc_samplers[index]

    def set(self, index, dynamics):
        """
        Set the dynamics object for a given index.

        Parameters
        ----------

        index: int
            The index of the replica.

        dynamics: sire.legacy.Convert.SOMMContext
            The dynamics object.
        """
        self._dynamics[index] = dynamics

    def delete(self, index):
        """
        Delete the dynamics object for a given index.

        Parameters
        ----------

        index: int
            The index of the replica.
        """
        self._dynamics[index] = None

    def save_openmm_state(self, index):
        """
        Save the state of the dynamics object.

        Parameters
        ----------

        index: int
            The index of the replica.
        """
        from openmm.unit import angstrom

        # Get the current OpenMM state.
        state = (
            self._dynamics[index]
            .context()
            .getState(getPositions=True, getVelocities=True)
        )

        # Store the state.
        self._openmm_states[index] = state

    def save_gcmc_state(self, index):
        """
        Save the current GCMC water state for the replica.

        Parameters
        ----------

        index: int
            The index of the replica.
        """
        # Get the GCMC sampler.
        gcmc_sampler = self._gcmc_samplers[index]

        # Store the state.
        self._gcmc_states[index] = gcmc_sampler.water_state()

    def get_states(self):
        """
        Get the states of the dynamics objects.

        Returns
        -------

        np.ndarray
            The states.
        """
        return self._states.copy()

    def set_states(self, states):
        """
        Set the states of the dynamics objects.

        Parameters
        ----------

        states: np.ndarray
            The new states.
        """
        self._states = states

    def mix_states(self):
        """
        Mix the states of the dynamics objects.
        """
        # Mix the states.
        for i, state in enumerate(self._states):
            # The state has changed.
            if i != state:
                _logger.debug(f"Replica {i} seeded from state {state}")
                self._dynamics[i].context().setState(self._openmm_states[state])

                # Swap the water state in the GCMCSamplers.
                if self._gcmc_samplers[i] is not None:
                    # Find the indices of the water states that differ.
                    water_idxs = _np.where(
                        self._gcmc_states[i] != self._gcmc_states[state]
                    )[0]

                    # Update the water state in the GCMCSampler.
                    self._gcmc_samplers[i].push()
                    self._gcmc_samplers[i]._set_water_state(
                        self._dynamics[i].context(),
                        indices=water_idxs,
                        states=self._gcmc_states[state][water_idxs],
                    )
                    self._gcmc_samplers[i].pop()

            # Update the swap matrix.
            old_state = self._old_states[i]
            self._num_swaps[old_state, state] += 1

        # Store the current states.
        self._old_states = self._states.copy()

    def get_proposed(self):
        """
        Return the number of proposed swaps between replicas.
        """
        return self._num_proposed

    def get_accepted(self):
        """
        Return the number of accepted swaps between replicas.
        """
        return self._num_accepted

    def get_swaps(self):
        """
        Return the swap matrix.
        """
        return self._num_swaps

    @staticmethod
    def _check_device_memory(device_index=0):
        """
        Check the memory usage of the specified GPU device.

        Parameters
        ----------

        index: int
            The index of the GPU device.
        """
        import pyopencl as cl

        # Get the device.
        platforms = cl.get_platforms()
        all_devices = []
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                all_devices.extend(devices)
            except:
                continue

        if device_index >= len(all_devices):
            msg = f"Device index {device_index} out of range. Found {len(all_devices)} GPU(s)."
            _logger.error(msg)
            raise IndexError(msg)

        device = all_devices[device_index]
        total = device.global_mem_size

        # NVIDIA: Use pynvml
        if "NVIDIA" in device.vendor:
            try:
                import pynvml

                pynvml.nvmlInit()

                # Find matching device by name
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)

                    if name in device.name or device.name in name:
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        pynvml.nvmlShutdown()
                        return (memory.used, memory.free, memory.total)

                pynvml.nvmlShutdown()
            except Exception as e:
                msg = f"Could not get NVIDIA GPU memory info for device {device_index}: {e}"
                _logger.error(msg)
                raise RuntimeError(msg) from e

        # AMD: Use OpenCL extension
        elif "AMD" in device.vendor or "Advanced Micro Devices" in device.vendor:
            try:
                free_memory_info = device.get_info(0x4038)
                free_kb = (
                    free_memory_info[0]
                    if isinstance(free_memory_info, list)
                    else free_memory_info
                )
                free = free_kb * 1024
                used = total - free
                return (used, free, total)
            except Exception as e:
                msg = (
                    f"Could not get AMD GPU memory info for device {device_index}: {e}"
                )
                _logger.error(msg)
                raise RuntimeError(msg) from e


class RepexRunner(_RunnerBase):
    """
    A class for running replica exchange simulations.
    """

    def __init__(self, system, config):
        """
        Constructor.

        Parameters
        ----------

        system: str, :class: `System <sire.system.System>`
            The perturbable system to be simulated. This can either be a path
            to a stream file, or a Sire system object.

        config: :class: `Config <somd2.config.Config>`
            The configuration options for the simulation.
        """

        # No support for non replica exchange simulations.
        if not config.replica_exchange:
            msg = (
                "The RepexRunner class can only be used for replica exchange simulations. "
                "Please set replica_exchange=True, or use the Runner class."
            )
            _logger.error(msg)
            raise ValueError(msg)

        if config.lambda_energy is not None:
            raise ValueError(
                "'lambda_energy' is not currently supported for replica exchange."
            )

        # Call the base class constructor.
        super().__init__(system, config)

        # Make sure we're using the CUDA or OpenCL platform.
        if self._config.platform not in ["cuda", "opencl"]:
            msg = (
                "Currently replica exchange simulations can only be "
                "run on the CUDA and OpenCL platforms."
            )
            _logger.error(msg)
            raise ValueError(msg)

        # Get the number of available GPUs.
        try:
            gpu_devices = self._get_gpu_devices(
                "cuda", self._config.oversubscription_factor
            )
        except Exception as e:
            _logger.error(f"Could not determine available GPU devices: {e}")
            raise e

        # We can only use replica exchange if we have a GPU.
        if len(gpu_devices) == 0:
            _logger.error("No GPUs available. Cannot run replica exchange.")

        # Set the number of GPUs.
        if self._config.max_gpus is None:
            self._num_gpus = len(gpu_devices)
        else:
            self._num_gpus = min(self._config.max_gpus, len(gpu_devices))

        # Store the name of the dynamics cache pickle file.
        self._repex_state = self._config.output_directory / "repex_state.pkl"

        # Store the name of the replica exchange swap acceptance matrix.
        self._repex_matrix = self._config.output_directory / "repex_matrix.txt"

        # Flag that we haven't equilibrated.
        self._is_equilibration = False

        # Store the default options.
        timestep = self._config.timestep
        constraint = self._config.constraint
        perturbable_constraint = self._config.perturbable_constraint

        # Don't use constraints during minimisation.
        if self._config.minimise and not self._config.minimisation_constraints:
            constraint = "none"
            perturbable_constraint = "none"

        if not self._is_restart and self._config.equilibration_time.value() > 0.0:
            self._is_equilibration = True

            # Don't use constraints during equilibration.
            if not self._config.equilibration_constraints:
                constraint = "none"
                perturbable_constraint = "none"

            # Update the timestep.
            timestep = self._config.equilibration_timestep

        # Update the initial constraint values.
        self._initial_constraint = constraint
        self._initial_perturbable_constraint = perturbable_constraint

        # Copy the dynamics keyword arguments.
        dynamics_kwargs = self._dynamics_kwargs.copy()

        # Overload the dynamics kwargs with any updated options.
        dynamics_kwargs.update(
            {
                "timestep": timestep,
                "constraint": constraint,
                "perturbable_constraint": perturbable_constraint,
            }
        )

        # Create the dynamics cache.
        if not self._is_restart:
            self._dynamics_cache = DynamicsCache(
                self._system,
                self._lambda_values,
                self._rest2_scale_factors,
                self._num_gpus,
                dynamics_kwargs,
                gcmc_kwargs=self._gcmc_kwargs,
                perturbed_positions=self._perturbed_positions,
                perturbed_box=self._perturbed_box,
                output_directory=self._config.output_directory,
            )
        else:
            # Check to see if the simulation is already complete.
            time = self._system[0].time()
            if time > self._config.runtime - self._config.timestep:
                _logger.success(f"Simulation already complete. Exiting.")
                _sys.exit(0)

            try:
                with open(self._repex_state, "rb") as f:
                    self._dynamics_cache = _pickle.load(f)
            except Exception as e:
                _logger.error(
                    f"Could not load dynamics cache from {self._repex_state}: {e}"
                )
                raise e

            # Make sure the number of replicas is the same.
            if len(self._dynamics_cache._lambdas) != self._config.num_lambda:
                _logger.error(
                    f"The number of replicas in the dynamics cache ({len(self._dynamics_cache._lambdas)}) "
                    f"does not match the number of replicas in the configuration ({self._config.num_lambda})."
                )

            # Create the dynamics objects.
            self._dynamics_cache._create_dynamics(
                self._system,
                self._lambda_values,
                self._rest2_scale_factors,
                self._num_gpus,
                self._dynamics_kwargs,
                gcmc_kwargs=self._gcmc_kwargs,
                output_directory=self._config.output_directory,
            )

            # Reset the state of the OpenMM contexts and GCMC samplers.
            for i in range(len(self._lambda_values)):
                dynamics, gcmc_sampler = self._dynamics_cache.get(i)

                # Reset the OpenMM state.
                dynamics.context().setState(self._dynamics_cache._openmm_states[i])

                # Reset the GCMC water state.
                if gcmc_sampler is not None:
                    gcmc_sampler.push()
                    gcmc_sampler._set_water_state(
                        dynamics.context(),
                        states=self._dynamics_cache._gcmc_states[i],
                        force=True,
                    )
                    gcmc_sampler.pop()

        # Conversion factor for reduced potential.
        kT = (_sr.units.k_boltz * self._config.temperature).to(_sr.units.kcal_per_mol)
        self._beta = 1.0 / kT

        # If restarting, subtract the time already run from the total runtime
        if self._config.restart:
            time = self._system[0].time()
            self._config.runtime = str(self._config.runtime - time)

            # Work out the current block number.
            if self._config.checkpoint_frequency.value() > 0.0:
                self._start_block = int(
                    round(time.value() / self._config.checkpoint_frequency.value(), 12)
                )
            else:
                self._start_block = 0
        else:
            self._start_block = 0

        from threading import Lock

        # Create a lock to guard the dynamics cache.
        self._lock = Lock()

    def __str__(self):
        """Return a string representation of the object."""
        return f"RepexRunner(system={self._system}, config={self._config})"

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def run(self):
        """
        Run the replica exchange simulation.
        """

        from math import ceil
        from time import time

        from concurrent.futures import ThreadPoolExecutor
        from itertools import repeat

        # Record the start time.
        start = time()

        # Work out the number of repex cycles.
        cycles = (self._config.runtime / self._config.energy_frequency).value()

        # Handle rounding errors to to internal time representation.
        if abs(cycles - round(cycles)) < 1e-6:
            cycles = int(round(cycles))
        # Round up to ensure we run at least the requested time.
        else:
            cycles = int(ceil(cycles))

        if self._config.checkpoint_frequency.value() > 0.0:
            # Calculate the number of blocks and the remainder time.
            frac = (self._config.runtime / self._config.checkpoint_frequency).value()

            # Handle the case where the runtime is less than the checkpoint frequency.
            if frac < 1.0:
                frac = 1.0
                self._config.checkpoint_frequency = str(self._config.runtime)

            num_blocks = int(frac)
            rem = round(frac - num_blocks, 12)

            # Work out the number of repex cycles per block.
            frac = (
                self._config.checkpoint_frequency.value()
                / self._config.energy_frequency.value()
            )

            # Handle the case where the checkpoint frequency is less than the energy frequency.
            if frac < 1.0:
                frac = 1.0
                self._config.checkpoint_frequency = str(self._config.energy_frequency)

            # Store the number of repex cycles per block.
            cycles_per_checkpoint = int(frac)

        # Otherwise, we don't checkpoint.
        else:
            cycles_per_checkpoint = cycles
            num_blocks = 1
            rem = 0

        # Store the number of concurrent workers.
        num_workers = self._num_gpus * self._config.oversubscription_factor

        # Store the number of workers to use for checkpointing.
        if self._config.num_checkpoint_workers is None:
            num_checkpoint_workers = num_workers
        else:
            num_checkpoint_workers = min(
                self._config.num_checkpoint_workers, num_workers
            )

        # Work out the required number of batches.
        num_batches = ceil(self._config.num_lambda / num_workers)
        num_checkpoint_batches = ceil(self._config.num_lambda / num_checkpoint_workers)

        # Create the replica list.
        replica_list = list(range(self._config.num_lambda))

        # Minimise at each lambda value.
        if self._config.minimise:
            for i in range(num_batches):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for success, index, e in executor.map(
                            self._minimise,
                            replica_list[i * num_workers : (i + 1) * num_workers],
                        ):
                            if not success:
                                msg = f"Minimisation failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                                if self._config.minimisation_errors:
                                    _logger.error(msg)
                                    raise e
                                else:
                                    _logger.warning(msg)
                    except KeyboardInterrupt:
                        _logger.error("Minimisation cancelled. Exiting.")
                        _sys.exit(1)

        # Equilibrate the system.
        if self._is_equilibration and not self._is_restart:
            for i in range(num_batches):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for success, index, e in executor.map(
                            self._equilibrate,
                            replica_list[i * num_workers : (i + 1) * num_workers],
                        ):
                            if not success:
                                _logger.error(
                                    f"Equilibration failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                                )
                                raise e
                    except KeyboardInterrupt:
                        _logger.error("Equilibration cancelled. Exiting.")
                        _sys.exit(1)

        # Current block number.
        block = self._start_block

        # Record the start time for the production block.
        prod_start = time()

        # Store the number of blocks per-frame. For GCMC, we need to write the
        # indices of the current ghost water residues each time a frame is saved.
        # For GCMC simulations, the frame frequency is guaranteed to be a multiple
        # of the energy frequency.
        cycles_per_frame = int(
            self._config.frame_frequency / self._config.energy_frequency
        )

        # Work out the number of cycles per GCMC move.
        if self._config.gcmc:
            cycles_per_gcmc = int(
                self._config.gcmc_frequency / self._config.energy_frequency
            )
        else:
            cycles_per_gcmc = cycles + 1

        # Perform the replica exchange simulation.
        for i in range(cycles):
            _logger.info(f"Running dynamics for cycle {i+1} of {cycles}")

            # Log the states. This is the replica index for the state (positions
            # and velocities) used to seed each replica for the current cycle.
            # For example:
            #   States: [ 2 0 1 3 4 5 6 7 8 9 10 ]
            # means that the final positions and velocities from replica 2 are
            # used to seed replica 0, those from replica 0 are used to seed
            # replica 1, and so on.
            _logger.info(f"States: {self._dynamics_cache.get_states()}")

            # Clear the results list.
            results = []

            # Whether to checkpoint.
            is_checkpoint = i > 0 and i % cycles_per_checkpoint == 0

            # Whether to perform a GCMC move before the dynamics block.
            is_gcmc = i % cycles_per_gcmc == 0

            # Whether a frame is saved at the end of the cycle.
            write_gcmc_ghosts = i > 0 and i % cycles_per_frame == 0

            # Run a dynamics block for each replica, making sure only each GPU is only
            # oversubscribed by a factor of self._config.oversubscription_factor.
            for j in range(num_batches):
                replicas = replica_list[j * num_workers : (j + 1) * num_workers]
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for result, index, energies in executor.map(
                            self._run_block,
                            replicas,
                            repeat(self._lambda_values),
                            repeat(is_gcmc),
                            repeat(write_gcmc_ghosts),
                        ):
                            if not result:
                                _logger.error(
                                    f"Dynamics failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {energies}"
                                )
                                raise energies
                            results.append((index, energies))
                    except KeyboardInterrupt:
                        _logger.error("Dynamics cancelled. Exiting.")
                        _sys.exit(1)

            # Checkpoint.
            if is_checkpoint or i == cycles - 1:
                # Create the lock.
                lock = _FileLock(self._lock_file)

                # Acquire the file lock to ensure that the checkpoint files are
                # in a consistent state if read by another process.
                with lock.acquire(timeout=self._config.timeout.to("seconds")):
                    # First backup existing checkpoint files.
                    for j in range(num_checkpoint_batches):
                        # Get the indices of the replicas in this batch.
                        replicas = replica_list[
                            j
                            * num_checkpoint_workers : (j + 1)
                            * num_checkpoint_workers
                        ]
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            try:
                                for index, error in executor.map(
                                    self._backup_checkpoint,
                                    replicas,
                                ):
                                    if not result:
                                        _logger.error(
                                            f"Backup failed for {_lam_sym} = "
                                            f"{self._lambda_values[index]:.5f}: {error}"
                                        )
                                        raise error
                            except KeyboardInterrupt:
                                _logger.error("Backup cancelled. Exiting.")
                                _sys.exit(1)

                    # Now write the new checkpoint files.
                    for j in range(num_checkpoint_batches):
                        # Get the indices of the replicas in this batch.
                        replicas = replica_list[
                            j
                            * num_checkpoint_workers : (j + 1)
                            * num_checkpoint_workers
                        ]
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            try:
                                for index, error in executor.map(
                                    self._checkpoint,
                                    replicas,
                                    repeat(self._lambda_values),
                                    repeat(block),
                                    repeat(num_blocks + int(rem > 0)),
                                    repeat(i == cycles - 1),
                                ):
                                    if not result:
                                        _logger.error(
                                            f"Checkpoint failed for {_lam_sym} = "
                                            f"{self._lambda_values[index]:.5f}: {error}"
                                        )
                                        raise error
                            except KeyboardInterrupt:
                                _logger.error("Checkpoint cancelled. Exiting.")
                                _sys.exit(1)

            if i < cycles:
                # Assemble and energy matrix from the results.
                _logger.info("Assembling energy matrix")
                energy_matrix = self._assemble_results(results)

                # Mix the replicas.
                _logger.info("Mixing replicas")
                self._dynamics_cache.set_states(
                    self._mix_replicas(
                        self._config.num_lambda,
                        energy_matrix,
                        self._dynamics_cache.get_proposed(),
                        self._dynamics_cache.get_accepted(),
                    )
                )
                self._dynamics_cache.mix_states()

                # This is a checkpoint cycle.
                if is_checkpoint:
                    # Update the block number.
                    block += 1

                    # Guard the repex state and transition matrix saving with a file lock.
                    lock = _FileLock(self._lock_file)
                    with lock.acquire(timeout=self._config.timeout.to("seconds")):
                        # Save the transition matrix.
                        _logger.info("Saving replica exchange transition matrix")
                        self._save_transition_matrix()

                        # Backup the dynamics cache pickle file, if it exists.
                        if self._repex_state.exists():
                            _copyfile(
                                self._repex_state,
                                self._repex_state.with_suffix(".pkl.bak"),
                            )

                        # Pickle the dynamics cache.
                        _logger.info("Saving replica exchange state")
                        with open(self._repex_state, "wb") as f:
                            _pickle.dump(self._dynamics_cache, f)

        # Record the end time for the production block.
        prod_end = time()

        lock = _FileLock(self._lock_file)
        with lock.acquire(timeout=self._config.timeout.to("seconds")):
            # Save the final transition matrix.
            _logger.info("Saving final replica exchange transition matrix")
            self._save_transition_matrix()

            # Backup the dynamics cache pickle file, if it exists.
            if self._repex_state.exists():
                _copyfile(
                    self._repex_state,
                    self._repex_state.with_suffix(".pkl.bak"),
                )

            # Pickle final state of the dynamics cache.
            _logger.info("Saving final replica exchange state")
            with open(self._repex_state, "wb") as f:
                _pickle.dump(self._dynamics_cache, f)

        # Record the end time.
        end = time()

        # Work how many fractional days the production block took.
        prod_time = (prod_end - prod_start) / 86400

        # Record the average production speed. (ns/day per replica)
        prod_speed = self._config.runtime.to("ns") / prod_time

        # Record the average production speed.
        _logger.info(f"Overall performance: {prod_speed:.2f} ns day-1")

        # Log the run time in minutes.
        _logger.success(
            f"Simulation finished. Run time: {(end - start) / 60:.2f} minutes"
        )

        # Delete all backup files from the working directory.
        self._cleanup()

    def _run_block(
        self,
        index,
        lambdas,
        is_gcmc=False,
        write_gcmc_ghosts=False,
    ):
        """
        Run a dynamics block for a given replica.

        Parameters
        ----------

        index: int
            The index of the replica.

        lambdas: np.ndarray
            The lambda values for each replica.

        num_blocks: int
            The total number of blocks.

        is_gcmc: bool
            Whether a GCMC move should be performed before the dynamics block.

        write_gcmc_ghosts: bool
            Whether to write the indices of GCMC ghost residues to
            file.

        Returns
        -------

        success: bool
            Whether the dynamics was successful.

        index: int
            The index of the replica.

        energies: np.ndarray
            The energies at each lambda value. If unsuccessful, the exception
            is returned.
        """

        # Get the lambda value.
        lam = lambdas[index]

        try:
            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(index)

            _logger.info(f"Running dynamics at {_lam_sym} = {lam:.5f}")

            # Draw new velocities from the Maxwell-Boltzmann distribution.
            dynamics.randomise_velocities()

            # Perform a GCMC move. For repex this needs to be done before the
            # dynamics block so that the final energies, which are used in the
            # repex acceptance criteria, are correct.
            if is_gcmc and gcmc_sampler is not None:
                # Push the PyCUDA context on top of the stack.
                gcmc_sampler.push()

                # Perform the GCMC move.
                _logger.info(f"Performing GCMC move at {_lam_sym} = {lam:.5f}")
                gcmc_sampler.move(dynamics.context())

                # Remove the PyCUDA context from the stack.
                gcmc_sampler.pop()

            # Run the dynamics.
            dynamics.run(
                self._config.energy_frequency,
                energy_frequency=self._config.energy_frequency,
                frame_frequency=self._config.frame_frequency,
                lambda_windows=lambdas,
                rest2_scale_factors=self._rest2_scale_factors,
                save_velocities=self._config.save_velocities,
                auto_fix_minimise=True,
                num_energy_neighbours=self._config.num_energy_neighbours,
                null_energy=self._config.null_energy,
                save_crash_report=self._config.save_crash_report,
                # GCMC specific options.
                excess_chemical_potential=(
                    self._mu_ex if gcmc_sampler is not None else None
                ),
                num_waters=(
                    _np.sum(gcmc_sampler.water_state())
                    if gcmc_sampler is not None
                    else None
                ),
            )

            # Set the state.
            self._dynamics_cache.save_openmm_state(index)

            # Save the GCMC state.
            if gcmc_sampler is not None:
                self._dynamics_cache.save_gcmc_state(index)
                # The frame frequency was hit, so write the indices of the
                # current ghost water residues to file.
                if write_gcmc_ghosts:
                    gcmc_sampler.write_ghost_residues()

            # Get the energy at each lambda value.
            energies = (
                dynamics._d.energy_trajectory()
                .to_pandas(to_alchemlyb=True, energy_unit="kcal/mol")
                .iloc[-1, :]
                .to_numpy()
            )

        except Exception as e:
            try:
                # Save the energy components for debugging purposes.
                self._save_energy_components(index, dynamics.context())
            except:
                pass
            return False, index, e

        # Return the index and the energies.
        return (
            True,
            index,
            energies,
        )

    def _minimise(self, index):
        """
        Minimise the system.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        success: bool
            Whether the minimisation was successful.

        index: int
            The index of the replica.

        exception: Exception
            The exception if the minimisation failed.
        """
        _logger.info(f"Minimising at {_lam_sym} = {self._lambda_values[index]:.5f}")

        try:
            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(index)

            if gcmc_sampler is not None:
                # Push the PyCUDA context on top of the stack.
                gcmc_sampler.push()

                _logger.info(
                    f"Pre-equilibrating with GCMC moves at {_lam_sym} = {self._lambda_values[index]:.5f}"
                )
                for i in range(100):
                    gcmc_sampler.move(dynamics.context())

                # Remove the PyCUDA context from the stack.
                gcmc_sampler.pop()

            # Minimise.
            dynamics.minimise(timeout=self._config.timeout)

            # If we're not equilibrating and the production constraints will change,
            # then we need to rebuild the context.
            if not self._is_equilibration:
                constraints_changed = (
                    self._initial_constraint != self._config.constraint
                ) or (
                    self._initial_perturbable_constraint
                    != self._config.perturbable_constraint
                )

                if constraints_changed:
                    # Commit the current system.
                    system = dynamics.commit()

                    # Delete the dynamics object.
                    self._dynamics_cache.delete(index)

                    # Work out the device index.
                    device = index % self._num_gpus

                    # Copy the dynamics keyword arguments.
                    dynamics_kwargs = self._dynamics_kwargs.copy()

                    # Overload the device and lambda value.
                    dynamics_kwargs["device"] = device
                    dynamics_kwargs["lambda_value"] = self._lambda_values[index]
                    dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]

                    # Create the production dynamics object.
                    dynamics = system.dynamics(**dynamics_kwargs)

                    # Reset the GCMC water state. The dynamics object is created from
                    # the original Sire system, so the water state in the context does
                    # not match the current GCMC water state.
                    if gcmc_sampler is not None:
                        self._reset_gcmc_sampler(gcmc_sampler, dynamics)

                    # Set the new dynamics object.
                    self._dynamics_cache.set(index, dynamics)

                    _logger.info(
                        f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )

        except Exception as e:
            return False, index, e

        return True, index, None

    def _equilibrate(self, index):
        """
        Equilibrate the system.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        success: bool
            Whether the equilibration was successful.

        index: int
            The index of the replica.

        exception: Exception
            The exception if the equilibration failed.
        """
        _logger.info(f"Equilibrating at {_lam_sym} = {self._lambda_values[index]:.5f}")

        try:
            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(index)

            if gcmc_sampler is not None:
                # Push the PyCUDA context on top of the stack.
                gcmc_sampler.push()

                _logger.info(
                    f"Equilibrating with GCMC moves at {_lam_sym} = {self._lambda_values[index]:.5f}"
                )
                for i in range(100):
                    gcmc_sampler.move(dynamics.context())

                # Remove the PyCUDA context from the stack.
                gcmc_sampler.pop()

                # Store the current water state.
                water_state = gcmc_sampler.water_state()

            # Work out whether the constraints have changed from the initial minimisation.
            if self._config.minimise:
                constraint = self._config.constraint
                perturbable_constraint = self._config.perturbable_constraint

                if not self._config.equilibration_constraints:
                    constraint = "none"
                    perturbable_constraint = "none"

                constraints_changed = (self._initial_constraint != constraint) or (
                    self._initial_perturbable_constraint != perturbable_constraint
                )

                # We need to create a new dynamics object if the constraints have changed.
                if constraints_changed:
                    _logger.info(
                        f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )

                    # Commit the current system.
                    system = dynamics.commit()

                    # Delete the current dynamics object.
                    self._dynamics_cache.delete(index)

                    # Work out the device index.
                    device = index % self._num_gpus

                    # Copy the dynamics keyword arguments.
                    dynamics_kwargs = self._dynamics_kwargs.copy()

                    # Overload the device and lambda value.
                    dynamics_kwargs["device"] = device
                    dynamics_kwargs["lambda_value"] = self._lambda_values[index]
                    dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]
                    dynamics_kwargs["constraint"] = constraint
                    dynamics_kwargs["perturbable_constraint"] = perturbable_constraint

                    # Create the new dynamics object.
                    dynamics = system.dynamics(**dynamics_kwargs)

                    # Reset the GCMC water state.
                    if gcmc_sampler is not None:
                        self._reset_gcmc_sampler(gcmc_sampler, dynamics)

                    # Update the dynamics object in the cache.
                    self._dynamics_cache.set(index, dynamics)

            # Equilibrate.
            dynamics.run(
                self._config.equilibration_time,
                energy_frequency=0,
                frame_frequency=0,
                save_velocities=False,
                auto_fix_minimise=True,
                save_crash_report=self._config.save_crash_report,
            )

            # Commit the system.
            system = dynamics.commit()

            # Reset the timer.
            if self._initial_time[index].value() != 0:
                system.set_time(self._initial_time[index])
            else:
                system.set_time(_sr.u("0ps"))

            # Delete the dynamics object.
            self._dynamics_cache.delete(index)

            # Work out the device index.
            device = index % self._num_gpus

            # Copy the dynamics keyword arguments.
            dynamics_kwargs = self._dynamics_kwargs.copy()

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = self._lambda_values[index]
            dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]

            # Create the production dynamics object.
            dynamics = system.dynamics(**dynamics_kwargs)

            # Reset the GCMC water state. The dynamics object is created from
            # the original Sire system, so the water state in the context does
            # not match the current GCMC water state.
            if gcmc_sampler is not None:
                self._reset_gcmc_sampler(gcmc_sampler, dynamics)

            # Set the new dynamics object.
            self._dynamics_cache.set(index, dynamics)

            _logger.info(
                f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
            )

        except Exception as e:
            try:
                # Save the energy components for debugging purposes.
                self._save_energy_components(index, dynamics.context())
            except:
                pass
            return False, index, e

        return True, index, None

    def _compute_energies(self, index):
        """
        Compute the energies for a given replica by updating the OpenMM state
        within the context and re-evaluating the potential energy.

        Energies are currently computed internally by Sire at the end of each
        dynamics block, but this approach incurs an overhead due to the cost of
        updating the force field parameters within the context when changing
        lambda. This alternaitve method is left here for performance testing.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        index: int
            The index of the replica.

        energies: np.ndarray
            The energies of the replica and each state.
        """
        _logger.info(
            f"Computing energies for {_lam_sym} = {self._lambda_values[index]:.5f}"
        )

        # Get the dynamics object.
        dynamics, _ = self._dynamics_cache.get(index)

        # Create an array to hold the energies.
        energies = _np.zeros(self._config.num_lambda)

        # Loop over the states.
        for i in range(self._config.num_lambda):
            # Set the state.
            dynamics.context().setState(self._dynamics_cache._openmm_states[i])
            dynamics._d._clear_state()

            # Compute and store the energy for this state.
            energies[i] = dynamics.current_potential_energy().value()

        # Reset the state.
        dynamics.context().setState(self._dynamics_cache._openmm_states[index])

        return index, energies

    def _assemble_results(self, results):
        """
        Assemble the results into a matrix.

        Parameters
        ----------

        results: list
            The results from the repex dynamics block.
        """
        # Create the matrix.
        matrix = _np.zeros((len(results), len(results)))

        # Fill the matrix. The energy returned by the dynamics block already
        # includes the pressure and grand canonical contributions.
        for i, energies in results:
            for j, energy in enumerate(energies):
                matrix[i, j] = self._beta * energy

        return matrix

    def _checkpoint(self, index, lambdas, block, num_blocks, is_final_block=False):
        """
        Checkpoint the simulation.

        Parameters
        ----------

        index: int
            The index of the replica.

        lambdas: np.ndarray
            The lambda values for each replica.

        block: int
            The current block number.

        num_blocks: int
            The total number of blocks in the simulation.

        is_final_block: bool
            Whether this is the final block.

        Returns
        -------

        index: int
            The index of the replica.

        exception: Exception
            The exception if the checkpoint failed.
        """
        try:
            # Get the lambda value.
            lam = lambdas[index]

            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(index)

            # Commit the current system.
            system = dynamics.commit()

            # If performing GCMC, then we need to flag the ghost waters.
            if gcmc_sampler is not None:
                system = gcmc_sampler._flag_ghost_waters(system)

            # Get the simulation speed.
            speed = dynamics.time_speed()

            # Call the base class checkpoint method to save the system state.
            with self._lock:
                index, error = super()._checkpoint(
                    system, index, block, speed, is_final_block=is_final_block
                )

                if error is not None:
                    return index, error

            # Delete all trajectory frames from the Sire system within the
            # dynamics object.
            dynamics._d._sire_mols.delete_all_frames()

            _logger.info(
                f"Finished block {block+1} of {self._start_block + num_blocks} "
                f"for {_lam_sym} = {lam:.5f}"
            )

            # Log the number of waters within the GCMC sampling volume.
            if gcmc_sampler is not None:
                # Push the PyCUDA context on top of the stack.
                gcmc_sampler.push()

                _logger.info(
                    f"Current number of waters in GCMC volume at {_lam_sym} = {lam:.5f} "
                    f"is {gcmc_sampler.num_waters()}"
                )

                # Remove the PyCUDA context from the stack.
                gcmc_sampler.pop()

            if is_final_block:
                _logger.success(f"{_lam_sym} = {lam:.5f} complete")

            return index, None

        except Exception as e:
            return index, e

    @staticmethod
    @_njit
    def _mix_replicas(num_replicas, energy_matrix, proposed, accepted):
        """
        Mix the replicas.

        Parameters
        ----------

        num_replicas: int
            The number of replicas.

        energy_matrix: np.ndarray
            The energy matrix for the replicas.

        Returns
        -------

        states: np.ndarray
            The new states.
        """

        # Adapted from OpenMMTools: https://github.com/choderalab/openmmtools

        # Set the states to the initial order.
        states = _np.arange(num_replicas)

        # Attempt swaps.
        for swap in range(num_replicas**3):
            # Choose two replicas to swap.
            replica_i = _np.random.randint(num_replicas)
            replica_j = _np.random.randint(num_replicas)

            # Get the current state.
            state_i = states[replica_i]
            state_j = states[replica_j]

            # Record that we have proposed a swap.
            proposed[state_i, state_j] += 1
            proposed[state_j, state_i] += 1

            # Get the energies.
            energy_ii = energy_matrix[replica_i, state_i]
            energy_jj = energy_matrix[replica_j, state_j]
            energy_ij = energy_matrix[replica_i, state_j]
            energy_ji = energy_matrix[replica_j, state_i]

            # Compute the log probability of the swap.
            log_p_swap = -(energy_ij + energy_ji) + energy_ii + energy_jj

            # Accept the swap and update the states.
            if log_p_swap >= 0 or _np.random.rand() < _np.exp(log_p_swap):
                # Swap the states.
                states[replica_i] = state_j
                states[replica_j] = state_i
                # Record the swap.
                accepted[state_i, state_j] += 1
                accepted[state_j, state_i] += 1

        return states

    def _save_transition_matrix(self):
        """
        Internal method to save the replica exchange transition matrix.
        """
        # Create the transition matrix estimate. Adapted from OpenMMTools:
        #   https://github.com/choderlab/openmmtools
        t_ij = _np.zeros((self._config.num_lambda, self._config.num_lambda))
        for i_state in range(self._config.num_lambda):
            swaps = self._dynamics_cache.get_swaps()
            denom = float((swaps[i_state, :].sum() + swaps[:, i_state].sum()))
            if denom > 0:
                for j_state in range(self._config.num_lambda):
                    t_ij[i_state, j_state] = (
                        swaps[i_state, j_state] + swaps[j_state, i_state]
                    ) / denom
            else:
                t[i_state, i_state] = 1.0

        # Backup the existing transition matrix, if it exists.
        if self._repex_matrix.exists():
            _copyfile(
                self._repex_matrix,
                self._repex_matrix.with_suffix(".txt.bak"),
            )

        # Save the replica exchange swap acceptance matrix.
        _np.savetxt(
            self._repex_matrix,
            t_ij,
            fmt="%.5f",
        )

    @staticmethod
    def _reset_gcmc_sampler(gcmc_sampler, dynamics):
        """
        Reset the GCMC sampler.

        Parameters
        ----------

        gcmc_sampler: sire.gcmc.GCMCSampler
            The GCMC sampler to reset.

        dynamics: sire.mol.Dynamics
            The dynamics object associated with the GCMC sampler.
        """
        # Reset the GCMC sampler. This resets the sampling statistics and
        # clears the associated OpenMM forces.
        gcmc_sampler.reset()

        # Push the PyCUDA context on top of the stack.
        gcmc_sampler.push()

        # Set the water state.
        gcmc_sampler._set_water_state(dynamics.context(), force=True)

        # Remove the PyCUDA context from the stack.
        gcmc_sampler.pop()

        # Re-bind the GCMC sampler to the dynamics object.
        gcmc_sampler.bind_dynamics(dynamics)
