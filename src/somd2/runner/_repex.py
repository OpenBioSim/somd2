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

__all__ = ["RepexRunner"]

from numba import njit as _njit

import numpy as _np
import pickle as _pickle

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
        self._openmm_volumes = [None] * len(lambdas)
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
            "_openmm_volumes": self._openmm_volumes,
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
        """

        # Copy the dynamics keyword arguments.
        dynamics_kwargs = dynamics_kwargs.copy()

        # Copy the GCMC keyword arguments.
        if gcmc_kwargs is not None:
            gcmc_kwargs = gcmc_kwargs.copy()

        # Initialise the dynamics object list.
        self._dynamics = []

        # Initialise the GCMC object list.
        self._gcmc = []

        # Create the dynamics objects in serial.
        for i, (lam, scale) in enumerate(zip(lambdas, rest2_scale_factors)):
            # Work out the device index.
            device = i % num_gpus

            # This is a restart, get the system for this replica.
            if isinstance(system, list):
                mols = system[i]
            # This is a new simulation.
            else:
                mols = system

            if gcmc_kwargs is not None:
                from loch import GCMCSampler

                ghost_file = str(output_directory / f"gcmc_{lam:.5f}.ghost")

                # Create the GCMC sampler.
                gcmc_sampler = GCMCSampler(
                    mols,
                    device=device,
                    lambda_value=lam,
                    ghost_file=ghost_file,
                    **gcmc_kwargs,
                )

                # Get the modified GCMC system.
                mols = gcmc_sampler.system()

                # Store the GCMC sampler.
                self._gcmc.append(gcmc_sampler)

                _logger.info(
                    f"Created GCMC sampler for lambda {lam:.5f} on device {device}"
                )

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = lam
            dynamics_kwargs["rest2_scale"] = scale

            # Create the dynamics object.
            try:
                dynamics = mols.dynamics(**dynamics_kwargs)
            except Exception as e:
                _logger.error(
                    f"Could not create dynamics object for lambda {lam:.5f}: {e}"
                )

            # Append the dynamics object.
            self._dynamics.append(dynamics)

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
        try:
            gcmc_sampler = self._gcmc[index]
        except:
            gcmc_sampler = None

        return self._dynamics[index], gcmc_sampler

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
            ._d.context()
            .getState(getPositions=True, getVelocities=True)
        )

        # Store the state.
        self._openmm_states[index] = state

        # Store the volume.
        self._openmm_volumes[index] = state.getPeriodicBoxVolume().value_in_unit(
            angstrom**3
        )

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
                self._dynamics[i]._d._omm_mols.setState(self._openmm_states[state])

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
            The perturbable system to be simulated. This can be either a path
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

        # Make sure we're using the CUDA platform.
        if self._config.platform != "cuda":
            msg = "Currently replica exchange simulations can only be run on the CUDA platform."
            _logger.error(msg)
            raise ValueError(msg)

        # Get the number of available GPUs.
        try:
            gpu_devices = self._get_gpu_devices("cuda")
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

        # Create the dynamics cache.
        if not self._is_restart:
            dynamics_kwargs = self._dynamics_kwargs.copy()

            if self._config.equilibration_time.value() > 0.0:
                self._is_equilibration = True

                # Overload the dynamics kwargs with the equilibration options.
                dynamics_kwargs.update(
                    {
                        "timestep": self._config.equilibration_timestep,
                        "constraint": (
                            "none"
                            if not self._config.equilibration_constraints
                            else self._config.constraint
                        ),
                        "perturbable_constraint": (
                            "none"
                            if not self._config.equilibration_constraints
                            else self._config.perturbable_constraint
                        ),
                    }
                )

            self._dynamics_cache = DynamicsCache(
                self._system,
                self._lambda_values,
                self._rest2_scale_factors,
                self._num_gpus,
                dynamics_kwargs,
                self._gcmc_kwargs,
                output_directory=self._config.output_directory,
            )
        else:
            # Check to see if the simulation is already complete.
            time = self._system[0].time()
            if time > self._config.runtime - self._config.timestep:
                _logger.success(f"Simulation already complete. Exiting.")
                exit(0)

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
                gcmc_kwargs=self._config.gcmc_kwargs,
                output_directory=self._config.output_directory,
            )

        # Conversion factor for reduced potential.
        kT = (_sr.units.k_boltz * self._config.temperature).to(_sr.units.kcal_per_mol)
        self._beta = 1.0 / kT

        # Store the pressure times Avaogadro's number.
        if self._config.pressure is not None:
            NA = 6.02214076e23 / _sr.units.mole
            self._pressure = (self._config.pressure * NA).value()
        else:
            self._pressure = None

        # If restarting, subtract the time already run from the total runtime
        if self._config.restart:
            time = self._system[0].time()
            self._config.runtime = str(self._config.runtime - time)

            # Work out the current block number.
            self._start_block = int(
                round(time.value() / self._config.checkpoint_frequency.value(), 12)
            )
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
        cycles = ceil(self._config.runtime / self._config.energy_frequency)

        if self._config.checkpoint_frequency.value() > 0.0:
            # Calculate the number of blocks and the remainder time.
            frac = (
                self._config.runtime.value() / self._config.checkpoint_frequency.value()
            )

            # Handle the case where the runtime is less than the checkpoint frequency.
            if frac < 1.0:
                frac = 1.0
                self._config.checkpoint_frequency = str(self._config.runtime)

            num_blocks = int(frac)
            rem = frac - num_blocks

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

        # Work out the required number of batches.
        num_batches = ceil(self._config.num_lambda / num_workers)

        # Create the replica list.
        replica_list = list(range(self._config.num_lambda))

        # Minimise at each lambda value.
        if self._config.minimise and not self._is_restart:
            for i in range(num_batches):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for success, index, e in executor.map(
                            self._minimise,
                            replica_list[i * num_workers : (i + 1) * num_workers],
                        ):
                            if not success:
                                _logger.error(
                                    f"Minimisation failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                                )
                                raise e
                    except KeyboardInterrupt:
                        _logger.error("Minimisation cancelled. Exiting.")
                        exit(1)

        # Equilibrate the system.
        if self._is_equilibration:
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
                        exit(1)

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

            # Whether a frame was saved after the previous block.
            write_gcmc_ghosts = i > 0 and (i - 1) % cycles_per_frame == 0

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
                            repeat(is_checkpoint),
                            repeat(i == 0),
                            repeat(i == cycles - 1),
                            repeat(block),
                            repeat(num_blocks + int(rem > 0)),
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
                        exit(1)

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

                    # Save the transition matrix.
                    _logger.info("Saving replica exchange transition matrix")
                    self._save_transition_matrix()

                    # Pickle the dynamics cache.
                    _logger.info("Saving replica exchange state")
                    with open(self._repex_state, "wb") as f:
                        _pickle.dump(self._dynamics_cache, f)

        # Record the end time for the production block.
        prod_end = time()

        # Save the final transition matrix.
        _logger.info("Saving final replica exchange transition matrix")
        self._save_transition_matrix()

        # Pickle final state of the dynamics cache.
        _logger.info("Saving final replica exchange state")
        with open(self._repex_state, "wb") as f:
            _pickle.dump(self._dynamics_cache, f)

        # Save the final GCMC ghost indices.
        if self._config.gcmc and i % cycles_per_frame == 0:
            for gcmc in self._dynamics_cache._gcmc:
                gcmc.write_ghost_residues()

        # Record the end time.
        end = time()

        # Work how many fractional days the production block took.
        prod_time = (prod_end - prod_start) / 86400

        # Record the average production speed. (ns/day per replica)
        prod_speed = self._config.runtime.to("ns") / prod_time

        # Record the average production speed.
        _logger.info(f"Average replica speed: {prod_speed:.2f} ns day-1")

        # Log the run time in minutes.
        _logger.success(
            f"Simulation finished. Run time: {(end - start) / 60:.2f} minutes"
        )

    def _run_block(
        self,
        index,
        lambdas,
        is_checkpoint,
        is_first_block,
        is_final_block,
        block,
        num_blocks,
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

        rest2_scale: np.ndarray
            The REST2 scaling factor for each replica.

        is_checkpoint: bool
            Whether to checkpoint.

        is_first_block: bool
            Whether this is the first block.

        is_final_block: bool
            Whether this is the final block.

        block: int
            The block number.

        num_blocks: int
            The total number of blocks.

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

            # Minimise the system if this is a restart simulation and this is
            # the first block.
            if is_first_block and self._is_restart:
                if self._config.minimise:
                    _logger.info(f"Minimising restart at {_lam_sym} = {lam:.5f}")
                    dynamics.minimise(timeout=self._config.timeout)

            _logger.info(f"Running dynamics for {_lam_sym} = {lam:.5f}")

            # Draw new velocities from the Maxwell-Boltzmann distribution.
            dynamics.randomise_velocities()

            # Perform a GCMC move. For repex this needs to be done before the
            # dynamics block so that the final energies, which are used in the
            # repex acceptance criteria, are correct.
            if gcmc_sampler is not None:
                # Push the PyCUDA context on top of the stack.
                gcmc_sampler.push()

                # The frame frequency was hit after the previous block, so we
                # need to write the current indices of the GCMC ghost residues
                # to file.
                if write_gcmc_ghosts:
                    gcmc_sampler.write_ghost_residues()

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
            )

            # Set the state.
            self._dynamics_cache.save_openmm_state(index)

            # Get the energies at each lambda value.
            energies = (
                dynamics._d.energy_trajectory()
                .to_pandas(to_alchemlyb=True, energy_unit="kcal/mol")
                .iloc[-1, :]
                .to_numpy()
            )

            # Checkpoint.
            if is_checkpoint or is_final_block:
                # Commit the current system.
                system = dynamics.commit()

                # Get the simulation speed.
                speed = dynamics.time_speed()

                # Checkpoint.
                with self._lock:
                    self._checkpoint(
                        system, index, block, speed, is_final_block=is_final_block
                    )

                # Delete all trajectory frames from the Sire system within the
                # dynamics object.
                dynamics._d._sire_mols.delete_all_frames()

                _logger.info(
                    f"Finished block {block+1} of {self._start_block + num_blocks} "
                    f"for {_lam_sym} = {lam:.5f}"
                )

                if is_final_block:
                    _logger.success(f"{_lam_sym} = {lam:.5f} complete")

        except Exception as e:
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

            # Equilibrate.
            dynamics.run(
                self._config.equilibration_time,
                energy_frequency=0,
                frame_frequency=0,
                save_velocities=False,
                auto_fix_minimise=True,
            )

            # Commit the system.
            system = dynamics.commit()

            # Reset the timer to zero.
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

            # Perform minimisation at the end of equilibration only if the
            # timestep is increasing, or the constraint is changing.
            if (self._config.timestep > self._config.equilibration_timestep) or (
                not self._config.equilibration_constraints
                and self._config.perturbable_constraint != "none"
            ):
                _logger.info(
                    f"Minimising at {_lam_sym} = {self._lambda_values[index]:.5f}"
                )
                dynamics.minimise(timeout=self._config.timeout)

            # Set the new dynamics object.
            self._dynamics_cache.set(index, dynamics)

            _logger.info(
                f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
            )

        except Exception as e:
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
            dynamics._d.context().setState(self._dynamics_cache._openmm_states[i])
            dynamics._d._clear_state()

            # Compute and store the energy for this state.
            energies[i] = dynamics.current_potential_energy().value()

        # Reset the state.
        dynamics._d.context().setState(self._dynamics_cache._openmm_states[index])

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

        # Fill the matrix.
        for i, energies in results:
            for j, energy in enumerate(energies):
                matrix[i, j] = self._beta * energy
                if self._pressure is not None:
                    matrix[i, j] += (
                        self._beta
                        * self._config.pressure
                        * self._dynamics_cache._openmm_volumes[j]
                    )

        return matrix

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

        # Save the replica exchange swap acceptance matrix.
        _np.savetxt(
            self._repex_matrix,
            t_ij,
            fmt="%.5f",
        )
