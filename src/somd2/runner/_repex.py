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

__all__ = ["RepexRunner"]

from numba import njit as _njit

import numpy as _np

import sire as _sr

from somd2 import _logger

from .._utils import _lam_sym

from ._base import RunnerBase as _RunnerBase


class DynamicsCache:
    """
    A class for caching dynamics objects.
    """

    def __init__(self, system, lambdas, num_gpus, dynamics_kwargs):
        """
        Constructor.

        Parameters
        ----------

        system: :class: `System <sire.system.System>`, List[:class: `System <sire.system.System>`]
            The perturbable system, or systems, to be simulated.

        lambdas: np.ndarray
            The lambda value for each replica.

        num_gpus: int
            The number of GPUs to use.

        dynamics_kwargs: dict
            A dictionary of default dynamics keyword arguments.
        """

        # Warn if the number of replicas is not a multiple of the number of GPUs.
        if len(lambdas) > num_gpus and len(lambdas) % num_gpus != 0:
            _logger.warning(
                "The number of replicas is not a multiple of the number of GPUs. "
                "This may result in suboptimal performance."
            )

        # Initialise attributes.
        self._dynamics = []
        self._lambdas = lambdas
        self._states = _np.array(range(len(lambdas)))
        self._openmm_states = [None] * len(lambdas)
        self._openmm_volumes = [None] * len(lambdas)

        # Copy the dynamics keyword arguments.
        dynamics_kwargs = dynamics_kwargs.copy()

        # Create the dynamics objects in serial.
        for i, lam in enumerate(lambdas):
            # Work out the device index.
            device = i % num_gpus

            # This is a restart, get the system for this replica.
            if isinstance(system, list):
                mols = system[i]
            # This is a new simulation.
            else:
                mols = system

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = lam

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
        Get the dynamics object for a given index.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        tuple
            The dynamics object for the replica.
        """
        return self._dynamics[index]

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
        state = self._dynamics[index]._d._omm_mols.getState(
            getPositions=True, getVelocities=True
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

        if config.restart:
            raise ValueError(
                "'restart' is not currently supported for replica exchange."
            )

        if config.lambda_energy is not None:
            raise ValueError(
                "'lambda_energy' is not currently supported for replica exchange."
            )

        # Call the base class constructor.
        super().__init__(system, config)

        # Get the number of available GPUs.
        gpu_devices = self._get_gpu_devices("cuda")

        # We can only use replica exchange if we have a GPU.
        if len(gpu_devices) == 0:
            _logger.error("No GPUs available. Cannot run replica exchange.")

        # Set the number of GPUs.
        if self._config.max_gpus is None:
            self._num_gpus = len(gpu_devices)
        else:
            self._num_gpus = min(self._config.max_gpus, len(gpu_devices))

        # Create the dynamics cache.
        self._dynamics_cache = DynamicsCache(
            self._system,
            self._lambda_values,
            self._num_gpus,
            self._dynamics_kwargs,
        )

        # Conversion factor for reduced potential.
        kT = (_sr.units.k_boltz * self._config.temperature).to(_sr.units.kcal_per_mol)
        self._beta = 1.0 / kT

        # Store the pressure times Avaogadro's number.
        NA = 6.02214076e23 / _sr.units.mole
        self._pressure = (self._config.pressure * NA).value()

        # If restarting, subtract the time already run from the total runtime
        if self._config.restart:
            self._config.runtime = str(self._config.runtime - self._system.time())

            # Work out the current block number.
            self._start_block = int(
                round(
                    self._system.time().value()
                    / self._config.checkpoint_frequency.value(),
                    12,
                )
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

        # Work out the required number of batches.
        num_batches = ceil(
            self._config.num_lambda
            / (self._num_gpus * self._config.oversubscription_factor)
        )

        # Create the replica list.
        replica_list = list(range(self._config.num_lambda))

        # Minimise at each lambda value. This is currently done in serial due to a
        # limitation in OpenMM.
        if self._config.minimise:
            for i in range(self._config.num_lambda):
                self._minimise(i)

        # Current block number.
        block = 0

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

            # Run a dynamics block for each replica, making sure only each GPU is only
            # oversubscribed by a factor of self._config.oversubscription_factor.
            for j in range(num_batches):
                replicas = replica_list[
                    j
                    * self._num_gpus
                    * self._config.oversubscription_factor : (j + 1)
                    * self._num_gpus
                    * self._config.oversubscription_factor
                ]
                with ThreadPoolExecutor() as executor:
                    try:
                        for result, index, energies in executor.map(
                            self._run_block,
                            replicas,
                            repeat(self._lambda_values),
                            repeat(is_checkpoint),
                            repeat(i == cycles - 1),
                            repeat(block),
                            repeat(num_blocks + int(rem > 0)),
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

            if i < cycles - 1:
                # Assemble and energy matrix from the results.
                _logger.info("Assembling energy matrix")
                energy_matrix = self._assemble_results(results)

                # Mix the replicas.
                _logger.info("Mixing replicas")
                self._dynamics_cache.set_states(
                    self._mix_replicas(
                        self._config.num_lambda,
                        energy_matrix,
                    )
                )
                self._dynamics_cache.mix_states()

                # Update the block number.
                if is_checkpoint:
                    block += 1

        # Record the end time.
        end = time()

        # Log the run time in minutes.
        _logger.success(
            f"Simulation finished. Run time: {(end - start) / 60:.2f} minutes"
        )

    def _run_block(
        self, index, lambdas, is_checkpoint, is_final_block, block, num_blocks
    ):
        """
        Run a dynamics block for a given replica.

        Parameters
        ----------

        index: int
            The index of the replica.

        lambdas: np.ndarray
            The lambda values for each replica.

        is_checkpoint: bool
            Whether to checkpoint.

        is_final_block: bool
            Whether this is the final block.

        block: int
            The block number.

        num_blocks: int
            The total number of blocks.

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

        _logger.info(f"Running dynamics for {_lam_sym} = {lam:.5f}")

        try:
            # Get the dynamics object.
            dynamics = self._dynamics_cache.get(index)

            # Run the dynamics.
            dynamics.run(
                self._config.energy_frequency,
                energy_frequency=self._config.energy_frequency,
                frame_frequency=self._config.frame_frequency,
                lambda_windows=lambdas,
                save_velocities=self._config.save_velocities,
                auto_fix_minimise=True,
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

                _logger.info(
                    f"Finished block {block+1} of {self._start_block + num_blocks} "
                    f"for {_lam_sym} = {lam:.5f}"
                )

                if is_final_block:
                    _logger.success(
                        f"{_lam_sym} = {lam:.5f} complete, speed = {speed:.2f} ns day-1"
                    )

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
            # Get the dynamics object.
            dynamics = self._dynamics_cache.get(index)

            # Minimise the system.
            dynamics.minimise(timeout=self._config.timeout)

        except Exception as e:
            return False, index, e

        return True, index, None

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
                matrix[i, j] = self._beta * (
                    energy + self._pressure * self._dynamics_cache._openmm_volumes[i]
                )

        return matrix

    @staticmethod
    @_njit
    def _mix_replicas(num_replicas, energy_matrix):
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

            # Get the energies.
            energy_ii = energy_matrix[replica_i, state_i]
            energy_jj = energy_matrix[replica_j, state_j]
            energy_ij = energy_matrix[replica_i, state_j]
            energy_ji = energy_matrix[replica_j, state_i]

            # Compute the log probability of the swap.
            log_p_swap = -(energy_ij + energy_ji) + energy_ii + energy_jj

            # Accept the swap and update the states.
            if log_p_swap >= 0 or _np.random.rand() < _np.exp(log_p_swap):
                states[replica_i] = state_j
                states[replica_j] = state_i

        return states
