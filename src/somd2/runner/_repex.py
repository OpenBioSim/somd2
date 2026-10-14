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


@_njit
def _mix_subset(replicas, energy_matrix, proposed, accepted):
    """
    Mix a subset of the replicas, leaving the others in place.

    Swaps are only proposed between the replicas in the subset, which is a
    valid Gibbs move for any subset, since the others are held fixed.

    Parameters
    ----------

    replicas: np.ndarray
        The indices of the replicas to mix.

    energy_matrix: np.ndarray
        The reduced energy of each replica at each state. Only the rows of the
        replicas being mixed are used.

    proposed: np.ndarray
        The number of proposed swaps between states, updated in place.

    accepted: np.ndarray
        The number of accepted swaps between states, updated in place.

    Returns
    -------

    states: np.ndarray
        The new states for every replica, where states[i] is the replica whose
        configuration seeds replica i.
    """

    # Adapted from OpenMMTools: https://github.com/choderalab/openmmtools

    num_mixed = len(replicas)

    # Set the states to the initial order.
    states = _np.arange(energy_matrix.shape[0])

    # Attempt swaps.
    for swap in range(num_mixed**3):
        # Choose two replicas to swap.
        replica_i = replicas[_np.random.randint(num_mixed)]
        replica_j = replicas[_np.random.randint(num_mixed)]

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

    # Here states[i] is the state that replica i's configuration moves to,
    # whereas the configurations are moved into fixed states, so invert.
    return _np.argsort(states)


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
        perturbed_system=None,
        xml_filenames=None,
        num_slots=None,
        update_constraints=True,
        constraint_lambda_index=None,
        gpu_devices=None,
        strided=False,
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

        perturbed_system: :class: `System <sire.system.System>`
            The perturbed end-state system used to seed starting coordinates for
            lambda > 0.5 replicas. If None, the perturbed state is not used.

        xml_filenames: list of str
            A list of file paths for the OpenMM XML output, one per replica.
            If None, XML files are not written.

        num_slots: int
            The number of dynamics objects (slots) to create. If None, then one
            is created per replica. If fewer, then each slot is re-used to
            propagate several replicas per cycle, changing its lambda value as
            it goes.

        update_constraints: bool
            Whether to update the constraints when changing the lambda value of
            a slot.

        constraint_lambda_index: int
            The index of the lambda value to create every context at, so that
            constrained bond lengths are the same for every replica. If None,
            each context is created at the lambda value of the first replica it
            hosts, which is only consistent between replicas when there is a
            context each. Only meaningful when 'update_constraints' is False,
            since the constraints are otherwise updated whenever lambda
            changes.

        gpu_devices: list
            The physical devices backing each OpenMM device index, i.e. the
            entries of CUDA_VISIBLE_DEVICES. Used to query the memory of the
            right device. If None, the OpenMM index is used directly.

        strided: bool
            Whether to assign replicas to slots in a strided rather than a
            contiguous pattern, so that any contiguous range of replicas is
            spread evenly over the slots.
        """

        num_replicas = len(lambdas)

        if num_slots is None:
            num_slots = num_replicas

        # Initialise attributes.
        self._lambdas = lambdas
        self._rest2_scale_factors = rest2_scale_factors
        self._num_replicas = num_replicas
        self._num_slots = num_slots
        self._update_constraints = update_constraints
        self._constraint_lambda_index = constraint_lambda_index
        self._gpu_devices = gpu_devices
        self._strided = strided
        self._states = _np.array(range(num_replicas))
        self._time = None
        # The simulation clock of each replica.
        self._clocks = [None] * num_replicas
        # The hot replicas of adaptive replica exchange, or None for all.
        self._hot_replicas = None
        self._cold_votes = [0] * num_replicas
        self._adaptive_data = None
        self._openmm_states = [None] * num_replicas
        # The molecules with virtual sites, so that each molecule's particles
        # can be found in the saved states. Set by the runner.
        self._virtual_sites = None
        self._gcmc_states = [None] * num_replicas
        # GCMC statistics for the whole simulation, keyed by lambda value.
        self._gcmc_stats = None
        self._energy_trajectories = [None] * num_replicas
        self._ghost_files = [None] * num_replicas
        # Waters in the GCMC volume, recorded per replica while it is resident
        # in its slot. Derived, so not stored in a checkpoint.
        self._gcmc_num_waters = [None] * num_replicas
        # Whether the last mix moved a replica's state, and so whether it must
        # be pushed into the context before the next block.
        self._state_moved = [False] * num_replicas
        self._terminal_flip_stats = [[0, 0] for _ in range(num_replicas)]
        self._num_proposed = _np.matrix(_np.zeros((num_replicas, num_replicas)))
        self._num_accepted = _np.matrix(_np.zeros((num_replicas, num_replicas)))
        self._num_swaps = _np.matrix(_np.zeros((num_replicas, num_replicas)))

        # Build the slot layout and the per-slot attributes.
        self._build_slot_layout()

        # Create the dynamics objects.
        self._create_dynamics(
            system,
            lambdas,
            rest2_scale_factors,
            num_gpus,
            dynamics_kwargs,
            gcmc_kwargs=gcmc_kwargs,
            output_directory=output_directory,
            perturbed_system=perturbed_system,
            xml_filenames=xml_filenames,
        )

    def _build_slot_layout(self):
        """
        Assign replicas to slots.

        Each slot is given a contiguous group of replicas, so that it only ever
        moves between neighbouring lambda values. That keeps the change in the
        force field parameters, and hence the chance of a constraint update
        forcing the context to be reinitialised, as small as possible.

        When strided, replica r is hosted by slot r % num_slots instead, so
        that the hot replicas of adaptive replica exchange, which form
        contiguous ranges, are spread evenly over the slots and GPUs.

        The layout is derived from the number of replicas and slots, so it is
        rebuilt rather than stored in a checkpoint.
        """
        self._gcmc_samplers = [None] * self._num_slots
        self._slot_replica = [None] * self._num_slots

        if self._strided:
            self._groups = [
                list(range(slot, self._num_replicas, self._num_slots))
                for slot in range(self._num_slots)
            ]
        else:
            self._groups = [
                [int(r) for r in group]
                for group in _np.array_split(
                    _np.arange(self._num_replicas), self._num_slots
                )
            ]

        # The slot that hosts each replica.
        self._replica_slot = [None] * self._num_replicas
        for slot, group in enumerate(self._groups):
            for replica in group:
                self._replica_slot[replica] = slot

    def __setstate__(self, state):
        """
        Set the state of the object.
        """

        # Checkpoints written before slots were introduced stored the states
        # unpermuted, with self._states holding the mapping to apply on
        # restart. They are detected by the absence of "_num_slots".
        is_legacy = "_num_slots" not in state

        for key, value in state.items():
            setattr(self, key, value)

        # Provide defaults for attributes added after the initial release,
        # so that old checkpoint files can still be loaded.
        n = len(self._lambdas)
        if not hasattr(self, "_gcmc_stats"):
            self._gcmc_stats = None
        if not hasattr(self, "_gcmc_states"):
            self._gcmc_states = [None] * n
        if not hasattr(self, "_terminal_flip_stats"):
            self._terminal_flip_stats = [[0, 0] for _ in range(n)]
        if not hasattr(self, "_time"):
            self._time = None
        if not hasattr(self, "_num_replicas"):
            self._num_replicas = n
        if not hasattr(self, "_energy_trajectories"):
            self._energy_trajectories = [None] * n
        if not hasattr(self, "_ghost_files"):
            self._ghost_files = [None] * n
        # Without these, replicas start from the shared clock with all hot.
        if not hasattr(self, "_clocks"):
            self._clocks = [None] * n
        if not hasattr(self, "_hot_replicas"):
            self._hot_replicas = None
        if not hasattr(self, "_cold_votes"):
            self._cold_votes = [0] * n
        if not hasattr(self, "_adaptive_data"):
            self._adaptive_data = None
        self._gcmc_num_waters = [None] * n

        # The slot layout is not pickled, since it is rebuilt by
        # _create_dynamics() when the run is restarted. Older checkpoints
        # predate slots entirely, in which case there was one per replica.
        if not hasattr(self, "_num_slots"):
            self._num_slots = n
        if not hasattr(self, "_update_constraints"):
            self._update_constraints = True
        if not hasattr(self, "_virtual_sites"):
            self._virtual_sites = None
        # Set by the runner on a restart, since it follows the configuration.
        if not hasattr(self, "_strided"):
            self._strided = False

        # Convert a legacy checkpoint to the current convention, in which the
        # stored state of a replica is its own, with the last mix already
        # applied. Legacy states hold the destination of each replica's
        # configuration, so are inverted.
        if is_legacy:
            sources = _np.argsort(self._states)
            self._openmm_states = [self._openmm_states[s] for s in sources]
            self._gcmc_states = [self._gcmc_states[s] for s in sources]

        # Every replica is seeded from its stored state on a restart, since the
        # contexts are created from the input system rather than the checkpoint.
        self._state_moved = [True] * n

        # Rebuild the slot layout, which is derived rather than stored.
        self._build_slot_layout()

        # Checkpoints written before a sampler could be re-used across lambda
        # values stored the GCMC statistics as a list of counters per replica.
        # Convert these to a single dictionary keyed by lambda value.
        if isinstance(self._gcmc_stats, list):
            converted = {}
            for lam, stats in zip(self._lambdas, self._gcmc_stats):
                if _RunnerBase._is_legacy_gcmc_stats(stats):
                    converted.update(_RunnerBase._convert_legacy_gcmc_stats(stats, lam))
            self._gcmc_stats = converted if converted else None

    def __getstate__(self):
        """
        Get the state of the object.
        """

        # Create the state dict.
        d = {
            "_lambdas": self._lambdas,
            "_rest2_scale_factors": self._rest2_scale_factors,
            "_num_replicas": self._num_replicas,
            "_num_slots": self._num_slots,
            "_update_constraints": self._update_constraints,
            "_states": self._states,
            "_time": self._time,
            "_clocks": self._clocks,
            "_hot_replicas": self._hot_replicas,
            "_cold_votes": self._cold_votes,
            "_adaptive_data": self._adaptive_data,
            "_openmm_states": self._openmm_states,
            "_virtual_sites": self._virtual_sites,
            # Don't pickle the GCMC samplers since they need to be recreated.
            "_gcmc_samplers": len(self._gcmc_samplers) * [None],
            "_gcmc_states": self._gcmc_states,
            "_gcmc_stats": self._gcmc_stats,
            "_terminal_flip_stats": self._terminal_flip_stats,
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
        perturbed_system=None,
        xml_filenames=None,
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

        perturbed_system: :class: `System <sire.system.System>`
            The perturbed end-state system used to seed starting coordinates for
            lambda > 0.5 replicas. If None, the perturbed state is not used.

        xml_filenames: list of str
            A list of file paths for the OpenMM XML output, one per replica.
            If None, XML files are not written.
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

        # Per-device memory tracking for estimation.
        device_mem = {}

        # The device of each slot. Slots are assigned round-robin.
        self._slot_devices = [i % num_gpus for i in range(self._num_slots)]
        contexts_per_device = [self.contexts_on(d) for d in range(num_gpus)]

        # Record the ghost file for each replica. A slot writes to the file of
        # whichever replica it currently hosts.
        if gcmc_kwargs is not None:
            self._ghost_files = [
                str(output_directory / f"gcmc_ghosts_{lam:.5f}.txt") for lam in lambdas
            ]

        # Create the dynamics objects in serial. Each slot is created at the
        # lambda value of the first replica that it hosts.
        for i in range(self._num_slots):
            # The replica that seeds this slot.
            seed = self._groups[i][0]

            # The replica in the middle of the group, used to choose which end
            # state the starting coordinates come from. A slot's context is
            # created from a single system and every replica it hosts starts
            # from that context, so taking the middle rather than the first
            # keeps any mismatch to at most half a group, next to the lambda
            # value at which the end state switches.
            middle = self._groups[i][len(self._groups[i]) // 2]

            lam = lambdas[seed]
            scale = rest2_scale_factors[seed]

            # Create the context at a common lambda value, so that constrained
            # bond lengths are the same for every replica. Only set when the
            # constraints aren't updated as the slot changes lambda.
            if self._constraint_lambda_index is None:
                build_lam = lam
            else:
                build_lam = lambdas[self._constraint_lambda_index]

            # Work out the device index.
            device = self._slot_devices[i]

            # Record baseline memory before the first slot on this device.
            if device not in device_mem:
                used_before, _, total_mem = self._check_device_memory(
                    self._physical_device(device)
                )
                device_mem[device] = {
                    "before": used_before,
                    "total": total_mem,
                    "count": 0,
                }

            # Whether to seed from the perturbed end state. Swapping the end
            # states reverses the lambda schedule, so the perturbed end state
            # is then at lambda = 0 rather than lambda = 1.
            seed_perturbed = (lambdas[middle] > 0.5) != dynamics_kwargs[
                "swap_end_states"
            ]

            # This is a restart, get the system for the seeding replica.
            if isinstance(system, list):
                mols = system[seed]
            # This is a new simulation. Use the perturbed system to seed the
            # starting coordinates and periodic space.
            elif perturbed_system is not None and seed_perturbed:
                mols = perturbed_system
            else:
                mols = system

            # Delete an existing trajectory frames.
            mols.delete_all_frames()

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = build_lam
            dynamics_kwargs["rest2_scale"] = scale

            if gcmc_kwargs is not None:
                try:
                    from loch import GCMCSampler
                except:
                    msg = "loch is not installed. GCMC sampling cannot be performed."
                    _logger.error(msg)

                # Create the GCMC sampler, telling it every lambda value that
                # this slot will host so that switching between them doesn't
                # need to build an OpenMM context.
                gcmc_sampler = GCMCSampler(
                    mols,
                    device=device,
                    lambda_value=lam,
                    rest2_scale=scale,
                    lambda_values=[lambdas[r] for r in self._groups[i]],
                    rest2_scales=[rest2_scale_factors[r] for r in self._groups[i]],
                    ghost_file=self._ghost_files[seed],
                    **gcmc_kwargs,
                )

                # Get the modified GCMC system.
                mols = gcmc_sampler.system()

                # Store the GCMC sampler.
                self._gcmc_samplers[i] = gcmc_sampler

                _logger.info(
                    f"Created GCMC sampler for lambda {lam:.5f} on device {device}"
                )

            # Create the dynamics object.
            try:
                dynamics = mols.dynamics(**dynamics_kwargs)
            except Exception as e:
                # Contexts are appended as they are created, so this is the
                # number that fit before the failure.
                num_created = len(self._dynamics)
                msg = (
                    f"Could not create dynamics object for lambda {lam:.5f} on "
                    f"device {device} after creating {num_created} of "
                    f"{self._num_slots} contexts: {e}"
                )
                # Failing partway through points at exhausted resources.
                # Failing on the first context does not, so don't advise on
                # the number of contexts in that case.
                if num_created > 0:
                    msg += f". {self._max_contexts_advice(num_created)}"
                _logger.error(msg)
                raise RuntimeError(msg) from e

            # Bind the GCMC sampler to the dynamics object. This allows the
            # dynamics object to reset the water state in its internal OpenMM
            # context following a crash recovery.
            if gcmc_kwargs is not None:
                gcmc_sampler.bind_dynamics(dynamics)

            # Append the dynamics object.
            self._dynamics.append(dynamics)

            # Write the OpenMM XML file to the output directory. This is
            # indexed by replica, so use the replica that seeded the slot.
            if xml_filenames is not None:
                _logger.info(
                    f"Writing OpenMM XML for lambda {lam:.5f} on device {device}"
                )
                dynamics.to_xml(xml_filenames[seed])

            # Track memory footprint for this device.
            info = device_mem[device]
            info["count"] += 1
            num_contexts = contexts_per_device[device]

            # Estimate memory after the first or second replica.
            if info["count"] == 1:
                used_mem, _, _ = self._check_device_memory(
                    self._physical_device(device)
                )
                info["after_first"] = used_mem

                if num_contexts == 1:
                    # Only one replica on this device, use actual measurement.
                    est_total = used_mem
                else:
                    # Wait for the second replica to get the marginal cost.
                    est_total = None

            elif info["count"] == 2:
                used_mem, _, _ = self._check_device_memory(
                    self._physical_device(device)
                )
                # The first replica includes one-time context overhead.
                # The marginal cost of subsequent replicas is the difference
                # between the second and first.
                first_cost = info["after_first"] - info["before"]
                marginal_cost = used_mem - info["after_first"]
                est_total = (
                    info["before"] + first_cost + marginal_cost * (num_contexts - 1)
                )
                _logger.info(
                    f"Memory per replica on device {device}: "
                    f"first = {first_cost / (1024**2):.0f} MiB, "
                    f"marginal = {marginal_cost / (1024**2):.0f} MiB"
                )
            else:
                est_total = None

            if est_total is not None:
                total_mem = info["total"]

                # If this exceeds the total memory, raise an error.
                if est_total > total_mem:
                    baseline = info["before"]
                    replica_cost = first_cost + marginal_cost * (num_contexts - 1)

                    # How many contexts fit on this device, given the measured
                    # cost of the first and the marginal cost of the rest. The
                    # slots are spread evenly over the devices, so the total is
                    # capped by what the busiest device can hold.
                    if marginal_cost > 0:
                        per_device = 1 + floor(
                            (total_mem - baseline - first_cost) / marginal_cost
                        )
                    else:
                        per_device = info["count"]

                    msg = (
                        f"Not enough memory on device {device} for all assigned replicas. "
                        f"Baseline usage before simulation: {baseline / (1024**3):.2f} GB "
                        f"Estimated replica memory: {replica_cost / (1024**3):.2f} GB, "
                        f"Total estimated: {est_total / (1024**3):.2f} GB, "
                        f"Available memory: {total_mem / (1024**3):.2f} GB. "
                        f"{self._max_contexts_advice(per_device * num_gpus)}"
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

            # Leave the slot marked as holding no replica, so that the first
            # call to load_replica() does the full setup (lambda value, GCMC
            # parameters, ghost file and sampling statistics) rather than
            # assuming the seeding replica is already fully installed.
            self._slot_replica[i] = None

        # Give each replica its own energy trajectory. These are seeded from a
        # slot's own so that the "ensemble" property is carried over. A slot
        # accumulates into the trajectory of whichever replica it hosts.
        for replica in range(self._num_replicas):
            slot = self._replica_slot[replica]
            self._energy_trajectories[replica] = self._dynamics[
                slot
            ]._d.energy_trajectory()

        # Seed the starting state for every replica from the context of the
        # slot that hosts it. The GCMC water state must be seeded too, since
        # load_replica() diffs against it: a replica with no stored state would
        # be skipped, leaving the sampler holding the water configuration of
        # whichever replica used the slot last.
        for replica in range(self._num_replicas):
            slot = self._replica_slot[replica]

            if self._openmm_states[replica] is None:
                self.save_openmm_state(slot, replica)

            if self._gcmc_samplers[slot] is not None and (
                self._gcmc_states[replica] is None
            ):
                self.save_gcmc_state(slot, replica)

    def slot_for(self, replica):
        """
        Return the index of the slot that hosts a given replica.

        Parameters
        ----------

        replica: int
            The index of the replica.

        Returns
        -------

        int
            The index of the slot.
        """
        return self._replica_slot[replica]

    def device_for(self, slot):
        """
        Return the index of the GPU that a slot's context is on.

        Parameters
        ----------

        slot: int
            The index of the slot.

        Returns
        -------

        int
            The device index.
        """
        return self._slot_devices[slot]

    def contexts_on(self, device):
        """
        Return the number of contexts (slots) on a GPU.

        Parameters
        ----------

        device: int
            The device index.

        Returns
        -------

        int
            The number of contexts.
        """
        return self._slot_devices.count(device)

    def get(self, slot):
        """
        Get the dynamics object (and GCMC sampler) for a given slot.

        When there is one slot per replica the slot and replica indices are
        the same. Otherwise use slot_for() to map a replica to its slot.

        Parameters
        ----------

        slot: int
            The index of the slot.

        Returns
        -------

        tuple
            The dynamics object for the slot and its GCMC sampler.
        """
        return self._dynamics[slot], self._gcmc_samplers[slot]

    def set(self, slot, dynamics):
        """
        Set the dynamics object for a given slot.

        Parameters
        ----------

        slot: int
            The index of the slot.

        dynamics: sire.legacy.Convert.SOMMContext
            The dynamics object.
        """
        self._dynamics[slot] = dynamics

    def delete(self, slot):
        """
        Delete the dynamics object for a given slot.

        Parameters
        ----------

        slot: int
            The index of the slot.
        """
        self._dynamics[slot] = None

    def save_openmm_state(self, slot, replica):
        """
        Save the state of a slot's dynamics object as the state of a replica.

        Parameters
        ----------

        slot: int
            The index of the slot.

        replica: int
            The index of the replica whose state this is.
        """

        # Get the current OpenMM state.
        state = (
            self._dynamics[slot]
            .context()
            .getState(getPositions=True, getVelocities=True)
        )

        # Store positions, velocities, and box vectors as compact numpy arrays
        # rather than the OpenMM State object, which serialises to XML when
        # pickled and is orders of magnitude larger.
        self._openmm_states[replica] = {
            "positions": state.getPositions(asNumpy=True),
            "velocities": state.getVelocities(asNumpy=True),
            "box": state.getPeriodicBoxVectors(asNumpy=True),
        }

    @staticmethod
    def _get_positions(state):
        """
        Return the positions from a saved OpenMM state.

        Parameters
        ----------

        state: dict or openmm.State
            The state to read. Dicts (new format) hold the positions directly.
            A bare openmm.State is accepted for backwards compatibility with
            old checkpoint files, as it is by _apply_openmm_state().

        Returns
        -------

        openmm.unit.Quantity
            The positions.
        """
        if isinstance(state, dict):
            return state["positions"]

        return state.getPositions(asNumpy=True)

    @staticmethod
    def _apply_openmm_state(context, state):
        """
        Apply a saved OpenMM state to a context.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to update.

        state: dict or openmm.State
            The state to apply. Dicts (new format) contain "positions",
            "velocities", and "box" numpy arrays. A bare openmm.State is
            accepted for backwards compatibility with old checkpoint files.

        Note that the step count and simulation time carried by an openmm.State
        are deliberately not restored for the dict format. They are held
        separately and applied by Dynamics._set_clock().
        """
        if isinstance(state, dict):
            # Set the box before the positions, since a barostat may have
            # changed it between the state being saved and restored.
            context.setPeriodicBoxVectors(*state["box"])
            context.setPositions(state["positions"])
            context.setVelocities(state["velocities"])
        else:
            # Legacy openmm.State from checkpoint files written before this
            # format change.
            context.setState(state)

    def save_gcmc_state(self, slot, replica):
        """
        Save the current GCMC water state of a slot as that of a replica.

        Parameters
        ----------

        slot: int
            The index of the slot.

        replica: int
            The index of the replica whose state this is.
        """
        # Get the GCMC sampler.
        gcmc_sampler = self._gcmc_samplers[slot]

        # Store the state.
        self._gcmc_states[replica] = gcmc_sampler.water_state()

    def get_clock(self, replica=None):
        """
        Get the simulation clock.

        Parameters
        ----------

        replica: int
            The index of the replica. If None, the clock of the first slot is
            returned, which is common to every replica until they have each
            been propagated and stored a clock of their own.

        Returns
        -------

        dict
            The clock, as returned by Dynamics._get_clock(), or None if the
            replica has no clock of its own.
        """
        if replica is not None:
            return self._clocks[replica]

        return self._dynamics[0]._get_clock()

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

    def load_replica(self, replica, clock=None):
        """
        Make a replica resident in its slot, ready for a dynamics block.

        This sets the slot's lambda value, pushes the replica's state into the
        OpenMM context, and points the slot at the replica's clock, energy
        trajectory, GCMC water state and ghost file.

        Work that isn't needed is skipped. When there is one slot per replica
        and no swap has taken place, the slot already holds everything the
        replica needs and this reduces to a handful of comparisons.

        Only touches the replica's own slot, so this is safe to call
        concurrently for replicas in different slots.

        Parameters
        ----------

        replica: int
            The index of the replica.

        clock: dict
            The simulation clock to restore, as returned by
            Dynamics._get_clock(). If None, the slot's clock is left alone.
        """
        slot = self._replica_slot[replica]
        dynamics = self._dynamics[slot]
        gcmc_sampler = self._gcmc_samplers[slot]

        # The replica that the slot currently holds.
        resident = self._slot_replica[slot]

        # The slot needs new positions and velocities if it is being handed a
        # different replica, or if the last mix moved this replica's state.
        if resident != replica or self._state_moved[replica]:
            self._apply_openmm_state(dynamics.context(), self._openmm_states[replica])

            # Positions have changed underneath the context, so any cached
            # energies are stale.
            dynamics.clear_energy_cache()

            self._state_moved[replica] = False

        # Set the lambda value and REST2 scaling factor. This is a no-op if
        # the slot is already at this lambda value.
        if resident != replica:
            dynamics.set_lambda(
                self._lambdas[replica],
                rest2_scale=self._rest2_scale_factors[replica],
                update_constraints=self._update_constraints,
            )

        # Restore the clock and point the slot at this replica's energy
        # trajectory, so that energies are accumulated against the replica
        # rather than the slot.
        if clock is not None:
            dynamics._set_clock(clock)
        dynamics.set_energy_trajectory(self._energy_trajectories[replica])

        if gcmc_sampler is not None:
            gcmc_sampler.push()
            try:
                # Swap the water state into the sampler. Diff against what the
                # sampler currently holds, which is the state of whichever
                # replica was last resident.
                target = self._gcmc_states[replica]
                if target is not None:
                    current = gcmc_sampler.water_state()
                    water_idxs = _np.where(current != target)[0]

                    if len(water_idxs) > 0:
                        gcmc_sampler._set_water_state(
                            dynamics.context(),
                            indices=water_idxs,
                            states=target[water_idxs],
                        )

                if resident != replica:
                    # Update the lambda dependent non-bonded parameters used to
                    # evaluate insertion and deletion energies, and append ghost
                    # residues to this replica's file. The sampler keeps its
                    # statistics per lambda value, so switching also switches
                    # to this replica's.
                    gcmc_sampler.set_lambda(
                        self._lambdas[replica], self._rest2_scale_factors[replica]
                    )
                    gcmc_sampler.set_ghost_file(self._ghost_files[replica])
            finally:
                gcmc_sampler.pop()

        self._slot_replica[slot] = replica

    def store_replica(self, replica):
        """
        Save the state of a replica back out of its slot, so that the slot can
        be handed to another replica.

        Parameters
        ----------

        replica: int
            The index of the replica.
        """
        slot = self._replica_slot[replica]

        self.save_openmm_state(slot, replica)
        self._clocks[replica] = self._dynamics[slot]._get_clock()

        if self._gcmc_samplers[slot] is not None:
            self.save_gcmc_state(slot, replica)

            # Count the waters against the slot's context rather than whichever
            # one the sampler happens to be bound to, which is unset after the
            # sampler has been reset.
            gcmc_sampler = self._gcmc_samplers[slot]
            gcmc_sampler.push()
            try:
                self._gcmc_num_waters[replica] = gcmc_sampler.num_waters(
                    context=self._dynamics[slot].context()
                )
            finally:
                gcmc_sampler.pop()

    def mix_states(self):
        """
        Apply the result of a replica mix.

        The states are permuted here, but not pushed into the OpenMM contexts.
        They are applied lazily by load_replica(), which is the only point at
        which a slot is known to be free. This does the same amount of work as
        applying them eagerly, since load_replica() pushes a state exactly when
        the mix moved it.

        The permutation must happen here rather than being resolved lazily
        through self._states. A slot is re-used within a cycle, so a replica
        may be loaded after another replica has already stored its post-run
        state; reading through the indirection at that point would pick up the
        new state rather than the pre-mix one.
        """
        # Permute the travelling state. This is a reference shuffle, so it is
        # cheap even for large systems. Statistics and output files stay with
        # the lambda window, so are not permuted.
        self._openmm_states = [self._openmm_states[state] for state in self._states]
        self._gcmc_states = [self._gcmc_states[state] for state in self._states]

        # Flag the replicas whose state moved, so that load_replica() knows it
        # has to push new positions and velocities into the context. The flags
        # accumulate, and are only cleared once the state has been pushed, so
        # that a replica mixed twice without being loaded in between isn't left
        # starting from whatever its context happens to hold.
        self._state_moved = [
            moved or bool(state != i)
            for i, (state, moved) in enumerate(zip(self._states, self._state_moved))
        ]

        # Update the swap matrix with each configuration's move.
        for i, state in enumerate(self._states):
            self._num_swaps[state, i] += 1

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

    def _physical_device(self, device):
        """
        Map an OpenMM device index to the physical device backing it.

        Parameters
        ----------

        device: int
            The OpenMM device index, which is relative to the visible set.

        Returns
        -------

        int, str
            The physical device, i.e. the matching entry from
            CUDA_VISIBLE_DEVICES. Falls back to the OpenMM index if the visible
            set is unknown, which is correct whenever it starts at zero and is
            contiguous.
        """
        gpu_devices = getattr(self, "_gpu_devices", None)

        if gpu_devices is None or device >= len(gpu_devices):
            return device

        return gpu_devices[device]

    @staticmethod
    def _max_contexts_advice(num_contexts):
        """
        Advice for a user whose replicas don't fit in GPU memory.

        Parameters
        ----------

        num_contexts: int
            The number of OpenMM contexts that are known to fit.
        """

        if num_contexts < 1:
            return (
                "A single OpenMM context does not fit on this device. Reduce "
                "the size of the system, or use a device with more memory."
            )

        return (
            f"Set 'max_contexts' to {num_contexts} or fewer to re-use contexts "
            "across lambda values, which removes the memory limit on the "
            "number of replicas. This also requires 'frame_frequency' to equal "
            "'checkpoint_frequency'."
        )

    @staticmethod
    def _check_device_memory(device=0):
        """
        Check the memory usage of the specified GPU device.

        Parameters
        ----------

        device: int, str
            The device to query. This is the physical device, i.e. an entry
            from CUDA_VISIBLE_DEVICES (or the equivalent for other platforms),
            not the index used by OpenMM. OpenMM numbers devices relative to
            the visible set, whereas pynvml and pyopencl enumerate all devices
            on the machine, so the two only agree when the visible set starts
            at zero and is contiguous. CUDA_VISIBLE_DEVICES entries may be
            either an index or a UUID.
        """

        device = str(device).strip()

        # A UUID cannot be used to index into the OpenCL device list.
        is_uuid = device.startswith("GPU-") or device.startswith("MIG-")
        device_index = None if is_uuid else int(device)

        # Try to use pyopencl to detect the GPU vendor.
        vendor = None
        ocl_device = None
        try:
            import pyopencl as cl

            if device_index is None:
                raise ValueError("Cannot index OpenCL devices by UUID")

            platforms = cl.get_platforms()
            all_devices = []
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    all_devices.extend(devices)
                except Exception:
                    continue

            if device_index < len(all_devices):
                ocl_device = all_devices[device_index]
                vendor = ocl_device.vendor
            else:
                msg = f"Device index {device_index} out of range. Found {len(all_devices)} GPU(s)."
                _logger.error(msg)
                raise IndexError(msg)
        except IndexError:
            raise
        except Exception:
            _logger.warning(
                "Could not query GPU platform via OpenCL; falling back to pynvml for NVIDIA detection."
            )

        # NVIDIA: Use pynvml (also used as fallback when OpenCL is unavailable).
        if vendor is None or "NVIDIA" in vendor:
            try:
                import pynvml

                pynvml.nvmlInit()
                if is_uuid:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(device.encode())
                else:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return (memory.used, memory.free, memory.total)
            except Exception as e:
                if vendor is None:
                    msg = f"Could not get GPU memory info for device {device} via OpenCL or pynvml: {e}"
                else:
                    msg = (
                        f"Could not get NVIDIA GPU memory info for device {device}: {e}"
                    )
                _logger.error(msg)
                raise RuntimeError(msg) from e

        # AMD: Use OpenCL extension.
        elif "AMD" in vendor or "Advanced Micro Devices" in vendor:
            try:
                import pyopencl as cl

                total = ocl_device.global_mem_size

                # cl_amd_device_attribute_query reports the free memory in
                # KBytes, as a list with the whole heap first.
                free_memory_info = ocl_device.get_info(
                    cl.device_info.GLOBAL_FREE_MEMORY_AMD
                )
                free_kb = (
                    free_memory_info[0]
                    if isinstance(free_memory_info, (list, tuple))
                    else free_memory_info
                )
                free = free_kb * 1024
                used = total - free
                return (used, free, total)
            except Exception as e:
                msg = f"Could not get AMD GPU memory info for device {device}: {e}"
                _logger.error(msg)
                raise RuntimeError(msg) from e

        # Any other vendor: we have no way of querying the free memory.
        else:
            msg = (
                f"Unable to query the memory of device {device}: "
                f"unsupported GPU vendor '{vendor}'."
            )
            _logger.error(msg)
            raise RuntimeError(msg)


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

        # Make sure we're using a GPU platform.
        if not self._is_gpu:
            msg = (
                "Currently replica exchange simulations can only be "
                "run on the CUDA, OpenCL, and HIP platforms."
            )
            _logger.error(msg)
            raise ValueError(msg)

        # The available devices were detected by the base class constructor.
        gpu_devices = self._gpu_devices

        # We can only use replica exchange if we have a GPU.
        if len(gpu_devices) == 0:
            _logger.error("No GPUs available. Cannot run replica exchange.")

        # Set the number of GPUs.
        if self._config.max_gpus is None:
            self._num_gpus = len(gpu_devices)
        else:
            self._num_gpus = min(self._config.max_gpus, len(gpu_devices))

        # The physical devices backing each OpenMM device index. OpenMM numbers
        # devices relative to the visible set, so index i is gpu_devices[i].
        self._gpu_devices = list(gpu_devices)[: self._num_gpus]

        # Work out how many OpenMM contexts (slots) to create. When there are
        # fewer slots than replicas, each slot is re-used to propagate several
        # replicas per cycle, changing its lambda value as it goes.
        self._num_replicas = len(self._lambda_values)

        # There is nothing to exchange with a single replica.
        if self._num_replicas < 2:
            msg = (
                "Replica exchange requires at least two "
                f"{_lam_sym} values. Please use the Runner class instead."
            )
            _logger.error(msg)
            raise ValueError(msg)

        self._set_num_slots()

        # The particles of the bond that is broken or made, whose distance is
        # used to rank the windows for adaptive replica exchange.
        self._ring_pair = (
            self._find_ring_pair()
            if self._config.adaptive_repex and self._is_ring_break
            else None
        )

        # Auto-generate a Boresch restraint for ABFE runs with no user-supplied
        # restraint. This must happen before the dynamics cache is built below,
        # since the per-replica OpenMM contexts it creates are fixed at
        # construction time and won't pick up a restraint added afterwards.
        if self._is_abfe_bound and self._config.restraints is None:
            try:
                restraints = self._generate_boresch_restraint(device=0)
            except Exception as e:
                msg = f"Unable to generate Boresch restraint for ABFE simulation: {e}"
                _logger.error(msg)
                raise RuntimeError(msg)
            self._config.restraints = restraints
            self._dynamics_kwargs["restraints"] = restraints

        # Store the name of the dynamics cache pickle file.
        self._repex_state = self._config.output_directory / "repex_state.pkl"

        # Store the name of the replica exchange swap acceptance matrix.
        self._repex_matrix = self._config.output_directory / "repex_matrix.txt"

        # Sentinel file written only after a fully successful run (dynamics +
        # trajectory consolidation + backup cleanup). Used to distinguish
        # "truly complete" from "complete dynamics but killed during cleanup".
        self._done_file = self._config.output_directory / "simulation.done"

        # Flag that we haven't equilibrated.
        self._is_equilibration = False

        # Store the default options.
        timestep = self._config.timestep
        constraint = self._config.constraint
        perturbable_constraint = self._config.perturbable_constraint

        # Don't use constraints during minimisation.
        if (
            self._config.minimise
            and not self._is_restart
            and not self._config.minimisation_constraints
        ):
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

        # On a fresh (non-restart) run, remove any leftover sentinel so that
        # a repeated run with --overwrite doesn't immediately exit as complete.
        if not self._is_restart and self._done_file.exists():
            self._done_file.unlink()

        # Create the dynamics cache.
        if not self._is_restart:
            xml_filenames = (
                [self._filenames[i]["xml"] for i in range(len(self._lambda_values))]
                if self._config.save_xml
                else None
            )
            self._dynamics_cache = DynamicsCache(
                self._system,
                self._lambda_values,
                self._rest2_scale_factors,
                self._num_gpus,
                dynamics_kwargs,
                gcmc_kwargs=self._gcmc_kwargs,
                perturbed_system=self._perturbed_system,
                output_directory=self._config.output_directory,
                xml_filenames=xml_filenames,
                num_slots=self._num_slots,
                update_constraints=self._config.update_constraints,
                constraint_lambda_index=self._constraint_lambda_index,
                gpu_devices=self._gpu_devices,
                strided=self._is_strided,
            )
            self._dynamics_cache._virtual_sites = self._virtual_sites

        else:
            _logger.debug("Restarting from file")

            # Load the dynamics cache first so we can read the simulation time
            # from it (new format). Old-format restarts with .s3 files fall
            # back to reading the time from the loaded Sire system.
            try:
                with open(self._repex_state, "rb") as f:
                    self._dynamics_cache = _pickle.load(f)
            except Exception as e:
                _logger.error(
                    f"Could not load dynamics cache from {self._repex_state}: {e}"
                )
                raise e

            # Set again, since states saved by older versions don't have it.
            self._dynamics_cache._virtual_sites = self._virtual_sites

            # Derive the simulation time: prefer the value stored in the
            # pickle (_time is set by the new-format _write_checkpoint_system);
            # fall back to the Sire system for old-format checkpoints.
            if self._dynamics_cache._time is not None and not isinstance(
                self._system, list
            ):
                time = self._dynamics_cache._time
            else:
                time = self._system[0].time()

            # Check to see if the simulation is already complete.
            if self._done_file.exists():
                # The runtime may have been extended beyond the previous run.
                # If so, clear the sentinel and continue.
                if time < self._config.runtime - self._config.timestep:
                    _logger.info(
                        "Runtime has been extended. Clearing completion sentinel."
                    )
                    self._done_file.unlink()
                else:
                    _logger.success("Simulation already complete. Exiting.")
                    _sys.exit(0)

            if time > self._config.runtime - self._config.timestep:
                # Dynamics finished but the process was killed before cleanup
                # completed (e.g. during DCD consolidation or backup removal).
                # Consolidate any remaining trajectory chunks and tidy up.
                _logger.warning(
                    "Simulation dynamics are complete but post-run cleanup was "
                    "not finished. Completing cleanup now."
                )
                self._consolidate_trajectories()
                self._cleanup()
                self._done_file.touch()
                _logger.success("Cleanup complete. Exiting.")
                _sys.exit(0)
            else:
                _logger.info(
                    f"Restarting at time {time}, time remaining = {self._config.runtime - time}"
                )

            # Make sure the number of replicas is the same.
            if len(self._dynamics_cache._lambdas) != self._config.num_lambda:
                _logger.error(
                    f"The number of replicas in the dynamics cache ({len(self._dynamics_cache._lambdas)}) "
                    f"does not match the number of replicas in the configuration ({self._config.num_lambda})."
                )

            # For new-format restarts, set the system time so that dynamics
            # objects are initialised with the correct integrator step count.
            if not isinstance(self._system, list):
                self._system.set_time(time)

            # The physical device list is not pickled, since the run may be
            # restarted against a different set of GPUs.
            self._dynamics_cache._gpu_devices = self._gpu_devices

            # Rebuild the slot layout from the current config, so that
            # 'max_contexts' can change on restart. Everything that is restored
            # is per-replica, so it doesn't depend on the grouping.
            self._dynamics_cache._num_slots = self._num_slots
            self._dynamics_cache._update_constraints = self._config.update_constraints
            self._dynamics_cache._constraint_lambda_index = (
                self._constraint_lambda_index
            )
            self._dynamics_cache._strided = self._is_strided
            self._dynamics_cache._build_slot_layout()

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

            # The OpenMM contexts are not reset here. Each replica's state is
            # pushed into its slot by load_replica() at the start of its first
            # block, which is the only point at which the slot is known to be
            # free.

            # Restore the sampling statistics. A sampler keeps only the lambda
            # values it visits, so each can be handed the whole simulation's.
            if self._dynamics_cache._gcmc_stats is not None:
                for slot in range(self._dynamics_cache._num_slots):
                    _, gcmc_sampler = self._dynamics_cache.get(slot)
                    gcmc_sampler.restore_stats(self._dynamics_cache._gcmc_stats)

        # Log the GCMC sphere centre for each replica. This uses the stored
        # state rather than the context, since a slot only holds the positions
        # of the replica it last hosted.
        import openmm.unit as _omm_unit

        for i, lam in enumerate(self._lambda_values):
            _, gcmc_sampler = self._dynamics_cache.get(self._dynamics_cache.slot_for(i))
            if gcmc_sampler is not None and gcmc_sampler._reference is not None:
                positions = DynamicsCache._get_positions(
                    self._dynamics_cache._openmm_states[i]
                )
                target = gcmc_sampler._get_target_position(
                    positions.value_in_unit(_omm_unit.angstrom)
                )
                _logger.info(
                    f"Initial GCMC sphere centre for lambda {lam:.5f}: "
                    f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] A"
                )

        # Conversion factor for reduced potential.
        kT = (_sr.units.k_boltz * self._config.temperature).to(_sr.units.kcal_per_mol)
        self._beta = 1.0 / kT

        # If restarting, subtract the time already run from the total runtime
        if self._config.restart:
            time = (
                self._dynamics_cache._time
                if (
                    self._dynamics_cache._time is not None
                    and not isinstance(self._system, list)
                )
                else self._system[0].time()
            )
            self._config.runtime = str(self._config.runtime - time)

            # Work out the current block number.
            if self._config.checkpoint_frequency.value() > 0.0:
                self._start_block = int(
                    round(time.value() / self._config.checkpoint_frequency.value(), 12)
                )
            else:
                self._start_block = 0
            self._start_time = time
        else:
            self._start_block = 0
            self._start_time = _sr.u("0 ps")

        if self._config.adaptive_repex:
            self._check_adaptive_config()

        # Create a terminal flip sampler per replica (if terminal groups were detected).
        if self._terminal_groups:
            from ._samplers import TerminalFlipSampler

            self._terminal_flip_samplers = [
                TerminalFlipSampler(
                    self._terminal_groups,
                    float(self._config.temperature.value()),
                )
                for _ in self._lambda_values
            ]
            _logger.info(
                f"Terminal flip samplers ready ({len(self._terminal_groups)} group(s))"
            )
        else:
            self._terminal_flip_samplers = None

        # Restore terminal flip sampler statistics from checkpoint (deferred
        # until here so that _terminal_flip_samplers is always initialised first).
        if self._is_restart and self._terminal_flip_samplers is not None:
            for i in range(len(self._lambda_values)):
                attempted, accepted = self._dynamics_cache._terminal_flip_stats[i]
                self._terminal_flip_samplers[i].reset(attempted, accepted)

        from threading import Lock

        # Create a lock to guard the dynamics cache.
        self._lock = Lock()

        # Systems committed while a replica was resident in its slot, awaiting
        # the end of cycle checkpoint. Only used when contexts are shared,
        # keyed by replica index and emptied by _checkpoint().
        self._committed = {}

    def _build_lambda(self, replica):
        """
        Return the lambda value to create a replica's context at.

        This is the replica's own lambda value, unless the constraints are
        fixed at a common one, in which case a rebuilt context has to use that
        too or it would pick up the constrained bond lengths of its own lambda
        value instead.

        Parameters
        ----------

        replica: int
            The index of the replica.

        Returns
        -------

        float
            The lambda value to create the context at.
        """
        if self._constraint_lambda_index is None:
            return self._lambda_values[replica]

        return self._lambda_values[self._constraint_lambda_index]

    def _replica_passes(self, cycle, replicas=None):
        """
        Work out which replicas to propagate in each pass of a cycle.

        A slot can only host one replica at a time, so each pass takes at most
        one replica from each slot. Groups are traversed in alternating
        directions on successive cycles, so that a slot always moves to a
        neighbouring lambda window, including across the cycle boundary. That
        keeps the change in force field parameters as small as possible.

        When there is one slot per replica there is a single pass containing
        every replica, which is the same as propagating them all together.

        Parameters
        ----------

        cycle: int
            The index of the current cycle.

        replicas: list of int
            The replicas to propagate. If None, every replica is propagated.

        Returns
        -------

        list of list of int
            The replicas to propagate in each pass.
        """
        groups = self._dynamics_cache._groups

        if replicas is not None:
            replicas = set(replicas)
            groups = [[r for r in group if r in replicas] for group in groups]
            groups = [group for group in groups if group]

        # Traverse the groups backwards on odd cycles.
        is_reversed = cycle % 2 == 1

        passes = []
        for i in range(max(len(group) for group in groups)):
            batch = []
            for group in groups:
                if i < len(group):
                    batch.append(group[len(group) - 1 - i] if is_reversed else group[i])
            passes.append(batch)

        return passes

    def _map_replicas(self, executor, step, func, replicas, *args):
        """
        Run a step for each replica in a thread pool, letting every call
        finish before raising the first error, so that files written under a
        lock are all written before it is released. An interrupt exits
        straight away, without waiting.

        Two replicas that share a slot must never be processed at the same
        time, so the replicas must come from a single pass, unless the step
        doesn't use the slot's context.

        Parameters
        ----------

        executor: concurrent.futures.ThreadPoolExecutor
            The thread pool to run the step in.

        step: str
            The name of the step, for logging.

        func: callable
            The step, which returns the index of the replica and the exception
            raised, if any. An exception it raises is treated the same way.

        replicas: list of int
            The replicas to run the step for.

        args: iterable
            The remaining arguments, an iterable of one per replica for each.
        """
        from concurrent.futures import wait

        try:
            futures = [
                executor.submit(func, *arguments) for arguments in zip(replicas, *args)
            ]
            wait(futures)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            _logger.error(f"{step} cancelled. Exiting.")
            _sys.exit(1)

        outcomes = []
        for replica, future in zip(replicas, futures):
            error = future.exception()
            outcomes.append((replica, error) if error else future.result())

        for index, error in outcomes:
            if error is not None:
                _logger.error(
                    f"{step} failed for {_lam_sym} = "
                    f"{self._lambda_values[index]:.5f}: {error}"
                )
                raise error

    def _set_num_slots(self):
        """
        Work out the number of OpenMM contexts (slots) to create, validating
        the configuration options that only apply when contexts are re-used
        across lambda values.

        Sets self._num_slots, self._is_cached and self._is_strided.
        """

        num_replicas = self._num_replicas

        if self._config.max_contexts is None:
            self._num_slots = num_replicas
        else:
            self._num_slots = min(self._config.max_contexts, num_replicas)

        # There is a context per replica, so nothing is re-used and all of the
        # constraints below are irrelevant. A context then keeps the lambda
        # value it was created at, so there is no need to fix the constraints
        # at a common one.
        self._is_cached = self._num_slots < num_replicas
        self._is_strided = self._is_cached and self._config.adaptive_repex
        self._constraint_lambda_index = None

        # Contexts run concurrently, so it's the number of them that determines
        # how evenly the GPUs are loaded, not the number of replicas.
        num_workers = self._num_gpus * self._config.oversubscription_factor

        if self._num_slots < num_workers:
            _logger.warning(
                f"The number of contexts ({self._num_slots}) is less than the "
                f"number of workers ({num_workers}). Some GPUs will be left idle."
            )
        elif self._num_slots % self._num_gpus != 0:
            _logger.warning(
                f"The number of contexts ({self._num_slots}) is not a multiple "
                f"of the number of GPUs ({self._num_gpus}). This may result in "
                "suboptimal performance."
            )

        if not self._is_cached:
            if self._config.max_contexts is not None:
                _logger.info(
                    f"Creating one OpenMM context per replica ({num_replicas})"
                )
            return

        # Frames can only be saved on checkpoint cycles when contexts are
        # re-used. Within a cycle a context propagates several replicas in
        # turn, so frames from different replicas would otherwise accumulate
        # in the same internal trajectory. Tying frames to checkpoints means
        # each one is written out and cleared before the context is handed to
        # the next replica.
        if (
            self._save_frames
            and self._config.frame_frequency != self._config.checkpoint_frequency
        ):
            msg = (
                "'frame_frequency' must equal 'checkpoint_frequency' when "
                "'max_contexts' is less than the number of replicas."
            )
            _logger.error(msg)
            raise ValueError(msg)

        # When the constraints aren't updated as a slot changes lambda, they stay
        # as they were when its context was created. Create every context at the
        # same lambda value, so that the constrained bond lengths are uniform
        # across replicas rather than depending on which slot a replica happens
        # to be assigned to. Only needed if they actually perturb.
        if not self._config.update_constraints and self._end_state_constraints_differ:
            # Which lambda value is used matters less than every replica using
            # the same one, since where the bonds actually perturb depends on
            # the lambda schedule.
            if self._config.constraint_lambda_index >= num_replicas:
                msg = (
                    f"'constraint_lambda_index' "
                    f"({self._config.constraint_lambda_index}) is out of range "
                    f"for {num_replicas} {_lam_sym} values."
                )
                _logger.error(msg)
                raise ValueError(msg)

            self._constraint_lambda_index = self._config.constraint_lambda_index
            _logger.warning(
                f"'update_constraints' is False. Constrained bond lengths will not "
                f"perturb with lambda, and are fixed at those of "
                f"{_lam_sym} = "
                f"{self._lambda_values[self._constraint_lambda_index]:.5f} "
                f"for every replica."
            )

        from math import ceil

        _logger.info(
            f"Re-using {self._num_slots} OpenMM context(s) across "
            f"{num_replicas} replicas: {ceil(num_replicas / self._num_slots)} "
            "pass(es) per cycle"
        )

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
        from concurrent.futures import ThreadPoolExecutor

        # Store the number of concurrent workers.
        num_workers = self._num_gpus * self._config.oversubscription_factor

        # Store the number of workers to use for checkpointing.
        if self._config.num_checkpoint_workers is None:
            num_checkpoint_workers = num_workers
        else:
            num_checkpoint_workers = min(
                self._config.num_checkpoint_workers, num_workers
            )

        # Persistent thread pools, reused across every batch and cycle.
        dynamics_executor = ThreadPoolExecutor(max_workers=num_workers)
        checkpoint_executor = ThreadPoolExecutor(max_workers=num_checkpoint_workers)

        executors = (dynamics_executor, checkpoint_executor)
        try:
            self._run(dynamics_executor, checkpoint_executor)
        except Exception:
            # Wait for running blocks, so that nothing still uses the contexts.
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            for executor in executors:
                executor.shutdown(wait=True)

    def _run(self, dynamics_executor, checkpoint_executor):
        """
        Run the replica exchange simulation.

        Parameters
        ----------

        dynamics_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for dynamics, with a worker for each replica that
            can be propagated at once.

        checkpoint_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for writing checkpoint files.
        """

        from math import ceil
        from time import time

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

        # Store the current checkpoint frequency.
        checkpoint_frequency = self._config.checkpoint_frequency

        if checkpoint_frequency.value() > 0.0:
            # Calculate the number of blocks and the remainder time.
            frac = (self._config.runtime / checkpoint_frequency).value()

            # Handle the case where the runtime is less than the checkpoint frequency.
            if frac < 1.0:
                frac = 1.0
                checkpoint_frequency = self._config.runtime

            num_blocks = int(frac)
            rem = round(frac - num_blocks, 12)

            # Work out the number of repex cycles per block.
            frac = (checkpoint_frequency / self._config.energy_frequency).value()

            # Handle the case where the checkpoint frequency is less than the energy frequency.
            if frac < 1.0:
                frac = 1.0
                checkpoint_frequency = self._config.energy_frequency

            # Store the number of repex cycles per block (may be fractional).
            cycles_per_checkpoint = frac

        # Otherwise, we don't checkpoint.
        else:
            cycles_per_checkpoint = float(cycles)
            num_blocks = 1
            rem = 0

        # Create the replica list.
        replica_list = list(range(self._config.num_lambda))

        # The GPU that propagates each replica, and how many replicas each GPU
        # can propagate at once, which are fixed for the run.
        replica_devices = self._replica_devices()
        device_capacity = self._device_capacity()

        # Minimise at each lambda value. Don't minimise on restart.
        if self._config.minimise and not self._is_restart:
            for batch in self._dynamics_batches(replica_devices):
                try:
                    for success, index, e in dynamics_executor.map(
                        self._minimise,
                        batch,
                    ):
                        if not success:
                            msg = f"Minimisation failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                            if self._config.minimisation_errors:
                                _logger.error(msg)
                                raise e
                            else:
                                _logger.warning(msg)
                except KeyboardInterrupt:
                    dynamics_executor.shutdown(wait=False, cancel_futures=True)
                    _logger.error("Minimisation cancelled. Exiting.")
                    _sys.exit(1)

        # Equilibrate the system.
        if self._is_equilibration and not self._is_restart:
            for batch in self._dynamics_batches(replica_devices):
                try:
                    for success, index, e in dynamics_executor.map(
                        self._equilibrate,
                        batch,
                    ):
                        if not success:
                            _logger.error(
                                f"Equilibration failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                            )
                            raise e
                except KeyboardInterrupt:
                    dynamics_executor.shutdown(wait=False, cancel_futures=True)
                    _logger.error("Equilibration cancelled. Exiting.")
                    _sys.exit(1)

        # Write a checkpoint immediately after equilibration so that a restart
        # after an early production crash doesn't need to re-equilibrate.
        if self._is_equilibration and not self._is_restart:
            # When contexts are shared, commit each replica while it is
            # resident so that every checkpoint file can still be written
            # under a single lock below.
            if self._is_cached:
                for batch in self._replica_passes(0):
                    self._map_replicas(
                        checkpoint_executor,
                        "Post-equilibration commit",
                        self._load_and_commit,
                        batch,
                    )

            lock = _FileLock(self._lock_file)
            with lock.acquire(timeout=self._config.timeout.to("seconds")):
                self._map_replicas(
                    checkpoint_executor,
                    "Post-equilibration checkpoint",
                    self._checkpoint,
                    replica_list,
                    repeat(self._lambda_values),
                    repeat(-1),
                    repeat(cycles),
                )

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

        # Work out the number of cycles per terminal flip move.
        if (
            self._config.terminal_flip_frequency is not None
            and self._terminal_flip_samplers is not None
        ):
            cycles_per_flip = max(
                1,
                round(
                    (
                        self._config.terminal_flip_frequency
                        / self._config.energy_frequency
                    ).value()
                ),
            )
        else:
            cycles_per_flip = cycles + 1

        # The cycles, numbered from one, that end with a checkpoint.
        checkpoint_cycles = self._checkpoint_cycles(cycles, cycles_per_checkpoint)
        checkpoint_set = set(checkpoint_cycles.tolist())

        # Whether the most recent cycle saved the replica exchange state.
        is_checkpoint = False

        num_replicas = self._config.num_lambda
        is_adaptive = self._config.adaptive_repex

        # The cycle on which each replica was last propagated, used to catch up
        # on the MC moves that fell due while it wasn't.
        last_cycle = [-1] * num_replicas

        # The number of energy samples each replica has collected since the
        # last checkpoint.
        num_samples = [0] * num_replicas

        if is_adaptive:
            self._reset_adaptive_data()
            self._restore_adaptive_data()
        else:
            # Adaptive replica exchange may have been switched off on restart.
            self._dynamics_cache._hot_replicas = None
            self._dynamics_cache._cold_votes = [0] * num_replicas

        # Perform the replica exchange simulation.
        for i in range(cycles):
            _logger.info(f"Running dynamics for cycle {i + 1} of {cycles}")

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
            is_checkpoint = (i + 1) in checkpoint_set

            # Whether the checkpoint files are written at the end of this cycle.
            do_checkpoint = is_checkpoint or i == cycles - 1

            # Only the hot replicas are propagated, except on mix and checkpoint
            # cycles.
            hot = self._dynamics_cache._hot_replicas
            is_full = (
                hot is None
                or (i + 1) % self._config.adaptive_mix_interval == 0
                or do_checkpoint
            )
            active = list(range(num_replicas)) if is_full else hot

            if not is_full:
                _logger.info(f"Propagating {len(active)} hot replicas: {active}")

            # Capture the simulation clock at the start of the cycle. This seeds
            # any replica that doesn't yet have a clock of its own.
            clock = self._dynamics_cache.get_clock()

            # Propagate the replicas, one pass at a time. Each pass takes at
            # most one replica from each slot, and there is a single pass when
            # there is a slot per replica.
            for batch in self._replica_passes(i, replicas=None if is_full else active):
                # Run a dynamics block for each replica in the pass, making sure
                # each GPU is only oversubscribed by a factor of
                # self._config.oversubscription_factor.
                for replicas in self._device_batches(batch, replica_devices):
                    # Whether each replica is due a GCMC or terminal flip move,
                    # including any that fell due while it wasn't propagated.
                    is_gcmc = [
                        self._is_due(last_cycle[r], i, cycles_per_gcmc)
                        for r in replicas
                    ]
                    is_terminal_flip = [
                        self._is_due(last_cycle[r], i, cycles_per_flip)
                        for r in replicas
                    ]

                    # Whether each replica saves a frame and writes its ghost
                    # waters, scheduled by the cycle since a cold replica's own
                    # clock falls behind.
                    is_frame = [
                        self._is_due(last_cycle[r], i, cycles_per_frame)
                        for r in replicas
                    ]

                    try:
                        for result, index, energies in dynamics_executor.map(
                            self._run_block,
                            replicas,
                            repeat(self._lambda_values),
                            is_gcmc,
                            is_frame,
                            is_terminal_flip,
                            repeat(self._config.save_energy_components),
                            repeat(clock),
                            (
                                is_frame
                                if is_adaptive and self._save_frames
                                else repeat(None)
                            ),
                        ):
                            if not result:
                                _logger.error(
                                    f"Dynamics failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {energies}"
                                )
                                raise energies
                            results.append((index, energies))
                    except KeyboardInterrupt:
                        dynamics_executor.shutdown(wait=False, cancel_futures=True)
                        _logger.error("Dynamics cancelled. Exiting.")
                        _sys.exit(1)

                # When contexts are shared, commit the replicas of this pass
                # while they are still resident in their slots. The committed
                # systems are held until the end of the cycle, so that every
                # checkpoint file is still written under a single lock. With a
                # context per replica everything is still resident once the
                # cycle finishes, so committing is left to _checkpoint().
                if do_checkpoint and self._is_cached:
                    self._map_replicas(
                        checkpoint_executor, "Commit", self._commit_replica, batch
                    )

            for index in active:
                last_cycle[index] = i
                num_samples[index] += 1

            # Assemble an energy matrix from the results.
            _logger.info("Assembling energy matrix")
            energy_matrix = self._assemble_results(results)

            # Record the energy differences used to rank the windows, once the
            # burn-in is complete.
            cycle_time = self._start_time + (i + 1) * self._config.energy_frequency
            is_burnt_in = cycle_time > self._config.adaptive_burn_in
            if is_adaptive and is_burnt_in:
                if (
                    cycle_time - self._config.energy_frequency
                    <= self._config.adaptive_burn_in
                ):
                    _logger.info("Adaptive replica exchange burn-in complete")
                self._record_adaptive_data(energy_matrix, active)

            # Mix the replicas that were propagated.
            _logger.info("Mixing replicas")
            self._dynamics_cache.set_states(
                _mix_subset(
                    _np.array(active, dtype=_np.int64),
                    energy_matrix,
                    self._dynamics_cache.get_proposed(),
                    self._dynamics_cache.get_accepted(),
                )
            )

            # This only permutes the stored states. They are pushed into the
            # contexts by load_replica() at the start of the next block, which
            # is also where the pre-run state for crash recovery is captured.
            self._dynamics_cache.mix_states()

            # Checkpoint. This happens once the whole cycle is complete, with
            # every checkpoint file written under a single lock, so that an
            # external process reading the output directory always sees a
            # coherent set rather than a mixture of new and old files.
            if do_checkpoint:
                # Rank the windows before the state is saved, so that a restart
                # carries on with the same hot replicas.
                if is_adaptive and is_checkpoint and is_burnt_in:
                    cold_rate = self._cold_rate(
                        i + 1,
                        cycles,
                        checkpoint_cycles,
                        self._config.adaptive_mix_interval,
                    )
                    self._rank_windows(
                        replica_devices,
                        device_capacity,
                        self._window_samples(),
                        cycles - (i + 1),
                        cold_rate,
                    )

                # Create the lock.
                lock = _FileLock(self._lock_file)

                # Acquire the file lock to ensure that the checkpoint files are
                # in a consistent state if read by another process.
                with lock.acquire(timeout=self._config.timeout.to("seconds")):
                    # First backup existing checkpoint files.
                    self._map_replicas(
                        checkpoint_executor,
                        "Backup",
                        self._backup_checkpoint,
                        replica_list,
                    )

                    # Now write the new checkpoint files.
                    self._map_replicas(
                        checkpoint_executor,
                        "Checkpoint",
                        self._checkpoint,
                        replica_list,
                        repeat(self._lambda_values),
                        repeat(block),
                        repeat(num_blocks + int(rem > 0)),
                        repeat(i == cycles - 1),
                        num_samples,
                    )

                    num_samples = [0] * num_replicas

                    # Replicas' own clocks fall behind while they are cold, so
                    # the simulation time is that of the cycle.
                    self._dynamics_cache._time = cycle_time

                    self._save_repex_state(final=not is_checkpoint)

            # This is a checkpoint cycle.
            if is_checkpoint:
                # Update the block number.
                block += 1

        # Record the end time for the production block.
        prod_end = time()

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

        # Write the sentinel file to signal that the run completed fully,
        # including trajectory consolidation and cleanup.
        self._done_file.touch()

    def _run_block(
        self,
        index,
        lambdas,
        is_gcmc=False,
        write_gcmc_ghosts=False,
        is_terminal_flip=False,
        save_energy_components=False,
        clock=None,
        save_frame=None,
    ):
        """
        Run a dynamics block for a given replica.

        The replica is made resident in its slot, propagated, then stored back
        out again so that the slot can be handed to the next replica. When
        there is one slot per replica, loading and storing reduce to the state
        bookkeeping that replica exchange does anyway.

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

        is_terminal_flip: bool
            Whether a terminal flip MC move should be performed before the
            dynamics block.

        save_energy_components: bool
            Whether to save the energy contribution of each force.

        clock: dict
            The simulation clock at the start of the cycle, as returned by
            Dynamics._get_clock(). Only used if the replica doesn't yet have a
            clock of its own.

        save_frame: bool
            Whether to save a frame at the end of the block. If None, frames
            are saved every 'frame_frequency' of the replica's own clock.

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
            # Make the replica resident in its slot, restoring its own clock so
            # that its energies and frames are never written over.
            own_clock = self._dynamics_cache.get_clock(index)
            self._dynamics_cache.load_replica(
                index, clock=clock if own_clock is None else own_clock
            )

            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(
                self._dynamics_cache.slot_for(index)
            )

            # Schedule the next frame for the end of this block, or never.
            if save_frame is not None:
                steps = max(
                    1,
                    int(
                        self._config.energy_frequency.to("ps")
                        / dynamics.timestep().to("ps")
                    ),
                )
                dynamics._d._next_save_frame = (
                    dynamics.current_step() + steps if save_frame else _sys.maxsize
                )
            # Let Sire reschedule frames if adaptive scheduling was switched off.
            elif dynamics._d._next_save_frame == _sys.maxsize:
                dynamics._d._next_save_frame = None

            auto_fix_minimise = self._config.auto_fix_minimise

            # Perform the GCMC move before dynamics so that the energies
            # computed during dynamics are consistent with the state used
            # for replica exchange mixing.
            if gcmc_sampler is not None and is_gcmc:
                gcmc_sampler.push()
                try:
                    _logger.info(f"Performing GCMC move at {_lam_sym} = {lam:.5f}")
                    gcmc_sampler.move(dynamics.context())
                finally:
                    gcmc_sampler.pop()

                # Record ghost residues immediately after the GCMC move so the
                # ghost state and frame (saved during dynamics) are consistent.
                if write_gcmc_ghosts:
                    self._save_ghost_residues(index, gcmc_sampler)

            # Perform a terminal flip move before dynamics if requested.
            if self._terminal_flip_samplers is not None and is_terminal_flip:
                _logger.info(f"Performing terminal flip move at {_lam_sym} = {lam:.5f}")
                self._terminal_flip_samplers[index].move(dynamics.context())

            # Snapshot the context state for crash recovery. The slot was just
            # seeded with this replica's state, and any MC move above may have
            # changed it again, so this is always required.
            if auto_fix_minimise:
                dynamics._d._pre_run_state = dynamics.context().getState(
                    getPositions=True, getVelocities=True
                )

            _logger.info(f"Running dynamics at {_lam_sym} = {lam:.5f}")

            # Draw new velocities from the Maxwell-Boltzmann distribution.
            if self._config.randomise_velocities:
                dynamics.randomise_velocities()

            # Run the dynamics.
            dynamics.run(
                self._config.energy_frequency,
                energy_frequency=self._config.energy_frequency,
                frame_frequency=self._config.frame_frequency,
                lambda_windows=lambdas,
                rest2_scale_factors=self._rest2_scale_factors,
                save_velocities=self._config.save_velocities,
                auto_fix_minimise=self._config.auto_fix_minimise,
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

            # Save the replica's state back out of the slot, so that the slot
            # can be handed to the next replica.
            self._dynamics_cache.store_replica(index)

            # Save the energy contribution for each force.
            if save_energy_components:
                self._save_energy_components(
                    index, dynamics.context(), dynamics.current_time().to("ns")
                )

            # Get the energy at each lambda value.
            energies = dynamics._current_energy_array()

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
            # Make the replica resident in its slot.
            self._dynamics_cache.load_replica(index)

            # Get the dynamics object (and GCMC sampler).
            slot = self._dynamics_cache.slot_for(index)
            dynamics, gcmc_sampler = self._dynamics_cache.get(slot)

            if gcmc_sampler is not None and not self._is_restart:
                gcmc_sampler.push()
                try:
                    _logger.info(
                        f"Pre-equilibrating with GCMC moves at {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )
                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())
                finally:
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
                    self._dynamics_cache.delete(slot)

                    # Work out the device index.
                    device = self._dynamics_cache.device_for(slot)

                    # Copy the dynamics keyword arguments.
                    dynamics_kwargs = self._dynamics_kwargs.copy()

                    # Overload the device and lambda value.
                    dynamics_kwargs["device"] = device
                    dynamics_kwargs["lambda_value"] = self._build_lambda(index)
                    dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]

                    # Create the production dynamics object.
                    dynamics = system.dynamics(**dynamics_kwargs)

                    # Reset the GCMC water state. The dynamics object is created from
                    # the original Sire system, so the water state in the context does
                    # not match the current GCMC water state.
                    if gcmc_sampler is not None:
                        self._reset_gcmc_sampler(gcmc_sampler, dynamics)

                    # Set the new dynamics object. The rebuilt object has a
                    # fresh, empty energy trajectory, but no energies are
                    # recorded during minimisation or equilibration, and
                    # load_replica() re-attaches the replica's trajectory
                    # before the first production block.
                    self._dynamics_cache.set(slot, dynamics)

                    _logger.info(
                        f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )

            # Save the minimised state back out of the slot.
            self._dynamics_cache.store_replica(index)

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
            # Make the replica resident in its slot.
            self._dynamics_cache.load_replica(index)

            # Get the dynamics object (and GCMC sampler).
            slot = self._dynamics_cache.slot_for(index)
            dynamics, gcmc_sampler = self._dynamics_cache.get(slot)

            if gcmc_sampler is not None:
                gcmc_sampler.push()
                try:
                    _logger.info(
                        f"Equilibrating with GCMC moves at {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )
                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())
                finally:
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
                    self._dynamics_cache.delete(slot)

                    # Work out the device index.
                    device = self._dynamics_cache.device_for(slot)

                    # Copy the dynamics keyword arguments.
                    dynamics_kwargs = self._dynamics_kwargs.copy()

                    # Overload the device and lambda value.
                    dynamics_kwargs["device"] = device
                    dynamics_kwargs["lambda_value"] = self._build_lambda(index)
                    dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]
                    dynamics_kwargs["timestep"] = self._config._equilibration_timestep
                    dynamics_kwargs["constraint"] = constraint
                    dynamics_kwargs["perturbable_constraint"] = perturbable_constraint

                    # Create the new dynamics object.
                    dynamics = system.dynamics(**dynamics_kwargs)

                    # Reset the GCMC water state.
                    if gcmc_sampler is not None:
                        self._reset_gcmc_sampler(gcmc_sampler, dynamics)

                    # Update the dynamics object in the cache.
                    self._dynamics_cache.set(slot, dynamics)

            # Equilibrate.
            dynamics.run(
                self._config.equilibration_time,
                energy_frequency=0,
                frame_frequency=0,
                save_velocities=False,
                auto_fix_minimise=self._config.auto_fix_minimise,
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
            self._dynamics_cache.delete(slot)

            # Work out the device index.
            device = self._dynamics_cache.device_for(slot)

            # Copy the dynamics keyword arguments.
            dynamics_kwargs = self._dynamics_kwargs.copy()

            # Overload the device and lambda value.
            dynamics_kwargs["device"] = device
            dynamics_kwargs["lambda_value"] = self._build_lambda(index)
            dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]

            # Create the production dynamics object.
            dynamics = system.dynamics(**dynamics_kwargs)

            # Reset the GCMC water state. The dynamics object is created from
            # the original Sire system, so the water state in the context does
            # not match the current GCMC water state.
            if gcmc_sampler is not None:
                self._reset_gcmc_sampler(gcmc_sampler, dynamics)

            # Set the new dynamics object.
            self._dynamics_cache.set(slot, dynamics)

            _logger.info(
                f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
            )

            # Save the equilibrated state back out of the slot.
            self._dynamics_cache.store_replica(index)

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
        dynamics, _ = self._dynamics_cache.get(self._dynamics_cache.slot_for(index))

        # Create an array to hold the energies.
        energies = _np.zeros(self._config.num_lambda)

        # Loop over the states.
        for i in range(self._config.num_lambda):
            # Set the state.
            DynamicsCache._apply_openmm_state(
                dynamics.context(), self._dynamics_cache._openmm_states[i]
            )
            dynamics._d._clear_state()

            # Compute and store the energy for this state.
            energies[i] = dynamics.current_potential_energy().value()

        # Reset the state.
        DynamicsCache._apply_openmm_state(
            dynamics.context(), self._dynamics_cache._openmm_states[index]
        )

        return index, energies

    def _assemble_results(self, results):
        """
        Assemble the results into a matrix.

        Parameters
        ----------

        results: list
            The results from the repex dynamics block.
        """
        # Create the matrix. Rows of replicas that weren't propagated are left
        # empty, since they aren't mixed.
        num_replicas = self._config.num_lambda
        matrix = _np.zeros((num_replicas, num_replicas))

        # Fill the matrix. The energy returned by the dynamics block already
        # includes the pressure and grand canonical contributions.
        for i, energies in results:
            for j, energy in enumerate(energies):
                matrix[i, j] = self._beta * energy

        return matrix

    @staticmethod
    def _checkpoint_cycles(num_cycles, cycles_per_checkpoint):
        """
        The cycles that end with a checkpoint: the first to reach each multiple
        of the number of cycles between checkpoints, which may not be a whole
        number.

        Parameters
        ----------

        num_cycles: int
            The number of cycles in the run.

        cycles_per_checkpoint: float
            The number of cycles between checkpoints.

        Returns
        -------

        np.ndarray
            The checkpoint cycles, numbered from one, in increasing order.
        """
        if num_cycles <= 0 or cycles_per_checkpoint <= 0:
            return _np.array([], dtype=int)

        num_checkpoints = int(num_cycles / cycles_per_checkpoint) + 1
        multiples = _np.arange(1, num_checkpoints + 1) * cycles_per_checkpoint
        cycles = _np.ceil(multiples - 1e-10).astype(int)
        return cycles[cycles <= num_cycles]

    @staticmethod
    def _cold_rate(first_cycle, num_cycles, checkpoint_cycles, mix_interval):
        """
        The share of the cycles left that propagate every window, and so the
        cold ones: the mix and checkpoint cycles, and the last cycle.

        Parameters
        ----------

        first_cycle: int
            The number of cycles run so far.

        num_cycles: int
            The number of cycles in the run.

        checkpoint_cycles: np.ndarray
            The checkpoint cycles, numbered from one, in increasing order.

        mix_interval: int
            The number of cycles between mix cycles.

        Returns
        -------

        float
            The share of cycles.
        """
        if first_cycle >= num_cycles:
            return 1.0

        # Mix cycles, then checkpoint cycles that aren't also mix cycles.
        full = num_cycles // mix_interval - first_cycle // mix_interval
        start, end = _np.searchsorted(
            checkpoint_cycles, [first_cycle, num_cycles], side="right"
        )
        checkpoints = checkpoint_cycles[start:end]
        full += int((checkpoints % mix_interval != 0).sum())

        # The last cycle always propagates every window.
        if num_cycles % mix_interval != 0 and (
            len(checkpoints) == 0 or checkpoints[-1] != num_cycles
        ):
            full += 1

        return full / (num_cycles - first_cycle)

    def _replica_devices(self):
        """
        The GPU that propagates each replica, which is that of its slot.

        Returns
        -------

        list of int
            The device index for each replica.
        """
        cache = self._dynamics_cache
        return [cache.device_for(cache.slot_for(r)) for r in range(self._num_replicas)]

    def _device_capacity(self):
        """
        The number of replicas each GPU can propagate at once: the
        oversubscription factor, or fewer if the GPU has fewer contexts.

        Returns
        -------

        list of int
            The capacity of each device.
        """
        return [
            max(
                1,
                min(
                    self._config.oversubscription_factor,
                    self._dynamics_cache.contexts_on(d),
                ),
            )
            for d in range(self._num_gpus)
        ]

    def _device_batches(self, replicas, devices):
        """
        Split replicas into batches that are propagated concurrently, each with
        at most 'oversubscription_factor' replicas on any GPU, so that every
        batch spreads its work over the GPUs.

        Two replicas that share a slot can't be propagated at once, so the
        replicas must come from a single pass, i.e. at most one per slot.

        Parameters
        ----------

        replicas: list of int
            The replicas to propagate, at most one per slot.

        devices: list of int
            The GPU that propagates each replica.

        Returns
        -------

        list of list of int
            The batches.
        """
        per_device = self._config.oversubscription_factor

        queues = {}
        for r in replicas:
            queues.setdefault(devices[r], []).append(r)

        batches = []
        while any(queues.values()):
            batch = []
            for queue in queues.values():
                batch += queue[:per_device]
                del queue[:per_device]
            batches.append(sorted(batch))
        return batches

    def _dynamics_batches(self, devices):
        """
        Yield batches of every replica that can be propagated concurrently: at
        most one per slot, and at most 'oversubscription_factor' per GPU.

        Parameters
        ----------

        devices: list of int
            The GPU that propagates each replica.

        Yields
        ------

        list of int
            A batch of replicas.
        """
        for batch in self._replica_passes(0):
            yield from self._device_batches(batch, devices)

    def _window_samples(self):
        """
        The number of energy samples each window has collected, from its clock.

        Returns
        -------

        list of int
            The number of samples for each window.
        """
        energy_frequency = self._config.energy_frequency.to("ps")
        return [
            (
                0
                if clock is None
                else round(clock["_current_time"].to("ps") / energy_frequency)
            )
            for clock in self._dynamics_cache._clocks
        ]

    def _check_adaptive_config(self):
        """
        Warn about settings that stop adaptive replica exchange having an
        effect, before any time is spent on minimisation and equilibration.
        """
        from math import floor

        checkpoint = self._config.checkpoint_frequency.to("ps")
        burn_in = self._config.adaptive_burn_in.to("ps")
        end_time = (self._start_time + self._config.runtime).to("ps")

        if checkpoint == 0.0:
            _logger.warning(
                "Adaptive replica exchange ranks the windows at checkpoints, but "
                "'checkpoint_frequency' is zero. Every window will be propagated "
                "every cycle."
            )
        # Windows first go cold at the second checkpoint after the burn-in.
        elif (floor(burn_in / checkpoint + 1e-9) + 2) * checkpoint >= end_time - 1e-6:
            _logger.warning(
                "There are fewer than two checkpoints after 'adaptive_burn_in' "
                "and before the end of the run, so no window will stop being "
                "hot. Every window will be propagated every cycle."
            )
        elif (
            checkpoint / self._config.energy_frequency.to("ps")
            <= self._config.adaptive_mix_interval
        ):
            _logger.warning(
                "Every window is propagated on checkpoint cycles, which are no "
                "further apart than 'adaptive_mix_interval'. Adaptive replica "
                "exchange will have little effect."
            )

    def _find_ring_pair(self):
        """
        Find the OpenMM particle indices of the bond that is broken or made by
        a ring-breaking perturbation.

        Returns
        -------

        tuple of int
            The particle indices, or None unless exactly one bond is broken or
            made, since the distance of a single bond is used.
        """
        atoms = self._system.atoms()
        pairs = []

        for mol in self._system["property is_perturbable"].molecules():
            for prop in ("ring_breaking_bonds", "ring_making_bonds"):
                if mol.has_property(prop):
                    flat = mol.property(prop).to_list()
                    for i in range(0, len(flat) - 1, 2):
                        pairs.append(
                            tuple(
                                atoms.find(mol.atoms()[int(idx)])
                                for idx in flat[i : i + 2]
                            )
                        )

        if len(pairs) != 1:
            if len(pairs) > 1:
                _logger.info(
                    f"{len(pairs)} bonds are broken or made, so the bond distance "
                    "is not used to rank windows for adaptive replica exchange."
                )
            return None

        # Virtual sites are added after the atoms of their molecule, so shift
        # the particles of any later molecule along.
        first = atoms.find(atoms[pairs[0][0]].molecule().atoms()[0])
        offset = 0
        if self._virtual_sites is not None:
            offset = int(sum(n for start, n in self._virtual_sites if start < first))

        pair = tuple(int(idx) + offset for idx in pairs[0])
        _logger.info(
            f"Ranking windows for adaptive replica exchange using the distance "
            f"between particles {pair[0]} and {pair[1]}"
        )

        return pair

    @staticmethod
    def _is_due(last_cycle, cycle, cycles_per_move):
        """
        Whether a periodic move falls due on a cycle, or fell due on one of
        the cycles since a replica was last propagated.

        Parameters
        ----------

        last_cycle: int
            The cycle on which the replica was last propagated, or -1.

        cycle: int
            The current cycle.

        cycles_per_move: int
            The number of cycles between moves.

        Returns
        -------

        bool
            Whether the move is due.
        """
        return (cycle + 1) // cycles_per_move > (last_cycle + 1) // cycles_per_move

    def _reset_adaptive_data(self):
        """
        Clear the data used to rank the windows for adaptive replica exchange.
        """
        num_replicas = self._config.num_lambda

        # The reduced energy difference to the lower and upper neighbouring
        # window, sampled in each window.
        self._adaptive_lower = [[] for _ in range(num_replicas)]
        self._adaptive_upper = [[] for _ in range(num_replicas)]

        # The distance of the bond that is broken or made, sampled in each
        # window.
        self._adaptive_distance = [[] for _ in range(num_replicas)]

        # The swap counts when the data was cleared.
        self._adaptive_proposed = self._dynamics_cache.get_proposed().copy()
        self._adaptive_accepted = self._dynamics_cache.get_accepted().copy()

    def _store_adaptive_data(self):
        """
        Store the data used to rank the windows in the dynamics cache, so that
        it is kept on restart.
        """
        self._dynamics_cache._adaptive_data = {
            "lower": [_np.array(x) for x in self._adaptive_lower],
            "upper": [_np.array(x) for x in self._adaptive_upper],
            "distance": [_np.array(x) for x in self._adaptive_distance],
            "proposed": self._adaptive_proposed,
            "accepted": self._adaptive_accepted,
        }

    def _restore_adaptive_data(self):
        """
        Restore the data used to rank the windows from the dynamics cache.
        """
        data = self._dynamics_cache._adaptive_data
        if data is None:
            return

        self._adaptive_lower = [list(x) for x in data["lower"]]
        self._adaptive_upper = [list(x) for x in data["upper"]]
        self._adaptive_distance = [list(x) for x in data["distance"]]
        self._adaptive_proposed = data["proposed"]
        self._adaptive_accepted = data["accepted"]

    def _record_adaptive_data(self, energy_matrix, replicas):
        """
        Record the reduced energy differences between neighbouring windows,
        and the distance of the bond that is broken or made, if any. Must be
        called before the replicas are mixed.

        Parameters
        ----------

        energy_matrix: np.ndarray
            The reduced energy of each replica at each window.

        replicas: list of int
            The replicas that were propagated.
        """
        num_replicas = self._config.num_lambda

        for k in replicas:
            row = energy_matrix[k]
            if k > 0:
                self._adaptive_lower[k].append(row[k - 1] - row[k])
            if k < num_replicas - 1:
                self._adaptive_upper[k].append(row[k + 1] - row[k])

            if self._ring_pair is not None:
                import openmm.unit as _omm_unit

                positions = DynamicsCache._get_positions(
                    self._dynamics_cache._openmm_states[k]
                )
                i, j = self._ring_pair
                pair = _np.asarray(positions[[i, j]].value_in_unit(_omm_unit.angstrom))
                self._adaptive_distance[k].append(
                    float(_np.linalg.norm(pair[1] - pair[0]))
                )

    @staticmethod
    def _statistical_inefficiency(x):
        """
        Estimate the statistical inefficiency of a time series, i.e. the number
        of samples per independent sample.

        Parameters
        ----------

        x: np.ndarray
            The time series.

        Returns
        -------

        float
            The statistical inefficiency.
        """
        n = len(x)
        if n < 2:
            return 1.0

        dx = x - x.mean()
        var = dx.dot(dx) / n
        if var == 0.0:
            return 1.0

        # The unnormalised autocovariance at every lag.
        f = _np.fft.rfft(dx, 2 * n)
        acov = _np.fft.irfft(f * _np.conj(f))[:n]

        # Sum the autocorrelation function up to its first non-positive value.
        g = 1.0
        for t in range(1, n):
            c = acov[t] / ((n - t) * var)
            if c <= 0.0:
                break
            g += 2.0 * c * (1.0 - t / n)

        return max(1.0, g)

    @staticmethod
    def _exp_estimate(du):
        """
        The free energy difference from exponential averaging of reduced
        energy differences.

        Parameters
        ----------

        du: np.ndarray
            The reduced energy differences.

        Returns
        -------

        float
            The reduced free energy difference.
        """
        m = (-du).max()
        return -(m + _np.log(_np.mean(_np.exp(-du - m))))

    @classmethod
    def _is_unconverged(cls, fwd, rev, g_fwd, g_rev):
        """
        Whether the free energy difference between a pair of neighbouring
        windows shows signs of poor sampling, regardless of its uncertainty.

        Parameters
        ----------

        fwd: np.ndarray
            The reduced energy difference to the upper window, sampled in the
            lower window.

        rev: np.ndarray
            The reduced energy difference to the lower window, sampled in the
            upper window.

        g_fwd, g_rev: float
            The statistical inefficiencies of fwd and rev.

        Returns
        -------

        bool
            Whether the pair is unconverged.
        """
        # The forward and reverse estimates disagree by more than 1 kT.
        if abs(cls._exp_estimate(fwd) + cls._exp_estimate(rev)) > 1.0:
            return True

        # The estimates from the two halves of the data differ by more than
        # 0.25 kT, and by more than twice their combined standard error.
        estimates = []
        errors = []
        for half in range(2):
            f = _np.array_split(fwd, 2)[half]
            r = _np.array_split(rev, 2)[half]
            estimates.append(0.5 * (cls._exp_estimate(f) - cls._exp_estimate(r)))
            errors.append(
                0.5 * _np.sqrt(f.var() * g_fwd / len(f) + r.var() * g_rev / len(r))
            )

        drift = abs(estimates[0] - estimates[1])
        return drift > 0.25 and drift > 2.0 * _np.hypot(*errors)

    def _rank_windows(
        self, window_devices, device_capacity, window_samples, cycles_left, cold_rate
    ):
        """
        Rank the windows by the uncertainty they contribute to the free energy
        and choose the hot replicas for adaptive replica exchange.

        Each pair of neighbouring windows is weighted by sigma * sqrt(g) of
        its reduced energy differences, which sets the uncertainty of its free
        energy difference for a fixed amount of sampling. Pairs with signs of
        poor sampling are always hot, as are windows with fewer than
        'adaptive_min_samples' independent samples of either the energy
        differences or, for a perturbation that breaks or makes a single bond,
        the bond distance. The remaining pairs are added in order of weight,
        with the windows either side of each so that exchange carries
        configurations into and out of them. With a context per replica, each
        set this gives is filled with the cold windows owning the most
        variance, until every GPU runs as many full batches as the busiest
        needs, since those windows take no extra time. A GPU's batches hold as
        many replicas as it can propagate at once. With shared contexts the set
        isn't filled, since an added window can need another pass. The set is
        then scored by the variance of the final free energy, given the samples
        each window has and will collect in the cycles left, multiplied by the
        compute of the whole run. The set with the lowest score is chosen. A
        window only stops being hot once it has been found cold at two
        consecutive rankings. Every window is hot if no set, including the
        windows kept hot, is better than that.

        The ranking uses all of the data since the burn-in, since the slowest
        windows decorrelate over far longer than a checkpoint interval.

        Parameters
        ----------

        window_devices: list of int
            The GPU that propagates each window.

        device_capacity: list of int
            The number of replicas each GPU can propagate at once.

        window_samples: list of int
            The number of energy samples each window has collected.

        cycles_left: int
            The number of cycles left in the run.

        cold_rate: float
            The share of the cycles left that propagate every window, and so
            the cold ones.
        """
        from math import ceil

        # There is nothing left to allocate, so keep the current hot replicas.
        if cycles_left <= 0:
            _logger.info("Not ranking windows: no cycles left, keeping hot replicas")
            return

        num_replicas = self._config.num_lambda
        lower = [_np.array(x) for x in self._adaptive_lower]
        upper = [_np.array(x) for x in self._adaptive_upper]

        min_samples = self._config.adaptive_min_samples

        # Every window needs enough samples to estimate its fluctuations.
        num_samples = min(len(x) for x in lower[1:] + upper[:-1])
        if num_samples < min_samples:
            _logger.info(f"Not ranking windows: {num_samples} samples of {min_samples}")
            return

        g_lower = [self._statistical_inefficiency(x) for x in lower]
        g_upper = [self._statistical_inefficiency(x) for x in upper]

        # Windows with too few independent samples can't be made cold.
        undersampled = []
        for k in range(num_replicas):
            series = []
            if k > 0:
                series.append((lower[k], g_lower[k]))
            if k < num_replicas - 1:
                series.append((upper[k], g_upper[k]))
            if min(len(x) / g for x, g in series) < min_samples:
                undersampled.append(k)

        # Nor can windows with a slow bond distance, which can bias the free
        # energy without showing in the energy differences.
        slow_distance = []
        if self._ring_pair is not None:
            for k, x in enumerate(self._adaptive_distance):
                x = _np.array(x)
                if len(x) / self._statistical_inefficiency(x) < min_samples:
                    slow_distance.append(k)

        proposed = self._dynamics_cache.get_proposed() - self._adaptive_proposed
        accepted = self._dynamics_cache.get_accepted() - self._adaptive_accepted

        # The variance per sample of each pair's free energy, from the samples
        # in its lower (forward) and upper (reverse) windows.
        fwd_variance = _np.zeros(num_replicas - 1)
        rev_variance = _np.zeros(num_replicas - 1)
        flagged = []
        for k in range(num_replicas - 1):
            fwd = upper[k]
            rev = lower[k + 1]
            fwd_variance[k] = fwd.var() * g_upper[k]
            rev_variance[k] = rev.var() * g_lower[k + 1]

            # Flag pairs that rarely exchange, or whose estimates disagree.
            num_proposed = proposed[k, k + 1]
            if (
                num_proposed > 0 and accepted[k, k + 1] / num_proposed < 0.05
            ) or self._is_unconverged(fwd, rev, g_upper[k], g_lower[k + 1]):
                flagged.append(k)

        weights = _np.sqrt(fwd_variance + rev_variance)

        forced = set(undersampled) | set(slow_distance)
        for k in flagged:
            forced.update((k, k + 1))

        def pad(windows):
            return windows | {
                r + d for r in windows for d in (-1, 1) if 0 <= r + d < num_replicas
            }

        all_hot = set(range(num_replicas))

        # The variance owned by each window: the forward side of the pair above
        # it and the reverse side of the pair below it.
        owned = _np.zeros(num_replicas)
        owned[:-1] += fwd_variance
        owned[1:] += rev_variance

        def fill(windows):
            # Make the cold windows owning the most variance hot, until each GPU
            # fills as many batches as the busiest needs. With shared contexts
            # an added window can need another pass, so it isn't free.
            if self._is_cached:
                return set(windows)
            counts = {d: 0 for d in window_devices}
            for r in windows:
                counts[window_devices[r]] += 1
            batches = max(ceil(n / device_capacity[d]) for d, n in counts.items())
            filled = set(windows)
            for r in sorted(all_hot - windows, key=lambda r: -owned[r]):
                d = window_devices[r]
                if counts[d] < batches * device_capacity[d]:
                    filled.add(r)
                    counts[d] += 1
            return filled

        total_variance = owned.sum()
        samples = _np.maximum(_np.asarray(window_samples, dtype=float), 1.0)

        def rates(windows):
            # The samples per cycle of each window for the rest of the run.
            is_hot = _np.zeros(num_replicas, dtype=bool)
            is_hot[list(windows)] = True
            return _np.where(is_hot, 1.0, cold_rate)

        def score(windows):
            # The compute of the whole run, in blocks of dynamics, and the
            # variance of the final free energy, from the samples at its end.
            rate = rates(windows)
            final = samples + rate * cycles_left
            cost = samples.sum() + rate.sum() * cycles_left
            variance = (fwd_variance / final[:-1] + rev_variance / final[1:]).sum()
            return cost, variance

        # A set must beat all hot by more than rounding error.
        all_hot_cost, all_hot_variance = score(all_hot)
        threshold = all_hot_cost * all_hot_variance * (1 - 1e-9)

        # Add pairs in order of weight, keeping the set with the lowest final
        # variance multiplied by the compute of the whole run, if better than
        # all hot. Sets are scored with the windows that will fill the workers.
        candidate = set(forced)
        chosen = None
        best = threshold
        order = list(_np.argsort(-weights)) if total_variance > 0 else []
        for k in [None] + order:
            if k is not None:
                if k in candidate and k + 1 in candidate:
                    continue
                candidate.update((int(k), int(k) + 1))
            windows = pad(candidate)
            if not windows:
                continue
            efficiency = _np.prod(score(fill(windows)))
            if efficiency < best:
                chosen, best = windows, efficiency

        votes = self._dynamics_cache._cold_votes
        if chosen is None:
            hot = all_hot
            votes[:] = [0] * num_replicas
        else:
            # Keep windows hot until they have been found cold twice in a row.
            previous = self._dynamics_cache._hot_replicas
            previous = all_hot if previous is None else set(previous)
            hot = set(chosen)
            for r in range(num_replicas):
                if r in chosen:
                    votes[r] = 0
                elif r in previous:
                    votes[r] += 1
                    if votes[r] < 2:
                        hot.add(r)

            # Windows kept hot can leave no gain over all hot.
            hot = fill(hot)
            if _np.prod(score(hot)) >= threshold:
                hot = all_hot

        hot_variance = score(hot)[1]
        hot = sorted(hot)
        self._dynamics_cache._hot_replicas = None if len(hot) == num_replicas else hot

        _logger.info(
            "Window pair weights (sigma * sqrt(g)): "
            + ", ".join(f"{w:.3f}" for w in weights)
        )
        if flagged:
            _logger.info(f"Unconverged window pairs: {flagged}")
        if undersampled:
            _logger.info(f"Undersampled windows: {undersampled}")
        if slow_distance:
            _logger.info(f"Windows with a slow bond distance: {slow_distance}")
        _logger.info(
            f"Hot replicas: {hot}" if len(hot) < num_replicas else "All replicas hot"
        )
        # The compute per cycle from now on and the final variance, relative to
        # every window being hot for the rest of the run.
        cost = rates(hot).sum() / num_replicas
        variance = hot_variance / all_hot_variance if all_hot_variance > 0 else 1.0
        _logger.info(
            f"Estimated cost per cycle {cost:.2f} and final variance "
            f"{variance:.2f} relative to every window being hot"
        )

    def _check_restart(self):
        """
        Check the output directory for a valid restart state.

        If per-replica checkpoint stream files (.s3) exist the base class is
        used to load them (old format, backwards compatible). Otherwise the
        repex state pickle is used and the original input system is returned
        directly, since positions and velocities come from the OpenMM states
        stored in the pickle.
        """
        from pathlib import Path as _Path_local

        checkpoint_path = _Path_local(self._filenames[0]["checkpoint"])
        if checkpoint_path.exists():
            _logger.info("Restarting from legacy stream file checkpoint.")
            return super()._check_restart()

        repex_state = self._config.output_directory / "repex_state.pkl"
        if not repex_state.exists():
            return False, self._system

        return True, self._system

    def _write_checkpoint_system(self, system, index, context=None, gcmc_sampler=None):
        """
        Record the current simulation time in the dynamics cache.

        For repex, per-replica stream files are not written. The simulation
        time is stored in the dynamics cache pickle instead, and positions and
        velocities are already stored as compact numpy arrays in the OpenMM
        state dict.
        """
        self._dynamics_cache._time = system.time()

    def _load_and_commit(self, index):
        """
        Make a replica resident in its slot and commit it, for the
        post-equilibration checkpoint.

        Equilibration stores every replica's state back out of its slot, so a
        replica has to be loaded again before it can be committed.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        index: int
            The index of the replica.

        exception: Exception
            The exception if the commit failed.
        """
        try:
            self._dynamics_cache.load_replica(index)
        except Exception as e:
            return index, e

        return self._commit_replica(index)

    def _commit_replica(self, index):
        """
        Commit a replica while it is still resident in its slot, holding the
        result until the checkpoint files are written at the end of the cycle.

        This is only used when contexts are shared between replicas. A slot
        holds the positions and trajectory frames of the replica it last
        hosted, so a replica has to be committed before its slot is handed on.
        Deferring only the file writes keeps every checkpoint file within a
        single lock, so that an external reader never sees a partial set.

        Parameters
        ----------

        index: int
            The index of the replica.

        Returns
        -------

        index: int
            The index of the replica.

        exception: Exception
            The exception if the commit failed.
        """
        try:
            slot = self._dynamics_cache.slot_for(index)
            dynamics, _ = self._dynamics_cache.get(slot)

            # commit() returns a clone, so the frames can be cleared straight
            # away, ready for the next replica to use the slot.
            system = dynamics.commit()
            speed = dynamics.time_speed()
            dynamics._d._sire_mols.delete_all_frames()

            with self._lock:
                self._committed[index] = (system, speed)

        except Exception as e:
            return index, e

        return index, None

    def _checkpoint(
        self,
        index,
        lambdas,
        block,
        num_blocks,
        is_final_block=False,
        num_energy_rows=None,
    ):
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

        num_energy_rows: int
            The number of energy samples the replica has collected since the
            last checkpoint. If None, a full block is assumed.

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
            slot = self._dynamics_cache.slot_for(index)
            dynamics, gcmc_sampler = self._dynamics_cache.get(slot)

            # Use the system committed by _commit_replica() if there is one.
            # When contexts are shared the replica is no longer resident in its
            # slot by the time the cycle finishes, so it was committed earlier,
            # while it still was.
            committed = self._committed.pop(index, None)

            if committed is None:
                # Commit the current system.
                system = dynamics.commit()

                # Get the simulation speed.
                speed = dynamics.time_speed()
            else:
                system, speed = committed

            # Call the base class checkpoint method to save the system state.
            with self._lock:
                index, error = super()._checkpoint(
                    system,
                    index,
                    block,
                    speed,
                    is_final_block=is_final_block,
                    num_energy_rows=num_energy_rows,
                )

                if error is not None:
                    return index, error

            # Delete the trajectory frames from the Sire system within the
            # dynamics object. A replica committed earlier, with shared
            # contexts, had them cleared then, so its slot isn't touched here.
            if committed is None:
                dynamics._d._sire_mols.delete_all_frames()

            if block == -1:
                _logger.info(
                    f"Writing post-equilibration checkpoint for {_lam_sym} = {lam:.5f}"
                )
            else:
                _logger.info(
                    f"Finished block {block + 1} of {self._start_block + num_blocks} "
                    f"for {_lam_sym} = {lam:.5f}"
                )

            # Log the number of waters within the GCMC sampling volume. Both
            # the water count and the statistics are read from what was
            # recorded while this replica was resident in its slot, since by
            # the time the cycle finishes the slot may hold another one.
            num_waters = self._dynamics_cache._gcmc_num_waters[index]
            if gcmc_sampler is not None and num_waters is not None:
                stats = gcmc_sampler.get_stats().get(gcmc_sampler.stats_key(lam))
                n_moves = stats["num_moves"] if stats is not None else 0
                acc_str = (
                    f", acceptance rate = {stats['num_accepted'] / n_moves:.3f}"
                    f" (ins = {stats['num_insertions']}, del = {stats['num_deletions']})"
                    if n_moves > 0
                    else ""
                )
                _logger.info(
                    f"Current number of waters in GCMC volume at {_lam_sym} = {lam:.5f} "
                    f"(time = {system.time().to('ns'):.4f} ns) is {num_waters}{acc_str}"
                )

            # Log terminal flip acceptance rate for this replica.
            if self._terminal_flip_samplers is not None:
                sampler = self._terminal_flip_samplers[index]
                _logger.info(
                    f"Terminal flip acceptance rate at {_lam_sym} = {lam:.5f}: "
                    f"{sampler.acceptance_rate:.3f} "
                    f"({sampler.num_accepted}/{sampler.num_attempted})"
                )

            if is_final_block:
                _logger.success(f"{_lam_sym} = {lam:.5f} complete")

            return index, None

        except Exception as e:
            return index, e

    def _consolidate_trajectories(self):
        """
        Consolidate any remaining trajectory chunk files into the final DCD.

        Called when a restart detects that dynamics completed but the process
        was killed before post-run cleanup finished. Safe to call when some
        replicas are already fully consolidated (no chunks left) — those are
        skipped automatically.
        """
        from glob import glob as _glob_local
        from pathlib import Path as _Path_local
        from shutil import copyfile as _copyfile_local

        if not self._config.save_trajectories:
            return

        for i in range(len(self._lambda_values)):
            traj_filename = self._filenames[i]["trajectory"]
            chunk_pattern = f"{self._filenames[i]['trajectory_chunk']}*"
            traj_chunks = sorted(_glob_local(chunk_pattern))

            # On a restart, prepend an existing final DCD as .prev so frames
            # from a previous (possibly partial) consolidation are preserved.
            path = _Path_local(traj_filename)
            if path.exists() and path.stat().st_size > 0:
                prev = f"{traj_filename}.prev"
                _copyfile_local(traj_filename, prev)
                traj_chunks = [prev] + traj_chunks

            if not traj_chunks:
                continue

            topology0 = self._filenames["topology0"]
            mols = _sr.load([topology0] + traj_chunks)
            _sr.save(mols.trajectory(), traj_filename, format=["DCD"])

            for chunk in traj_chunks:
                _Path_local(chunk).unlink()

    def _merge_gcmc_stats(self):
        """
        Merge the GCMC sampling statistics from every sampler.

        A sampler accumulates statistics for each lambda value it visits, so
        the results are gathered into a single dictionary keyed by lambda
        value. Samplers only report the lambda values they visit, so the keys
        are disjoint and the merge order doesn't matter.

        Returns
        -------

        dict
            The statistics for each lambda value, or None if not using GCMC.
        """
        stats = {}

        for slot in range(self._dynamics_cache._num_slots):
            _, gcmc_sampler = self._dynamics_cache.get(slot)
            if gcmc_sampler is not None:
                stats.update(gcmc_sampler.get_stats())

        return stats if stats else None

    def _save_repex_state(self, final=False):
        """
        Save the transition matrix and pickle the dynamics cache, backing up
        the previous pickle. Must be called with the file lock held, alongside
        the checkpoint files.

        Parameters
        ----------

        final: bool
            Whether this is the final save of the run, for logging.
        """
        label = "final replica exchange" if final else "replica exchange"

        _logger.info(f"Saving {label} transition matrix")
        self._save_transition_matrix()

        if self._repex_state.exists():
            _copyfile(
                self._repex_state,
                self._repex_state.with_suffix(".pkl.bak"),
            )

        _logger.info(f"Saving {label} state")
        self._save_sampler_stats()
        if self._config.adaptive_repex:
            self._store_adaptive_data()
        else:
            self._dynamics_cache._adaptive_data = None
        with open(self._repex_state, "wb") as f:
            _pickle.dump(self._dynamics_cache, f)

    def _save_sampler_stats(self):
        """
        Save GCMC and terminal flip sampler statistics to the dynamics cache
        prior to pickling.
        """
        self._dynamics_cache._gcmc_stats = self._merge_gcmc_stats()

        if self._terminal_flip_samplers is not None:
            self._dynamics_cache._terminal_flip_stats = [
                [s.num_attempted, s.num_accepted] for s in self._terminal_flip_samplers
            ]

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
                t_ij[i_state, i_state] = 1.0

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

        gcmc_sampler.push()
        try:
            # Set the water state.
            gcmc_sampler._set_water_state(dynamics.context(), force=True)
        finally:
            gcmc_sampler.pop()

        # Re-bind the GCMC sampler to the dynamics object.
        gcmc_sampler.bind_dynamics(dynamics)
