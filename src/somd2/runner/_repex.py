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
        perturbed_system=None,
        xml_filenames=None,
        num_slots=None,
        update_constraints=True,
        constraint_lambda_index=None,
        gpu_devices=None,
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
        """

        num_replicas = len(lambdas)

        if num_slots is None:
            num_slots = num_replicas

        # Warn if the number of slots is not a multiple of the number of GPUs.
        if num_slots > num_gpus and num_slots % num_gpus != 0:
            _logger.warning(
                "The number of contexts is not a multiple of the number of GPUs. "
                "This may result in suboptimal performance."
            )

        # Initialise attributes.
        self._lambdas = lambdas
        self._rest2_scale_factors = rest2_scale_factors
        self._num_replicas = num_replicas
        self._num_slots = num_slots
        self._update_constraints = update_constraints
        self._constraint_lambda_index = constraint_lambda_index
        self._gpu_devices = gpu_devices
        self._states = _np.array(range(num_replicas))
        self._time = None
        self._openmm_states = [None] * num_replicas
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
        self._terminal_flip_stats = [[0, 0]] * num_replicas
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

        The layout is derived from the number of replicas and slots, so it is
        rebuilt rather than stored in a checkpoint.
        """
        self._gcmc_samplers = [None] * self._num_slots
        self._slot_replica = [None] * self._num_slots

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
            self._terminal_flip_stats = [[0, 0]] * n
        if not hasattr(self, "_time"):
            self._time = None
        if not hasattr(self, "_num_replicas"):
            self._num_replicas = n
        if not hasattr(self, "_energy_trajectories"):
            self._energy_trajectories = [None] * n
        if not hasattr(self, "_ghost_files"):
            self._ghost_files = [None] * n
        self._gcmc_num_waters = [None] * n

        # The slot layout is not pickled, since it is rebuilt by
        # _create_dynamics() when the run is restarted. Older checkpoints
        # predate slots entirely, in which case there was one per replica.
        if not hasattr(self, "_num_slots"):
            self._num_slots = n
        if not hasattr(self, "_update_constraints"):
            self._update_constraints = True

        # Convert a legacy checkpoint to the current convention, in which the
        # stored state of a replica is its own, with the last mix already
        # applied.
        if is_legacy:
            self._openmm_states = [self._openmm_states[s] for s in self._states]
            self._gcmc_states = [self._gcmc_states[s] for s in self._states]

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
            "_openmm_states": self._openmm_states,
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

        # Work out how many slots are assigned to each device.
        # Slots are assigned round-robin, so the first (num_slots % num_gpus)
        # devices get one extra slot.
        base = floor(self._num_slots / num_gpus)
        remainder = self._num_slots % num_gpus
        contexts_per_device = [
            base + (1 if i < remainder else 0) for i in range(num_gpus)
        ]

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
            device = i % num_gpus

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

            # This is a restart, get the system for the seeding replica.
            if isinstance(system, list):
                mols = system[seed]
            # This is a new simulation. For lambda > 0.5, use the perturbed
            # system to seed the starting coordinates and periodic space.
            elif perturbed_system is not None and lambdas[middle] > 0.5:
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
                msg = f"Could not create dynamics object for lambda {lam:.5f} on device {device}: {e}"
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
                    msg = (
                        f"Not enough memory on device {device} for all assigned replicas. "
                        f"Baseline usage before simulation: {baseline / (1024**3):.2f} GB "
                        f"Estimated replica memory: {replica_cost / (1024**3):.2f} GB, "
                        f"Total estimated: {est_total / (1024**3):.2f} GB, "
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

    def get_clock(self):
        """
        Get the simulation clock.

        Every replica advances by the same amount each cycle, so the clock is
        common to all of them and is read from the first slot.

        Returns
        -------

        dict
            The clock, as returned by Dynamics._get_clock().
        """
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

    def mix_states(self, old_states):
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

        Parameters
        ----------

        old_states : numpy.ndarray
            The state indices from before the last replica mix.
        """
        # Permute the travelling state. This is a reference shuffle, so it is
        # cheap even for large systems. Statistics and output files stay with
        # the lambda window, so are not permuted.
        self._openmm_states = [self._openmm_states[state] for state in self._states]
        self._gcmc_states = [self._gcmc_states[state] for state in self._states]

        # Flag the replicas whose state moved, so that load_replica() knows it
        # has to push new positions and velocities into the context.
        self._state_moved = [bool(state != i) for i, state in enumerate(self._states)]

        # Update the swap matrix.
        for i, state in enumerate(self._states):
            self._num_swaps[old_states[i], state] += 1

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
                total = ocl_device.global_mem_size
                free_memory_info = ocl_device.get_info(0x4038)
                free_kb = (
                    free_memory_info[0]
                    if isinstance(free_memory_info, list)
                    else free_memory_info
                )
                free = free_kb * 1024
                used = total - free
                return (used, free, total)
            except Exception as e:
                msg = f"Could not get AMD GPU memory info for device {device}: {e}"
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

        # The physical devices backing each OpenMM device index. OpenMM numbers
        # devices relative to the visible set, so index i is gpu_devices[i].
        self._gpu_devices = list(gpu_devices)[: self._num_gpus]

        # Work out how many OpenMM contexts (slots) to create. When there are
        # fewer slots than replicas, each slot is re-used to propagate several
        # replicas per cycle, changing its lambda value as it goes.
        self._num_replicas = len(self._lambda_values)
        self._set_num_slots()

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
            )

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
        else:
            self._start_block = 0

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

    def _replica_passes(self, cycle):
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

        Returns
        -------

        list of list of int
            The replicas to propagate in each pass.
        """
        groups = self._dynamics_cache._groups

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

    def _safe_batches(self, num_workers, cycle=0):
        """
        Yield batches of replicas that can be processed concurrently.

        Two replicas that share a slot must never be processed at the same
        time, since they would be using the same dynamics object and OpenMM
        context. Batches are therefore taken from within a single pass, which
        holds at most one replica per slot, and then split by the number of
        workers.

        With one slot per replica there is a single pass containing every
        replica, so this is just a split by the number of workers.

        Parameters
        ----------

        num_workers: int
            The maximum number of replicas in a batch.

        cycle: int
            The index of the current cycle, which sets the traversal order.

        Yields
        ------

        list of int
            A batch of replicas that is safe to process concurrently.
        """
        from math import ceil

        for batch in self._replica_passes(cycle):
            for i in range(ceil(len(batch) / num_workers)):
                yield batch[i * num_workers : (i + 1) * num_workers]

    def _set_num_slots(self):
        """
        Work out the number of OpenMM contexts (slots) to create, validating
        the configuration options that only apply when contexts are re-used
        across lambda values.

        Sets self._num_slots and self._is_cached.
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
        self._constraint_lambda_index = None

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

        num_workers = self._num_gpus * self._config.oversubscription_factor

        if self._num_slots < num_workers:
            _logger.warning(
                f"'max_contexts' ({self._num_slots}) is less than the number of "
                f"workers ({num_workers}). Some GPUs will be left idle."
            )
        elif self._num_slots % self._num_gpus != 0:
            _logger.warning(
                f"'max_contexts' ({self._num_slots}) is not a multiple of the "
                f"number of GPUs ({self._num_gpus}). This may result in "
                "suboptimal performance."
            )

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
        num_checkpoint_batches = ceil(self._config.num_lambda / num_checkpoint_workers)

        # Persistent thread pools, reused across every batch and cycle.
        dynamics_executor = ThreadPoolExecutor(max_workers=num_workers)
        checkpoint_executor = ThreadPoolExecutor(max_workers=num_checkpoint_workers)

        # Create the replica list.
        replica_list = list(range(self._config.num_lambda))

        # Minimise at each lambda value.
        if self._config.minimise:
            for batch in self._safe_batches(num_workers):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for success, index, e in executor.map(
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
                        _logger.error("Minimisation cancelled. Exiting.")
                        _sys.exit(1)

        # Equilibrate the system.
        if self._is_equilibration and not self._is_restart:
            for batch in self._safe_batches(num_workers):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        for success, index, e in executor.map(
                            self._equilibrate,
                            batch,
                        ):
                            if not success:
                                _logger.error(
                                    f"Equilibration failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                                )
                                raise e
                    except KeyboardInterrupt:
                        _logger.error("Equilibration cancelled. Exiting.")
                        _sys.exit(1)

        # Write a checkpoint immediately after equilibration so that a restart
        # after an early production crash doesn't need to re-equilibrate.
        if self._is_equilibration and not self._is_restart:
            # When contexts are shared, commit each replica while it is
            # resident so that every checkpoint file can still be written
            # under a single lock below.
            if self._is_cached:
                for batch in self._safe_batches(num_checkpoint_workers):
                    try:
                        for index, error in checkpoint_executor.map(
                            self._load_and_commit,
                            batch,
                        ):
                            if error is not None:
                                msg = (
                                    f"Post-equilibration commit failed for {_lam_sym} = "
                                    f"{self._lambda_values[index]:.5f}:\n{error}"
                                )
                                _logger.error(msg)
                                raise error
                    except KeyboardInterrupt:
                        checkpoint_executor.shutdown(wait=False, cancel_futures=True)
                        _logger.error("Post-equilibration commit cancelled. Exiting.")
                        _sys.exit(1)

            lock = _FileLock(self._lock_file)
            with lock.acquire(timeout=self._config.timeout.to("seconds")):
                for j in range(num_checkpoint_batches):
                    replicas = replica_list[
                        j * num_checkpoint_workers : (j + 1) * num_checkpoint_workers
                    ]
                    try:
                        for index, error in checkpoint_executor.map(
                            self._checkpoint,
                            replicas,
                            repeat(self._lambda_values),
                            repeat(-1),
                            repeat(cycles),
                        ):
                            if error is not None:
                                msg = (
                                    f"Post-equilibration checkpoint failed for {_lam_sym} = "
                                    f"{self._lambda_values[index]:.5f}:\n{error}"
                                )
                                _logger.error(msg)
                                raise error
                    except KeyboardInterrupt:
                        checkpoint_executor.shutdown(wait=False, cancel_futures=True)
                        _logger.error(
                            "Post-equilibration checkpoint cancelled. Exiting."
                        )
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

        # Initialise the threshold for the next checkpoint cycle. This is a float
        # to handle non-integer ratios between the checkpoint and energy frequencies.
        next_checkpoint = cycles_per_checkpoint

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

            # Whether to checkpoint. Use a float threshold to correctly handle
            # non-integer ratios between the checkpoint and energy frequencies.
            is_checkpoint = (i + 1) >= next_checkpoint - 1e-10

            # Whether to perform a GCMC move before the dynamics block.
            is_gcmc = (i + 1) % cycles_per_gcmc == 0

            # Whether to perform a terminal flip move before the dynamics block.
            is_terminal_flip = (i + 1) % cycles_per_flip == 0

            # Whether a frame is saved at the end of the cycle.
            write_gcmc_ghosts = (i + 1) % cycles_per_frame == 0

            # Current simulation time in ns for energy components saving.
            time_ns = (
                (
                    self._start_block * checkpoint_frequency
                    + (i + 1) * self._config.energy_frequency
                ).to("ns")
                if self._config.save_energy_components
                else None
            )

            # Whether the checkpoint files are written at the end of this cycle.
            do_checkpoint = is_checkpoint or i == cycles - 1

            # Capture the simulation clock at the start of the cycle. Every
            # replica advances by the same amount each cycle, so a slot has to
            # be rewound to this point before it propagates the next replica.
            clock = self._dynamics_cache.get_clock()

            # Propagate the replicas, one pass at a time. Each pass takes at
            # most one replica from each slot, and there is a single pass when
            # there is a slot per replica.
            for batch in self._replica_passes(i):
                # Run a dynamics block for each replica in the pass, making sure
                # each GPU is only oversubscribed by a factor of
                # self._config.oversubscription_factor.
                for j in range(ceil(len(batch) / num_workers)):
                    replicas = batch[j * num_workers : (j + 1) * num_workers]
                    try:
                        for result, index, energies in dynamics_executor.map(
                            self._run_block,
                            replicas,
                            repeat(self._lambda_values),
                            repeat(is_gcmc),
                            repeat(write_gcmc_ghosts),
                            repeat(is_terminal_flip),
                            repeat(time_ns),
                            repeat(clock),
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
                    for j in range(ceil(len(batch) / num_checkpoint_workers)):
                        replicas = batch[
                            j * num_checkpoint_workers : (j + 1)
                            * num_checkpoint_workers
                        ]
                        try:
                            for index, error in checkpoint_executor.map(
                                self._commit_replica,
                                replicas,
                            ):
                                if error:
                                    _logger.error(
                                        f"Commit failed for {_lam_sym} = "
                                        f"{self._lambda_values[index]:.5f}: {error}"
                                    )
                                    raise error
                        except KeyboardInterrupt:
                            checkpoint_executor.shutdown(
                                wait=False, cancel_futures=True
                            )
                            _logger.error("Commit cancelled. Exiting.")
                            _sys.exit(1)

            # Checkpoint. This happens once the whole cycle is complete, with
            # every checkpoint file written under a single lock, so that an
            # external process reading the output directory always sees a
            # coherent set rather than a mixture of new and old files.
            if do_checkpoint:
                # Create the lock.
                lock = _FileLock(self._lock_file)

                # Acquire the file lock to ensure that the checkpoint files are
                # in a consistent state if read by another process.
                with lock.acquire(timeout=self._config.timeout.to("seconds")):
                    # First backup existing checkpoint files.
                    for j in range(num_checkpoint_batches):
                        # Get the indices of the replicas in this batch.
                        replicas = replica_list[
                            j * num_checkpoint_workers : (j + 1)
                            * num_checkpoint_workers
                        ]
                        try:
                            for index, error in checkpoint_executor.map(
                                self._backup_checkpoint,
                                replicas,
                            ):
                                if error:
                                    _logger.error(
                                        f"Backup failed for {_lam_sym} = "
                                        f"{self._lambda_values[index]:.5f}: {error}"
                                    )
                                    raise error
                        except KeyboardInterrupt:
                            checkpoint_executor.shutdown(
                                wait=False, cancel_futures=True
                            )
                            _logger.error("Backup cancelled. Exiting.")
                            _sys.exit(1)

                    # Now write the new checkpoint files.
                    for j in range(num_checkpoint_batches):
                        # Get the indices of the replicas in this batch.
                        replicas = replica_list[
                            j * num_checkpoint_workers : (j + 1)
                            * num_checkpoint_workers
                        ]
                        try:
                            for index, error in checkpoint_executor.map(
                                self._checkpoint,
                                replicas,
                                repeat(self._lambda_values),
                                repeat(block),
                                repeat(num_blocks + int(rem > 0)),
                                repeat(i == cycles - 1),
                            ):
                                if error:
                                    _logger.error(
                                        f"Checkpoint failed for {_lam_sym} = "
                                        f"{self._lambda_values[index]:.5f}: {error}"
                                    )
                                    raise error
                        except KeyboardInterrupt:
                            checkpoint_executor.shutdown(
                                wait=False, cancel_futures=True
                            )
                            _logger.error("Checkpoint cancelled. Exiting.")
                            _sys.exit(1)

            # Assemble an energy matrix from the results.
            _logger.info("Assembling energy matrix")
            energy_matrix = self._assemble_results(results)

            # Mix the replicas.
            _logger.info("Mixing replicas")
            old_states = self._dynamics_cache.get_states()
            self._dynamics_cache.set_states(
                self._mix_replicas(
                    self._config.num_lambda,
                    energy_matrix,
                    self._dynamics_cache.get_proposed(),
                    self._dynamics_cache.get_accepted(),
                )
            )

            # This only permutes the stored states. They are pushed into the
            # contexts by load_replica() at the start of the next block, which
            # is also where the pre-run state for crash recovery is captured.
            self._dynamics_cache.mix_states(old_states)

            # This is a checkpoint cycle.
            if is_checkpoint:
                # Update the block number.
                block += 1

                # Advance the checkpoint threshold.
                next_checkpoint += cycles_per_checkpoint

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
                    self._save_sampler_stats()
                    with open(self._repex_state, "wb") as f:
                        _pickle.dump(self._dynamics_cache, f)

        dynamics_executor.shutdown(wait=True)
        checkpoint_executor.shutdown(wait=True)

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
            if self._terminal_flip_samplers is not None:
                self._dynamics_cache._terminal_flip_stats = [
                    [s.num_attempted, s.num_accepted]
                    for s in self._terminal_flip_samplers
                ]
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
        time_ns=None,
        clock=None,
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

        time_ns: float or None
            The current simulation time in nanoseconds, used when saving energy
            components. If None, energy components are not saved.

        clock: dict
            The simulation clock at the start of the cycle, as returned by
            Dynamics._get_clock(). Every replica advances by the same amount
            each cycle, so a slot must be rewound to the start of the cycle
            before propagating the next replica.

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
            # Make the replica resident in its slot.
            self._dynamics_cache.load_replica(index, clock=clock)

            # Get the dynamics object (and GCMC sampler).
            dynamics, gcmc_sampler = self._dynamics_cache.get(
                self._dynamics_cache.slot_for(index)
            )

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

                # Write ghost residues immediately after the GCMC move so the
                # ghost state and frame (saved during dynamics) are consistent.
                if write_gcmc_ghosts:
                    gcmc_sampler.write_ghost_residues()

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
            if self._config.save_energy_components and time_ns is not None:
                self._save_energy_components(index, dynamics.context(), time_ns)

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
                    device = slot % self._num_gpus

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
                    device = slot % self._num_gpus

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
            device = slot % self._num_gpus

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

                # Compute the current number of waters in the GCMC sampling
                # volume after equilibration.
                gcmc_sampler.push()
                try:
                    gcmc_sampler.num_waters(context=dynamics.context())
                finally:
                    gcmc_sampler.pop()

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
        # Create the matrix.
        matrix = _np.zeros((len(results), len(results)))

        # Fill the matrix. The energy returned by the dynamics block already
        # includes the pressure and grand canonical contributions.
        for i, energies in results:
            for j, energy in enumerate(energies):
                matrix[i, j] = self._beta * energy

        return matrix

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
                    system, index, block, speed, is_final_block=is_final_block
                )

                if error is not None:
                    return index, error

            # Delete all trajectory frames from the Sire system within the
            # dynamics object. This is a no-op when the replica was committed
            # earlier, since the frames were cleared then to free the slot.
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
                    f"is {num_waters}{acc_str}"
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

        gcmc_sampler.push()
        try:
            # Set the water state.
            gcmc_sampler._set_water_state(dynamics.context(), force=True)
        finally:
            gcmc_sampler.pop()

        # Re-bind the GCMC sampler to the dynamics object.
        gcmc_sampler.bind_dynamics(dynamics)
