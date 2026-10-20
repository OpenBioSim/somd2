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

__all__ = ["DynamicsCache"]

import numpy as _np

from somd2 import _logger

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
        strided=False,
        virtual_sites=None,
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

        virtual_sites: np.ndarray
            The molecules with virtual sites, so that each molecule's particles
            can be found in the saved states.
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
        self._virtual_sites = virtual_sites
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
        # Not pickled, since the runner sets it from the configuration.
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
        Set the dynamics object for a given slot. The slot then holds no
        replica, so the next one loaded into it is set up in full.

        Parameters
        ----------

        slot: int
            The index of the slot.

        dynamics: sire.legacy.Convert.SOMMContext
            The dynamics object.
        """
        self._dynamics[slot] = dynamics
        self.release(slot)

    def release(self, slot):
        """
        Mark a slot as holding no replica, so that the next replica loaded
        into it has its state, lambda value and GCMC water state pushed in
        full, e.g. after a failed step left the context in a bad state.

        Parameters
        ----------

        slot: int
            The index of the slot.
        """
        self._slot_replica[slot] = None

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

    def prepare_restart(
        self,
        system,
        lambdas,
        rest2_scale_factors,
        num_gpus,
        dynamics_kwargs,
        gcmc_kwargs=None,
        output_directory=None,
        num_slots=None,
        update_constraints=True,
        constraint_lambda_index=None,
        gpu_devices=None,
        strided=False,
        virtual_sites=None,
    ):
        """
        Prepare a cache loaded from a checkpoint to carry on the simulation.
        The settings that aren't stored in a checkpoint are taken from the
        current configuration, so that the number of contexts and the GPUs can
        change on restart. The slot layout is rebuilt, the dynamics objects are
        created, and the GCMC statistics are handed to the new samplers.

        The OpenMM contexts are not reset here. Each replica's state is pushed
        into its slot by load_replica() at the start of its first block, which
        is the only point at which the slot is known to be free.

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

        num_slots: int
            The number of dynamics objects (slots) to create. See the
            constructor.

        update_constraints: bool
            Whether to update the constraints when changing the lambda value of
            a slot.

        constraint_lambda_index: int
            The index of the lambda value to create every context at. See the
            constructor.

        gpu_devices: list
            The physical devices backing each OpenMM device index.

        strided: bool
            Whether to assign replicas to slots in a strided pattern.

        virtual_sites: np.ndarray
            The molecules with virtual sites. Set again, since checkpoints
            written by older versions don't store them.
        """
        self._gpu_devices = gpu_devices
        self._num_slots = self._num_replicas if num_slots is None else num_slots
        self._update_constraints = update_constraints
        self._constraint_lambda_index = constraint_lambda_index
        self._strided = strided
        self._virtual_sites = virtual_sites
        self._build_slot_layout()

        self._create_dynamics(
            system,
            lambdas,
            rest2_scale_factors,
            num_gpus,
            dynamics_kwargs,
            gcmc_kwargs=gcmc_kwargs,
            output_directory=output_directory,
        )

        # A sampler keeps only the lambda values it visits, so each can be
        # handed the whole simulation's statistics.
        if self._gcmc_stats is not None:
            for slot in range(self._num_slots):
                _, gcmc_sampler = self.get(slot)
                gcmc_sampler.restore_stats(self._gcmc_stats)

    def stored_positions(self, replica):
        """
        Return the positions in the stored state of a replica.

        Parameters
        ----------

        replica: int
            The index of the replica.

        Returns
        -------

        positions: openmm.unit.Quantity
            The positions.
        """
        return self._get_positions(self._openmm_states[replica])

    def gcmc_num_waters(self, replica):
        """
        Return the number of waters in the GCMC volume, recorded when a replica
        was last propagated.

        Parameters
        ----------

        replica: int
            The index of the replica.

        Returns
        -------

        int
            The number of waters, or None if not recorded.
        """
        return self._gcmc_num_waters[replica]

    def reset_hot_replicas(self):
        """
        Make every replica hot, clearing the votes for adaptive replica
        exchange.
        """
        self._hot_replicas = None
        self._cold_votes = [0] * self._num_replicas

    @property
    def num_replicas(self):
        """
        The number of replicas.
        """
        return self._num_replicas

    @property
    def num_slots(self):
        """
        The number of dynamics objects (slots).
        """
        return self._num_slots

    @property
    def groups(self):
        """
        The replicas hosted by each slot.
        """
        return self._groups

    @property
    def time(self):
        """
        The simulation time at the last checkpoint, or None.
        """
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def hot_replicas(self):
        """
        The hot replicas of adaptive replica exchange, or None if every
        replica is hot.
        """
        return self._hot_replicas

    @hot_replicas.setter
    def hot_replicas(self, hot_replicas):
        self._hot_replicas = hot_replicas

    @property
    def cold_votes(self):
        """
        The number of consecutive rankings at which each replica has been
        found cold.
        """
        return self._cold_votes

    @cold_votes.setter
    def cold_votes(self, cold_votes):
        self._cold_votes = cold_votes

    @property
    def adaptive_data(self):
        """
        The data used to rank the windows for adaptive replica exchange, or
        None.
        """
        return self._adaptive_data

    @adaptive_data.setter
    def adaptive_data(self, adaptive_data):
        self._adaptive_data = adaptive_data

    @property
    def gcmc_stats(self):
        """
        The GCMC statistics for the whole simulation, keyed by lambda value, or
        None.
        """
        return self._gcmc_stats

    @gcmc_stats.setter
    def gcmc_stats(self, gcmc_stats):
        self._gcmc_stats = gcmc_stats

    @property
    def terminal_flip_stats(self):
        """
        The number of terminal flip moves attempted and accepted by each
        replica.
        """
        return self._terminal_flip_stats

    @terminal_flip_stats.setter
    def terminal_flip_stats(self, terminal_flip_stats):
        self._terminal_flip_stats = terminal_flip_stats

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
