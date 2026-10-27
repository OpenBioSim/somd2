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

from ._adaptive import WindowRanker as _WindowRanker
from ._adaptive import cold_rate as _cold_rate
from ._base import RunnerBase as _RunnerBase

# Imported under its own name, since checkpoints written before it moved
# pickle it as somd2.runner._repex.DynamicsCache.
from ._dynamics_cache import DynamicsCache


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

        # Add the Boresch restraint before the dynamics cache is built below,
        # since the per-replica OpenMM contexts it creates are fixed at
        # construction time and won't pick up a restraint added afterwards.
        self._add_boresch_restraint(device=0)

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

        if not self._is_restart and any(
            t.value() > 0.0 for t in self._equilibration_times
        ):
            self._is_equilibration = True

            # Don't use constraints during equilibration.
            if not self._config.equilibration_constraints:
                constraint = "none"
                perturbable_constraint = "none"

            # Update the timestep.
            timestep = self._config.equilibration_timestep

        # Equilibration and production constraints, and whether the initial
        # contexts must be rebuilt to use them.
        production_constraints = (
            self._config.constraint,
            self._config.perturbable_constraint,
        )
        if self._config.equilibration_constraints:
            self._equilibration_constraints = production_constraints
        else:
            self._equilibration_constraints = ("none", "none")

        initial_constraints = (constraint, perturbable_constraint)
        self._is_equilibration_rebuild = (
            initial_constraints != self._equilibration_constraints
        )
        self._is_production_rebuild = initial_constraints != production_constraints

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
                virtual_sites=self._virtual_sites,
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
            # pickle (set by the new-format _write_checkpoint_system); fall
            # back to the Sire system for old-format checkpoints.
            if self._dynamics_cache.time is not None and not isinstance(
                self._system, list
            ):
                time = self._dynamics_cache.time
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
            if self._dynamics_cache.num_replicas != self._config.num_lambda:
                _logger.error(
                    f"The number of replicas in the dynamics cache ({self._dynamics_cache.num_replicas}) "
                    f"does not match the number of replicas in the configuration ({self._config.num_lambda})."
                )

            # For new-format restarts, set the system time so that dynamics
            # objects are initialised with the correct integrator step count.
            if not isinstance(self._system, list):
                self._system.set_time(time)

            # The settings that aren't stored are taken from the current
            # configuration, so that the GPUs and 'max_contexts' can change on
            # restart. Everything that is restored is per-replica, so it doesn't
            # depend on the slot layout.
            self._dynamics_cache.prepare_restart(
                self._system,
                self._lambda_values,
                self._rest2_scale_factors,
                self._num_gpus,
                self._dynamics_kwargs,
                gcmc_kwargs=self._gcmc_kwargs,
                output_directory=self._config.output_directory,
                num_slots=self._num_slots,
                update_constraints=self._config.update_constraints,
                constraint_lambda_index=self._constraint_lambda_index,
                gpu_devices=self._gpu_devices,
                strided=self._is_strided,
                virtual_sites=self._virtual_sites,
            )

        # Log the GCMC sphere centre for each replica. This uses the stored
        # state rather than the context, since a slot only holds the positions
        # of the replica it last hosted.
        import openmm.unit as _omm_unit

        for i, lam in enumerate(self._lambda_values):
            _, gcmc_sampler = self._dynamics_cache.get(self._dynamics_cache.slot_for(i))
            if gcmc_sampler is not None and gcmc_sampler._reference is not None:
                positions = self._dynamics_cache.stored_positions(i)
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
                self._dynamics_cache.time
                if (
                    self._dynamics_cache.time is not None
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
                attempted, accepted = self._dynamics_cache.terminal_flip_stats[i]
                self._terminal_flip_samplers[i].reset(attempted, accepted)

        from threading import Lock

        # Create a lock to guard the committed systems.
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
        groups = self._dynamics_cache.groups

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
        from time import time

        # Record the start time.
        start = time()

        cycles, cycles_per_checkpoint, total_blocks = self._cycle_counts()

        # The GPU that propagates each replica, and how many replicas each GPU
        # can propagate at once, which are fixed for the run.
        replica_devices = self._replica_devices()
        device_capacity = self._device_capacity()

        self._prepare_replicas(
            dynamics_executor, checkpoint_executor, replica_devices, cycles
        )

        # Current block number.
        block = self._start_block

        # Record the start time for the production block.
        prod_start = time()

        cycles_per_frame, cycles_per_gcmc, cycles_per_flip = self._move_cycles(cycles)

        # The cycles, numbered from one, that end with a checkpoint.
        checkpoint_cycles = self._checkpoint_cycles(cycles, cycles_per_checkpoint)
        checkpoint_set = set(checkpoint_cycles.tolist())

        num_replicas = self._config.num_lambda
        is_adaptive = self._config.adaptive_repex

        # The cycles on which the windows are ranked for adaptive replica
        # exchange, which are the checkpoint cycles unless set separately. The
        # last cycle is left out, since no cycles are left to allocate.
        rank_cycles = _np.array([], dtype=int)
        rank_frequency = self._config.adaptive_rank_frequency
        if is_adaptive and rank_frequency is None:
            rank_cycles = checkpoint_cycles
        elif is_adaptive:
            energy_frequency = self._config.energy_frequency
            rank_cycles = self._rank_cycles(
                cycles,
                round((self._start_time / energy_frequency).value()),
                round((rank_frequency / energy_frequency).value()),
            )
        rank_cycles = rank_cycles[rank_cycles < cycles]
        rank_set = set(rank_cycles.tolist())

        # The cycle on which each replica was last propagated, used to catch up
        # on the MC moves that fell due while it wasn't.
        last_cycle = [-1] * num_replicas

        # The number of energy samples each replica has collected since the
        # last checkpoint.
        num_samples = [0] * num_replicas

        self._start_adaptive()

        # Converted once, since unit conversions are slow.
        start_ns = self._start_time.to("ns")
        energy_frequency_ns = self._config.energy_frequency.to("ns")

        # Perform the replica exchange simulation.
        for i in range(cycles):
            _logger.info(
                f"Running dynamics for cycle {i + 1} of {cycles} "
                f"(time = {start_ns + i * energy_frequency_ns:.4f} ns)"
            )

            # Log the states. This is the replica index for the state (positions
            # and velocities) used to seed each replica for the current cycle.
            # For example:
            #   States: [ 2 0 1 3 4 5 6 7 8 9 10 ]
            # means that the final positions and velocities from replica 2 are
            # used to seed replica 0, those from replica 0 are used to seed
            # replica 1, and so on.
            _logger.info(f"States: {self._dynamics_cache.get_states()}")

            # Whether to checkpoint.
            is_checkpoint = (i + 1) in checkpoint_set

            # Whether the checkpoint files are written at the end of this cycle.
            do_checkpoint = is_checkpoint or i == cycles - 1

            # Only the hot replicas are propagated, except on mix and checkpoint
            # cycles.
            hot = self._dynamics_cache.hot_replicas
            is_full = (
                hot is None
                or (i + 1) % self._config.adaptive_mix_interval == 0
                or do_checkpoint
            )
            active = list(range(num_replicas)) if is_full else hot

            if not is_full:
                _logger.info(f"Propagating {len(active)} hot replicas: {active}")

            results = self._propagate_cycle(
                dynamics_executor,
                checkpoint_executor,
                cycle=i,
                replicas=None if is_full else active,
                replica_devices=replica_devices,
                last_cycle=last_cycle,
                move_cycles=(cycles_per_frame, cycles_per_gcmc, cycles_per_flip),
                do_checkpoint=do_checkpoint,
            )

            for index in active:
                last_cycle[index] = i
                num_samples[index] += 1

            # Assemble an energy matrix from the results.
            _logger.info("Assembling energy matrix")
            energy_matrix = self._assemble_results(results)

            cycle_time = self._start_time + (i + 1) * self._config.energy_frequency
            is_burnt_in = cycle_time > self._config.adaptive_burn_in
            if is_adaptive:
                self._record_cycle(i, cycle_time, energy_matrix, active, rank_cycles)

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

            # Rank the windows before any checkpoint is written, so that a
            # restart carries on with the same hot replicas.
            if is_adaptive and (i + 1) in rank_set and is_burnt_in:
                self._rank_windows(
                    replica_devices,
                    device_capacity,
                    self._window_samples(),
                    cycles - (i + 1),
                    _cold_rate(
                        i + 1,
                        cycles,
                        checkpoint_cycles,
                        self._config.adaptive_mix_interval,
                    ),
                )

            if do_checkpoint:
                self._write_cycle_checkpoint(
                    checkpoint_executor,
                    block,
                    total_blocks,
                    i == cycles - 1,
                    num_samples,
                    cycle_time,
                    is_checkpoint,
                )
                num_samples = [0] * num_replicas

            # This is a checkpoint cycle.
            if is_checkpoint:
                # Update the block number.
                block += 1

        self._finish(start, prod_start)

    def _cycle_counts(self):
        """
        Work out the number of replica exchange cycles in the run, between
        checkpoints, and the number of checkpoint blocks.

        Returns
        -------

        cycles: int
            The number of cycles.

        cycles_per_checkpoint: float
            The number of cycles between checkpoints, which may be fractional.

        total_blocks: int
            The number of checkpoint blocks, including a final partial block.
        """
        from math import ceil

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

            # Store the number of repex cycles per block (may be fractional).
            cycles_per_checkpoint = frac

        # Otherwise, we don't checkpoint.
        else:
            cycles_per_checkpoint = float(cycles)
            num_blocks = 1
            rem = 0

        return cycles, cycles_per_checkpoint, num_blocks + int(rem > 0)

    def _prepare_replicas(
        self, dynamics_executor, checkpoint_executor, replica_devices, cycles
    ):
        """
        Minimise and equilibrate the replicas, unless this is a restart, and
        write a checkpoint after equilibration so that a restart after an
        early production crash doesn't need to re-equilibrate.

        Parameters
        ----------

        dynamics_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for dynamics.

        checkpoint_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for writing checkpoint files.

        replica_devices: list of int
            The GPU that propagates each replica.

        cycles: int
            The number of cycles in the run.
        """
        from itertools import repeat

        # Minimise at each lambda value. Don't minimise on restart.
        if self._config.minimise and not self._is_restart:
            for batch in self._setup_batches(replica_devices):
                try:
                    for results in dynamics_executor.map(self._minimise_slot, batch):
                        self._report_minimisation(results)
                except KeyboardInterrupt:
                    dynamics_executor.shutdown(wait=False, cancel_futures=True)
                    _logger.error("Minimisation cancelled. Exiting.")
                    _sys.exit(1)

        # Equilibrate the system.
        if self._is_equilibration and not self._is_restart:
            for batch in self._setup_batches(replica_devices):
                try:
                    for results in dynamics_executor.map(self._equilibrate_slot, batch):
                        for success, index, e in results:
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
                with self._checkpoint_threads():
                    for batch in self._replica_passes(0):
                        self._map_replicas(
                            checkpoint_executor,
                            "Post-equilibration commit",
                            self._load_and_commit,
                            batch,
                        )

            lock = _FileLock(self._lock_file)
            with (
                lock.acquire(timeout=self._config.timeout.to("seconds")),
                self._checkpoint_threads(),
            ):
                self._save_replica_topology(is_final_block=False)

                self._map_replicas(
                    checkpoint_executor,
                    "Post-equilibration checkpoint",
                    self._checkpoint,
                    list(range(self._config.num_lambda)),
                    repeat(self._lambda_values),
                    repeat(-1),
                    repeat(cycles),
                )

    def _move_cycles(self, cycles):
        """
        Work out the number of cycles between frames, GCMC moves and terminal
        flip moves.

        Parameters
        ----------

        cycles: int
            The number of cycles in the run.

        Returns
        -------

        cycles_per_frame, cycles_per_gcmc, cycles_per_flip: int
            The number of cycles between each. A move that isn't used is never
            due.
        """
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

        return cycles_per_frame, cycles_per_gcmc, cycles_per_flip

    def _start_adaptive(self):
        """
        Set up the ranking of windows for adaptive replica exchange, restoring
        the data from a restart, or make every replica hot if it isn't used.
        """
        num_replicas = self._config.num_lambda

        if self._config.adaptive_repex:
            self._ranker = _WindowRanker(
                num_replicas,
                self._config.adaptive_min_samples,
                self._dynamics_cache.get_proposed(),
                self._dynamics_cache.get_accepted(),
                is_ring_break=self._ring_pair is not None,
                is_cached=self._is_cached,
            )
            # A burn-in raised on restart starts the adaptation again.
            if self._start_time >= self._config.adaptive_burn_in:
                self._ranker.restore(self._dynamics_cache.adaptive_data)
            else:
                self._dynamics_cache.reset_hot_replicas()
        else:
            # Adaptive replica exchange may have been switched off on restart.
            self._dynamics_cache.reset_hot_replicas()

    def _propagate_cycle(
        self,
        dynamics_executor,
        checkpoint_executor,
        cycle,
        replicas,
        replica_devices,
        last_cycle,
        move_cycles,
        do_checkpoint,
    ):
        """
        Propagate the replicas for a cycle, one pass at a time. Each pass takes
        at most one replica from each slot, and there is a single pass when
        there is a slot per replica.

        Parameters
        ----------

        dynamics_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for dynamics.

        checkpoint_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for writing checkpoint files.

        cycle: int
            The index of the cycle.

        replicas: list of int
            The replicas to propagate. If None, every replica is propagated.

        replica_devices: list of int
            The GPU that propagates each replica.

        last_cycle: list of int
            The cycle on which each replica was last propagated, or -1.

        move_cycles: tuple of int
            The number of cycles between frames, GCMC moves and terminal flip
            moves.

        do_checkpoint: bool
            Whether the checkpoint files are written at the end of the cycle.

        Returns
        -------

        list of tuple
            The index and energies of each replica that was propagated.
        """
        from itertools import repeat

        cycles_per_frame, cycles_per_gcmc, cycles_per_flip = move_cycles
        is_adaptive = self._config.adaptive_repex
        results = []

        # Capture the simulation clock at the start of the cycle. This seeds
        # any replica that doesn't yet have a clock of its own.
        clock = self._dynamics_cache.get_clock()

        for batch in self._replica_passes(cycle, replicas=replicas):
            # Run a dynamics block for each replica in the pass, making sure
            # each GPU is only oversubscribed by a factor of
            # self._config.oversubscription_factor.
            for batch_replicas in self._device_batches(batch, replica_devices):
                # Whether each replica is due a GCMC or terminal flip move,
                # including any that fell due while it wasn't propagated.
                is_gcmc = [
                    self._is_due(last_cycle[r], cycle, cycles_per_gcmc)
                    for r in batch_replicas
                ]
                is_terminal_flip = [
                    self._is_due(last_cycle[r], cycle, cycles_per_flip)
                    for r in batch_replicas
                ]

                # Whether each replica saves a frame and writes its ghost
                # waters, scheduled by the cycle since a cold replica's own
                # clock falls behind.
                is_frame = [
                    self._is_due(last_cycle[r], cycle, cycles_per_frame)
                    for r in batch_replicas
                ]

                try:
                    for result, index, energies in dynamics_executor.map(
                        self._run_block,
                        batch_replicas,
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

            # When contexts are shared, commit the replicas of this pass while
            # they are still resident in their slots. The committed systems are
            # held until the end of the cycle, so that every checkpoint file is
            # still written under a single lock. With a context per replica
            # everything is still resident once the cycle finishes, so
            # committing is left to _checkpoint().
            if do_checkpoint and self._is_cached:
                with self._checkpoint_threads():
                    self._map_replicas(
                        checkpoint_executor, "Commit", self._commit_replica, batch
                    )

        return results

    def _record_cycle(self, cycle, cycle_time, energy_matrix, replicas, rank_cycles):
        """
        Record the data used to rank the windows for adaptive replica exchange,
        logging when the burn-in is complete and counting swaps from then on.
        Must be called before the replicas are mixed.

        Parameters
        ----------

        cycle: int
            The index of the cycle.

        cycle_time: sire.units.GeneralUnit
            The simulation time at the end of the cycle.

        energy_matrix: np.ndarray
            The reduced energy of each replica at each window.

        replicas: list of int
            The replicas that were propagated.

        rank_cycles: np.ndarray
            The cycles on which the windows are ranked, numbered from one, in
            increasing order.
        """
        # Nothing is recorded until the burn-in is complete.
        if cycle_time <= self._config.adaptive_burn_in:
            return

        if cycle_time - self._config.energy_frequency <= self._config.adaptive_burn_in:
            ranked = rank_cycles[rank_cycles >= cycle + 1]
            if len(ranked) > 0:
                ranked_time = self._start_time + int(ranked[0]) * (
                    self._config.energy_frequency
                )
                _logger.info(
                    "Adaptive replica exchange burn-in complete, first "
                    f"ranking at {ranked_time.to('ns'):.4f} ns"
                )
            else:
                _logger.info(
                    "Adaptive replica exchange burn-in complete, but no "
                    "ranking of the windows remains"
                )
            self._ranker.set_swap_baseline(
                self._dynamics_cache.get_proposed(),
                self._dynamics_cache.get_accepted(),
            )
        self._ranker.record(energy_matrix, replicas, self._bond_distances(replicas))

    def _write_cycle_checkpoint(
        self,
        checkpoint_executor,
        block,
        num_blocks,
        is_final_block,
        num_samples,
        cycle_time,
        is_checkpoint,
    ):
        """
        Write the checkpoint files once a cycle is complete, every one under a
        single lock, so that an external process reading the output directory
        always sees a coherent set rather than a mixture of new and old files.

        Parameters
        ----------

        checkpoint_executor: concurrent.futures.ThreadPoolExecutor
            The thread pool for writing checkpoint files.

        block: int
            The index of the checkpoint block.

        num_blocks: int
            The number of checkpoint blocks in the run.

        is_final_block: bool
            Whether this is the last cycle of the run.

        num_samples: list of int
            The number of energy samples each replica has collected since the
            last checkpoint.

        cycle_time: sire.units.GeneralUnit
            The simulation time at the end of the cycle.

        is_checkpoint: bool
            Whether this is a checkpoint cycle, rather than a final save.
        """
        from itertools import repeat

        replica_list = list(range(self._config.num_lambda))

        lock = _FileLock(self._lock_file)
        with (
            lock.acquire(timeout=self._config.timeout.to("seconds")),
            self._checkpoint_threads(),
        ):
            # First backup existing checkpoint files.
            self._map_replicas(
                checkpoint_executor,
                "Backup",
                self._backup_checkpoint,
                replica_list,
            )

            self._save_replica_topology(is_final_block)

            # Now write the new checkpoint files.
            self._map_replicas(
                checkpoint_executor,
                "Checkpoint",
                self._checkpoint,
                replica_list,
                repeat(self._lambda_values),
                repeat(block),
                repeat(num_blocks),
                repeat(is_final_block),
                num_samples,
            )

            # Replicas' own clocks fall behind while they are cold, so the
            # simulation time is that of the cycle.
            self._dynamics_cache.time = cycle_time

            self._save_repex_state(final=not is_checkpoint)

    def _finish(self, start, prod_start):
        """
        Log the performance and run time, remove the backup files, and mark
        the run as complete.

        Parameters
        ----------

        start: float
            The time at which the run started, in seconds.

        prod_start: float
            The time at which production started, in seconds.
        """
        from time import time

        end = time()

        # Record the average production speed. (ns/day per replica)
        prod_speed = self._config.runtime.to("ns") / ((end - prod_start) / 86400)
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

            if gcmc_sampler is not None:
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

            # Save the minimised state back out of the slot.
            self._dynamics_cache.store_replica(index)

        except Exception as e:
            # The next load restores the replica's stored state over whatever
            # the failure left in the context.
            self._dynamics_cache.release(self._dynamics_cache.slot_for(index))
            return False, index, e

        return True, index, None

    def _report_minimisation(self, results):
        """
        Log any failed minimisations, raising the first error if they are
        treated as errors.

        Parameters
        ----------

        results: list of tuple
            The success, index and exception for each replica.
        """
        for success, index, e in results:
            if not success:
                msg = f"Minimisation failed for {_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                if self._config.minimisation_errors:
                    _logger.error(msg)
                    raise e
                else:
                    _logger.warning(msg)

    def _minimise_slot(self, replicas):
        """
        Minimise the replicas hosted by a slot, one after another. If there is
        no equilibration and the production constraints differ from those used
        for minimisation, the slot is then given a production dynamics object.

        Parameters
        ----------

        replicas: list of int
            The replicas hosted by the slot.

        Returns
        -------

        list of tuple
            The success, index and exception for each replica.
        """
        results = [self._minimise(index) for index in replicas]

        # There's no need to rebuild if a failed minimisation stops the run.
        is_failed = not all(success for success, _, _ in results)
        if is_failed and self._config.minimisation_errors:
            return results

        if not self._is_equilibration and self._is_production_rebuild:
            # Load a valid state to build from, in case the last minimisation
            # failed and released the slot.
            index = replicas[-1]
            try:
                self._dynamics_cache.load_replica(index)
                self._rebuild_dynamics(index)
            except Exception as e:
                self._report_minimisation(results)
                _logger.error(
                    f"Could not create the production dynamics object for "
                    f"{_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                )
                raise

        return results

    def _rebuild_dynamics(self, index, **overrides):
        """
        Replace the dynamics object of a replica's slot with a new one, built
        from the slot's current state, e.g. to change the constraints or
        timestep. Afterwards the slot holds no replica.

        Parameters
        ----------

        index: int
            The index of the replica.

        overrides: dict
            Dynamics keyword arguments that replace those used for production.
        """
        slot = self._dynamics_cache.slot_for(index)
        dynamics, gcmc_sampler = self._dynamics_cache.get(slot)

        # Commit the current system.
        system = dynamics.commit()

        # Delete the current dynamics object.
        self._dynamics_cache.delete(slot)

        # Copy the dynamics keyword arguments, overloading the device and lambda value.
        dynamics_kwargs = self._dynamics_kwargs.copy()
        dynamics_kwargs["device"] = self._dynamics_cache.device_for(slot)
        dynamics_kwargs["lambda_value"] = self._build_lambda(index)
        dynamics_kwargs["rest2_scale"] = self._rest2_scale_factors[index]
        dynamics_kwargs.update(overrides)

        # Create the new dynamics object.
        dynamics = system.dynamics(**dynamics_kwargs)

        # Reset the GCMC water state. The dynamics object is created from the
        # Sire system, so the water state in the context does not match it.
        if gcmc_sampler is not None:
            self._reset_gcmc_sampler(gcmc_sampler, dynamics)

        # Set the new dynamics object, holding no replica, so that the next load
        # sets up its state, lambda value and energy trajectory.
        self._dynamics_cache.set(slot, dynamics)
        self._dynamics_cache.release(slot)

        _logger.info(
            f"Created dynamics object for {_lam_sym} = {self._lambda_values[index]:.5f}"
        )

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
        time = self._equilibration_times[index]

        # The replica is still loaded and stored when its time is zero, so
        # that the slot's resident replica is the same as for other windows.
        is_equilibrated = time.value() > 0.0
        if is_equilibrated:
            _logger.info(
                f"Equilibrating at {_lam_sym} = {self._lambda_values[index]:.5f}"
            )

        try:
            # Make the replica resident in its slot, with its own clock.
            clock = self._dynamics_cache.get_clock(index)
            self._dynamics_cache.load_replica(index, clock=clock)

            # Get the dynamics object (and GCMC sampler).
            slot = self._dynamics_cache.slot_for(index)
            dynamics, gcmc_sampler = self._dynamics_cache.get(slot)

            # Equilibration doesn't count towards the simulation time, so the
            # clock is restored afterwards.
            if clock is None:
                clock = dynamics._get_clock()

            if is_equilibrated and gcmc_sampler is not None:
                gcmc_sampler.push()
                try:
                    _logger.info(
                        f"Equilibrating with GCMC moves at {_lam_sym} = {self._lambda_values[index]:.5f}"
                    )
                    for i in range(100):
                        gcmc_sampler.move(dynamics.context())
                finally:
                    gcmc_sampler.pop()

            # Equilibrate.
            if is_equilibrated:
                dynamics.run(
                    time,
                    energy_frequency=0,
                    frame_frequency=0,
                    save_velocities=False,
                    auto_fix_minimise=self._config.auto_fix_minimise,
                    save_crash_report=self._config.save_crash_report,
                )

            dynamics._set_clock(clock)

            # Save the equilibrated state back out of the slot.
            self._dynamics_cache.store_replica(index)

        except Exception as e:
            return False, index, e

        return True, index, None

    def _equilibrate_slot(self, replicas):
        """
        Equilibrate the replicas hosted by a slot, one after another, in a
        single dynamics object, then give the slot one for production. This
        creates two dynamics objects per slot, rather than per replica.

        Parameters
        ----------

        replicas: list of int
            The replicas hosted by the slot.

        Returns
        -------

        list of tuple
            The success, index and exception for each replica equilibrated.
        """
        # Rebuild the slot's dynamics object for equilibration, starting from
        # the first replica.
        if self._is_equilibration_rebuild:
            index = replicas[0]
            constraint, perturbable_constraint = self._equilibration_constraints
            try:
                self._dynamics_cache.load_replica(index)
                self._rebuild_dynamics(
                    index,
                    timestep=self._config._equilibration_timestep,
                    constraint=constraint,
                    perturbable_constraint=perturbable_constraint,
                )
            except Exception as e:
                _logger.error(
                    f"Could not create the equilibration dynamics object for "
                    f"{_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
                )
                raise

        results = []
        for index in replicas:
            results.append(self._equilibrate(index))
            if not results[-1][0]:
                return results

        # Rebuild the slot's dynamics object for production, from the last
        # replica, which is still resident.
        index = replicas[-1]
        try:
            self._rebuild_dynamics(index)
        except Exception as e:
            _logger.error(
                f"Could not create the production dynamics object for "
                f"{_lam_sym} = {self._lambda_values[index]:.5f}: {e}"
            )
            raise

        return results

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
    def _rank_cycles(num_cycles, first_cycle, cycles_per_rank):
        """
        The cycles on which the windows are ranked, when ranking has its own
        frequency. These fall on multiples of the number of cycles between
        rankings counted from the start of the simulation, so that a restart
        keeps the same schedule.

        Parameters
        ----------

        num_cycles: int
            The number of cycles in this run.

        first_cycle: int
            The number of cycles run before this run started.

        cycles_per_rank: int
            The number of cycles between rankings.

        Returns
        -------

        np.ndarray
            The cycles of this run, numbered from one, in increasing order.
        """
        cycles = _np.arange(
            cycles_per_rank, first_cycle + num_cycles + 1, cycles_per_rank
        )
        return cycles[cycles > first_cycle] - first_cycle

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

    def _setup_batches(self, devices):
        """
        Yield batches of slots to minimise or equilibrate at once, at most
        'oversubscription_factor' per GPU. Each slot is given by the replicas
        it hosts, which are set up one after another.

        Parameters
        ----------

        devices: list of int
            The GPU that propagates each replica.

        Yields
        ------

        list of list of int
            A batch of slots.
        """
        groups = self._dynamics_cache.groups

        # A slot's replicas share its GPU, so its first replica stands in for it.
        firsts = [group[0] for group in groups]

        for batch in self._device_batches(firsts, devices):
            yield [groups[self._dynamics_cache.slot_for(r)] for r in batch]

    def _window_samples(self):
        """
        The number of energy samples each window has collected, from its clock.

        Returns
        -------

        list of int
            The number of samples for each window.
        """
        energy_frequency = self._config.energy_frequency.to("ps")
        clocks = [
            self._dynamics_cache.get_clock(replica)
            for replica in range(self._dynamics_cache.num_replicas)
        ]
        return [
            (
                0
                if clock is None
                else round(clock["_current_time"].to("ps") / energy_frequency)
            )
            for clock in clocks
        ]

    def _check_adaptive_config(self):
        """
        Validate the ranking frequency, and warn about settings that stop
        adaptive replica exchange having an effect, before any time is spent
        on minimisation and equilibration.
        """
        from math import floor, isclose

        checkpoint = self._config.checkpoint_frequency.to("ps")
        rank_frequency = self._config.adaptive_rank_frequency
        rank = checkpoint if rank_frequency is None else rank_frequency.to("ps")
        energy_frequency = self._config.energy_frequency.to("ps")
        burn_in = self._config.adaptive_burn_in.to("ps")
        end_time = (self._start_time + self._config.runtime).to("ps")

        if rank_frequency is not None:
            ratio = rank / energy_frequency
            if not isclose(ratio, round(ratio), abs_tol=1e-4):
                msg = "'adaptive_rank_frequency' must be a multiple of 'energy_frequency'."
                _logger.error(msg)
                raise ValueError(msg)

            # Each ranking is over the whole history, and two consecutive
            # rankings only smooth out noise if they are a few cycles apart.
            if round(ratio) < self._config.adaptive_mix_interval:
                msg = (
                    "'adaptive_rank_frequency' must be at least "
                    "'adaptive_mix_interval' cycles."
                )
                _logger.error(msg)
                raise ValueError(msg)

        if rank == 0.0:
            _logger.warning(
                "Adaptive replica exchange ranks the windows at checkpoints, but "
                "'checkpoint_frequency' is zero and 'adaptive_rank_frequency' "
                "isn't set. Every window will be propagated every cycle."
            )
        # Windows first go cold at the second ranking after the burn-in.
        elif (floor(burn_in / rank + 1e-9) + 2) * rank >= end_time - 1e-6:
            _logger.warning(
                "There are fewer than two rankings after 'adaptive_burn_in' "
                "and before the end of the run, so no window will stop being "
                "hot. Every window will be propagated every cycle."
            )
        elif (
            checkpoint > 0.0
            and checkpoint / energy_frequency <= self._config.adaptive_mix_interval
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
        # Legacy restarts load a system per replica, which share a topology.
        system = self._system[0] if isinstance(self._system, list) else self._system
        atoms = system.atoms()
        pairs = []

        for mol in system["property is_perturbable"].molecules():
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

    def _bond_distances(self, replicas):
        """
        The distance of the bond that is broken or made in each replica, for
        a ring-breaking perturbation. Must be called before the replicas are
        mixed.

        Parameters
        ----------

        replicas: list of int
            The replicas that were propagated.

        Returns
        -------

        dict
            The distance in Angstrom for each replica, or None if there is no
            single bond that is broken or made.
        """
        if self._ring_pair is None:
            return None

        import openmm.unit as _omm_unit

        i, j = self._ring_pair
        distances = {}
        for k in replicas:
            positions = self._dynamics_cache.stored_positions(k)
            pair = _np.asarray(positions[[i, j]].value_in_unit(_omm_unit.angstrom))
            distances[k] = float(_np.linalg.norm(pair[1] - pair[0]))

        return distances

    def _rank_windows(
        self, window_devices, device_capacity, window_samples, cycles_left, cold_rate
    ):
        """
        Rank the windows and choose the hot replicas for adaptive replica
        exchange, which are kept in the dynamics cache so that a restart
        carries on with them. See WindowRanker.rank().

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
        hot, votes = self._ranker.rank(
            window_devices,
            device_capacity,
            window_samples,
            cycles_left,
            cold_rate,
            self._dynamics_cache.get_proposed(),
            self._dynamics_cache.get_accepted(),
            self._dynamics_cache.hot_replicas,
            self._dynamics_cache.cold_votes,
        )
        self._dynamics_cache.hot_replicas = hot
        self._dynamics_cache.cold_votes = votes

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
        self._dynamics_cache.time = system.time()

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

        When contexts are shared between replicas, a slot holds the positions
        and trajectory frames of the replica it last hosted, so a replica has
        to be committed before its slot is handed on. Deferring only the file
        writes keeps every checkpoint file within a single lock, so that an
        external reader never sees a partial set. It is also used to commit
        the first replica early, when the GCMC topologies are saved from it.

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

    def _save_replica_topology(self, is_final_block):
        """
        Save the GCMC topologies from the first replica when they are due. The
        replicas share the topology files, so they are saved once, before the
        replicas are checkpointed, rather than by each replica.

        Parameters
        ----------

        is_final_block: bool
            Whether this is the final block of the simulation.
        """
        if not self._is_gcmc_topology_due(is_final_block):
            return

        try:
            # Commit the replica now if it isn't already, so that _checkpoint()
            # re-uses the committed system.
            if 0 not in self._committed:
                _, error = self._commit_replica(0)
                if error is not None:
                    raise error

            system, _ = self._committed[0]
            self._save_gcmc_topology(system)
        except Exception as e:
            _logger.error(f"Saving the GCMC topologies failed: {e}")
            raise

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

            # Free the committed system with the GIL released, before deleting
            # frames, which would otherwise copy every molecule it shares.
            system_time = system.time()
            system._free()

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
            num_waters = self._dynamics_cache.gcmc_num_waters(index)
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
                    f"(time = {system_time.to('ns'):.4f} ns) is {num_waters}{acc_str}"
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

        if not self._config.save_trajectories:
            return

        for i in range(len(self._lambda_values)):
            traj_filename = self._filenames[i]["trajectory"]
            chunk_pattern = f"{self._filenames[i]['trajectory_chunk']}*"
            traj_chunks = sorted(_glob_local(chunk_pattern))

            if not traj_chunks:
                continue

            self._join_trajectory_chunks(traj_chunks, traj_filename, append=True)

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

        for slot in range(self._dynamics_cache.num_slots):
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
            self._dynamics_cache.adaptive_data = self._ranker.to_dict()
        else:
            self._dynamics_cache.adaptive_data = None
        with open(self._repex_state, "wb") as f:
            _pickle.dump(self._dynamics_cache, f)

    def _save_sampler_stats(self):
        """
        Save GCMC and terminal flip sampler statistics to the dynamics cache
        prior to pickling.
        """
        self._dynamics_cache.gcmc_stats = self._merge_gcmc_stats()

        if self._terminal_flip_samplers is not None:
            self._dynamics_cache.terminal_flip_stats = [
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
