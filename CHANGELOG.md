Changelog
=========

[2026.3.0](https://github.com/openbiosim/somd2/compare/2026.2.0...2026.3.0) - XXXX
----------------------------------------------------------------------------------

* Please add an item to this CHANGELOG for any new features or bug fixes when creating a PR.
* Increase the default `cutoff` from 7.5 Å to 9 Å, matching common practice and giving faster PME on current GPUs, since the shorter cutoff shifts too much work onto the reciprocal space grid [#209](https://github.com/OpenBioSim/somd2/pull/209).
* Buffer energy components and write them at checkpoint time, rather than rewriting the parquet file on every energy save, which cost a few milliseconds per replica per cycle and grew with the length of the run [#212](https://github.com/OpenBioSim/somd2/pull/212).
* Silence Sire's progress bars when a runner is constructed rather than when `somd2` is imported, so that importing `somd2` as a library no longer changes how Sire reports progress [#215](https://github.com/OpenBioSim/somd2/pull/215).
* Save the replica exchange state once at the end of a run rather than twice when the last cycle is a checkpoint cycle, and include the GCMC statistics in the final save [#218](https://github.com/OpenBioSim/somd2/pull/218).
* Fall back to the Aldeghi Boresch restraint search protocol when the default RXRX protocol can't be used, e.g. for ligands with no N/O atoms to act as hydrogen-bond partners. Topology failures are now detected before the restraint search trajectory is run [#223](https://github.com/OpenBioSim/somd2/pull/223).
* Write the replica exchange state and GCMC ghost residues under the same file lock as the checkpoint files, so that a process reading the output directory can't see files from different cycles. The regular runner's final remainder block now backs up its checkpoint and saves sampler statistics, matching the other blocks [#225](https://github.com/OpenBioSim/somd2/pull/225).
* Return the `annihilate` and `decouple` lambda schedules from `Config.lambda_schedule`, built from the current soft-core settings, rather than `None` until a runner is created [#229](https://github.com/OpenBioSim/somd2/pull/229).
* Fixed `charge_scale_factor` being ignored for the `charge_scaled_morph` lambda schedule, which always used a factor of 0.2 [#229](https://github.com/OpenBioSim/somd2/pull/229).
* Fixed the ABFE lambda schedules ignoring the `restraint_lever` of a user-supplied Boresch restraint, which could leave the restraint uncoupled from the schedule [#229](https://github.com/OpenBioSim/somd2/pull/229).
* Fixed replica exchange applying the inverse of the accepted permutation when mixing, which sent configurations to the wrong lambda windows whenever accepted swaps formed a cycle of three or more replicas. The replica exchange transition matrix now also records each configuration's move [#232](https://github.com/OpenBioSim/somd2/pull/232).
* Add an `auto` option for `softcore_form`, which is now the default. It uses the Beutler form for the `annihilate` and `decouple` lambda schedules and the Zacharias form otherwise, including for custom schedules. The resolved form is recorded in the config file, so ABFE runs started with the previous default must set `softcore_form="zacharias"` to restart [#234](https://github.com/OpenBioSim/somd2/pull/234).
* Add `pme_alpha`, `pme_grid` and `pme_spacing` options to set the PME parameters explicitly, a `pme_tolerance` option for the Ewald error tolerance, and a `tune_pme` option, on by default, which picks the smallest PME grid that is at least as accurate as the parameters chosen from `pme_tolerance` on the CUDA and OpenCL platforms. The tuned parameters and their relative force error are logged, saved to `pme_parameters.yaml`, and reused on restart. Restarts of runs without this file use the default parameters [#235](https://github.com/OpenBioSim/somd2/pull/235).
* Add a `somd2-view` web viewer for monitoring and analysing one or more output directories, showing progress, end-state depictions, the MBAR free energy and overlap, replica exchange and sampler statistics, and the lambda schedule. Use the `--view` option to launch it alongside a simulation [#236](https://github.com/OpenBioSim/somd2/pull/236).
* Fixed alchemical ion creation when an existing counter-ion is perturbed to water. The water was recorded as the alchemical ion, so it got the wrong lambda schedule, and the same counter-ion was reused for every unit of charge. The furthest counter-ions from the perturbable molecule are now used, once each, whenever the system has no free ion of the required type [#239](https://github.com/OpenBioSim/somd2/pull/239).
* Raise an error when the `annihilate` or `decouple` lambda schedule is used with a perturbable molecule created by `sire.morph.annihilate()`, which also removes the bonded terms. ABFE ligands should be created with `sire.morph.decouple()`, leaving the choice between decoupling and annihilation to the lambda schedule [#239](https://github.com/OpenBioSim/somd2/pull/239).
* Save the GCMC end-state topologies, which include the additional waters, at the first checkpoint of each run as well as the last, and keep them on restart, so that they match the trajectory and checkpoints while a simulation is running, if it crashes, or if a restart never reaches a checkpoint [#244](https://github.com/OpenBioSim/somd2/pull/244).
* Add `gcmc_batch_size` and `gcmc_num_attempts` options to set the number of GCMC trials evaluated in parallel per batch and in total per move, which were previously fixed at the defaults of the GCMC sampler. Both can be changed on restart, e.g. to reduce wasted trials when more than one is accepted per move [#246](https://github.com/OpenBioSim/somd2/pull/246).
* Add adaptive replica exchange, enabled with `adaptive_repex`, which concentrates sampling on the lambda windows that are slowest to converge. Windows are ranked periodically by the uncertainty they contribute to the free energy, and only the highest ranked are propagated and exchanged every cycle, with the rest joining every `adaptive_mix_interval` cycles [#247](https://github.com/OpenBioSim/somd2/pull/247).
* Require `frame_frequency` to be a multiple of `gcmc_frequency` for GCMC simulations, so that every frame has a matching ghost water record. Previously some combinations silently saved frames without one [#247](https://github.com/OpenBioSim/somd2/pull/247).
* Fixed the regular runner's final block when the `runtime` isn't a multiple of the `checkpoint_frequency`, which ran a single integration step rather than the remaining time, without GCMC moves. Each checkpoint now appends exactly the energies sampled since the last one, so energies are no longer duplicated, or lost on restart when `checkpoint_frequency` is zero. Blocks that aren't a whole number of GCMC or terminal flip intervals no longer overrun, and energy components are time stamped correctly after a restart [#248](https://github.com/OpenBioSim/somd2/pull/248).
* Require the `runtime`, `checkpoint_frequency`, `energy_frequency`, `frame_frequency`, `gcmc_frequency` and `terminal_flip_frequency` to be multiples of the `timestep` for the regular runner, which runs in whole integration steps. GCMC and terminal flip moves now keep their frequency across checkpoints [#248](https://github.com/OpenBioSim/somd2/pull/248).
* Speed up checkpointing, which could take minutes for large systems. Sire now uses every CPU thread while checkpoint files are written, unless `max_sire_threads` is set, and replica exchange checkpoints replicas in parallel, writing the shared GCMC topologies once rather than once per replica. The regular runner's worker processes now also apply the Sire thread limit, which previously only took effect in the parent process [#251](https://github.com/OpenBioSim/somd2/pull/251).
* Reduce the setup time of replica exchange simulations that re-use OpenMM contexts across lambda values. Minimisation and equilibration now create their dynamics objects once per context, rather than once per replica [#255](https://github.com/OpenBioSim/somd2/pull/255).
* Further speed up checkpointing, and join the final trajectory without reloading the topology [#257](https://github.com/OpenBioSim/somd2/pull/257).
* Record frame times in final trajectories, rather than zero for every frame [#257](https://github.com/OpenBioSim/somd2/pull/257).
* Fixed the Boresch restraint for ABFE and the Morse restraints for ring-breaking perturbations not being generated when any other restraint was passed. They are now generated unless restraints of the same type are passed, and added to the others [#259](https://github.com/OpenBioSim/somd2/pull/259).
* Fixed alchemical ions changing their charge linearly over the whole lambda range for the `annihilate`, `decouple`, `charge_scaled_morph`, `ring_break_morph` and `reverse_ring_break_morph` lambda schedules, rather than alongside the perturbable molecule. A warning is raised for custom lambda schedules, where alchemical ions still use the `standard_morph` schedule [#263](https://github.com/OpenBioSim/somd2/pull/263).
* Add a `ghostly_kwargs` option to pass options to `ghostly` when applying ghost atom modifications, e.g. `--ghostly-kwargs k_soft=3 stiffen_rotamers=true` on the command line. Options are validated against those supported by `ghostly`. Systems that have already been modified with `ghostly` raise an error unless `ghost_modifications` is disabled, or `force=True` is passed in `ghostly_kwargs` [#264](https://github.com/OpenBioSim/somd2/pull/264).
* Allow `equilibration_time` to be a list of times, one for each lambda window, e.g. `--equilibration-time 0ps 100ps 100ps 0ps` on the command line. Windows with a time of zero are not equilibrated. A single time is still used for every window [#265](https://github.com/OpenBioSim/somd2/pull/265).

[2026.2.0](https://github.com/openbiosim/somd2/compare/2026.1.0...2026.2.0) - Sep 2026
--------------------------------------------------------------------------------------

* Add support for generating Boresch restraints for absolute binding free energy calculations [#166](https://github.com/OpenBioSim/somd2/pull/166).
* Give alchemical ions their own plain morph lambda schedule so they interpolate correctly under non-standard lambda schedules [#169](https://github.com/OpenBioSim/somd2/pull/169).
* Persist alchemical ion identity across restarts so the same molecule is reused regardless of GCMC state [#172](https://github.com/OpenBioSim/somd2/pull/172).
* Use perisistent `ThreadPoolExector` objects within the main replica exchange dynamics block [#175](https://github.com/OpenBioSim/somd2/pull/175).
* Allow `oversubscription_factor` to change on restart [#177](https://github.com/OpenBioSim/somd2/pull/177).
* Restrict energy component decomposition to force groups that are used for integration [#180](https://github.com/OpenBioSim/somd2/pull/180).
* Parallelise replica mixing [#181](https://github.com/OpenBioSim/somd2/pull/181).
* Fixed the replica exchange GPU memory check querying the wrong device when `CUDA_VISIBLE_DEVICES` does not start at zero, since OpenMM numbers devices relative to the visible set whereas `pynvml` enumerates all of them [#183](https://github.com/OpenBioSim/somd2/issues/183).
* Store GCMC sampling statistics per lambda value, converting those from earlier checkpoints on restart [#184](https://github.com/OpenBioSim/somd2/pull/184).
* Link restart systems to the reference end state rather than the perturbed one, since that is the coordinate set that dynamics maintains. Perturbable molecules were otherwise resumed from the coordinates they were built with [#189](https://github.com/OpenBioSim/somd2/pull/189).
* Add `max_contexts` to cap the number of OpenMM contexts used for replica exchange, re-using each across lambda values so that GPU memory no longer limits the number of replicas [#191](https://github.com/OpenBioSim/somd2/pull/191).
* Skip minimisation on restart [#191](https://github.com/OpenBioSim/somd2/pull/191).
* Pre-equilibrate the water with GCMC moves before minimising in the regular `Runner`, making it consistent with the `RepexRunner`, which already did so to stop the geometry relaxing into a dry pocket [#191](https://github.com/OpenBioSim/somd2/pull/191).
* Add a `precision` option for GPU platforms, defaulting to `single` [#191](https://github.com/OpenBioSim/somd2/pull/191).
* Add support for generating Morse restraints for ring-breaking perturbations [#194](https://github.com/OpenBioSim/somd2/pull/194).
* Remove the unused `kappa` lever equations from the ring-breaking/making lambda schedules [#195](https://github.com/OpenBioSim/somd2/pull/195).
* Accept stream file paths for the `restraints` and `lambda_schedule` configuration options, so they can be set from the command line [#198](https://github.com/OpenBioSim/somd2/pull/198).
* Account for off-site charges (virtual sites) when computing the charge difference between the end states. They are held as a molecule property rather than on the atoms, so a charge-preserving perturbation could appear to change charge and be given spurious alchemical ions [#200](https://github.com/OpenBioSim/somd2/pull/200).
* Handle `num_lambda=1`, which previously raised a `ZeroDivisionError` when generating the lambda values. The `RepexRunner` now rejects a single lambda window, since there is nothing to exchange with and the regular `Runner` is faster [#203](https://github.com/OpenBioSim/somd2/pull/203).
* Detect the available GPUs once in the base runner and re-use the list, rather than the `RepexRunner` querying `CUDA_VISIBLE_DEVICES` regardless of the chosen platform. Replica exchange is now also permitted on the HIP platform [#206](https://github.com/OpenBioSim/somd2/pull/206).
* Query the free memory of AMD GPUs with `CL_DEVICE_GLOBAL_FREE_MEMORY_AMD` rather than `CL_DEVICE_BOARD_NAME_AMD`, which returns the device name [#206](https://github.com/OpenBioSim/somd2/pull/206).

[2026.1.0](https://github.com/openbiosim/somd2/compare/2025.1.0...2026.1.0) - Jun 2026
--------------------------------------------------------------------------------------

* Improve constraint handling during minimisation and equilibration [#80](https://github.com/OpenBioSim/somd2/pull/80)
* Add support for GCMC on the OpenCL platform [#115](https://github.com/OpenBioSim/somd2/pull/115)
* Expose ring-breaking/making lambda schedules [#129](https://github.com/OpenBioSim/somd2/pull/129)
* Add support for Terminal Flip Monte Carlo [#138](https://github.com/OpenBioSim/somd2/pull/138)
* Add support for per-force energy decomposition [#143](https://github.com/OpenBioSim/somd2/pull/143)
* Add support for long-range dispersion correction and Beutler softcore [#147](https://github.com/OpenBioSim/somd2/pull/147)
* Add support for GCMC in the osmotic ensemble [#151](https://github.com/OpenBioSim/somd2/pull/151)
* Improve handling of simulation restarts via a `.done` sentinel file [#153](https://github.com/OpenBioSim/somd2/pull/153)
* Reduce checkpoint memory footprint by storing `NumPy` arrays in the replica exchange state pickle file [#155](https://github.com/OpenBioSim/somd2/pull/155)
* Remove redundant `s3` checkpoint files [#157](https://github.com/OpenBioSim/somd2/pull/157)
* Unconditionally apply AMBER water topology conversion to ensure fully rigid water constraints [#163](https://github.com/OpenBioSim/somd2/pull/163)

[2025.1.0](https://github.com/OpenBioSim/somd2/releases/tag/2025.1.0) - Nov 2025
-------------------------------------------------------------------------------

* Initial public release.
