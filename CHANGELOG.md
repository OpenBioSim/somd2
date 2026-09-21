Changelog
=========

[2026.3.0](https://github.com/openbiosim/somd2/compare/2026.2.0...2026.3.0) - XXXX
----------------------------------------------------------------------------------

* Please add an item to this CHANGELOG for any new features or bug fixes when creating a PR.
* Increase the default `cutoff` from 7.5 Å to 9 Å, matching common practice and giving faster PME on current GPUs, since the shorter cutoff shifts too much work onto the reciprocal space grid [#209](https://github.com/OpenBioSim/somd2/pull/209).
* Buffer energy components and write them at checkpoint time, rather than rewriting the parquet file on every energy save, which cost a few milliseconds per replica per cycle and grew with the length of the run [#212](https://github.com/OpenBioSim/somd2/pull/212).
* Silence Sire's progress bars when a runner is constructed rather than when `somd2` is imported, so that importing `somd2` as a library no longer changes how Sire reports progress [#215](https://github.com/OpenBioSim/somd2/pull/215).
* Save the replica exchange state once at the end of a run rather than twice when the last cycle is a checkpoint cycle, and include the GCMC statistics in the final save [#218](https://github.com/OpenBioSim/somd2/pull/218).

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
