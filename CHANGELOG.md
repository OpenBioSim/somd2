Changelog
=========

[2026.2.0](https://github.com/openbiosim/somd2/compare/2026.1.0...2026.2.0) - ********
--------------------------------------------------------------------------------------

* Please add an item to this CHANGELOG for any new features or bug fixes when creating a PR.
* Add support for generating Boresch restraints for absolute binding free energy calculations [#166](https://github.com/OpenBioSim/somd2/pull/166).
* Give alchemical ions their own plain morph lambda schedule so they interpolate correctly under non-standard lambda schedules [#169](https://github.com/OpenBioSim/somd2/pull/169).
* Persist alchemical ion identity across restarts so the same molecule is reused regardless of GCMC state [#172](https://github.com/OpenBioSim/somd2/pull/172).
* Use perisistent `ThreadPoolExector` objects within the main replica exchange dynamics block [#175](https://github.com/OpenBioSim/somd2/pull/175).
* Allow `oversubscription_factor` to change on restart [#177](https://github.com/OpenBioSim/somd2/pull/177).
* Restrict energy component decomposition to force groups that are used for integration [#180](https://github.com/OpenBioSim/somd2/pull/180).

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
