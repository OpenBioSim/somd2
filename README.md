<p align="center">
    <picture align="center">
        <img alt="SOMD2" src="./.img/somd2.png" width="50%"/>
    </picture>
</p>

# SOMD2

[![GitHub Actions](https://github.com/openbiosim/somd2/actions/workflows/devel.yaml/badge.svg)](https://github.com/openbiosim/somd2/actions/workflows/devel.yaml)
[![Conda Version](https://anaconda.org/openbiosim/somd2/badges/downloads.svg)](https://anaconda.org/openbiosim/somd2)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Open-source GPU accelerated molecular dynamics engine for alchemical free-energy
simulations. Built on top of [Sire](https://github.com/OpenBioSim/sire) and [OpenMM](https://github.com/openmm/openmm).

## Features

- **Perturbations**: relative binding free energies,
  [absolute binding free energies](#absolute-binding-free-energies),
  [ring-breaking](#ring-breaking-perturbations),
  [charge-change](#charge-change-perturbations), and protein mutations.
- **[GCMC](#gcmc)**: grand canonical Monte Carlo water sampling.
- **[Replica exchange](#replica-exchange)**: Hamiltonian replica exchange
  between lambda windows.
- **[REST2](#rest2)**: replica exchange with solute scaling.
- **[Terminal ring flips](#terminal-ring-flip-monte-carlo)**: Monte Carlo moves
  to improve sampling of terminal aromatic rings.
- **[Ghost atom modifications](#ghost-atom-modifications)**: modification of
  ghost atom bonded terms to avoid spurious coupling to the physical system.
- **[Multiple GPUs](#running-somd2-using-one-or-more-gpus)**: lambda windows are
  distributed across the available devices, with optional
  [oversubscription](#gpu-oversubscription).

## Installation

### Conda package

Install `somd2` directly from the `openbiosim` channel:

```
conda install -c conda-forge -c openbiosim somd2
```

Or, for the development version:

```
conda install -c conda-forge -c openbiosim/label/dev somd2
```

### Installing from source (standalone)

To install from source using [pixi](https://pixi.sh), which will
automatically create an environment with all required dependencies
(including pre-built [Sire](https://github.com/OpenBioSim/sire),
[BioSimSpace](https://github.com/OpenBioSim/biosimspace),
[Ghostly](https://github.com/OpenBioSim/ghostly), and
[Loch](https://github.com/OpenBioSim/loch)):

```
git clone https://github.com/openbiosim/somd2
cd somd2
pixi install
pixi shell
pip install -e .
```

### Installing from source (full OpenBioSim development)

If you are developing across the full OpenBioSim stack, first install
[Sire](https://github.com/OpenBioSim/sire) from source by following the
instructions [here](https://github.com/OpenBioSim/sire#installation), then
activate its pixi environment:

```
pixi shell --manifest-path /path/to/sire/pixi.toml -e dev
```

You may also need to install other packages from source, e.g.
[BioSimSpace](https://github.com/OpenBioSim/biosimspace),
[Ghostly](https://github.com/OpenBioSim/ghostly), and
[Loch](https://github.com/OpenBioSim/loch):

```
pip install -e /path/to/biosimspace
pip install -e /path/to/ghostly
pip install -e /path/to/loch
```

Then install `somd2` into the environment:

```
pip install -e .
```

> [!NOTE]
> Pixi does not run conda post-link scripts, so the `ocl-icd-system`
> symlink needed for OpenCL won't be created automatically. After
> creating the environment (or after a pixi update), run the following
> to fix this:
>
> ```bash
> pixi shell
> ln -sfn /etc/OpenCL/vendors "${CONDA_PREFIX}/etc/OpenCL/vendors/ocl-icd-system"
> ```

### Testing

You should now have a `somd2` executable in your path. To test, run:

```
somd2 --help
```

### Keeping up to date

During a development cycle the OpenBioSim packages are pinned only to a
`YYYY.N.0.dev` version, not to a specific build. `somd2` and its dependencies
therefore need to be kept in sync, so always update the whole stack together
rather than `somd2` alone.

For a conda install, update everything in one go:

```
conda update -c conda-forge -c openbiosim/label/dev sire biosimspace ghostly loch somd2
```

For a standalone pixi install, pull the latest `somd2` and refresh the
pre-built dependencies:

```
git pull
pixi update
```

For a full source install, `git pull` in *every* repository you have installed
(`sire`, `biosimspace`, `ghostly`, `loch` and `somd2`), not just `somd2`. Since
`sire` is compiled, you will also need to rebuild it.

## Development

Pre-commit hooks are used to ensure consistent code formatting and linting.
To set up pre-commit in your development environment:

```
pixi shell -e dev
pre-commit install
```

This will run [ruff](https://docs.astral.sh/ruff/) formatting and linting
checks automatically on each commit. To run the checks manually against all
files:

```
pre-commit run --all-files
```

## Usage

In order to run an alchemical free-energy simulation you will need to
first create a stream file containing the *perturbable* system of interest.
This can be created using [BioSimSpace](https://github.com/OpenBioSim/biosimspace).
For example, following the tutorial
[here](https://biosimspace.openbiosim.org/tutorials/hydration_freenrg.html).
Once the system is created, it can be streamed to file using, e.g.:

```python
import BioSimSpace as BSS

BSS.Stream.save(system, "perturbable_system")
```

You can then run a simulation with:

```
somd2 perturbable_system.bss
```

The help message provides information on all of the supported options, along
with their default values. Options can be specified on the command line, or
using a YAML configuration file, passed with the `--config` option. Any options
explicitly set on the command line will override those set via the config file.

An example perturbable system for a methane to ethanol perturbation in solvent
can be found [here](https://sire.openbiosim.org/m/merged_molecule.s3.bz2).
This is a `bzip2` compressed file that will need to be extracted before use.

A larger collection of input files and end-to-end tutorials, covering everything
from a simple charge-change validation system to full case studies, can be found
in the [somd2_examples](https://github.com/OpenBioSim/somd2_examples) repository.

### Running SOMD2 using one or more GPUs

In order to run using GPUs you will first need to set the relevant environment
variable. For example, to run using 4 CUDA enabled GPUs set `CUDA_VISIBLE_DEVICES=0,1,2,3`
(for OpenCL and HIP use `OPENCL_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` respectively).

By default SOMD2 will run using the CPU platform, however if the relevant
environment variable has been set (as above) the new platform will be detected
and set. In the case that this detection fails, or if there are multiple platforms
available, the `--platform` option can be set (for example `--platform cuda`).

By default, SOMD2 will automatically manage the distribution of lambda windows
across all listed devices. In order to restrict the number of devices used
the `--max-gpus` option can be set, for example setting `--max-gpus 2` while
`CUDA_VISIBLE_DEVICES` are set as above would restrict SOMD2 to using only
GPUs 0 and 1.

## Replica exchange

SOMD2 supports Hamiltonian replica exchange (HREX) simulations, which can be
enabled using the `--replica-exchange` option. By default, dynamics contexts are
created up-front for all replicas, so this can be memory intensive. As such,
replica exchange is intended for use on multi-GPU nodes with a large amount of
memory. It is also possible to oversubscribe the GPUs, i.e. have more than one
replica running on a GPU at a time. This can be controlled via the
`--oversubscription-factor` option, e.g. a value of 2 would allow 2 replicas to
run on each GPU at a time. This requires the NVIDIA multi-process service (MPS)
to be enabled, see [GPU oversubscription](#gpu-oversubscription) below.

If the number of replicas you want doesn't fit in GPU memory, use the
`--max-contexts` option to cap the number of contexts that are created. Each
context is then re-used to propagate several replicas per cycle, changing its
lambda value as it goes, so the number of replicas is no longer limited by
memory. For example, `--num-lambda 24 --max-contexts 4` runs 24 replicas using
the memory of 4. This costs some performance, since the replicas sharing a
context run one after another rather than at the same time, so only use it when
one context per replica won't fit. When contexts are re-used, `--frame-frequency`
must equal `--checkpoint-frequency`.

For optimal performance, it is recommended that the number of contexts, i.e. the
number of replicas, or `--max-contexts` if it is set, be a multiple of the number
of GPUs, and no smaller than the number of GPUs multiplied by the
oversubscription factor. SOMD2 will warn you if this isn't the case.

Changing the lambda value of a context requires it to be reinitialised whenever a
constrained bond length actually perturbs with lambda, which is slow. If this
overhead is significant, pass `--no-update-constraints` to freeze the
constrained bond lengths at those of a single lambda value, chosen with
`--constraint-lambda-index`. Both options are ignored unless contexts are being
re-used.

The swap frequency for replica exchange is controlled by the `--energy-frequency`
option, i.e. we compute the energies for all replicas at this frequency, then
attempt to mix the replicas. A larger value will improve performance, but may
reduce the efficiency of the exchange.

## REST2

We also support Replica Exchange with Solute Scaling
([REST2](https://pubs.acs.org/doi/10.1021/jp204407d)) simulations to facilitate sampling for perturbations
involving conformational changes, e.g. ring flips. This can be enabled
using the `--rest2-scale` option, which specifies the "temperature" of the
REST2 region relative to the rest of the system. By default, the REST2 region
comprises *all* atoms in perturbable molecules, but can be controlled via the
`--rest2-selection` option. This should be a `Sire` selection string that specifies
additional atoms of interest, i.e. those in regular, non-perturbable molecules.
If the selection does contain atoms within perturbable molecules, then only
those atoms within the perturbable molecules will be considered as part of the
REST2 region, i.e. you can select a sub-set of atoms within a perturbable
molecule to be scaled.

By default, the REST2 schedule is a triangular function that starts and ends
at 1.0, with a peak at the middle of the lambda schedule corresponding to
the value of `--rest2-scale`. By passing multiple values for `--rest2-scale`, the
user can fully control the schedule. When doing so, the number of values must
match the number of lambda windows.

## GCMC

SOMD2 also supports grand canonical Monte Carlo (GCMC) water sampling using
the [loch](https://github.com/OpenBioSim/loch) package. This can be enabled
using the `--gcmc` option. To define a GCMC region, use the `--gcmc-selection`
option, which should be a `Sire` selection string that specifies the atoms
defining the centre of geometry for the GCMC region. The radius of the GCMC
sphere can be controlled using the `--gcmc-radius` option. To see all GCMC
related options, run:

```
somd2 --help | grep -A2 '  --gcmc'
```

> [!NOTE]
> GCMC is only supported when using the CUDA or OpenCL platforms.

When using the CUDA platform, make sure that `nvcc` is in your `PATH`. If you
require a different `nvcc` to that provided by conda, you can set the
`PYCUDA_NVCC` environment variable to point to the desired `nvcc` binary.
Depending on your setup, you may also need to install the `cuda-nvvm` package
from `conda-forge`.

## Terminal ring flip Monte Carlo

SOMD2 supports terminal ring flip Monte Carlo (MC) moves to improve sampling
of terminal aromatic rings in perturbable ligands, as described in
[this paper](https://doi.org/10.26434/chemrxiv-2025-2zkx5).
Each move attempts a discrete rotation of a terminal ring around the bond
connecting it to the rest of the molecule, accepted or rejected via the
Metropolis criterion. Terminal ring groups are detected automatically from
the molecular connectivity of perturbable molecules.

To enable terminal flip MC, set the frequency at which moves are attempted:

```
somd2 perturbable_system.bss --terminal-flip-frequency "1 ps"
```

The flip angle for each group is determined automatically from the ring
geometry. To override this for all groups:

```
somd2 perturbable_system.bss --terminal-flip-frequency "1 ps" --terminal-flip-angle "180 degrees"
```

## Lambda schedules

The way that the perturbation is applied across the lambda coordinate is
controlled by the `--lambda-schedule` option, which defaults to
`standard_morph`, which is intended for use with relative binding free
energy (RBFE) simulations. The available schedules are:

| Schedule | Description |
| --- | --- |
| `standard_morph` | Linear interpolation between the two end states. |
| `charge_scaled_morph` | As above, but with charges scaled at intermediate lambda values. |
| `annihilate` | Absolute binding free energies, removing all non-bonded interactions. |
| `decouple` | Absolute binding free energies, removing only intermolecular interactions. |
| `ring_break_morph` | Ring-breaking perturbations. |
| `reverse_ring_break_morph` | Ring-making perturbations, i.e. the reverse of the above. |

For the `annihilate`, `decouple`, and ring-breaking schedules, appropriate
restraints can be generated automatically. See the sections below.

## Absolute binding free energies

Absolute binding free energy (ABFE) calculations are supported using the
`annihilate` and `decouple` lambda schedules. Both first discharge the ligand,
then remove its Lennard-Jones interactions: `annihilate` removes all non-bonded
interactions, including those within the ligand, whereas `decouple` retains the
intramolecular terms.

```
somd2 perturbable_system.bss --lambda-schedule decouple
```

The ligand must be restrained within the binding site. If no restraints are
passed, a Boresch restraint is generated automatically for the bound leg, i.e.
when the system contains both a protein and water. This is done by minimising
the system, running a short trajectory at lambda = 0, then choosing the anchor
atoms and force constants from it. The length of this trajectory and the
frequency at which frames are saved can be controlled with the
`--restraint-search-time` and `--restraint-search-frequency` options. By
default the receptor anchor atoms are chosen from the protein backbone; use
`--restraint-search-receptor-selection` to pass a `Sire` selection string
instead.

The restraint is written to `abfe_restraint.s3` in the output directory and is
reloaded on restart, since the accumulated free energy corresponds to that
particular restraint. The standard state correction is logged and written to
the metadata of the energy trajectory, so analysis code can apply it without
needing to scan the log.

> [!NOTE]
> The Beutler soft-core form, enabled with `--softcore-form beutler`, is only
> supported with the ABFE schedules, or a custom schedule.

## Ring-breaking perturbations

Perturbations that break (or form) a ring are supported using the
`ring_break_morph` schedule, or `reverse_ring_break_morph` for the ring-making
direction.

```
somd2 perturbable_system.bss --lambda-schedule ring_break_morph
```

These perturbations require a pair of Morse restraints on the atoms of the bond
that is broken. If no restraints are passed, both are generated automatically.
A "hard" Morse potential replaces the harmonic bond, inheriting its force
constant and equilibrium length, and is switched off as a weaker "soft" Morse
restraint holds the fragment in place. Their well depths and the force constant
of the soft restraint can be controlled with the `--morse-hard-well-depth`,
`--morse-soft-well-depth`, and `--morse-soft-force-constant` options.

Unlike the ABFE restraints, these are regenerated on each run rather than being
cached, since they are derived from the bond parameters alone and are therefore
identical every time.

> [!NOTE]
> The defaults are a reasonable starting point, but ring-breaking
> perturbations are demanding. A non-uniform spacing of lambda values, set with
> `--lambda-values`, is typically needed to obtain good overlap around the point
> at which the bond is broken. The
> [alchemate](https://github.com/akalpokas/alchemate) package provides
> workflows for iteratively optimising the lambda schedule.

## Charge-change perturbations

Perturbations that change the net charge of the system are handled
automatically using the co-alchemical ion method. The charge difference between
the two end states is computed when the system is loaded, and, if it is
non-zero, a number of water molecules equal to the absolute charge difference
are perturbed into counter-ions alongside the main perturbation, keeping the
total charge constant at every lambda value. The waters furthest from the
perturbable molecule are chosen, and the ion type is picked to offset the
charge change, re-using the parameters of a free ion already present in the
system where possible.

No options are needed to enable this. The automatically detected value can be
overridden with `--charge-difference`, which takes the perturbed charge minus
the reference charge:

```
somd2 perturbable_system.bss --charge-difference -1
```

The molecules chosen as alchemical ions are written to `alchemical_ions.npz` in
the output directory and reused on restart, so that ion selection does not
depend on anything that might have changed between runs.

Since a co-alchemical ion is only meaningful in the bulk, SOMD2 can restrain it
away from the perturbable region. Passing a distance to
`--coalchemical-restraint-dist` adds an inverse-distance restraint between each
ion and the atom closest to the centre of geometry of the perturbable molecule,
preventing the ion from drifting into the binding site and interacting with the
protein or ligand:

```
somd2 perturbable_system.bss --coalchemical-restraint-dist "10 A"
```

> [!NOTE]
> These restraints are *added* to any others in use. Restraints passed via the
> Python API, and those generated automatically for the ABFE and ring-breaking
> schedules described above, are all retained.

## Debugging with energy components

To help diagnose simulation instabilities, SOMD2 can record the potential
energy contribution from each OpenMM force group. This is enabled with the
`--save-energy-components` flag:

```
somd2 perturbable_system.bss --save-energy-components
```

One Parquet file per λ window is written to the output directory, named
`energy_components_<lambda>.parquet`. Times are in nanoseconds and energies in
kcal/mol; both are stored as schema metadata in the file.

The recording interval depends on the runner and active samplers:

- **Replica exchange**: always `energy-frequency`
- **Standard runner, no MC**: `energy-frequency`
- **Standard runner, with MC**: the shortest active MC frequency, i.e.
  `gcmc-frequency`, `terminal-flip-frequency`, or the smaller of the two
  when both are active

> [!NOTE]
> Energy components are written more frequently than checkpoint files and are
> not guarded by the file lock, so they may lead the checkpoint files by up
> to one `checkpoint-frequency` interval when copying output mid-simulation.

## Copying output files during a simulation

When SOMD2 writes checkpoint files it acquires an exclusive
[file lock](https://py-filelock.readthedocs.io) on `somd2.lock` inside the output
directory. This guarantees that checkpoint files are always in a consistent
state on disk.

If you want to copy the output directory while a simulation is running (for
example, to create a backup or to inspect intermediate results), acquire the
same lock first so that you do not copy files mid-write. On Linux/macOS this
can be done with the `flock` command:

```bash
flock /path/to/output/somd2.lock cp -r /path/to/output /destination
```

Or from Python using the [filelock](https://pypi.org/project/filelock/) package
(which `somd2` already depends on):

```python
from filelock import FileLock

with FileLock("/path/to/output/somd2.lock"):
    # copy files here
    ...
```

> [!NOTE]
> The `--timeout` option (default: `300 s`) controls how long SOMD2 will
> wait to re-acquire the lock after your copy completes. If you hold the lock
> for longer than this, the simulation will raise a `Timeout` error.

## Analysis

Simulation output will be written to the directory specified using the
`--output-directory` parameter. This will contain a number of files, including
[Parquet files](https://en.wikipedia.org/wiki/Apache_Parquet) for the energy
trajectories of each λ window. These can be processed using
[BioSimSpace](https://github.com/OpenBioSim/biosimspace) as follows:

```python
import BioSimSpace as BSS

pmf1, overlap1 = BSS.FreeEnergy.Relative.analyse("output1")
```

(Here we assume that the output directory is called `output1`.)

To compute the relative free-energy difference between two legs, e.g.
legs 1 and 2, you can use:

```python
pmf2, overlap2 = BSS.FreeEnergy.Relative.analyse("output2")

free_nrg = BSS.FreeEnergy.Relative.difference(pmf1, pmf2)
```

## Truncated MBAR analysis

When running HREX with a large number of replicas it can become computationally
expensive to compute energies. (We need the energies of each replica at each
lambda value.) As a shortcut, it's possible to truncate the neighbourhood of
windows for which we compute energies, then use a large null energy for the
remaining windows. This can be controlled via the `--num-energy-neighbours` option.
For example, setting this to 2 would compute energies for the current window and
its two neighbours on either side. The value assigned to the remaining windows
can be controlled via the `--null-energy` option. The number of neighbours should
be chosen as a trade off between accuracy and computational cost. A value of around
20% of the number of replicas has been found to be a good starting point.

## Ghost atom modifications

We support modification of ghost atom bonded terms to avoid spurious coupling
to the physical system using the approach described in
[this](https://pubs.acs.org/doi/10.1021/acs.jctc.0c01328) paper.
These are enabled by default, but can be disabled using the `--no-ghost-modifications`
option. Modifications are implemented using the [ghostly](https://github.com/OpenBioSim/ghostly)
package.

## Note for SOMD1 users

SOMD2 can be run in SOMD1 *compatibility* mode by passing the
`--somd1-compatibility` command-line option to the `somd2` executable. This ensures
that the perturbation used is consistent with the approach from SOMD1, i.e.
it uses the same modifications for bonded-terms involving dummy atoms as SOMD1.

Finally, it is also possible to run SOMD2 using an existing SOMD1 perturbation
file. To do so, you will also need to create a stream file representing the
λ = 0 state. For existing input generated by `prepareFEP.py`, this can be done as
follows. (This assumes that the output has a prefix `somd1`.)

```python
import BioSimSpace as BSS

# Load the lambda = 0 state from prepareFEP.py
system = BSS.IO.readMolecules(["somd1.prm7", "somd1.rst7"], reduce_box=True)

# Write a stream file.
BSS.Stream.save(system, "somd1")
```

(This will write a stream file called `somd1.bss`.)

This can then be run with SOMD2 using the following:

```
somd2 somd1.bss --pert-file somd1.pert --somd1-compatibility
```

(This only shows the limited options required. Others will take default values and can be set accordingly.)

If you want to load an existing system from a perturbation file and use the
new SOMD2 [ghost atom bonded-term modifications](https://github.com/OpenBioSim/ghostly),
then simply omit the `--somd1-compatibility` option.

## GPU oversubscription

If you have an NVIDIA GPU that supports the multi-process service (MPS), you can
oversubscribe the GPU to run multiple OpenMM contexts on the same GPU at once,
increasing the throughput of your simulation. To do this, you will need to first
enable MPS by running the following command:

```
nvidia-cuda-mps-control -d
```

The number of contexts that can be run in parallel is then controlled by the
`--oversubscription-factor` option, which defaults to 1.

More details on MPS, including tuning options, can be found in the following
[technical blog](https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/).

## Python API

SOMD2 can also be used as a Python API, allowing it to be embedded
within other Python scripts.

## Known issues

If using the regular `Runner` class via the Python API, then you will need to
guard calls to its `run()` method within a `if __name__ == "__main__":` block
since it uses multiprocessing with the `spawn` start method.

During a checkpoint cycle trajectory frames are stored in memory before being
paged to disk. When running replica exchange simulations with a large number
of replicas this can lead to exceeding the temporary file storage limit on
some systems, causing the simulation to hang. This can be resolved by either
reducing the frequency at which frames are stored, or checkpointing more
frequently. (Frames are written to disk and cleared from memory at each
checkpoint.)

PyMBAR uses JAX by default for GPU acceleration, which can cause issues in
some environments. If you encounter issues when analysing simulation output,
try setting the `PYMBAR_DISABLE_JAX` environment variable to `1`.
