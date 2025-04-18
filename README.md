# SOMD2

[![GitHub Actions](https://github.com/openbiosim/somd2/actions/workflows/main.yaml/badge.svg)](https://github.com/openbiosim/somd2/actions/workflows/main.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Open-source GPU accelerated molecular dynamics engine for alchemical free-energy
simulations. Built on top of [Sire](https://github.com/OpenBioSim/sire) and [OpenMM](https://github.com/openmm/openmm). The code is still under active development and is not yet ready for general use.

## Installation

First create a conda environment using the provided environment file:

```
conda env create -f environment.yaml
```

(We recommend using [Miniforge](https://github.com/conda-forge/miniforge).)

Now install `somd2` into the environment:

```
conda activate somd2
pip install --editable .
```

You should now have a `somd2` executable in your path. To test, run:

```
somd2 --help
```

## Usage

In order to run an alchemical free-energy simulation you will need to
first create a stream file containing the _perturbable_ system of interest.
This can be created using [BioSimSpace](https://github.com/OpenBioSim/biosimspace). For example, following the tutorial
[here](https://biosimspace.openbiosim.org/versions/2023.4.0/tutorials/hydration_freenrg.html). Once the system is created, it can be streamed to file using, e.g.:

```python
import BioSimSpace as BSS

BSS.Stream.save(system, "perturbable_system")
```

You can then run a simulation with:

```
somd2 perturtbable_system.bss
```

The help message provides information on all of the supported options, along
with their default values. Options can be specified on the command line, or
using a YAML configuration file, passed with the `--config` option. Any options
explicity set on the command line will override those set via the config file.

An example perturbable system for a methane to ethanol perturbation in solvent
can be found [here](https://sire.openbiosim.org/m/merged_molecule.s3.bz2).
This is a `bzip2` compressed file that will need to be extracted before use.

#### Running SOMD2 using one or more GPUs

In order to run using GPUs you will first need to set the relevant environment
variable. For example, to run using 4 CUDA enabled GPUS set `CUDA_VISIBLE_DEVICES=0,1,2,3`
(for openCL and HIP use `OPENCL_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` respectively).

By default `SOMD2` will run using the CPU platform, however if the relevant
environment variable has been set (as above) the new platform will be detected
and set. In the case that this detection fails, or if there are multiple platforms
available, the `--platform` option can be set (for example `--platform cuda`).

By default, `SOMD2` will automatically manage the distribution of lambda windows
across all listed devices. In order to restrict the number of devices used
the `--max_gpus` option can be set, for example setting `max_gpus=2` while
`CUDA_VISIBLE_DEVICES` are set as above would restrict `SOMD2` to using only
GPUs 0 and 1.

## Replica exchange

`SOMD2` supports Hamiltonian replica exchange (HREX) simulations, which can be
enabled using the `--replica-exchange` option. Note that dynamics contexts will
be created up-front for all replicas, so this can be memory intensive. As such,
replica exchange is intended for use on multi-GPU nodes with a large amount of
memory. For optimal performance, it is recommended that the number of replicas
be a multiple of the number of GPUs. It is also possible to oversubscribe the
GPUs, i.e. have more than one replica running on a GPU at a time. This can be
controlled via the `--oversubscription-factor` option, e.g. a value of 2 would
allow 2 replicas to run on each GPU at a time.

The swap frequency for replica exchange is controlled by the `energy-frequency`
option, i.e. we compute the energies for all replicas at this frequency, then
attempt to mix the replicas. A larger value will improve performance, but may
reduce the efficiency of the exchange.

## REST2

We also support Replica Exchange with Solute Scaling
([REST2](https://pubs.acs.org/doi/10.1021/jp204407d)) simulations to facilitate sampling for perturbations
involving conformational changes, e.g.  ring flips. This can be enabled
using the `--rest2-scale` option, which specifies the "temperature" of the
REST2 region relative to the rest of the system. By default, the REST2 region
comprises _all_ atoms in perturbable molecules, but can be controlled via the
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

## Analysis

Simulation output will be written to the directory specified using the
`--output-directory` parameter. This will contain a number of files, including
Parquet files for the energy trajectories of each λ window. These can be
processed using [BioSimSpace](https://github.com/OpenBioSim/biosimspace) as follows:

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
lamdba value.) As a shortcut, it's possible to truncate the neighbourhood of
windows for which we compute energies, then use a large null energy for the
remaining windows. This can be controlled via the `--num-energy-neighbours` option.
For example, setting this to 2 would compute energies for the current window and
its two neighbours on either side. The value assigned to the remaining windows
can be controlled via the `--null-energy` option. The number of neighbours should
be chosen as a trade off between accuracy and computational cost. A value of around
20% of the number of replicas has been found to be a good starting point.

## Note for SOMD1 users

For existing users of `somd1`, it's possible to generate input for `somd2` by passing
`--somd2 True` to the `prepareFEP.py` setup script. This will write a `somd2` compatible
stream file.

Additionally, `somd2` can be run in `somd1` _compatibility_ mode by passing the
``--somd1-compatibility`` command-line option to the `somd2` executable. This ensures
that the perturbation used is consistent with the approach from `somd1`, i.e.
it uses the same modifications for bonded-terms involving dummy atoms as `somd1`.

Finally, it is also possible to run `somd2` using an existing `somd1` perturbation
file. To do so, you will also need to create a stream file representating the
λ = 0 state. For existing input generated by `prepareFEP.py`, this can be done as
follows. (This assumes that the output has a prefix `somd1`.)

```python
import BioSimSpace as BSS

# Load the lambda = 0 state from prepareFEP.py
system = BSS.IO.readMolecules(["somd1.prm7", "somd1.rst7"])

# Write a stream file.
BSS.Stream.save(system, "somd1")
```

(This will write a stream file called `somd1.bss`.)

This can then be run with `somd2` using the following:

```
somd2 somd1.bss --pert-file somd1.pert --somd1-compatibility
```

(This only shows the limited options required. Others will take default values and can be set accordingly.)

If you want to load an existing system from a perturbation file and use the
new `somd2` ghost atom bonded-term modifications, then simply omit the
`--somd1-compatibility` option.
