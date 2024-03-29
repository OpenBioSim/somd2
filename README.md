# SOMD2

[![GitHub Actions](https://github.com/openbiosim/somd2/actions/workflows/main.yaml/badge.svg)](https://github.com/openbiosim/somd2/actions/workflows/main.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Open-source GPU accelerated molecular dynamics engine for alchemical free-energy
simulations. Built on top of [Sire](https://github.com/OpenBioSim/sire) and [OpenMM](https://github.com/openmm/openmm). The code is still under active development and is not yet ready for general use.

## Installation

First create a conda environment using the provided environment file:

```
mamba create -f environment.yaml
```

(We recommend using [Miniforge](https://github.com/conda-forge/miniforge).)

Now install `somd2` into the environment:

```
mamba activate somd2
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
