# SOMD2

Open-source GPU accelerated molecular dynamics engine for alchemical free-energy
simulations. Built on top of [Sire](https://github.com/OpenBioSim/sire) and [OpenMM](https://github.com/openmm/openmm). The code is still under active development and is not yet ready for general use.

## Installation

First create a conda environment using the provided environment file:

```
mamba create -f environment.yaml
```

(We recommend using [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).)

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
explicity set on the command line with override those set via the config file.

## Analysis

Simulation output will be written to the directory specified using the
`--output-directory` parameter. This will contain a number of files, including
Parquet files for the energy trajectories of each λ window. These can be
processed using [BioSimSpace](https://github.com/OpenBioSim/biosimspace) as follows:

```python
import BioSimSpace as BSS

pmf, overlap = BSS.FreeEnergy.Relative.analyse("output")
```

(Here we assume that the output directory is called `output`.)
