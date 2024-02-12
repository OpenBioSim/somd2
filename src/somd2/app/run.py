######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2024
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

"""
The somd2 command line program.

Usage:
    To get the help for this program and list all of the
    arguments (with defaults) use:

    somd2 --help
"""


def cli():
    """
    SOMD2: Command line interface.
    """

    from argparse import Namespace

    from somd2 import _logger
    from somd2.config import Config
    from somd2.runner import Runner

    from somd2.io import yaml_to_dict

    # Store the somd2 version.
    from somd2._version import __version__

    # Store the sire version.
    from sire import __version__ as sire_version
    from sire import __revisionid__ as sire_revisionid

    # Generate the parser.
    parser = Config._create_parser()

    # Add simulation specific positional arguments.
    parser.add_argument(
        "system",
        type=str,
        help="Path to a stream file containing the perturbable system, "
        "or the reference system. If a reference system, then this must be "
        "combined with a perturbation file via the --pert-file argument.",
    )
    parser.add_argument(
        "--pert-file",
        type=str,
        required=False,
        help="Path to a file containing the perturbation to apply "
        "to the reference system.",
    )

    # Parse the arguments into a dictionary.
    args = vars(parser.parse_args())

    # Pop the YAML config, system, and pert file from the arguments dictionary.
    config = args.pop("config")
    system = args.pop("system")
    pert_file = args.pop("pert_file")

    # If set, read the YAML config file.
    if config is not None:
        # Convert the YAML config to a dictionary.
        config = yaml_to_dict(config)

        # Reparse the command-line arguments using the existing config
        # as a Namespace. Any non-default arguments from the command line
        # will override those in the config.
        args = vars(parser.parse_args(namespace=Namespace(**config)))

        # Re-pop the YAML config, system, and pert file from the arguments
        # dictionary.
        args.pop("config")
        args.pop("system")
        if pert_file is None:
            pert_file = args.pop("pert_file")

    # Instantiate a Config object to validate the arguments.
    config = Config(**args)

    # Log the versions of somd2 and sire.
    _logger.info(f"somd2 version: {__version__}")
    _logger.info(f"sire version: {sire_version}+{sire_revisionid}")

    # Try to apply the perturbation to the reference system.
    if pert_file is not None:
        _logger.info(f"Applying perturbation to reference system: {pert_file}")
        system = apply_pert(system, pert_file)

    # Instantiate a Runner object to run the simulation.
    runner = Runner(system, config)

    # Run the simulation.
    runner.run()


def apply_pert(system, pert_file):
    """
    Helper function to apply a perturbation to a reference system.

    Parameters
    ----------

    system: str
        Path to a stream file containing the reference system.

    pert_file: str
        Path to a stream file containing the perturbation to apply to the
        reference system.

    Returns
    -------

    system: sire.system.System
        The perturbable system.
    """

    if not isinstance(system, str):
        raise TypeError("'system' must be of type 'str'.")

    if not isinstance(pert_file, str):
        raise TypeError("'pert_file' must be of type 'str'.")

    import os as _os

    if not _os.path.isfile(system):
        raise FileNotFoundError(f"'{system}' does not exist.")

    if not _os.path.isfile(pert_file):
        raise FileNotFoundError(f"'{pert_file}' does not exist.")

    from sire import stream as _stream
    from sire import morph as _morph

    # Load the reference system.
    try:
        system = _stream.load(system)
    except Exception as e:
        raise ValueError(f"Failed to load the reference 'system': {e}")

    # Get the non-water molecules in the system.
    non_waters = system["not water"]

    # Try to apply the perturbation to each non-water molecule.
    is_pert = False
    for mol in non_waters:
        try:
            pert_mol = _morph.create_from_pertfile(mol, pert_file)
            is_pert = True
            break
        except:
            pass

    if not is_pert:
        raise ValueError(f"Failed to apply the perturbation in '{pert_file}'.")

    # Replace the reference molecule with the perturbed molecule.
    system.remove(mol)
    system.add(pert_mol)

    return system
