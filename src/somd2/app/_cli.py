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

"""
SOMD2 command line interface.
"""

__all__ = ["somd2"]


def somd2():
    """
    SOMD2: Command line interface.
    """

    from argparse import Namespace
    from sys import exit

    from somd2 import _logger
    from somd2.config import Config
    from somd2.runner import Runner, RepexRunner

    from somd2.io import yaml_to_dict

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

    # Parse the arguments into a dictionary.
    args = vars(parser.parse_args())

    # Pop the YAML config and system from the arguments dictionary.
    config = args.pop("config")
    system = args.pop("system")

    # If set, read the YAML config file.
    if config is not None:
        # Convert the YAML config to a dictionary.
        config = yaml_to_dict(config)

        # Reparse the command-line arguments using the existing config
        # as a Namespace. Any non-default arguments from the command line
        # will override those in the config.
        args = vars(parser.parse_args(namespace=Namespace(**config)))

        # Re-pop the YAML config and system from the arguments dictionary.
        args.pop("config")
        args.pop("system")

    # Instantiate a Config object to validate the arguments.
    config = Config(**args)

    # Instantiate a Runner object to run the simulation.
    if config.replica_exchange:
        runner = RepexRunner(system, config)
    else:
        runner = Runner(system, config)

    # Run the simulation.
    try:
        runner.run()
    except Exception as e:
        _logger.error(f"An error occurred during the simulation: {e}")
        exit(1)
