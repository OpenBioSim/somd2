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

__all__ = ["somd2", "somd2_view"]


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

    # Add an option to launch the viewer alongside the simulation.
    parser.add_argument(
        "--view",
        action="store_true",
        help="Launch a web viewer for the output directory while the simulation runs.",
    )
    parser.add_argument(
        "--view-port",
        type=int,
        default=8000,
        help="The port for the web viewer. If it is in use, the next free "
        "port is used.",
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

    # Pop the viewer options from the arguments dictionary.
    view = args.pop("view")
    view_port = args.pop("view_port")

    # Instantiate a Config object to validate the arguments.
    config = Config(**args)

    # Instantiate a Runner object to run the simulation.
    if config.replica_exchange:
        runner = RepexRunner(system, config)
    else:
        runner = Runner(system, config)

    # Run the viewer in its own process so that it doesn't compete with the
    # simulation for the GIL.
    viewer = None
    if view:
        import os
        import subprocess
        import sys

        from somd2._viewer import find_port

        port = find_port(view_port)
        command = [
            sys.executable,
            "-m",
            "somd2._viewer",
            str(config.output_directory),
            "--port",
            str(port),
            "--parent-pid",
            str(os.getpid()),
        ]

        # Without a local display, the browser module may fall back to a
        # text-mode browser that would take over the terminal. Over SSH, any
        # display is remote.
        over_ssh = os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
        has_display = sys.platform in ("darwin", "win32") or (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if has_display and not over_ssh:
            command.append("--open")

        # The URL is logged here, so the viewer's own output isn't needed.
        viewer = subprocess.Popen(command, stdout=subprocess.DEVNULL)
        _logger.info(
            f"Viewer running at http://127.0.0.1:{port}. Once the simulation "
            "ends, it stops shortly after no page is open."
        )

    # Run the simulation. The viewer stops itself once it is no longer needed,
    # except on Windows where it can't detect that the simulation has ended.
    try:
        runner.run()
    except Exception as e:
        _logger.error(f"An error occurred during the simulation: {e}")
        exit(1)
    finally:
        if viewer is not None and sys.platform == "win32":
            viewer.terminate()


def somd2_view():
    """
    SOMD2 viewer: Command line interface.
    """

    import os
    from argparse import SUPPRESS, ArgumentParser
    from sys import exit

    # JAX can segfault, and holds on to the GPU after an MBAR analysis, which
    # would stop simulations creating contexts on it. Must be set before pymbar
    # is imported.
    os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

    from somd2._viewer import find_port, serve

    parser = ArgumentParser(
        prog="somd2-view",
        description="Serve a web viewer for SOMD2 output directories.",
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="SOMD2 output directories, or directories containing them.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The address to bind to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="The port to listen on. If it is in use, the next free port is used.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the viewer in a web browser.",
    )
    # Used by 'somd2 --view', so that the viewer exits with the simulation.
    parser.add_argument("--parent-pid", type=int, default=None, help=SUPPRESS)
    args = parser.parse_args()

    try:
        # 'somd2 --view' has already chosen the port.
        port = args.port if args.parent_pid is not None else find_port(args.port)
        serve(
            args.paths,
            host=args.host,
            port=port,
            open_browser=args.open,
            parent_pid=args.parent_pid,
        )
    except (OSError, RuntimeError, ValueError) as e:
        exit(f"somd2-view: {e}")
