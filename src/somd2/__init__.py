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

# Disable Sire progress bars until we work out the best way to handle
# them for the SOMD2 runner, i.e. when running multiple dynamics objects
# in parallel.
from sire.base import ProgressBar as _ProgressBar

_ProgressBar.set_silent()
del _ProgressBar

from loguru import logger as _logger

from . import runner

# Store the somd2 version.
from ._version import __version__

# Store the sire version.
from sire import __version__ as _sire_version
from sire import __revisionid__ as _sire_revisionid

# Determine whether we're being run interactively.
try:
    _shell = get_ipython().__class__.__name__
    if _shell == "ZMQInteractiveShell":
        _is_interactive = True  # Jupyter notebook or qtconsole
    elif _shell == "TerminalInteractiveShell":
        _is_interactive = True  # Terminal running IPython
    else:
        _is_interactive = False  # Other type (?)
    del _shell
except NameError:
    _is_interactive = False  # Probably standard Python interpreter

if not _is_interactive:
    # Log the versions of somd2 and sire.
    _logger.info(f"somd2 version: {__version__}")
    _logger.info(f"sire version: {_sire_version}+{_sire_revisionid}")

del _is_interactive
