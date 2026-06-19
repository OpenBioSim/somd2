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

# Store the BioSimSpace version.
from BioSimSpace import __version__ as _biosimspace_version

# Store the ghostly version.
from ghostly import __version__ as _ghostly_version

# Store the loch version.
from loch import __version__ as _loch_version


def get_versions():
    """
    Return the versions of SOMD2 and the OpenBioSim packages that it depends
    on.

    Returns
    -------

    versions: dict
        A dictionary mapping package name to version string.
    """
    return {
        "somd2": __version__,
        "sire": f"{_sire_version}+{_sire_revisionid}",
        "biosimspace": _biosimspace_version,
        "ghostly": _ghostly_version,
        "loch": _loch_version,
    }
