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
The viewer's cache of results that are slow to compute, kept between runs.
"""

__all__ = ["cache_dir", "clear_cache"]

import os as _os
from pathlib import Path as _Path


def cache_dir():
    """
    The viewer's cache directory, following the XDG base directory spec.
    """
    base = _os.environ.get("XDG_CACHE_HOME", "")
    # The spec says a relative path should be ignored.
    if not _os.path.isabs(base):
        base = _os.path.join(_os.path.expanduser("~"), ".cache")
    return _Path(base) / "somd2" / "viewer"


def clear_cache():
    """
    Remove everything in the viewer's cache, returning its directory.
    """
    import shutil

    path = cache_dir()
    shutil.rmtree(path, ignore_errors=True)
    return path
