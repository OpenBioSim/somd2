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
Logging of errors in the viewer, so that users can report them.
"""

__all__ = ["configure", "describe", "report"]

import sys as _sys
import threading as _threading

from loguru import logger as _loguru

_logger = _loguru.bind(somd2_viewer=True)

# Errors already logged for each context, since most are hit again on every
# refresh, and a limit in case their descriptions keep changing.
_reported = {}
_reported_lock = _threading.Lock()
_MAX_PER_CONTEXT = 5


def configure(log_file=None, level="INFO"):
    """
    Send the viewer's log to a file, or to stderr, and record the versions in
    use, which are only shown at the INFO level.
    """
    from .. import get_versions

    _loguru.remove()
    _loguru.add(
        _sys.stderr if log_file is None else str(log_file),
        level=level,
        filter=lambda record: record["extra"].get("somd2_viewer", False),
        diagnose=False,
    )
    versions = ", ".join(f"{k} {v}" for k, v in get_versions().items())
    _logger.info(f"Viewer started ({versions})")


def describe(error):
    """
    A one-line description of an error, including its type.
    """
    message = str(error)
    name = type(error).__name__
    return f"{name}: {message}" if message else name


def report(error, context):
    """
    Log an error with its traceback, once for each context and description,
    and return its description for the page.
    """
    description = describe(error)
    with _reported_lock:
        seen = _reported.setdefault(context, set())
        if description in seen or len(seen) > _MAX_PER_CONTEXT:
            action = None
        elif len(seen) < _MAX_PER_CONTEXT:
            seen.add(description)
            action = "log"
        else:
            # Marks the context as full, so this is only logged once.
            seen.add(None)
            action = "suppress"
    if action == "log":
        _logger.opt(exception=error).error(f"{context}: {description}")
    elif action == "suppress":
        _logger.error(f"{context}: further errors are not logged ({description})")
    return description
