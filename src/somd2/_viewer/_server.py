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
A minimal HTTP server for the SOMD2 viewer.
"""

__all__ = ["serve"]

import json as _json
import sys as _sys
import threading as _threading
import time as _time
from http.server import BaseHTTPRequestHandler as _BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path as _Path
from urllib.parse import urlparse as _urlparse

from ._data import Simulation as _Simulation
from ._data import discover as _discover

_static = _Path(__file__).parent / "_static"


class _Registry:
    """
    The runs found below the requested paths, re-discovered periodically so
    that new output directories are picked up.
    """

    # The minimum time between searches of the requested paths.
    _interval = 15.0

    def __init__(self, paths):
        self._paths = paths
        self._lock = _threading.Lock()
        self._simulations = {}
        self._last_search = None
        self.refresh(strict=True)

    def refresh(self, strict=False):
        now = _time.monotonic()
        with self._lock:
            if (
                self._last_search is not None
                and now - self._last_search < self._interval
            ):
                return list(self._simulations.values())
            self._last_search = now
        found = _discover(self._paths, strict=strict)
        with self._lock:
            by_path = {s.path: s for s in self._simulations.values()}
            simulations = {}
            for path, name in found:
                sim = by_path.get(path) or _Simulation(path, name)
                simulations[sim.id] = sim
            self._simulations = simulations
            return list(simulations.values())

    def get(self, id):
        with self._lock:
            return self._simulations.get(id)


class _Handler(_BaseHTTPRequestHandler):
    def do_GET(self):
        parts = [p for p in _urlparse(self.path).path.split("/") if p]
        registry = self.server.registry
        self.server.last_request = _time.monotonic()

        try:
            if not parts:
                self._send(
                    (_static / "index.html").read_bytes(), "text/html; charset=utf-8"
                )
            elif parts == ["api", "version"]:
                # Lets the page reload itself when the server or page changes.
                mtime = (_static / "index.html").stat().st_mtime_ns
                self._json({"version": f"{self.server.start_time}-{mtime}"})
            elif parts == ["api", "simulations"]:
                self._json([sim.overview() for sim in registry.refresh()])
            elif len(parts) in (3, 4, 5) and parts[:2] == ["api", "simulation"]:
                sim = registry.get(parts[2])
                if sim is None:
                    self.send_error(404)
                elif len(parts) == 3:
                    self._json(sim.summary())
                elif parts[3:] == ["depictions"]:
                    self._json(sim.depictions())
                elif len(parts) == 5 and parts[3] == "components":
                    try:
                        lam = float(parts[4])
                    except ValueError:
                        self.send_error(400)
                        return
                    self._json(sim.components(lam))
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass
        except Exception as e:
            import traceback

            traceback.print_exc()
            self._json({"error": str(e)}, status=500)

    def _json(self, obj, status=200):
        self._send(
            _json.dumps(obj, allow_nan=False).encode(), "application/json", status
        )

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _watch_parent(server, parent_pid, idle_timeout, interval=2.0):
    """
    Once the parent process ends, however it ends, exit after no page has
    contacted the server for 'idle_timeout' seconds. Open pages poll every few
    seconds, so the viewer stays up while anyone is watching. An orphaned
    process is re-parented, so its parent process ID changes.
    """
    import os

    def watch():
        while os.getppid() == parent_pid:
            _time.sleep(interval)
        ended = _time.monotonic()
        while _time.monotonic() - max(server.last_request, ended) < idle_timeout:
            _time.sleep(interval)
        os._exit(0)

    _threading.Thread(target=watch, daemon=True).start()


def serve(
    paths,
    host="127.0.0.1",
    port=8000,
    open_browser=False,
    parent_pid=None,
    idle_timeout=600.0,
):
    """
    Serve the viewer for a set of SOMD2 output directories.

    Parameters
    ----------

    paths: [str]
        Output directories, or directories containing them.

    host: str
        The address to bind to.

    port: int
        The port to listen on. Use 0 to pick a free port.

    open_browser: bool
        Whether to open the viewer in a web browser.

    parent_pid: int
        The ID of a process, e.g. a simulation, that the viewer outlives only
        while a page is open. Not supported on Windows.

    idle_timeout: float
        How long, in seconds, the viewer keeps running after 'parent_pid' has
        ended once no page has contacted it.
    """
    server = _ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.registry = _Registry(paths)
    server.start_time = _time.time_ns()
    server.last_request = _time.monotonic()

    if parent_pid is not None and _sys.platform != "win32":
        _watch_parent(server, parent_pid, idle_timeout)

    url = f"http://{host}:{server.server_address[1]}"
    print(f"SOMD2 viewer running at {url}", flush=True)

    if open_browser:
        import webbrowser

        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
