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

__all__ = ["find_port", "serve"]

import json as _json
import sys as _sys
import threading as _threading
import time as _time
from http.server import BaseHTTPRequestHandler as _BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path as _Path
from urllib.parse import unquote as _unquote
from urllib.parse import urlparse as _urlparse

from ._data import Simulation as _Simulation
from ._data import _clean
from ._data import discover as _discover
from ._log import configure as _configure_log
from ._log import report as _report
from ._network import build_network as _build_network
from ._network import find_network as _find_network
from ._network import read_network as _read_network
from ._summary import build_summary as _build_summary

_static = _Path(__file__).parent / "_static"

# Required on a request to stop the viewer. Browsers won't send a custom header
# to another site without its permission, so a web page can't stop it.
_SHUTDOWN_HEADER = "X-SOMD2-Viewer"


def _is_loopback(host):
    import ipaddress

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Registry:
    """
    The runs found below the requested paths, re-discovered periodically so
    that new output directories are picked up.
    """

    # The minimum time between searches of the requested paths.
    _interval = 15.0

    def __init__(self, paths):
        self.paths = paths
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
        found = _discover(self.paths, strict=strict)
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
    # The minimum time, in seconds, between forced analyses of a run.
    _force_interval = 60.0

    def _allowed_host(self):
        """
        Whether a request to a viewer bound to loopback was addressed to it by
        a loopback name, which stops a website reading it by pointing its own
        domain at 127.0.0.1. Other requests are refused.
        """
        if not self.server.loopback:
            return True
        host = self.headers.get("Host")
        if host is None:
            return True
        try:
            name = _urlparse(f"//{host}").hostname
        except ValueError:
            name = None
        if name is not None and _is_loopback(name):
            return True
        self.send_error(403)
        return False

    def _analysis_interval(self):
        """
        Results for the summary, network and repeats are only brought up to
        date while a page showing them asks for them to be analysed.
        Refreshing the page forces it for runs not analysed in the last minute.
        """
        query = _urlparse(self.path).query.split("&")
        if "analyse=1" not in query:
            return None
        interval = self.server.summary_interval
        if "force=1" in query:
            interval = min(interval, self._force_interval)
        return interval

    def do_GET(self):
        if not self._allowed_host():
            return
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
            elif parts == ["api", "status"]:
                self._json({"somd2_viewer": True, "idle": self.server.orphaned})
            elif parts == ["api", "paths"]:
                self._json([str(_Path(p).resolve()) for p in registry.paths])
            elif parts == ["logo.png"]:
                self._send((_static / "somd2.png").read_bytes(), "image/png")
            elif parts == ["api", "simulations"]:
                self._json([sim.overview() for sim in registry.refresh()])
            elif parts == ["api", "network"] and self.server.network is None:
                self.send_error(404)
            elif len(parts) == 3 and parts[:2] == ["api", "group"]:
                group = _unquote(parts[2])
                interval = self._analysis_interval()
                members = sorted(
                    (s for s in registry.refresh() if s.group() == group),
                    key=lambda s: s.name,
                )
                runs = [s.group_entry(interval) for s in members]
                self._json({"id": group, "runs": runs})
            elif parts in (["api", "summary"], ["api", "network"]):
                interval = self._analysis_interval()
                entries = [sim.summary_entry(interval) for sim in registry.refresh()]
                summary = _build_summary(entries)
                summary["network"] = self.server.network is not None
                if parts[1] == "summary":
                    self._json(_clean(summary))
                else:
                    edges = _read_network(self.server.network)
                    network = _build_network(edges, summary)
                    network["file"] = str(self.server.network)
                    for key in ("available", "pending", "updating", "excluded"):
                        network[key] = summary[key]
                    self._json(_clean(network))
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
            path = _urlparse(self.path).path
            reason = _report(e, f"Couldn't handle the request for {path}")
            self._json({"error": reason}, status=500)

    def do_POST(self):
        if not self._allowed_host():
            return
        parts = [p for p in _urlparse(self.path).path.split("/") if p]
        local = self.client_address[0] in ("127.0.0.1", "::1")

        if parts == ["api", "closed"]:
            # Sent by a page as it is closed. Not counted as a request, so the
            # viewer can tell whether any other page is still open.
            self.server.closed_at = _time.monotonic()
            self._send(b"", "text/plain", status=204)
        elif (
            parts == ["api", "shutdown"]
            and local
            and self.headers.get(_SHUTDOWN_HEADER) == "1"
        ):
            # Lets a new viewer take over the port from one whose simulation has
            # ended. A viewer for a running simulation is never stopped.
            if not self.server.orphaned:
                self._json({"stopped": False}, status=409)
                return
            self._json({"stopped": True})
            _threading.Thread(target=self._stop).start()
        else:
            self.send_error(404)

    @staticmethod
    def _stop():
        import os

        # Give the reply time to be sent before exiting.
        _time.sleep(0.2)
        os._exit(0)

    def _json(self, obj, status=200):
        self._send(
            _json.dumps(obj, allow_nan=False).encode(), "application/json", status
        )

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Lets a page tell that it is talking to a different viewer.
        self.send_header("X-Viewer-Instance", str(self.server.start_time))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _port_is_free(port):
    import errno
    import socket

    # Browsers use both addresses for 'localhost', and a port forwarded over
    # SSH may only hold the IPv6 one.
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        with s:
            # Match the server, which can bind while closed connections linger.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((address, port))
            except OSError as e:
                # IPv6 may not be enabled.
                if family == socket.AF_INET6 and e.errno == errno.EADDRNOTAVAIL:
                    continue
                return False
    return True


def _take_over(port, timeout=5.0):
    """
    Ask the process on a port to stop if it is a somd2 viewer whose simulation
    has ended. Returns whether the port is now free.
    """
    import urllib.request

    url = f"http://127.0.0.1:{port}/api"
    try:
        with urllib.request.urlopen(f"{url}/status", timeout=1) as r:
            status = _json.load(r)
        if not (status.get("somd2_viewer") and status.get("idle")):
            return False
        request = urllib.request.Request(
            f"{url}/shutdown", method="POST", headers={_SHUTDOWN_HEADER: "1"}
        )
        urllib.request.urlopen(request, timeout=1).close()
    except Exception:
        return False

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if _port_is_free(port):
            return True
        _time.sleep(0.1)
    return False


def find_port(start, attempts=100):
    """
    Return the first usable port from 'start'. A port held by a somd2 viewer
    whose simulation has ended is taken over, so that viewers don't pile up.
    """
    for port in range(start, start + attempts):
        if _port_is_free(port) or _take_over(port):
            return port
    raise RuntimeError(
        f"No free port for the viewer in {start}-{start + attempts - 1}."
    )


def _watch_parent(server, parent_pid, idle_timeout, close_grace, interval=2.0):
    """
    Once the parent process ends, however it ends, exit when no page is open:
    either 'close_grace' seconds after the last page reported it was closed,
    with no requests since, or after 'idle_timeout' seconds without a request,
    in case a page couldn't report it. An orphaned process is re-parented, so
    its parent process ID changes.
    """
    import os

    def watch():
        while os.getppid() == parent_pid:
            _time.sleep(interval)
        server.orphaned = True
        ended = _time.monotonic()
        while True:
            now = _time.monotonic()
            last = max(server.last_request, ended)
            closed = server.closed_at
            if now - last >= idle_timeout or (
                closed is not None and last < closed and now - closed >= close_grace
            ):
                os._exit(0)
            _time.sleep(interval)

    _threading.Thread(target=watch, daemon=True).start()


def serve(
    paths,
    host="127.0.0.1",
    port=8000,
    open_browser=False,
    parent_pid=None,
    idle_timeout=600.0,
    close_grace=30.0,
    summary_interval=600.0,
    network=None,
    log_file=None,
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

    close_grace: float
        How long, in seconds, the viewer keeps running after 'parent_pid' has
        ended and the last open page was closed, in case it is reloaded.

    summary_interval: float
        The minimum time, in seconds, between analyses of a run for the
        summary page.

    network: str
        A file listing the edges of a perturbation network, one per line as
        'ligand_a ligand_b'. By default, a 'network.dat' directly in one of
        the paths is used.

    log_file: str
        The file that errors are logged to. By default, they are written to
        stderr.
    """
    _configure_log(log_file)
    network = _find_network(paths, network)
    if network is not None and not network.is_file():
        raise ValueError(f"Network file not found: {network}")

    server = _ThreadingHTTPServer((host, port), _Handler)
    server.loopback = _is_loopback(host)
    server.network = network
    server.daemon_threads = True
    server.registry = _Registry(paths)
    server.start_time = _time.time_ns()
    server.last_request = _time.monotonic()
    server.closed_at = None
    server.orphaned = False
    server.summary_interval = summary_interval

    if parent_pid is not None and _sys.platform != "win32":
        _watch_parent(server, parent_pid, idle_timeout, close_grace)

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
