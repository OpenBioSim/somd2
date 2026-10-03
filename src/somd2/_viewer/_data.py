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
Discovery and loading of SOMD2 output directories for the viewer.
"""

__all__ = ["Simulation", "discover"]

import hashlib as _hashlib
import json as _json
import math as _math
import pickle as _pickle
import re as _re
import threading as _threading
import time as _time
from collections import deque as _deque
from datetime import datetime as _datetime
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

import numpy as _np

# Serialise the free-energy analyses, since each can use a lot of memory.
_analysis_lock = _threading.Lock()

_line_re = _re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s*\| \S+ - (.*)$"
)
_cycle_re = _re.compile(r"^Running dynamics for cycle (\d+) of (\d+)")
_states_re = _re.compile(r"^States: \[([\d\s]*)\]")
_block_re = _re.compile(r"^Finished block (\d+) of (\d+) for \S+ = ([\d.]+)")
_perf_re = _re.compile(r"^Overall performance: ([\d.]+) ns day-1")
_correction_re = _re.compile(r"Standard state correction: (-?[\d.]+) kcal mol-1")


def discover(paths, strict=True):
    """
    Find the SOMD2 output directories below a set of paths.

    Parameters
    ----------

    paths: [str]
        Output directories, or directories containing them.

    strict: bool
        Whether to raise an error for a path that isn't a directory, rather
        than skipping it.

    Returns
    -------

    directories: [(pathlib.Path, str)]
        The output directories, along with a display name for each.
    """
    found = {}
    for root in paths:
        root = _Path(root).resolve()
        if not root.is_dir():
            if strict:
                raise ValueError(f"'{root}' is not a directory.")
            continue
        if _find_config(root) is not None:
            dirs = [root]
        else:
            dirs = sorted({c.parent for c in root.rglob("config*.yaml")})
            # An explicitly requested directory may not have any output yet.
            if not dirs:
                dirs = [root]
        for d in dirs:
            if d == root:
                name = root.name
            else:
                name = f"{root.name}/{d.relative_to(root)}"
            found.setdefault(d, name)
    return list(found.items())


def _find_config(path):
    """
    Return the most recent config file in an output directory, if any.
    """
    configs = list(path.glob("config*.yaml"))
    if not configs:
        return None
    return max(configs, key=lambda p: p.stat().st_mtime)


def _stamp(*paths):
    """
    A key that changes whenever any of the files are modified.
    """
    key = []
    for p in paths:
        try:
            s = _Path(p).stat()
            key.append((str(p), s.st_mtime_ns, s.st_size))
        except OSError:
            key.append((str(p), None, None))
    return tuple(key)


def _to_ns(value):
    """
    Convert a time string from the config, e.g. '2 ps', to nanoseconds.
    """
    import sire as _sr

    try:
        return _sr.u(str(value)).to("ns")
    except Exception:
        return None


@_lru_cache(maxsize=4096)
def _parse_second(ts):
    return _datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def _parse_timestamp(ts):
    # Many log lines share the same second, so only the milliseconds vary.
    return _parse_second(ts[:19]) + int(ts[20:23]) / 1000


def _clean(obj):
    """
    Make an object JSON serialisable, replacing non-finite floats with None.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (float, _np.floating)):
        return float(obj) if _math.isfinite(obj) else None
    return obj


class _RepexState:
    """
    Stand-in for the pickled DynamicsCache, so that its attributes can be read
    without recreating any dynamics objects.
    """

    def __setstate__(self, state):
        self.__dict__.update(state)


class _RepexUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if module == "somd2.runner._repex" and name == "DynamicsCache":
            return _RepexState
        return super().find_class(module, name)


class _LogMonitor:
    """
    Incrementally parse a SOMD2 log file as it is written.
    """

    def __init__(self, path):
        self._path = path
        self._reset()

    def _reset(self):
        self._offset = 0
        self._buffer = b""
        self._pending_states = None
        self.states = []
        self._saved_states = 0
        self.standard_state_correction = None
        self.messages = _deque(maxlen=20)
        self._new_session(None)

    def _new_session(self, timestamp):
        self.session_start = timestamp
        self.production_start = None
        self.cycles = []
        self.blocks = []
        self.performance = None
        self.finished = False
        self.last_progress = None
        self.last_error = None

    def update(self):
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < self._offset:
            self._reset()
        if size == self._offset:
            return
        with open(self._path, "rb") as f:
            f.seek(self._offset)
            data = self._buffer + f.read()
            self._offset = f.tell()
        lines = data.split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            self._parse(line.decode("utf-8", errors="replace").rstrip())

    def _parse(self, line):
        # Large state arrays are wrapped over several lines by NumPy.
        if self._pending_states is not None:
            self._pending_states += " " + line
            if "]" in line:
                self._add_states(self._pending_states)
                self._pending_states = None
            return

        match = _line_re.match(line)
        if match is None:
            return
        stamp = match.group(1)
        level = match.group(2)
        message = match.group(3)

        if message.startswith("somd2 version:"):
            # A restart resumes from the last checkpoint, so the permutations
            # logged after it were discarded.
            del self.states[self._saved_states :]
            self._new_session(_parse_timestamp(stamp))
        elif message.startswith("Saving replica exchange state"):
            self._saved_states = len(self.states)
        elif message.startswith("States: ["):
            if "]" in message:
                self._add_states(message)
            else:
                self._pending_states = message
        elif m := _cycle_re.match(message):
            timestamp = _parse_timestamp(stamp)
            self.cycles.append((timestamp, int(m.group(1)), int(m.group(2))))
            self.last_progress = timestamp
            if self.production_start is None:
                self.production_start = timestamp
        elif m := _block_re.match(message):
            timestamp = _parse_timestamp(stamp)
            self.blocks.append((timestamp, int(m.group(1)), int(m.group(2))))
            self.last_progress = timestamp
        elif message.startswith("Running dynamics at"):
            self.last_progress = _parse_timestamp(stamp)
            if self.production_start is None:
                self.production_start = self.last_progress
        elif m := _perf_re.match(message):
            self.performance = float(m.group(1))
        elif message.startswith("Simulation finished"):
            self.finished = True
        elif m := _correction_re.search(message):
            self.standard_state_correction = float(m.group(1))

        if level in ("WARNING", "ERROR", "CRITICAL"):
            self.messages.append((_parse_timestamp(stamp), level, message))
            if level != "WARNING":
                self.last_error = _parse_timestamp(stamp)

    def _add_states(self, message):
        match = _states_re.match(message)
        if match is not None:
            self.states.append([int(x) for x in match.group(1).split()])


class Simulation:
    """
    A SOMD2 output directory, with cached access to its current results.
    """

    def __init__(self, path, name):
        self.path = _Path(path)
        self.name = name
        self.id = _hashlib.sha1(str(self.path).encode()).hexdigest()[:10]
        self._lock = _threading.RLock()
        self._depict_lock = _threading.Lock()
        self._cache = {}
        self._log = None
        self._analysis = None
        self._analysis_stamp = None
        self._analysis_running = False
        self._convergence_result = None
        self._convergence_stamp = None
        self._convergence_time = 0.0
        self._convergence_interval = 300.0

    def _cached(self, key, stamp, func):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and entry[0] == stamp:
                return entry[1]
        value = func()
        with self._lock:
            self._cache[key] = (stamp, value)
        return value

    def config(self):
        """
        Return the most recent config for the run as a dictionary.
        """
        path = _find_config(self.path)
        if path is None:
            return None

        def load():
            from ..io import yaml_to_dict

            config = yaml_to_dict(str(path))
            if not isinstance(config, dict):
                raise ValueError(f"'{path.name}' isn't a SOMD2 config file.")
            return config

        return self._cached("config", _stamp(path), load)

    def _lambdas(self, config):
        """
        Return the sampled and energy lambda values, as set up by the runner.
        """
        if config.get("lambda_values"):
            values = [round(float(x), 5) for x in config["lambda_values"]]
        elif int(config.get("num_lambda", 11)) == 1:
            values = [0.0]
        else:
            n = int(config.get("num_lambda", 11))
            values = [round(i / (n - 1), 5) for i in range(n)]
        if config.get("lambda_energy"):
            energy = [round(float(x), 5) for x in config["lambda_energy"]]
        else:
            energy = list(values)
        for value in values:
            if value not in energy:
                energy.append(value)
        return values, energy

    def _log_monitor(self, config):
        log_file = config.get("log_file")
        if not log_file:
            return None
        path = self.path / log_file
        with self._lock:
            if self._log is None or self._log._path != path:
                self._log = _LogMonitor(path)
            self._log.update()
            return self._log

    def windows(self):
        """
        Return the energy trajectory information for each sampled window.
        """
        import pyarrow.parquet as _pq

        windows = []
        for path in sorted(self.path.glob("energy_traj_*.parquet")):

            def read(path=path):
                meta = _pq.read_metadata(path)
                somd2 = _json.loads(meta.metadata[b"somd2"])
                return {
                    "lambda": float(somd2["lambda"]),
                    "samples": meta.num_rows,
                    "speed": somd2.get("speed"),
                    "standard_state_correction": somd2.get("standard_state_correction"),
                }

            try:
                windows.append(self._cached(f"window:{path.name}", _stamp(path), read))
            except Exception:
                # The file may be part way through being replaced.
                continue
        return sorted(windows, key=lambda w: w["lambda"])

    def _progress(self, config, lambda_values, windows, log):
        runtime = _to_ns(config.get("runtime"))
        energy_frequency = _to_ns(config.get("energy_frequency"))
        checkpoint_frequency = _to_ns(config.get("checkpoint_frequency"))

        done = {}
        for w in windows:
            if energy_frequency is not None:
                t = w["samples"] * energy_frequency
                done[w["lambda"]] = min(t, runtime) if runtime else t
        total = runtime * len(lambda_values) if runtime else None
        completed = sum(done.get(lam, 0.0) for lam in lambda_values)

        now = _time.time()
        progress = {
            "runtime_ns": runtime,
            "expected_samples": (
                round(runtime / energy_frequency)
                if runtime and energy_frequency
                else None
            ),
            "completed_ns": completed,
            "fraction": completed / total if total else None,
            "eta_seconds": None,
            "throughput_ns_day": None,
            "throughput_label": None,
            "performance_ns_day": None,
            "cycle": None,
            "num_cycles": None,
            "last_update_seconds": None,
            "status": "waiting",
            "messages": [],
        }

        if log is None:
            if total and completed >= total:
                progress["status"] = "finished"
            return progress

        try:
            idle = now - log._path.stat().st_mtime
        except OSError:
            idle = None
        progress["last_update_seconds"] = idle
        progress["performance_ns_day"] = log.performance
        progress["messages"] = [
            {"time": t, "level": level, "message": message}
            for t, level, message in log.messages
        ]

        interval = None
        if config.get("replica_exchange") and log.cycles:
            last_time, cycle, num_cycles = log.cycles[-1]
            progress["cycle"] = cycle
            progress["num_cycles"] = num_cycles
            times = [c[0] for c in log.cycles[-51:]]
            if len(times) > 1:
                interval = (times[-1] - times[0]) / (len(times) - 1)
                remaining = (num_cycles - cycle + 1) * interval - (now - last_time)
                progress["eta_seconds"] = max(remaining, 0.0)
                if energy_frequency is not None and interval > 0:
                    progress["throughput_ns_day"] = energy_frequency / interval * 86400
                    progress["throughput_label"] = "per replica"
        elif log.blocks and log.production_start is not None:
            elapsed = log.blocks[-1][0] - log.production_start
            if elapsed > 0 and checkpoint_frequency is not None:
                rate = len(log.blocks) * checkpoint_frequency / elapsed
                interval = elapsed / len(log.blocks)
                if total is not None:
                    progress["eta_seconds"] = max(total - completed, 0.0) / rate
                progress["throughput_ns_day"] = rate * 86400
                progress["throughput_label"] = "all windows"

        if log.finished:
            progress["status"] = "finished"
            progress["eta_seconds"] = 0.0
        elif log.last_error is not None and (
            log.last_progress is None or log.last_error >= log.last_progress
        ):
            # An error with no progress since is treated as a crash.
            progress["status"] = "error"
            progress["eta_seconds"] = None
        elif idle is not None:
            threshold = max(300.0, 10 * interval) if interval else 1800.0
            if idle < threshold:
                progress["status"] = "running"
            else:
                progress["status"] = "stopped"
                progress["eta_seconds"] = None

        return progress

    def _completeness(self, config, lambda_values, lambda_energy, windows):
        """
        Whether the windows in this directory form a complete free-energy
        calculation.

        Replica exchange always samples every window in one directory, and
        MBAR can include energy states that aren't sampled. The regular runner
        can be split across directories, so every energy state must have data.
        """
        if len(lambda_energy) < 2:
            return False, "A free energy needs at least two λ windows."
        required = lambda_values if config.get("replica_exchange") else lambda_energy
        sampled = {w["lambda"] for w in windows if w["samples"] > 1}
        missing = sorted(lam for lam in required if lam not in sampled)
        if not missing:
            return True, None
        if not windows:
            return False, "No energy data has been written yet."
        if config.get("replica_exchange"):
            return False, "Waiting for the first checkpoint."
        return False, (
            f"No data for λ = {', '.join(f'{x:.5f}' for x in missing)}. "
            "This directory only holds a subset of the λ windows, so a free "
            "energy can't be estimated from it on its own."
        )

    def analysis(self, analysable, reason, start=True):
        """
        Return the most recent MBAR analysis. If 'start' is True, a new one is
        started in the background if the energy data has changed.
        """
        if not analysable:
            return {"status": "unavailable", "reason": reason}

        stamp = _stamp(*sorted(self.path.glob("energy_traj_*.parquet")))
        with self._lock:
            if (
                start
                and not self._analysis_running
                and (stamp != self._analysis_stamp or self._convergence_due(stamp))
            ):
                self._analysis_running = True
                _threading.Thread(
                    target=self._run_analysis, args=(stamp,), daemon=True
                ).start()
            result = dict(self._analysis or {"status": "pending"})
            result["updating"] = self._analysis_running
            if result["status"] == "done":
                result["convergence"] = self._convergence_result or {
                    "status": "pending"
                }
            return result

    def _convergence_due(self, stamp):
        """
        Whether the convergence analysis is out of date and hasn't been run
        within the last '_convergence_interval' seconds.
        """
        return (
            stamp != self._convergence_stamp
            and _time.time() - self._convergence_time >= self._convergence_interval
        )

    def _run_analysis(self, stamp):
        if stamp != self._analysis_stamp:
            self._run_mbar(stamp)
        with self._lock:
            run_convergence = (
                self._analysis is not None
                and self._analysis["status"] == "done"
                and self._convergence_due(stamp)
            )
        if run_convergence:
            convergence = self._convergence()
            convergence["time"] = _time.time()
            with self._lock:
                self._convergence_result = convergence
                self._convergence_stamp = stamp
                self._convergence_time = convergence["time"]
        with self._lock:
            self._analysis_running = False

    def _run_mbar(self, stamp):
        try:
            with _analysis_lock:
                import warnings as _warnings

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    import BioSimSpace as _BSS

                    pmf, overlap = _BSS.FreeEnergy.Relative.analyse(
                        str(self.path), estimator="MBAR"
                    )
            result = {
                "status": "done",
                "lambda": [p[0] for p in pmf],
                "pmf": [p[1].kcal_per_mol().value() for p in pmf],
                "error": [p[2].kcal_per_mol().value() for p in pmf],
                "overlap": (
                    _np.asarray(overlap).tolist() if overlap is not None else None
                ),
                "time": _time.time(),
            }
            result["free_energy"] = result["pmf"][-1] - result["pmf"][0]
            result["free_energy_error"] = result["error"][-1]
        except Exception as e:
            result = {"status": "error", "reason": str(e), "time": _time.time()}
        with self._lock:
            self._analysis = result
            self._analysis_stamp = stamp

    def _convergence(
        self,
        num=10,
        min_samples=5,
        max_samples=1000,
        error_tol=3.0,
        num_bootstraps=10,
    ):
        """
        Forward and backward convergence of the MBAR free energy, using
        increasing fractions of the data from the start and end of each window.
        Each window is subsampled to at most 'max_samples', since only the
        trend is needed. Points whose analytic error exceeds 'error_tol' (kT),
        which is unreliable when overlap is poor, use a small bootstrap instead.
        """
        import pyarrow.parquet as _pq

        try:
            with _analysis_lock:
                import warnings as _warnings

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    import BioSimSpace as _BSS
                    from alchemlyb import concat as _concat
                    from alchemlyb.estimators import MBAR as _MBAR
                    from loguru import logger as _loguru

                    _loguru.disable("alchemlyb")

                    def fit(sample, f_k):
                        sample = _concat(sample)
                        mbar = _MBAR(initial_f_k=f_k).fit(sample)
                        f_k = mbar.delta_f_.iloc[0, :]
                        error = mbar.d_delta_f_.iloc[0, -1]
                        bootstrapped = bool(
                            not _math.isfinite(error) or error > error_tol
                        )
                        if bootstrapped:
                            error = (
                                _MBAR(n_bootstraps=num_bootstraps, initial_f_k=f_k)
                                .fit(sample)
                                .d_delta_f_.iloc[0, -1]
                            )
                        return f_k, f_k.iloc[-1], error, bootstrapped

                    windows = []
                    for path in self.path.glob("energy_traj_*.parquet"):
                        meta = _json.loads(_pq.read_metadata(path).metadata[b"somd2"])
                        windows.append(
                            (float(meta["lambda"]), float(meta["temperature"]), path)
                        )
                    windows.sort()
                    temperature = windows[0][1]
                    data = [
                        _BSS.FreeEnergy.Relative._somd2_extract(
                            path, T=T, estimator="MBAR"
                        )
                        for _, T, path in windows
                    ]
                    data = [d.iloc[:: max(1, -(-len(d) // max_samples))] for d in data]

                    # Each fraction needs enough samples per window to fit.
                    num = min(num, min(len(d) for d in data) // min_samples)
                    if num < 2:
                        return {
                            "status": "unavailable",
                            "reason": "Not enough samples yet.",
                        }
                    # Each fit starts from the previous one in the same direction.
                    kT = 0.0019872043 * temperature
                    result = {"status": "done", "fraction": []}
                    for direction in ("forward", "backward"):
                        values, errors, flags = [], [], []
                        f_k = None
                        for i in range(1, num + 1):
                            sample = []
                            for d in data:
                                n = max(1, len(d) * i // num)
                                sample.append(
                                    d.iloc[:n]
                                    if direction == "forward"
                                    else d.iloc[-n:]
                                )
                            f_k, value, error, bootstrapped = fit(sample, f_k)
                            values.append(value * kT)
                            errors.append(error * kT)
                            flags.append(bootstrapped)
                        result[direction] = values
                        result[f"{direction}_error"] = errors
                        result[f"{direction}_bootstrapped"] = flags
                    result["fraction"] = [i / num for i in range(1, num + 1)]
            return result
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def _repex_state(self):
        """
        Load the attributes of the pickled replica exchange state.
        """
        path = self.path / "repex_state.pkl"
        backup = self.path / "repex_state.pkl.bak"

        def load():
            # The state is rewritten in place, so fall back to the backup if
            # it is caught part way through.
            for p in (path, backup):
                try:
                    with open(p, "rb") as f:
                        state = _RepexUnpickler(f).load()
                    return {
                        "proposed": _np.asarray(state._num_proposed),
                        "accepted": _np.asarray(state._num_accepted),
                        "gcmc_stats": getattr(state, "_gcmc_stats", None),
                        "terminal_flip_stats": getattr(
                            state, "_terminal_flip_stats", None
                        ),
                    }
                except Exception:
                    continue
            return None

        return self._cached("repex_state", _stamp(path), load)

    def _repex(self, log, lambda_values, repex_state):
        result = {}

        path = self.path / "repex_matrix.txt"

        def load_matrix():
            for p in (path, path.with_suffix(".txt.bak")):
                try:
                    return _np.loadtxt(p, ndmin=2).tolist()
                except Exception:
                    continue
            return None

        result["transition_matrix"] = self._cached(
            "repex_matrix", _stamp(path), load_matrix
        )

        if repex_state is not None:
            proposed = repex_state["proposed"]
            accepted = repex_state["accepted"]
            pairs = []
            for i in range(len(proposed) - 1):
                p = proposed[i, i + 1] + proposed[i + 1, i]
                a = accepted[i, i + 1] + accepted[i + 1, i]
                pairs.append(a / p if p > 0 else None)
            result["neighbour_acceptance"] = pairs

        if log is not None and log.states:
            result.update(self._replica_paths(log.states, len(lambda_values)))

        return result

    @staticmethod
    def _replica_paths(states, num_states, max_points=500):
        """
        Follow each replica through state space using the logged per-cycle
        permutations, where states[k] is the source of window k.
        """
        labels = _np.arange(num_states)
        paths = []
        for perm in states:
            if len(perm) != num_states:
                continue
            labels = labels[perm]
            paths.append(_np.argsort(labels))
        if not paths:
            return {}
        paths = _np.array(paths)

        round_trips = []
        for r in range(num_states):
            trips = 0
            seen_top = False
            started = False
            for s in paths[:, r]:
                if s == 0:
                    if started and seen_top:
                        trips += 1
                    started = True
                    seen_top = False
                elif s == num_states - 1 and started:
                    seen_top = True
            round_trips.append(trips)

        stride = max(1, len(paths) // max_points)
        return {
            "num_cycles": len(paths),
            "stride": stride,
            "replica_states": paths[::stride].T.tolist(),
            "round_trips": round_trips,
        }

    def _samplers(self, config, lambda_values, repex_state):
        gcmc = {}
        flips = {}

        def add_gcmc(stats, lam):
            if not stats:
                return
            if isinstance(stats, list):
                for lam_i, s in zip(lambda_values, stats):
                    add_gcmc(s, lam_i)
                return
            if "num_moves" in stats:
                stats = {f"{lam:.5f}": stats}
            for key, value in stats.items():
                gcmc[float(key)] = value

        if repex_state is not None:
            add_gcmc(repex_state["gcmc_stats"], None)
            if repex_state["terminal_flip_stats"] is not None:
                for lam, s in zip(lambda_values, repex_state["terminal_flip_stats"]):
                    flips[lam] = s
        else:
            for lam in lambda_values:
                path = self.path / f"sampler_stats_{lam:.5f}.pkl"
                if not path.exists():
                    continue

                def load(path=path):
                    try:
                        with open(path, "rb") as f:
                            return _pickle.load(f)
                    except Exception:
                        return None

                stats = self._cached(f"sampler:{path.name}", _stamp(path), load)
                if not stats:
                    continue
                add_gcmc(stats.get("gcmc"), lam)
                if "terminal_flip" in stats:
                    flips[lam] = stats["terminal_flip"]

        result = {}
        if config.get("gcmc") and gcmc:
            rows = []
            for lam in sorted(gcmc):
                row = {"lambda": lam}
                row.update(
                    {k: v for k, v in gcmc[lam].items() if isinstance(v, (int, float))}
                )
                moves = row.get("num_moves")
                if moves:
                    row["acceptance"] = row.get("num_accepted", 0) / moves
                rows.append(row)
            result["gcmc"] = rows
        if config.get("terminal_flip_frequency") and flips:
            result["terminal_flip"] = [
                {
                    "lambda": lam,
                    "attempted": s[0],
                    "accepted": s[1],
                    "acceptance": s[1] / s[0] if s[0] else None,
                }
                for lam, s in sorted(flips.items())
            ]
        return result

    def _schedule(self, config, lambda_energy):
        name = config.get("lambda_schedule") or "standard_morph"

        def load():
            schedule = _resolve_schedule(config)
            df = schedule.get_lever_values(num_lambda=101)
            levers = {col: df[col].tolist() for col in df.columns if col != "stage"}
            is_keyword = isinstance(name, str) and len(name) < 64
            return {
                "name": name if is_keyword else "custom",
                "description": str(schedule),
                "lambda": df.index.tolist(),
                "stage": df["stage"].tolist() if "stage" in df.columns else None,
                "levers": levers,
            }

        key = tuple(
            str(config.get(k))
            for k in (
                "lambda_schedule",
                "charge_scale_factor",
                "softcore_form",
                "beutler_fix_epsilon",
                "restraints",
            )
        )
        try:
            result = self._cached("schedule", key, load)
        except Exception as e:
            result = {
                "name": name,
                "levers": None,
                "description": None,
                "error": str(e),
            }

        rest2 = config.get("rest2_scale")
        if rest2 is not None:
            if isinstance(rest2, (int, float)):
                factors = [
                    1.0 + (float(rest2) - 1.0) * (1.0 - 2.0 * abs(lam - 0.5))
                    for lam in lambda_energy
                ]
            else:
                factors = [float(x) for x in rest2]
            if len(factors) == len(lambda_energy) and any(
                abs(f - 1.0) > 1e-4 for f in factors
            ):
                pairs = sorted(zip(lambda_energy, factors))
                result = dict(result)
                result["rest2"] = {
                    "lambda": [p[0] for p in pairs],
                    "scale": [p[1] for p in pairs],
                    "selection": config.get("rest2_selection"),
                }
        return result

    def components(self, lam, max_points=1000):
        """
        Return the energy components for a window as a function of time.
        """
        import pyarrow.parquet as _pq

        path = self.path / f"energy_components_{float(lam):.5f}.parquet"
        if not path.exists():
            return {"status": "unavailable", "reason": "No energy components."}

        def load():
            # The file is rewritten at each checkpoint, so fall back to the
            # backup if it is caught part way through.
            for p in (path, _Path(str(path) + ".bak")):
                try:
                    df = _pq.read_table(p).to_pandas()
                    break
                except Exception:
                    continue
            else:
                return {"status": "error", "reason": "Couldn't read the file."}
            df = df.sort_values("time")
            stride = max(1, len(df) // max_points)
            df = df.iloc[::stride]
            return {
                "status": "done",
                "lambda": float(lam),
                "num_samples": len(df) * stride,
                "time": df["time"].tolist(),
                "components": {c: df[c].tolist() for c in df.columns if c != "time"},
            }

        return _clean(self._cached(f"components:{path.name}", _stamp(path), load))

    def _restraints(self, config):
        """
        Return a text summary of each restraint, from the config and the
        auto-generated ABFE restraint file.
        """
        auto = self.path / "abfe_restraint.s3"
        serialised = config.get("restraints") or []

        def load():
            import sire as _sr

            from ..config import Config as _Config

            restraints = []
            # Auto-generated ring-breaking restraints are also stored in the
            # config, so these can't be labelled as user-defined.
            n = len(serialised)
            sources = [
                (
                    "From the configuration" + (f" ({i} of {n})" if n > 1 else ""),
                    value,
                )
                for i, value in enumerate(serialised, start=1)
            ]
            if auto.exists():
                sources.append(("Auto-generated ABFE restraint", None))
            for source, value in sources:
                try:
                    if value is None:
                        restraint = _sr.stream.load(str(auto))
                    else:
                        restraint = _Config._from_string(value, "restraints")
                    text = str(restraint)
                except Exception as e:
                    text = f"Couldn't read this restraint: {e}"
                restraints.append({"source": source, "text": text})
            return restraints

        return self._cached("restraints", (str(serialised), _stamp(auto)), load)

    def depictions(self):
        """
        Return depictions of the perturbed molecules at each end state.
        """
        from ._depict import depict

        top0 = self.path / "system0.prm7"
        top1 = self.path / "system1.prm7"
        if not (top0.exists() and top1.exists()):
            return {"status": "unavailable", "reason": "No end-state topologies yet."}

        def load():
            try:
                return {"status": "done", "molecules": depict(top0, top1)}
            except Exception as e:
                return {"status": "error", "reason": str(e)}

        # Concurrent requests wait for the first, rather than repeating it.
        with self._depict_lock:
            return self._cached("depictions", _stamp(top0, top1), load)

    def _unreadable(self, error):
        """
        A placeholder for a run whose output can't be read.
        """
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "status": "error",
            "reason": f"Couldn't read this output directory: {error}",
        }

    def overview(self):
        """
        A brief summary of the run, for listing alongside other runs.
        """
        try:
            return self._overview()
        except Exception as e:
            return self._unreadable(e)

    def summary(self):
        """
        Everything the viewer shows for a single run.
        """
        try:
            return self._summary()
        except Exception as e:
            return self._unreadable(e)

    def _overview(self):
        config = self.config()
        if config is None:
            return {"id": self.id, "name": self.name, "status": "waiting"}
        lambda_values, lambda_energy = self._lambdas(config)
        windows = self.windows()
        log = self._log_monitor(config)
        progress = self._progress(config, lambda_values, windows, log)
        analysable, reason = self._completeness(
            config, lambda_values, lambda_energy, windows
        )
        # Only the selected run is analysed, so this reports the last result.
        analysis = self.analysis(analysable, reason, start=False)
        return _clean(
            {
                "id": self.id,
                "name": self.name,
                "path": str(self.path),
                "replica_exchange": bool(config.get("replica_exchange")),
                "status": progress["status"],
                "fraction": progress["fraction"],
                "eta_seconds": progress["eta_seconds"],
                "free_energy": analysis.get("free_energy"),
                "free_energy_error": analysis.get("free_energy_error"),
            }
        )

    def _summary(self):
        config = self.config()
        if config is None:
            return {
                "id": self.id,
                "name": self.name,
                "path": str(self.path),
                "status": "waiting",
            }

        lambda_values, lambda_energy = self._lambdas(config)
        windows = self.windows()
        log = self._log_monitor(config)
        progress = self._progress(config, lambda_values, windows, log)
        analysable, reason = self._completeness(
            config, lambda_values, lambda_energy, windows
        )
        is_repex = bool(config.get("replica_exchange"))
        repex_state = self._repex_state() if is_repex else None

        pme = None
        pme_path = self.path / "pme_parameters.yaml"
        if pme_path.exists():
            try:
                from ..io import yaml_to_dict

                pme = self._cached(
                    "pme", _stamp(pme_path), lambda: yaml_to_dict(str(pme_path))
                )
            except Exception:
                pme = None

        corrections = {
            w["standard_state_correction"]
            for w in windows
            if w.get("standard_state_correction")
        }

        return _clean(
            {
                "id": self.id,
                "name": self.name,
                "path": str(self.path),
                "replica_exchange": is_repex,
                "status": progress["status"],
                "progress": progress,
                "lambda_values": lambda_values,
                "lambda_energy": sorted(lambda_energy),
                "windows": windows,
                "has_components": any(self.path.glob("energy_components_*.parquet")),
                "analysis": self.analysis(analysable, reason),
                "standard_state_correction": (
                    float(corrections.pop())
                    if len(corrections) == 1
                    else log.standard_state_correction
                    if log is not None
                    else None
                ),
                "repex": (
                    self._repex(log, lambda_values, repex_state) if is_repex else None
                ),
                "samplers": self._samplers(config, lambda_values, repex_state),
                "schedule": self._schedule(config, lambda_energy),
                "pme": pme,
                "restraints": self._restraints(config),
                "config": {
                    k: (
                        "(see the Restraints section)"
                        if k == "restraints" and v
                        else _config_value(v)
                    )
                    for k, v in sorted(config.items())
                },
            }
        )


def _config_value(value):
    """
    Format a config value for display, abbreviating serialised objects.
    """
    if isinstance(value, str) and len(value) > 120:
        return value[:40] + f"… ({len(value)} characters)"
    if isinstance(value, list):
        return [_config_value(v) for v in value]
    return value


def _resolve_schedule(config):
    """
    Build the lambda schedule named in a config, as the Config class does.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    value = str(config.get("lambda_schedule") or "standard_morph").strip()
    keyword = value.lower()
    if keyword == "standard_morph":
        return _LambdaSchedule.standard_morph()
    if keyword == "charge_scaled_morph":
        return _LambdaSchedule.charge_scaled_morph(
            float(config.get("charge_scale_factor", 0.2))
        )
    if keyword in ("ring_break_morph", "reverse_ring_break_morph"):
        from .._utils import _schedules

        return getattr(_schedules, keyword)()
    from ..config import Config as _Config

    if keyword in ("annihilate", "decouple"):
        import sire as _sr

        from .._utils import _schedules

        # The 'auto' soft-core form resolves to Beutler for these schedules.
        fix_epsilon = config.get("softcore_form", "auto") in (
            "auto",
            "beutler",
        ) and bool(config.get("beutler_fix_epsilon", True))
        levers = {
            restraint.restraint_lever()
            for restraint in (
                _Config._from_string(r, "restraints")
                for r in config.get("restraints") or []
            )
            if isinstance(restraint, _sr.mm.BoreschRestraints)
        }
        return getattr(_schedules, keyword)(
            fix_epsilon=fix_epsilon,
            restraint_lever=levers.pop() if len(levers) == 1 else "split",
        )

    return _Config._from_string(value, "lambda_schedule")
