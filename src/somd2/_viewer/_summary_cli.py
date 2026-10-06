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
A summary of SOMD2 output directories for the command line, as on the
viewer's summary page.
"""

__all__ = ["collect", "format_text", "to_json", "to_tables", "write_csv"]

import time as _time

# Changed whenever the structure of the JSON output changes.
_JSON_VERSION = 1

# ANSI escape codes for problems in the text tables.
_WARN = "\033[33m"
_RESET = "\033[0m"

_VALUE = "value (kcal/mol)"
_ERROR = "error (kcal/mol)"


def collect(paths, analyse=True, progress=None):
    """
    Identify, and optionally analyse, every run below the paths, waiting
    until all are done.

    Parameters
    ----------

    paths: [str]
        Output directories, or directories containing them.

    analyse: bool
        Whether to estimate free energies, rather than only report progress.

    progress: callable
        Called as progress(done, total) while waiting.

    Returns
    -------

    summary: dict
        The summary, as from build_summary(), with the path of each run.
    """
    from ._data import Simulation, discover
    from ._summary import build_summary

    sims = [Simulation(p, n) for p, n in discover(paths)]

    # Each run is analysed at most once, so that runs still writing data
    # don't keep the summary waiting.
    interval = 0 if analyse else None
    while True:
        entries = [s.summary_entry(interval) for s in sims]
        interval = None
        busy = sum(
            bool(e.get("fingerprint_pending") or e.get("updating")) for e in entries
        )
        if progress is not None:
            progress(len(sims) - busy, len(sims))
        if not busy:
            break
        _time.sleep(1)

    summary = build_summary(entries)
    paths = {s.id: str(s.path) for s in sims}
    for sim in summary["simulations"]:
        for run in sim["runs"]:
            run["path"] = paths[run["id"]]
    for run in summary["excluded"]:
        run["path"] = paths[run["id"]]
    return summary


def to_json(summary, paths):
    """
    A flat, versioned form of the summary for other programs, with
    simulations referred to by name rather than by internal identifiers.
    """
    from datetime import datetime, timezone

    names = {s["id"]: s["name"] for s in summary["simulations"]}
    problems = {s["id"]: s["problems"] for s in summary["simulations"]}

    simulations = []
    for s in summary["simulations"]:
        simulations.append(
            {
                "name": s["name"],
                "leg": s["leg"],
                "schedule": s["schedule"],
                "absolute": s["absolute"],
                "num_runs": len(s["runs"]),
                "num_finished": s["num_finished"],
                "num_analysed": s["num_analysed"],
                "progress": s["progress"],
                "free_energy": s["free_energy"],
                "free_energy_error": s["free_energy_error"],
                "corrected_free_energy": s["corrected_free_energy"],
                "corrected_free_energy_error": s["corrected_free_energy_error"],
                "problems": s["problems"],
                "runs": [
                    {
                        k: r.get(k)
                        for k in (
                            "name",
                            "path",
                            "status",
                            "progress",
                            "free_energy",
                            "free_energy_error",
                        )
                    }
                    for r in s["runs"]
                ],
            }
        )

    binding = [
        {
            "name": leg["name"],
            "quantity": leg["quantity"],
            "value": leg["value"],
            "error": leg["error"],
            "bound": names[leg["bound"]],
            "free": names[leg["free"]],
            "note": leg["note"],
            "problems": {
                "bound": problems[leg["bound"]],
                "free": problems[leg["free"]],
            },
        }
        for leg in summary["legs"]
    ]

    hydration = [
        {
            "name": h["name"],
            "value": h["value"],
            "error": h["error"],
            "free": names[h["free"]],
            "vacuum": names[h["vacuum"]] if h["vacuum"] else None,
            "decouple": h["decouple"],
            "note": h["note"],
            "problems": {
                "free": problems[h["free"]],
                "vacuum": problems[h["vacuum"]] if h["vacuum"] else [],
            },
        }
        for h in summary["hydration"]
    ]

    return {
        "version": _JSON_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paths": [str(p) for p in paths],
        "units": "kcal/mol",
        "simulations": simulations,
        "binding": binding,
        "hydration": hydration,
        "excluded": [
            {"name": r["name"], "path": r["path"], "reason": r["reason"]}
            for r in summary["excluded"]
        ],
        "ambiguous": summary["ambiguous"],
    }


def to_tables(report):
    """
    The summary as tables of (header, rows), keyed by name, from the JSON
    form. Free energies are in kcal/mol.
    """

    def joined(problems):
        return "; ".join(problems)

    tables = {}
    tables["binding"] = (
        ["system", "quantity", _VALUE, _ERROR, "bound leg", "free leg", "problems"],
        [
            [
                b["name"],
                b["quantity"],
                b["value"],
                b["error"],
                b["bound"],
                b["free"],
                joined(
                    [f"bound: {p}" for p in b["problems"]["bound"]]
                    + [f"free: {p}" for p in b["problems"]["free"]]
                    + ([b["note"]] if b["note"] else [])
                ),
            ]
            for b in report["binding"]
        ],
    )
    tables["hydration"] = (
        ["system", _VALUE, _ERROR, "free leg", "vacuum leg", "problems"],
        [
            [
                h["name"],
                h["value"],
                h["error"],
                h["free"],
                h["vacuum"] or ("not needed (decouple)" if h["decouple"] else None),
                joined(
                    [f"free: {p}" for p in h["problems"]["free"]]
                    + [f"vacuum: {p}" for p in h["problems"]["vacuum"]]
                    + ([h["note"]] if h["note"] else [])
                ),
            ]
            for h in report["hydration"]
        ],
    )
    tables["simulations"] = (
        [
            "simulation",
            "leg",
            "runs",
            "finished",
            "analysed",
            "progress (%)",
            _VALUE,
            _ERROR,
            "problems",
        ],
        [
            [
                s["name"],
                s["leg"],
                s["num_runs"],
                s["num_finished"],
                s["num_analysed"],
                None if s["progress"] is None else round(100 * s["progress"], 1),
                s["free_energy"],
                s["free_energy_error"],
                joined(s["problems"]),
            ]
            for s in report["simulations"]
        ],
    )
    tables["excluded"] = (
        ["run", "path", "reason"],
        [[r["name"], r["path"], r["reason"]] for r in report["excluded"]],
    )
    return tables


def format_text(report, colour=False):
    """
    The summary as plain text tables, for a terminal, with problems shown in
    colour if requested.
    """

    def warn(text):
        return f"{_WARN}{text}{_RESET}" if colour else text

    titles = {
        "binding": "Binding free energies",
        "hydration": "Hydration free energies",
        "simulations": "Simulations",
        "excluded": "Runs left out",
    }

    def cell(value):
        if value is None:
            return "-"
        # Free energies are already formatted, so this is e.g. progress.
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    sections = []
    for key, (header, rows) in to_tables(report).items():
        if not rows:
            continue
        # A value and its error are shown together, each aligned on its digits.
        if _VALUE in header:
            i = header.index(_VALUE)
            values = [None if r[i] is None else f"{r[i]:.2f}" for r in rows]
            errors = [None if r[i + 1] is None else f"{r[i + 1]:.2f}" for r in rows]
            value_width = max((len(v) for v in values if v), default=1)
            error_width = max((len(e) for e in errors if e), default=0)

            def combined(value, error):
                if value is None:
                    return "-".rjust(value_width)
                if error is None:
                    return value.rjust(value_width)
                return f"{value.rjust(value_width)} ± {error.rjust(error_width)}"

            header = header[:i] + ["ΔG (kcal/mol)"] + header[i + 2 :]
            rows = [
                r[:i] + [combined(v, e)] + r[i + 2 :]
                for r, v, e in zip(rows, values, errors)
            ]
        # Other numbers are right-aligned.
        numeric = [
            all(v is None or isinstance(v, (int, float)) for v in column)
            for column in zip(*rows)
        ]
        cells = [[cell(v) for v in row] for row in rows]
        widths = [max(len(c) for c in column) for column in zip(header, *cells)]
        problems = header.index("problems") if "problems" in header else None

        def line(row, highlight=False):
            parts = [
                c.rjust(w) if n else c.ljust(w) for c, w, n in zip(row, widths, numeric)
            ]
            # Coloured after padding, so that the escape codes don't count
            # towards the width.
            if highlight and problems is not None and row[problems]:
                text = parts[problems].rstrip()
                parts[problems] = warn(text) + parts[problems][len(text) :]
            return "  ".join(parts).rstrip()

        lines = [titles[key], "", line(header)]
        lines.append("  ".join("-" * w for w in widths))
        lines.extend(line(row, highlight=True) for row in cells)
        sections.append("\n".join(lines))

    if report["ambiguous"]:
        sections.append(
            warn(
                f"{report['ambiguous']} legs couldn't be paired, since more than "
                "one bound or free leg has matching settings."
            )
        )
    if not sections:
        return "No SOMD2 output found."
    return "\n\n".join(sections)


def write_csv(report, directory):
    """
    Write each table of the summary to a CSV file in a directory, returning
    the files written.
    """
    import csv
    from pathlib import Path

    def field(value):
        # Stops spreadsheets treating text, e.g. a run name, as a formula.
        if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
            return f"'{value}"
        return value

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for key, (header, rows) in to_tables(report).items():
        path = directory / f"{key}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows([[field(v) for v in row] for row in rows])
        written.append(path)
    return written
