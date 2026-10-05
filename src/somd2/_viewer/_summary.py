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
Grouping of runs into repeats and legs for the viewer's summary page.
"""

__all__ = ["build_summary", "leg_settings_key", "settings_key", "topology_fingerprint"]

import hashlib as _hashlib
import json as _json
import math as _math
import threading as _threading

# Thresholds for flagging problems.
_MIN_OVERLAP = 0.03
_MIN_TRANSITION = 0.05


def _hash(value):
    return _hashlib.sha1(_json.dumps(value).encode()).hexdigest()[:16]


_cache = {}
_cache_lock = _threading.Lock()


def _file_key(path):
    """
    Hash a topology file, skipping its first line, which holds the date.
    """
    digest = _hashlib.sha1()
    with open(path, "rb") as f:
        f.readline()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology_fingerprint(topology0, topology1):
    """
    Fingerprint the end-state topologies of a run. Topologies with identical
    contents, e.g. those of repeats, reuse the same fingerprint, which is also
    kept on disk so that it is available straight away next time.
    """
    key = (_file_key(topology0), _file_key(topology1))
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    fingerprint = _read_cached(key)
    if fingerprint is None:
        fingerprint = _fingerprint(topology0, topology1)
        _write_cached(key, fingerprint)
    with _cache_lock:
        _cache[key] = fingerprint
    return fingerprint


# Changed whenever the fingerprint does, so that cached ones are recomputed.
_FINGERPRINT_VERSION = 1


def _cached_path(key):
    from ._cache import cache_dir

    name = f"v{_FINGERPRINT_VERSION}-{key[0]}-{key[1]}.json"
    return cache_dir() / "fingerprints" / name


def _read_cached(key):
    try:
        return _json.loads(_cached_path(key).read_text())
    except (OSError, ValueError):
        return None


def _write_cached(key, fingerprint):
    import contextlib
    import os

    from ._log import report

    path = _cached_path(key)
    # Written then renamed, so that another viewer never reads part of it.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(_json.dumps(fingerprint))
        os.replace(temporary, path)
    except OSError as e:
        with contextlib.suppress(OSError):
            temporary.unlink()
        report(e, f"Couldn't write to the cache in {path.parent}")


def _fingerprint(topology0, topology1):
    """
    Fingerprint the end-state topologies of a run.

    Only molecules that aren't water at either end state are used, since GCMC
    runs add waters to the topologies when they finish.

    Returns
    -------

    fingerprint: dict
        Fingerprints of the whole system and of the perturbed molecules, and
        whether the system is a bound leg, i.e. contains a protein.
    """
    import sire as _sr

    systems = [_sr.load(str(t), show_warnings=False) for t in (topology0, topology1)]

    def non_water(system):
        numbers = {mol.number() for mol in system.molecules("not water")}
        return {
            i for i, mol in enumerate(system.molecules()) if mol.number() in numbers
        }

    def molecule_fingerprint(mol):
        bonds = sorted(
            (b.atom0().index().value(), b.atom1().index().value()) for b in mol.bonds()
        )
        return _hash(
            [
                [e.num_protons() for e in mol.property("element").to_list()],
                [str(t) for t in mol.property("ambertype").to_list()],
                [round(q.value(), 4) for q in mol.property("charge").to_list()],
                bonds,
            ]
        )

    indices = sorted(non_water(systems[0]) & non_water(systems[1]))
    mols0 = systems[0].molecules()
    mols1 = systems[1].molecules()
    pairs = [
        (molecule_fingerprint(mols0[i]), molecule_fingerprint(mols1[i]))
        for i in indices
    ]

    perturbed = [p for p in pairs if p[0] != p[1]]
    large = sum(1 for i in indices if mols0[i].num_atoms() >= 3)
    return {
        "system": _hash(pairs),
        "perturbation": _hash(perturbed) if perturbed else None,
        # The leg, as named in SOMD1: a protein, water only, or neither.
        "leg": (
            "bound" if large > 1 else "free" if len(indices) < len(mols0) else "vacuum"
        ),
    }


def _settings(config, ignore):
    """
    The settings in a config, except those ignored and the output directory,
    as JSON values.
    """
    # Restart validation treats None and False as equal, so do the same here.
    kept = {
        k: False if v is None else v
        for k, v in sorted(config.items())
        if k not in ignore and k != "output_directory"
    }
    return _json.loads(_json.dumps(kept, default=str))


def settings_key(config, ignore):
    """
    Return a key for the settings that must match between repeats, i.e. all
    except those that may change on restart, and the output directory.
    """
    return _hash(_settings(config, ignore))


def leg_settings_key(config, ignore):
    """
    Return a key for the settings that must match between the bound and free
    legs of a perturbation, i.e. as for repeats, but also ignoring options
    that are often set per leg, such as restraints and the λ windows, and GCMC,
    which is only used for the bound leg.
    """
    return settings_key(config, _leg_specific(config) | set(ignore))


def _leg_specific(config):
    per_leg = {
        "restraints",
        "num_lambda",
        "lambda_values",
        "lambda_energy",
        "num_energy_neighbours",
        "null_energy",
    }
    return {
        k
        for k in config
        if k in per_leg or k.startswith("restraint_search") or k.startswith("gcmc")
    }


def vacuum_settings(config, ignore):
    """
    Return the settings that must match between the free and vacuum legs of a
    hydration free energy, i.e. as for bound and free legs, but also ignoring
    pressure and the dispersion correction, which aren't applied without water.
    """
    no_water = {
        "pressure",
        "barostat_frequency",
        "surface_tension",
        "use_dispersion_correction",
    }
    return _settings(config, _leg_specific(config) | set(ignore) | no_water)


def _is_cutoff_option(key):
    return key in ("cutoff", "cutoff_type", "tune_pme") or key.startswith("pme_")


def vacuum_settings_keys(settings):
    """
    Return keys for the settings from vacuum_settings(). The second key also
    ignores the cutoff and PME options, which are disabled for a vacuum leg run
    without a periodic box.
    """
    return _hash(settings), _hash(
        {k: v for k, v in settings.items() if not _is_cutoff_option(k)}
    )


def _common_name(names):
    """
    The longest common leading path of a set of run names, or the first name
    and a count if they have none.
    """
    parts = [n.split("/") for n in names]
    common = []
    for level in zip(*parts):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    if common:
        return "/".join(common)
    return names[0] + (f" + {len(names) - 1} more" if len(names) > 1 else "")


def _mean_and_error(values, errors):
    """
    The mean of a set of repeats and its standard error, or the MBAR error
    for a single value.
    """
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, errors[0]
    sd = _math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd / _math.sqrt(n)


def _mean_progress(runs):
    """
    The average fraction of the requested simulation time completed by a set
    of repeats.
    """
    values = [r["progress"] for r in runs if r.get("progress") is not None]
    return sum(values) / len(values) if values else None


def _usable(run):
    """
    Whether a run has a free energy and an error to average.
    """
    return (
        run.get("free_energy") is not None and run.get("free_energy_error") is not None
    )


def _group_problems(runs):
    problems = []
    uncorrected = [
        r for r in runs if r.get("auto_restraint") and r.get("correction") is None
    ]
    if uncorrected:
        problems.append(
            f"no restraint correction found for {len(uncorrected)} "
            f"{'run' if len(uncorrected) == 1 else 'runs'}"
        )
    failed = [r for r in runs if r["status"] in ("error", "stopped")]
    if failed:
        problems.append(
            f"{len(failed)} {'run has' if len(failed) == 1 else 'runs have'} stopped"
        )
    overlaps = [r["min_overlap"] for r in runs if r.get("min_overlap") is not None]
    if overlaps and min(overlaps) < _MIN_OVERLAP:
        problems.append(f"poor overlap ({min(overlaps):.3f})")
    transitions = [
        r["min_transition"] for r in runs if r.get("min_transition") is not None
    ]
    if transitions and min(transitions) < _MIN_TRANSITION:
        problems.append(f"poor replica mixing ({min(transitions):.3f})")
    done = [r for r in runs if _usable(r)]
    if len(done) >= 2:
        values = [r["free_energy"] for r in done]
        mean = sum(values) / len(values)
        sd = _math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        rms = _math.sqrt(sum(r["free_energy_error"] ** 2 for r in done) / len(done))
        if rms > 0 and sd > 2 * rms:
            problems.append("repeats disagree beyond their errors")
    return problems


def build_summary(entries):
    """
    Group runs into repeats of the same simulation, and pair the bound and
    free legs of the same perturbation.

    Parameters
    ----------

    entries: [dict]
        One entry per run, from Simulation.summary_entry().

    Returns
    -------

    summary: dict
        The simulations, legs, runs left out because of an error, and
        whether there is anything to summarise.
    """
    groups = {}
    excluded = []
    pending = sum(bool(e.get("fingerprint_pending")) for e in entries)
    for entry in entries:
        fingerprint = entry.get("fingerprint")
        reason = entry.get("reason")
        if reason is None and fingerprint and "error" in fingerprint:
            reason = f"Couldn't identify the system: {fingerprint['error']}"
        if reason is not None:
            excluded.append(
                {"id": entry["id"], "name": entry["name"], "reason": reason}
            )
            continue
        if not fingerprint or not entry.get("settings"):
            continue
        key = (fingerprint["system"], entry["settings"])
        groups.setdefault(key, []).append(entry)

    simulations = []
    for key, runs in groups.items():
        done = [r for r in runs if _usable(r)]
        # Runs without a known restraint correction are left out.
        corrected = [
            (r["free_energy"] + (r["correction"] or 0.0), r["free_energy_error"])
            for r in done
            if not r.get("auto_restraint") or r.get("correction") is not None
        ]
        mean, error = _mean_and_error(
            [r["free_energy"] for r in done], [r["free_energy_error"] for r in done]
        )
        corrected_mean, corrected_error = _mean_and_error(
            [v for v, _ in corrected], [e for _, e in corrected]
        )
        first = runs[0]["fingerprint"]
        simulations.append(
            {
                "id": "|".join(key),
                "name": _common_name([r["name"] for r in runs]),
                "perturbation": first["perturbation"],
                "leg": first["leg"],
                "absolute": runs[0]["absolute"],
                "auto_restraint": all(r.get("auto_restraint") for r in runs),
                "leg_settings": runs[0]["leg_settings"],
                "vacuum_settings": runs[0]["vacuum_settings"],
                "vacuum_values": runs[0].get("vacuum_values") or {},
                "schedule": runs[0]["schedule"],
                "cutoff_type": runs[0]["cutoff_type"],
                "runs": [
                    {
                        k: r.get(k)
                        for k in (
                            "id",
                            "name",
                            "status",
                            "progress",
                            "free_energy",
                            "free_energy_error",
                        )
                    }
                    for r in runs
                ],
                "progress": _mean_progress(runs),
                "num_finished": sum(r["status"] == "finished" for r in runs),
                "num_analysed": len(done),
                "free_energy": mean,
                "free_energy_error": error,
                "corrected_free_energy": corrected_mean,
                "corrected_free_energy_error": corrected_error,
                "problems": _group_problems(runs),
            }
        )
    simulations.sort(key=lambda s: s["name"])

    # Pair legs with matching settings. Each bound leg, e.g. with and without
    # GCMC, is paired with the free leg when there is only one.
    candidates = {}
    for sim in simulations:
        if sim["perturbation"]:
            key = (sim["perturbation"], sim["leg_settings"])
            candidates.setdefault(key, []).append(sim)
    legs = []
    unpaired = 0
    for sims in candidates.values():
        bound = [s for s in sims if s["leg"] == "bound"]
        free = [s for s in sims if s["leg"] == "free"]
        if len(free) == 1:
            legs.extend(_leg(b, free[0]) for b in bound)
        elif bound and free:
            unpaired += len(sims)
    legs.sort(key=lambda leg: leg["name"])

    hydration = _hydration(simulations, legs)
    # Only needed to explain unmatched vacuum legs.
    for sim in simulations:
        del sim["vacuum_values"]

    return {
        "available": bool(legs)
        or bool(hydration)
        or any(len(s["runs"]) > 1 for s in simulations),
        "pending": pending,
        "updating": any(e.get("updating") for e in entries),
        "simulations": simulations,
        "legs": legs,
        "hydration": hydration,
        "ambiguous": unpaired,
        "excluded": sorted(excluded, key=lambda e: e["name"]),
    }


def _vacuum_match(free, vacuum):
    """
    Whether a vacuum leg was run with the same settings as a free leg. The
    cutoff only has to match if the vacuum leg was run in a periodic box,
    since it is otherwise disabled.
    """
    index = 1 if vacuum["cutoff_type"] == "none" else 0
    return free["vacuum_settings"][index] == vacuum["vacuum_settings"][index]


def _differences(free, vacuum):
    """
    The settings that stop a vacuum leg matching a free leg.
    """
    a, b = free["vacuum_values"], vacuum["vacuum_values"]
    ignore_cutoff = vacuum["cutoff_type"] == "none"
    return sorted(
        k
        for k in set(a) | set(b)
        if a.get(k) != b.get(k) and not (ignore_cutoff and _is_cutoff_option(k))
    )


def _hydration(simulations, legs):
    """
    Absolute hydration free energies from free legs, which need a vacuum leg
    unless the decouple schedule was used.
    """
    results = []
    vacuum = [s for s in simulations if s["leg"] == "vacuum" and s["perturbation"]]
    binding = {leg["free"] for leg in legs}
    for f in simulations:
        if f["leg"] != "free" or not f["absolute"]:
            continue
        result = {
            "name": f["name"],
            "free": f["id"],
            "free_name": f["name"],
            "vacuum": None,
            "vacuum_name": None,
            "decouple": f["schedule"] == "decouple",
            "value": None,
            "error": None,
            "note": None,
        }
        if f["schedule"] == "decouple":
            # ΔG_hyd = −ΔG_free
            if f["free_energy"] is not None:
                result["value"] = -f["free_energy"]
                result["error"] = f["free_energy_error"]
        else:
            same = [v for v in vacuum if v["perturbation"] == f["perturbation"]]
            matches = [v for v in same if _vacuum_match(f, v)]
            if not matches:
                if same:
                    closest = min(same, key=lambda v: len(_differences(f, v)))
                    differences = _differences(f, closest)
                    result["note"] = "No vacuum leg has matching settings." + (
                        f" {closest['name']} differs in: {', '.join(differences)}."
                        if differences
                        else ""
                    )
                elif vacuum and f["id"] not in binding:
                    result["note"] = "No vacuum leg found for this molecule."
                else:
                    # E.g. the free leg of an ABFE campaign.
                    continue
            elif len(matches) > 1:
                result["note"] = "More than one vacuum leg has matching settings."
            else:
                v = matches[0]
                result["vacuum"] = v["id"]
                result["vacuum_name"] = v["name"]
                result["name"] = _common_name([f["name"], v["name"]])
                # ΔG_hyd = ΔG_vacuum − ΔG_free
                if f["free_energy"] is not None and v["free_energy"] is not None:
                    result["value"] = v["free_energy"] - f["free_energy"]
                    result["error"] = _math.hypot(
                        v["free_energy_error"], f["free_energy_error"]
                    )
        results.append(result)
    results.sort(key=lambda r: r["name"])
    return results


def _leg(b, f):
    """
    The relative or absolute binding free energy from a bound and free leg.
    """
    leg = {
        "name": _common_name([b["name"], f["name"]]),
        "bound": b["id"],
        "free": f["id"],
        "bound_name": b["name"],
        "free_name": f["name"],
        "absolute": b["absolute"],
        "value": None,
        "error": None,
        "note": None,
    }
    if b["absolute"]:
        leg["quantity"] = "ΔG bind"
        if not b["auto_restraint"]:
            leg["note"] = (
                "Only shown when the bound leg's restraint was generated automatically."
            )
        elif b["corrected_free_energy"] is not None and f["free_energy"] is not None:
            # ΔG_bind = ΔG_free − ΔG_bound − ΔG_correction
            leg["value"] = f["free_energy"] - b["corrected_free_energy"]
            leg["error"] = _math.hypot(
                f["free_energy_error"], b["corrected_free_energy_error"]
            )
    else:
        leg["quantity"] = "ΔΔG"
        if b["free_energy"] is not None and f["free_energy"] is not None:
            leg["value"] = b["free_energy"] - f["free_energy"]
            leg["error"] = _math.hypot(b["free_energy_error"], f["free_energy_error"])
    return leg
