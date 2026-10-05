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
Perturbation networks for the viewer's network page.
"""

__all__ = ["build_network", "find_network", "read_network"]

import math as _math
from pathlib import Path as _Path

# The number of combined standard errors beyond which a cycle closure or
# hysteresis is flagged.
_THRESHOLD = 2.0


def find_network(paths, network=None):
    """
    Return the network file, either as given, or a 'network.dat' found
    directly in one of the paths.
    """
    if network is not None:
        return _Path(network).resolve()
    for path in paths:
        candidate = _Path(path) / "network.dat"
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_network(path):
    """
    Read the edges from a network file, one per line as 'ligand_a ligand_b',
    with any further columns, e.g. the λ windows, ignored.
    """
    edges = []
    seen = set()
    for line in _Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or line.lstrip().startswith("#"):
            continue
        edge = (fields[0], fields[1])
        if edge not in seen and edge[::-1] not in seen:
            seen.add(edge)
            edges.append(edge)
    return edges


# Separators accepted between the two ligand names in an edge's directory.
_SEPARATORS = ("~", "-", "_", "->", "--", "__", "~~", "_to_", "-to-")


def _directory_names(edges):
    """
    Map each directory name that an edge's runs could sit under to the edge
    and direction, e.g. 'a~b' to ('a', 'b') and 'b-a' to ('b', 'a'). Names
    that could belong to more than one edge map to None.
    """
    names = {}
    for a, b in edges:
        for pair in ((a, b), (b, a)):
            for sep in _SEPARATORS:
                name = pair[0] + sep + pair[1]
                if name in names and names[name] != pair:
                    names[name] = None
                else:
                    names[name] = pair
    return names


def _edge_directory(names, directories):
    """
    The single edge directory that all of a set of run names sit under, as a
    pair of ligand names in the direction run, if there is one.
    """
    candidates = None
    for name in names:
        parts = {directories[p] for p in name.split("/") if p in directories}
        candidates = parts if candidates is None else candidates & parts
    if candidates and len(candidates) == 1:
        return candidates.pop()
    return None


def _direction(a, b, legs, sims, by_id):
    """
    The results for running the perturbation from ligand a to ligand b.
    """
    matching = [leg for leg in legs if leg["edge"] == (a, b)]
    started = [s for s in sims if s["edge"] == (a, b)]
    result = {
        "status": "not started",
        "value": None,
        "error": None,
        "problems": [],
        "runs": [],
        "bound": None,
        "free": None,
    }
    if not started:
        return result

    runs = [r for s in started for r in s["runs"]]
    result["runs"] = [
        {"id": r["id"], "name": r["name"], "status": r["status"]} for r in runs
    ]
    result["problems"] = [p for s in started for p in s["problems"]]

    if len(matching) > 1:
        result["status"] = "problem"
        result["problems"].append("more than one pair of legs")
    elif matching:
        leg = matching[0]
        result["bound"] = by_id[leg["bound"]]["name"]
        result["free"] = by_id[leg["free"]]["name"]
        result["value"], result["error"] = leg["value"], leg["error"]

    finished = bool(runs) and all(r["status"] == "finished" for r in runs)
    if finished and not matching and result["status"] != "problem":
        result["problems"].append("runs finished, but no matching pair of legs")
    if result["status"] != "problem":
        if result["problems"]:
            result["status"] = "problem"
        elif finished:
            result["status"] = "finished"
        else:
            result["status"] = "running"
    return result


def _combine(forward, reverse):
    """
    Combine the forward and negated reverse results with inverse-variance
    weights.
    """
    estimates = []
    if forward["value"] is not None and forward["error"]:
        estimates.append((forward["value"], forward["error"]))
    if reverse["value"] is not None and reverse["error"]:
        estimates.append((-reverse["value"], reverse["error"]))
    if not estimates:
        return None, None
    weights = [1 / e**2 for _, e in estimates]
    value = sum(w * v for w, (v, _) in zip(weights, estimates)) / sum(weights)
    return value, 1 / _math.sqrt(sum(weights))


def _cycles(ligands, edges):
    """
    Independent cycles of the edges with results, one for each edge that
    isn't part of a spanning tree, with the closure around each.
    """
    adjacency = {lig: [] for lig in ligands}
    for edge in edges:
        if edge["combined"]["value"] is None:
            continue
        a, b = edge["a"], edge["b"]
        adjacency[a].append((b, edge, 1))
        adjacency[b].append((a, edge, -1))

    parent = {}
    cycles = []
    used = set()
    for root in ligands:
        if root in parent:
            continue
        parent[root] = None
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour, edge, sign in adjacency[node]:
                key = (edge["a"], edge["b"])
                if key in used:
                    continue
                used.add(key)
                if neighbour in parent:
                    cycles.append(_closure(parent, node, neighbour, edge, sign))
                else:
                    parent[neighbour] = (node, edge, sign)
                    stack.append(neighbour)
    return cycles


def _path_to_root(parent, node):
    path = [node]
    while parent[node] is not None:
        node = parent[node][0]
        path.append(node)
    return path


def _closure(parent, start, end, edge, sign):
    """
    The closure of the cycle formed by adding an edge from 'start' to 'end'
    to the spanning tree.
    """
    up = _path_to_root(parent, start)
    down = _path_to_root(parent, end)
    common = next(n for n in up if n in set(down))
    # Walk start → common → end along the tree, then back to start by the edge.
    path = up[: up.index(common) + 1] + down[: down.index(common)][::-1]

    value = 0.0
    variance = 0.0
    members = []

    def step(edge, sign):
        nonlocal value, variance
        value += sign * edge["combined"]["value"]
        variance += edge["combined"]["error"] ** 2
        members.append((edge["a"], edge["b"]))

    for node in up[: up.index(common)]:
        _, e, s = parent[node]
        step(e, -s)
    for node in down[: down.index(common)][::-1]:
        _, e, s = parent[node]
        step(e, s)
    step(edge, -sign)

    error = _math.sqrt(variance)
    return {
        "ligands": path,
        "edges": members,
        "value": value,
        "error": error,
        "flagged": abs(value) > _THRESHOLD * error,
    }


def build_network(edges, summary):
    """
    Attach the summary's results to the edges of a network.

    Parameters
    ----------

    edges: [(str, str)]
        The edges, as pairs of ligand names.

    summary: dict
        The summary, from build_summary().

    Returns
    -------

    network: dict
        The ligands, the edges with the results in each direction, and the
        cycles with their closures.
    """
    directories = _directory_names(edges)
    sims = []
    for sim in summary["simulations"]:
        names = [r["name"] for r in sim["runs"]]
        sims.append(dict(sim, edge=_edge_directory(names, directories)))
    by_id = {s["id"]: s for s in sims}

    legs = []
    for leg in summary["legs"]:
        if leg["absolute"]:
            continue
        names = [
            r["name"]
            for s in (by_id[leg["bound"]], by_id[leg["free"]])
            for r in s["runs"]
        ]
        legs.append(dict(leg, edge=_edge_directory(names, directories)))

    ligands = []
    for a, b in edges:
        for lig in (a, b):
            if lig not in ligands:
                ligands.append(lig)

    results = []
    for a, b in edges:
        forward = _direction(a, b, legs, sims, by_id)
        reverse = _direction(b, a, legs, sims, by_id)
        value, error = _combine(forward, reverse)
        hysteresis = None
        if forward["value"] is not None and reverse["value"] is not None:
            h = forward["value"] + reverse["value"]
            herror = _math.hypot(forward["error"] or 0.0, reverse["error"] or 0.0)
            hysteresis = {
                "value": h,
                "error": herror,
                "flagged": herror > 0 and abs(h) > _THRESHOLD * herror,
            }
        results.append(
            {
                "a": a,
                "b": b,
                "forward": forward,
                "reverse": reverse,
                "combined": {"value": value, "error": error},
                "hysteresis": hysteresis,
            }
        )

    cycles = _cycles(ligands, results)
    flagged = {e for c in cycles if c["flagged"] for e in c["edges"]}
    for edge in results:
        edge["in_flagged_cycle"] = (edge["a"], edge["b"]) in flagged

    return {
        "ligands": ligands,
        "edges": results,
        "cycles": cycles,
    }
