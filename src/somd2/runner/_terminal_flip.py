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

# Adapted from the terminal ring flip MC implemenation in GrandFEP:
# https://github.com/deGrootLab/GrandFEP
# (Released under the MIT License.)

__all__ = ["TerminalFlipSampler", "detect_terminal_groups"]

import numpy as _np

import sire.legacy.Mol as _Mol

from somd2 import _logger
from somd2._utils import _delta_sym


def _auto_flip_angle(mol, anchor_idx, pivot_idx, ring_neighbor_idxs):
    """
    Compute the flip angle for a terminal group from the molecular geometry.

    The angle is measured between the two ring neighbours of the pivot,
    projected onto the plane perpendicular to the rotation axis (anchor →
    pivot). For a planar C₂-symmetric ring this is 180°; for higher-symmetry
    rings it will be smaller.

    Parameters
    ----------

    mol : sire.legacy.Mol.Molecule
        The perturbable molecule.

    anchor_idx : int
        Molecule-local index of the anchor atom.

    pivot_idx : int
        Molecule-local index of the pivot atom.

    ring_neighbor_idxs : list of int
        Molecule-local indices of the two ring atoms directly bonded to the
        pivot (i.e. the ortho atoms for a benzene ring).

    Returns
    -------

    float
        Raw angle in degrees between the projected ring-neighbour vectors.
    """

    def _coords(idx):
        v = mol.atom(_Mol.AtomIdx(idx)).property("coordinates")
        return _np.array([v.x().value(), v.y().value(), v.z().value()])

    anchor = _coords(anchor_idx)
    pivot = _coords(pivot_idx)
    n1 = _coords(ring_neighbor_idxs[0])
    n2 = _coords(ring_neighbor_idxs[1])

    # Unit rotation axis from anchor to pivot.
    k = pivot - anchor
    k = k / _np.linalg.norm(k)

    # Project each ring-neighbour displacement onto the plane perp to k.
    v1 = n1 - pivot
    v1_perp = v1 - _np.dot(v1, k) * k

    v2 = n2 - pivot
    v2_perp = v2 - _np.dot(v2, k) * k

    # Angle between the two projected vectors.
    cos_angle = _np.dot(v1_perp, v2_perp) / (
        _np.linalg.norm(v1_perp) * _np.linalg.norm(v2_perp)
    )
    return float(_np.degrees(_np.arccos(_np.clip(cos_angle, -1.0, 1.0))))


def _round_to_symmetry_angle(raw_angle, tolerance=10.0):
    """
    Round ``raw_angle`` to the nearest crystallographic symmetry angle
    (360°/n for n = 2 … 12). Returns ``None`` if the closest match is more
    than ``tolerance`` degrees away, indicating that the ring has no useful
    rotational symmetry.

    Parameters
    ----------

    raw_angle : float
        Measured angle in degrees.

    tolerance : float
        Maximum deviation (degrees) from a symmetry angle. Default is 10.0.

    Returns
    -------

    float or None
        The nearest symmetry angle in degrees, or None if none is close enough.
    """
    symmetry_angles = [360.0 / n for n in range(2, 13)]
    diffs = [abs(raw_angle - a) for a in symmetry_angles]
    min_idx = int(_np.argmin(diffs))
    if diffs[min_idx] > tolerance:
        return None
    return symmetry_angles[min_idx]


def detect_terminal_groups(system, flip_angle=None):
    """
    Detect terminal ring groups in perturbable molecules using Sire's native
    connectivity.

    A terminal ring group is identified by a bond between a non-ring atom
    (the anchor) and a ring atom (the pivot), where the ring side of the bond
    is connected to the rest of the molecule only through that single bond.
    The mobile atoms are all atoms reachable from the pivot when the
    anchor-pivot bond is cut.

    Parameters
    ----------

    system : sire system or molecule group
        The Sire system containing perturbable molecules.

    flip_angle : float or None
        The flip angle in degrees. If None (the default), the angle is
        determined automatically from the geometry of each terminal group
        by measuring the angle between the two ring neighbours of the pivot
        projected perpendicular to the rotation axis, then rounding to the
        nearest crystallographic symmetry angle (360°/n for n = 2..12). If
        a float is given it overrides the geometric measurement for all
        groups.

    Returns
    -------

    list of tuple
        Each entry is (angle, [anchor_idx, pivot_idx, mobile_idx_0, ...])
        where all indices are absolute atom indices corresponding to OpenMM
        atom ordering.
    """
    terminal_groups = []

    # Get the perturbable molecules.
    try:
        pert_mols = system.molecules("property is_perturbable")
    except Exception:
        _logger.warning(
            "No perturbable molecules found. Terminal flip detection skipped."
        )
        return terminal_groups

    # All atoms in the system, used to obtain absolute (OpenMM) atom indices.
    all_atoms = system.atoms()

    for mol in pert_mols:
        try:
            connectivity = mol.property("connectivity")
        except Exception:
            _logger.warning(f"Molecule {mol} has no 'connectivity' property. Skipping.")
            continue

        # Skip molecules whose connectivity changes between end states (e.g.
        # ring-breaking/growing perturbations). Terminal groups detected from
        # the lambda=0 connectivity would be invalid at lambda=1.
        try:
            conn0 = mol.property("connectivity0")
            conn1 = mol.property("connectivity1")
            if conn0 != conn1:
                _logger.warning(
                    f"Molecule {mol} has different connectivity at lambda=0 and "
                    "lambda=1 (ring-breaking/growing perturbation). Skipping "
                    "terminal flip detection for this molecule."
                )
                continue
        except Exception:
            pass

        num_atoms = mol.num_atoms()
        seen_bonds = set()

        for i in range(num_atoms):
            atom_i_idx = _Mol.AtomIdx(i)

            # Only consider non-ring atoms as anchors.
            if connectivity.in_ring(atom_i_idx):
                continue

            # Skip dead-end atoms (e.g. hydrogen bonded only to a ring
            # carbon): a valid anchor must be part of a chain, so it needs
            # at least two connections (one to the pivot, one elsewhere).
            if len(connectivity.connections_to(atom_i_idx)) < 2:
                continue

            for neighbor_idx in connectivity.connections_to(atom_i_idx):
                j = neighbor_idx.value()

                # Only consider ring atoms as pivots.
                if not connectivity.in_ring(_Mol.AtomIdx(j)):
                    continue

                # Avoid processing the same bond twice.
                bond_key = (min(i, j), max(i, j))
                if bond_key in seen_bonds:
                    continue
                seen_bonds.add(bond_key)

                # Collect mobile atoms via BFS from the pivot, not crossing
                # the anchor. The pivot itself does not move (it is the
                # rotation centre), so it is excluded from the mobile list.
                mobile = _bfs_mobile(connectivity, i, j, num_atoms)

                if not mobile:
                    continue

                # Determine the flip angle for this group.
                if flip_angle is not None:
                    group_angle = flip_angle
                else:
                    # Find the two ring neighbours of the pivot (mobile atoms
                    # directly bonded to the pivot that are in the ring).
                    mobile_set = set(mobile)
                    pivot_idx_obj = _Mol.AtomIdx(j)
                    ring_neighbors = [
                        n.value()
                        for n in connectivity.connections_to(pivot_idx_obj)
                        if n.value() in mobile_set
                        and connectivity.in_ring(_Mol.AtomIdx(n.value()))
                    ]

                    if len(ring_neighbors) != 2:
                        _logger.warning(
                            f"Expected 2 ring neighbours for pivot atom {j}, "
                            f"found {len(ring_neighbors)}. Skipping group."
                        )
                        continue

                    raw = _auto_flip_angle(mol, i, j, ring_neighbors)
                    group_angle = _round_to_symmetry_angle(raw)

                    if group_angle is None:
                        _logger.warning(
                            f"Terminal group at pivot atom {j} has no recognised "
                            f"rotational symmetry (raw angle = {raw:.1f}°). "
                            "Skipping group."
                        )
                        continue

                    _logger.debug(
                        f"Terminal group at pivot atom {j}: auto-detected flip "
                        f"angle = {group_angle}° (raw = {raw:.1f}°)."
                    )

                # Map molecule-local indices to absolute system indices.
                anchor_abs = all_atoms.find(mol.atom(atom_i_idx))
                pivot_abs = all_atoms.find(mol.atom(_Mol.AtomIdx(j)))
                mobile_abs = [all_atoms.find(mol.atom(_Mol.AtomIdx(k))) for k in mobile]

                terminal_groups.append(
                    (group_angle, [anchor_abs, pivot_abs] + mobile_abs)
                )

    return terminal_groups


def _bfs_mobile(connectivity, anchor_idx, pivot_idx, num_atoms):
    """
    Breadth-first search from ``pivot_idx``, not crossing ``anchor_idx``.

    Returns a sorted list of atom indices for atoms that will be rotated
    (all reachable atoms except the anchor and the pivot itself, since the
    pivot is the fixed rotation centre).

    Parameters
    ----------

    connectivity : sire.legacy.Mol.Connectivity
        The molecular connectivity object.

    anchor_idx : int
        Index of the anchor atom (defines the rotation axis start; fixed).

    pivot_idx : int
        Index of the pivot atom (rotation centre; fixed).

    num_atoms : int
        Total number of atoms in the molecule.

    Returns
    -------

    list of int
        Sorted list of mobile atom indices.
    """
    visited = {anchor_idx, pivot_idx}
    queue = [pivot_idx]

    while queue:
        current = queue.pop(0)
        for neighbor in connectivity.connections_to(_Mol.AtomIdx(current)):
            n = neighbor.value()
            if n not in visited:
                visited.add(n)
                queue.append(n)

    # Exclude the anchor and pivot; only mobile atoms are rotated.
    return sorted(visited - {anchor_idx, pivot_idx})


class TerminalFlipSampler:
    """
    Monte Carlo sampler for terminal ring flip moves.

    Each move selects one terminal group at random and attempts to rotate
    its mobile atoms by ±``flip_angle`` degrees around the bond axis from
    the anchor atom to the pivot atom. The move is accepted or rejected
    according to the Metropolis criterion.

    The rotation uses Rodrigues' rotation formula::

        v_rot = v·cos θ + (k × v)·sin θ + k·(k·v)·(1 − cos θ)

    where ``k`` is the unit vector along the rotation axis (anchor → pivot)
    and ``v`` is the displacement of a mobile atom from the pivot.

    The sign of ``flip_angle`` is chosen uniformly at random so that the
    proposal is symmetric, satisfying detailed balance for any angle.
    """

    def __init__(self, terminal_groups, temperature):
        """
        Parameters
        ----------

        terminal_groups : list of tuple
            Each entry is (angle, [anchor_idx, pivot_idx, mobile_idx_0, ...])
            where indices are absolute OpenMM atom indices.

        temperature : float
            Simulation temperature in Kelvin.
        """
        self._terminal_groups = terminal_groups

        # kBT in kJ/mol (R = 8.314462618e-3 kJ mol-1 K-1).
        self._kBT = 8.314462618e-3 * temperature

        self._num_attempted = 0
        self._num_accepted = 0

    def _rotate(self, context, group_idx, angle):
        """
        Rotate the mobile atoms of a terminal group by ``angle`` degrees
        around the anchor-to-pivot axis, updating the context in place.

        Parameters
        ----------

        context : openmm.Context
            The active OpenMM context.

        group_idx : int
            Index into ``self._terminal_groups`` selecting the group to rotate.

        angle : float
            Rotation angle in degrees.
        """
        from openmm import unit as _omm_unit

        _, atom_indices = self._terminal_groups[group_idx]

        positions = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(_omm_unit.nanometer)
        )

        theta = _np.deg2rad(angle)
        cos_t = _np.cos(theta)
        sin_t = _np.sin(theta)

        # Anchor (axis start, fixed) and pivot (rotation centre, fixed).
        p0 = positions[atom_indices[0]]
        p1 = positions[atom_indices[1]]

        # Unit rotation axis from anchor to pivot.
        axis = p1 - p0
        axis = axis / _np.linalg.norm(axis)

        # Rotate mobile atoms using Rodrigues' formula.
        new_positions = positions.copy()
        for atom_idx in atom_indices[2:]:
            v = positions[atom_idx] - p1
            new_positions[atom_idx] = (
                p1
                + v * cos_t
                + _np.cross(axis, v) * sin_t
                + axis * _np.dot(axis, v) * (1.0 - cos_t)
            )

        context.setPositions(new_positions * _omm_unit.nanometer)

    def move(self, context):
        """
        Attempt one terminal flip Monte Carlo move.

        A terminal group is chosen at random. The mobile atoms are rotated
        by ±``flip_angle`` around the anchor-to-pivot axis. The move is
        accepted with Metropolis probability ``min(1, exp(-ΔE / kBT))``.

        Parameters
        ----------

        context : openmm.Context
            The active OpenMM context.
        """
        from openmm import unit as _omm_unit

        if not self._terminal_groups:
            return

        self._num_attempted += 1

        # Randomly select one terminal group.
        group_idx = _np.random.randint(len(self._terminal_groups))
        angle, _ = self._terminal_groups[group_idx]

        # Retrieve current positions and energy before the move.
        state = context.getState(getPositions=True, getEnergy=True)
        old_positions = state.getPositions(asNumpy=True).value_in_unit(
            _omm_unit.nanometer
        )
        e_old = state.getPotentialEnergy().value_in_unit(_omm_unit.kilojoule_per_mole)

        # Random sign gives a symmetric proposal (detailed balance).
        signed_angle = float(_np.random.choice([-1, 1])) * angle
        self._rotate(context, group_idx, signed_angle)

        # Evaluate the energy of the proposed configuration.
        e_new = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(_omm_unit.kilojoule_per_mole)
        )

        # Metropolis acceptance criterion.
        delta_e = (e_new - e_old) / self._kBT
        if delta_e <= 0.0 or _np.random.random() < _np.exp(-delta_e):
            self._num_accepted += 1
            _logger.debug(
                f"Terminal flip accepted (group {group_idx}, "
                f"{_delta_sym} = {e_new - e_old:.2f} kJ/mol, "
                f"acc = {min(1.0, _np.exp(-delta_e)):.3f})"
            )
        else:
            context.setPositions(old_positions * _omm_unit.nanometer)
            _logger.debug(
                f"Terminal flip rejected (group {group_idx}, "
                f"{_delta_sym} = {e_new - e_old:.2f} kJ/mol, "
                f"acc = {_np.exp(-delta_e):.3f})"
            )

    @property
    def num_attempted(self):
        """Total number of terminal flip moves attempted."""
        return self._num_attempted

    @property
    def num_accepted(self):
        """Total number of terminal flip moves accepted."""
        return self._num_accepted

    @property
    def acceptance_rate(self):
        """Fraction of attempted moves that were accepted."""
        if self._num_attempted == 0:
            return 0.0
        return self._num_accepted / self._num_attempted
