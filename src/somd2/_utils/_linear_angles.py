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

__all__ = ["find_linear_angles", "torsion_window"]

# Angles with an equilibrium value of at least this many degrees are linear.
_LINEAR_ANGLE = 170.0


def find_linear_angles(mol, max_angle=_LINEAR_ANGLE):
    """
    Find the angles of a perturbable molecule that are linear at only one end
    state, and whether non-zero dihedrals or impropers pass through them, i.e.
    have three consecutive atoms that are the atoms of the angle.

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    max_angle : float
        The equilibrium angle, in degrees, at or above which an angle is
        linear.

    Returns
    -------

    angles : list of ((int, int, int), int, bool, bool)
        The atom indices of each angle, with the central atom in the middle,
        the end state at which it is linear, and whether a non-zero dihedral
        or improper passes through it at the other end state and at the end
        state where it's linear.
    """
    from math import degrees

    from sire.legacy.CAS import Symbol
    from sire.legacy.MM import AmberAngle

    info = mol.info()
    theta = Symbol("theta")

    def potentials(name):
        if not mol.has_property(name):
            return []
        return mol.property(name).potentials()

    def theta0s(state):
        result = {}
        for p in potentials(f"angle{state}"):
            i, j, k = (
                info.atom_idx(a).value() for a in (p.atom0(), p.atom1(), p.atom2())
            )
            try:
                theta0 = degrees(AmberAngle(p.function(), theta).theta0())
            except Exception:
                continue
            result[(min(i, k), j, max(i, k))] = theta0
        return result

    def spanned(state):
        result = set()
        for name in ("dihedral", "improper"):
            for p in potentials(f"{name}{state}"):
                if p.function().is_zero():
                    continue
                idx = [
                    info.atom_idx(a).value()
                    for a in (p.atom0(), p.atom1(), p.atom2(), p.atom3())
                ]
                result.add(frozenset(idx[:3]))
                result.add(frozenset(idx[1:]))
        return result

    theta0 = [theta0s(0), theta0s(1)]

    linear = []
    for key in sorted(theta0[0].keys() & theta0[1].keys()):
        linear0 = theta0[0][key] >= max_angle
        linear1 = theta0[1][key] >= max_angle
        if linear0 != linear1:
            linear.append((key, 0 if linear0 else 1))

    if not linear:
        return []

    terms = [spanned(0), spanned(1)]

    return [
        (key, state, frozenset(key) in terms[1 - state], frozenset(key) in terms[state])
        for key, state in linear
    ]


def torsion_window(states):
    """
    Return the fraction of each morph stage over which torsions should change,
    given the end states at which angles become linear.

    Parameters
    ----------

    states : set of int
        The end states at which angles become linear.

    Returns
    -------

    window : (float, float)
        The start and end of the window.
    """
    if states == {1}:
        return 0.0, 0.5
    if states == {0}:
        return 0.5, 1.0
    if states == {0, 1}:
        return 0.25, 0.75
    raise ValueError("'states' must be a non-empty subset of {0, 1}")
