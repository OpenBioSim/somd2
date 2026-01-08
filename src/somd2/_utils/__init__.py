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

import platform as _platform

if _platform.system() == "Windows":
    _lam_sym = "lambda"
else:
    _lam_sym = "λ"

del _platform


def _has_ghost(mol, idxs, is_lambda1=False):
    """
    Internal function to check whether any atom is a ghost.

    Parameters
    ----------

    mol : sire.legacy.Mol.Molecule
        The molecule.

    idxs : [sire.legacy.Mol.AtomIdx]
        A list of atom indices.

    is_lambda1 : bool
        Whether to check the lambda = 1 state.

    Returns
    -------

    has_ghost : bool
        Whether a ghost atom is present.
    """

    import sire.legacy.Mol as _SireMol

    # We need to check by ambertype too since this molecule may have been
    # created via sire.morph.create_from_pertfile, in which case the element
    # property will have been set to the end state with the largest mass, i.e.
    # may no longer by a ghost.
    if is_lambda1:
        element_prop = "element1"
        ambertype_prop = "ambertype1"
    else:
        element_prop = "element0"
        ambertype_prop = "ambertype0"

    element_ghost = _SireMol.Element(0)
    ambertype_ghost = "du"

    # Check whether an of the atoms is a ghost.
    for idx in idxs:
        if (
            mol.atom(idx).property(element_prop) == element_ghost
            or mol.atom(idx).property(ambertype_prop) == ambertype_ghost
        ):
            return True

    return False


def _is_ghost(mol, idxs, is_lambda1=False):
    """
    Internal function to return whether each atom is a ghost.

    Parameters
    ----------

    mol : sire.legacy.Mol.Molecule
        The molecule.

    idxs : [sire.legacy.Mol.AtomIdx]
        A list of atom indices.

    is_lambda1 : bool
        Whether to check the lambda = 1 state.

    Returns
    -------

    is_ghost : [bool]
        Whether each atom is a ghost.
    """

    import sire.legacy.Mol as _SireMol

    # We need to check by ambertype too since this molecule may have been
    # created via sire.morph.create_from_pertfile, in which case the element
    # property will have been set to the end state with the largest mass, i.e.
    # may no longer by a ghost.
    if is_lambda1:
        element_prop = "element1"
        ambertype_prop = "ambertype1"
    else:
        element_prop = "element0"
        ambertype_prop = "ambertype0"

    if is_lambda1:
        element_prop = "element1"
        ambertype_prop = "ambertype1"
    else:
        element_prop = "element0"
        ambertype_prop = "ambertype0"

    element_ghost = _SireMol.Element(0)
    ambertype_ghost = "du"

    # Initialise a list to store the state of each atom.
    is_ghost = []

    # Check whether each of the atoms is a ghost.
    for idx in idxs:
        is_ghost.append(
            mol.atom(idx).property(element_prop) == element_ghost
            or mol.atom(idx).property(ambertype_prop) == ambertype_ghost
        )

    return is_ghost
