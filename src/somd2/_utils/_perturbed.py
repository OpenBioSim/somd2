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

__all__ = ["perturbed_atoms", "perturbed_selection"]


def perturbed_atoms(mol):
    """
    Return the indices of the atoms of a perturbable molecule that change
    between the end states: atoms whose element changes, including ghost
    atoms, and heavy atoms whose Lennard-Jones parameters change. Hydrogens
    whose Lennard-Jones parameters change are left out, since this is often
    only a change in atom type caused by changes elsewhere in the molecule. If
    no atoms change, then all atoms are returned.

    Parameters
    ----------

    mol: sire.mol.Molecule
        The perturbable molecule.

    Returns
    -------

    indices: list of int
        The indices of the perturbed atoms within the molecule.
    """

    def changes(atom):
        element = atom.property("element0").symbol()
        if element != atom.property("element1").symbol():
            return True
        return element != "H" and atom.property("LJ0") != atom.property("LJ1")

    indices = [i for i, atom in enumerate(mol.atoms()) if changes(atom)]
    return indices or list(range(mol.num_atoms()))


def _neighbours(mol, indices, bonds):
    """
    Return the indices of the atoms within a number of bonds of the given
    atoms, in either end state, including the atoms themselves.
    """
    from sire.legacy.Mol import AtomIdx as _AtomIdx

    names = ["connectivity0", "connectivity1"]
    if not all(mol.has_property(name) for name in names):
        names = ["connectivity"]
    connectivities = [mol.property(name) for name in names]

    region = set(indices)
    shell = set(indices)
    for _ in range(bonds):
        shell = {
            n.value()
            for i in shell
            for c in connectivities
            for n in c.connections_to(_AtomIdx(i))
        } - region
        region |= shell

    return sorted(region)


def perturbed_selection(system, bonds=0):
    """
    Return a selection string for the perturbed atoms of the perturbable
    molecules in the system, and the atoms within a number of bonds of them.
    Perturbable waters and single-atom ions, e.g. co-alchemical ions, are left
    out unless they are the only perturbable molecules. Molecules are selected
    by index, rather than number, so that the selection also works for the
    system reloaded from file.

    Parameters
    ----------

    system: sire.system.System
        The system containing the perturbable molecules.

    bonds: int
        The number of bonds from the perturbed atoms to extend the selection.

    Returns
    -------

    selection: str
        The selection string.
    """
    perturbable = system.molecules("property is_perturbable")
    try:
        solutes = system.molecules("property is_perturbable and not water")
    except KeyError:
        solutes = []
    solutes = [mol for mol in solutes if mol.num_atoms() > 1] or perturbable

    molecules = system.molecules()
    parts = []
    for mol in solutes:
        indices = _neighbours(mol, perturbed_atoms(mol), bonds)
        atoms = ",".join(str(i) for i in indices)
        parts.append(f"(molidx {molecules.find(mol)} and atomidx {atoms})")
    return " or ".join(parts)
