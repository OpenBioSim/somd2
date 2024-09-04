######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2024
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

from sire.system import System as _System
from sire.legacy.System import System as _LegacySystem

import sire.legacy.MM as _SireMM
import sire.legacy.Mol as _SireMol

from somd2 import _logger
from ._somd1 import _is_dummy
from . import _lam_sym


def _appy_boresch(system):
    """
    Apply Boresch modifications to ghost atom bonded terms to avoid non-physical
    coupling between the ghost atoms and the physical region.

    Parameters
    ----------

    system : sire.system.System, sire.legacy.System.System
        The system containing the molecules to be perturbed.

    Returns
    -------

    system : sire.legacy.System.System
        The updated system.
    """

    _logger.debug(f"Applying Boresch modifications to ghost atom bonded terms:")

    # Check the system is a Sire system.
    if not isinstance(system, (_System, _LegacySystem)):
        raise TypeError(
            "'system' must of type 'sire.system.System' or 'sire.legacy.System.System'"
        )

    # Extract the legacy system.
    if isinstance(system, _LegacySystem):
        system = _System(system)

    # Clone the system.
    system = system.clone()

    # Search for perturbable molecules.
    try:
        pert_mols = system.molecules("property is_perturbable")
    except KeyError:
        raise KeyError("No perturbable molecules in the system")

    for mol in pert_mols:
        # Store the molecule info.
        info = mol.info()

        # Extract the connectivity property. This is currently the same
        # for both end states since we aren't dealing with ring breaking.
        # The logic is easy to adapt to separate connectivity objects.
        connectivity = mol.property("connectivity")

        # Find the indices of the dummy atoms at each end state.
        du0 = [
            _SireMol.AtomIdx(i)
            for i, x in enumerate(
                _is_dummy(mol, [_SireMol.AtomIdx(i) for i in range(mol.num_atoms())])
            )
            if x
        ]
        du1 = [
            _SireMol.AtomIdx(i)
            for i, x in enumerate(
                _is_dummy(
                    mol,
                    [_SireMol.AtomIdx(i) for i in range(mol.num_atoms())],
                    is_lambda1=True,
                )
            )
            if x
        ]

        # Work out the physical bridge atoms at lambda = 0. These are the atoms
        # that connect dummy atoms to the physical region.
        bridges0 = {}
        for du in du0:
            for c in connectivity.connections_to(du):
                if not _is_dummy(mol, [c])[0]:
                    if c not in bridges0:
                        bridges0[c] = [du]
                    else:
                        bridges0[c].append(du)
        # Work out the indices of the other physical atoms that are connected to
        # the bridge atoms.
        physical0 = {}
        for b in bridges0:
            physical0[b] = []
            for c in connectivity.connections_to(b):
                if not _is_dummy(mol, [c])[0]:
                    physical0[b].append(c)

        # Repeat the above for lambda = 1.
        bridges1 = {}
        for du in du1:
            for c in connectivity.connections_to(du):
                if not _is_dummy(mol, [c], is_lambda1=True)[0]:
                    if c not in bridges1:
                        bridges1[c] = [du]
                    else:
                        bridges1[c].append(du)
        physical1 = {}
        for b in bridges1:
            physical1[b] = []
            for c in connectivity.connections_to(b):
                if not _is_dummy(mol, [c], is_lambda1=True)[0]:
                    physical1[b].append(c)

        # Log the results for each end state.

        if len(bridges0) > 0:
            _logger.debug("Ghost atom bridges at lambda = 0")
            for i, b in enumerate(bridges0):
                _logger.debug(f"  Bridge {i}: {b}")
                _logger.debug(f"  dummies: {bridges0[b]}")
                _logger.debug(f"  physical: {physical0[b]}")
                _logger.debug(f"  type: {len(physical0[b])}")

        if len(bridges1) > 0:
            _logger.debug("Ghost atom bridges at lambda = 1")
            for i, b in enumerate(bridges1):
                _logger.debug(f"  Bridge {i}: {b}")
                _logger.debug(f"  dummies: {bridges1[b]}")
                _logger.debug(f"  physical: {physical1[b]}")
                _logger.debug(f"  type: {len(physical1[b])}")

        # Now process the bridges.

        for b in bridges0:

            # Determine the type of junction.
            junction = len(physical0[b])

            # Terminal junction.
            if junction == 1:
                mol = _terminal(mol, b, bridges0[b], physical0[b])

            # Dual junction.
            elif junction == 2:
                pass

            # Triple junction.
            elif junction == 3:
                pass

        # Update the molecule in the system.
        system.update(mol)

    # Return the updated system.
    return system


def _terminal(mol, bridge, dummies, physical, is_lambda1=False):
    """
    Apply Boresch modifications to a terminal junction.

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    bridge : sire.legacy.Mol.AtomIdx
        The physical bridge atom.

    dummies : List[sire.legacy.Mol.AtomIdx]
        The list of dummy atoms connected to the bridge atom.

    physical : List[sire.legacy.Mol.AtomIdx]
        The list of physical atoms connected to the bridge atom.

    is_lambda1 : bool, optional
        Whether the junction is at lambda = 1.

    Returns
    -------

    mol : sire.mol.Molecule
        The updated molecule.
    """

    _logger.debug(
        f"Applying Boresch modifications to terminal dummy junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Store the molecular connectivity.
    connectivity = mol.property("connectivity")

    # First, we need to work out the physical atoms two atoms away from the
    # bridge atom.
    physical2 = []
    # Loop over the physical atoms connected to the bridge atom.
    for p in physical:
        # Loop over the atoms connected to the physical atom.
        for c in connectivity.connections_to(p):
            # If the atom is not a dummy atom or the bridge atom itself, we have
            # found a physical atom two atoms away from the bridge atom.
            if not _is_dummy(mol, [c], is_lambda1)[0] and c != bridge:
                if c not in physical2:
                    physical2.append(c)

    # Get the end state dihedral functions.
    prop = "dihedral0" if not is_lambda1 else "dihedral1"
    dihedrals = mol.property(prop)

    # Initialise a container to store the updated dihedrals.
    new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

    # For the first physical2 atom, remove all dihedrals that reach into
    # the dummy group(s).
    idx = physical2[0]
    for d in dihedrals.potentials():
        idx0 = info.atom_idx(d.atom0())
        idx1 = info.atom_idx(d.atom1())
        idx2 = info.atom_idx(d.atom2())
        idx3 = info.atom_idx(d.atom3())
        if (idx0 == idx and idx3 in dummies) or (idx3 == idx and idx0 in dummies):
            _logger.debug(f"  Removing dihedral: {d}")
        else:
            new_dihedrals.set(idx0, idx1, idx2, idx3, d.function())

    # Set the updated dihedrals.
    mol = mol.edit().set_property(prop, new_dihedrals).molecule().commit()

    # Return the updated molecule.
    return mol
