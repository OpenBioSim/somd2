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

from . import _is_dummy
from . import _lam_sym


def _boresch(system):
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
                _logger.debug(f"  Bridge {i}: {b.value()}")
                _logger.debug(
                    f"  dummies: [{','.join([str(x.value()) for x in bridges0[b]])}]"
                )
                _logger.debug(
                    f"  physical: [{','.join([str(x.value()) for x in physical0[b]])}]"
                )
                _logger.debug(f"  type: {len(physical0[b])}")

        if len(bridges1) > 0:
            _logger.debug("Ghost atom bridges at lambda = 1")
            for i, b in enumerate(bridges1):
                _logger.debug(f"  Bridge {i}: {b.value()}")
                _logger.debug(
                    f"  dummies: [{','.join([str(x.value()) for x in bridges1[b]])}]"
                )
                _logger.debug(
                    f"  physical: [{','.join([str(x.value()) for x in physical1[b]])}]"
                )
                _logger.debug(f"  type: {len(physical1[b])}")

        # Now process the bridges.

        # First lambda = 0.
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
                mol = _triple(mol, b, bridges0[b], physical0[b])

        # Now lambda = 1.
        for b in bridges1:
            junction = len(physical1[b])

            if junction == 1:
                mol = _terminal(mol, b, bridges1[b], physical1[b], is_lambda1=True)

            elif junction == 2:
                pass

            elif junction == 3:
                mol = _triple(mol, b, bridges1[b], physical1[b], is_lambda1=True)

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
    for p in dihedrals.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())
        idx3 = info.atom_idx(p.atom3())
        if (idx0 == idx and idx3 in dummies) or (idx3 == idx and idx0 in dummies):
            _logger.debug(
                f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
            )
        else:
            new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

    # Set the updated dihedrals.
    mol = mol.edit().set_property(prop, new_dihedrals).molecule().commit()

    # Return the updated molecule.
    return mol


def _triple(mol, bridge, dummies, physical, is_lambda1=False):
    """
    Apply Boresch modifications to a triple junction.

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
        f"Applying Boresch modifications to triple dummy junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Store the molecular connectivity.
    connectivity = mol.property("connectivity")

    # Property suffix based on the end state.
    suffix = "0" if not is_lambda1 else "1"

    # Store the element of the bridge atom.
    element = mol.atom(bridge).property("element" + suffix)

    # Planar junction.
    if element == _SireMol.Element("C"):
        _logger.debug("  Planar junction.")

        # First remove all bonded terms between one of the physical atoms
        # and the dummy group.

        # Store the index of the first physical atom.
        idx = physical[0]

        # Get the end state bond functions.
        angles = mol.property("angle" + suffix)
        dihedrals = mol.property("dihedral" + suffix)
        impropers = mol.property("improper" + suffix)

        # Initialise a container to store the updated bonded terms.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())
        new_impropers = _SireMM.FourAtomFunctions(mol.info())

        # Angles.
        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idxs = [idx0, idx1, idx2]

            if idx in idxs and any([x in dummies for x in idxs]):
                _logger.debug(
                    f"  Removing angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], {p.function()}"
                )

            else:
                new_angles.set(idx0, idx1, idx2, p.function())

        # Dihedrals.
        for p in dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            idxs = [idx0, idx1, idx2, idx3]

            if idx in idxs and any([x in dummies for x in idxs]):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )

            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Impropers.
        for p in impropers.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            idxs = [idx0, idx1, idx2, idx3]

            if idx in idxs and any([x in dummies for x in idxs]):
                _logger.debug(
                    f"  Removing improper: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )

            else:
                new_impropers.set(idx0, idx1, idx2, idx3, p.function())

        # Next we modify the angle terms between the remaining physical and
        # dummy atoms so that the equilibrium angle is 90 degrees.
        new_new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        for angle in new_angles.potentials():
            idx0 = info.atom_idx(angle.atom0())
            idx1 = info.atom_idx(angle.atom1())
            idx2 = info.atom_idx(angle.atom2())

            if (
                idx0 in dummies
                and idx1 == idx
                and idx2 in physical[1:]
                or idx0 in physical[1:]
                and idx1 == idx
                and idx2 in dummies
            ):
                from math import pi
                from sire.legacy.CAS import Symbol

                theta0 = pi / 2.0

                # Create the new angle function.
                amber_angle = _SireMM.AmberAngle(100, theta0)

                # Generate the new angle expression.
                expression = amber_angle.expression.to_expression(Symbol("theta"))

                # Set the equilibrium angle to 90 degrees.
                new_new_angles.set(idx0, idx1, idx2, expression)

                _logger.debug(
                    f"  Modifying angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], {angle.function()}"
                )

            else:
                new_new_angles.set(idx0, idx1, idx2, angle.function())

        # Update the molecule.
        mol = (
            mol.edit()
            .set_property("angle" + suffix, new_new_angles)
            .molecule()
            .commit()
        )
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit
        )

    # Non-planar junction.
    elif element == _SireMol.Element("N"):
        _logger.debug("  Non-planar junction.")

        # First, modify the force constants of the angle terms between the dummy
        # atoms and the physical system to be very low.

        # Get the end state angle functions.
        angles = mol.property("angle" + suffix)

        # Initialise a container to store the updated angle functions.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())

        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if (
                idx0 in dummies
                and idx2 in physical
                or idx2 in dummies
                and idx0 in physical
            ):
                from sire.legacy.CAS import Symbol

                theta = Symbol("theta")

                # Cast the angle to an Amber angle to get the expression.
                amber_angle = _SireMM.AmberAngle(p.function(), theta)

                # Create a new Amber angle with a modified force constant.
                amber_angle = _SireMM.AmberAngle(5, amber_angle.theta0())

                # Generate the new angle expression.
                expression = amber_angle.to_expression(theta)

                # Set the force constant to a very low value.
                new_angles.set(idx0, idx1, idx2, expression)

                _logger.debug(
                    f"  Modifying angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], {p.function()}"
                )

            else:
                new_angles.set(idx0, idx1, idx2, p.function())

        # Next, remove all dihedral starting from the dummy atoms and ending in
        # the physical system.

        # Get the end state dihedral functions.
        dihedrals = mol.property("dihedral" + suffix)
        improper = mol.property("improper" + suffix)

        # Initialise containers to store the updated dihedral and improper functions.
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())
        new_impropers = _SireMM.FourAtomFunctions(mol.info())

        for p in dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            idxs = [idx0, idx1, idx2, idx3]

            # If there is one dummy atom, then this dihedral must begin or terminate
            # at the dummy atom.
            num_dummies = len([x for x in idxs if x in dummies])
            if num_dummies == 1:
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )

            elif num_dummies > 1:
                _logger.debug("Unhandled case: multiple dummy atoms in dihedral")
                new_impropers.set(idx0, idx1, idx2, idx3, p.function())

            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        for p in improper.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            idxs = [idx0, idx1, idx2, idx3]

            if len([x for x in idxs if x in dummies]) > 0:
                _logger.debug(
                    f"  Removing improper: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )

            elif num_dummies > 1:
                _logger.debug("Unhandled case: multiple dummy atoms in improper")
                new_impropers.set(idx0, idx1, idx2, idx3, p.function())

            else:
                new_impropers.set(idx0, idx1, idx2, idx3, p.function())

        # Update the molecule.
        mol = mol.edit().set_property("angle" + suffix, new_angles).molecule().commit()
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit()
        )
        mol = (
            mol.edit()
            .set_property("improper" + suffix, new_impropers)
            .molecule()
            .commit()
        )

    # Return the updated molecule.
    return mol
