######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2025
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

__all__ = ["boresch"]

from sire.system import System as _System
from sire.legacy.System import System as _LegacySystem

import sire.legacy.MM as _SireMM
import sire.legacy.Mol as _SireMol
import sire.morph as _morph

from somd2 import _logger

from . import _is_ghost
from . import _lam_sym


def boresch(system, k_hard=100, k_soft=5, optimise_angles=True, num_optimise=10):
    """
    Apply Boresch modifications to ghost atom bonded terms to avoid non-physical
    coupling between the ghost atoms and the physical region.

    Parameters
    ----------

    system : sire.system.System, sire.legacy.System.System
        The system containing the molecules to be perturbed.

    k_hard : float, optional
        The force constant to use to when setting angle terms involving ghost
        atoms to 90 degrees to avoid flapping. (In kcal/mol/rad^2)

    k_soft : float, optional
        The force constant to use when setting angle terms involving ghost atoms
        for non-planar triple junctions. (In kcal/mol/rad^2)

    optimise_angles : bool, optional
        Whether to optimise the equilibrium value of the angle terms involving
        ghost atoms for non-planar triple junctions.

    num_optimise : int, optional
        The number of repeats to average over when optimising the equilibrium
        value of the angle terms involving ghost atoms for non-planar triple
        junctions.

    Returns
    -------

    system : sire.system.System
        The updated system.

    Notes
    -----

    For technical details, please refer to the original publication:
        https://pubs.acs.org/doi/10.1021/acs.jctc.0c01328
    """

    _logger.info(f"Applying Boresch modifications to ghost atom bonded terms")

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

        # Generate the end state connectivity objects.
        connectivity0 = _create_connectivity(_morph.link_to_reference(mol))
        connectivity1 = _create_connectivity(_morph.link_to_perturbed(mol))

        # Find the indices of the ghost atoms at each end state.
        ghosts0 = [
            _SireMol.AtomIdx(i)
            for i, x in enumerate(
                _is_ghost(mol, [_SireMol.AtomIdx(i) for i in range(mol.num_atoms())])
            )
            if x
        ]
        ghosts1 = [
            _SireMol.AtomIdx(i)
            for i, x in enumerate(
                _is_ghost(
                    mol,
                    [_SireMol.AtomIdx(i) for i in range(mol.num_atoms())],
                    is_lambda1=True,
                )
            )
            if x
        ]

        # Work out the physical bridge atoms at lambda = 0. These are the atoms
        # that connect ghost atoms to the physical region.
        bridges0 = {}
        for ghost in ghosts0:
            for c in connectivity0.connections_to(ghost):
                if not _is_ghost(mol, [c])[0]:
                    if c not in bridges0:
                        bridges0[c] = [ghost]
                    else:
                        bridges0[c].append(ghost)
        # Work out the indices of the other physical atoms that are connected to
        # the bridge atoms, sorted by the atom index.
        physical0 = {}
        for b in bridges0:
            physical0[b] = []
            for c in connectivity0.connections_to(b):
                if not _is_ghost(mol, [c])[0]:
                    physical0[b].append(c)
        for b in physical0:
            physical0[b].sort(key=lambda x: x.value())

        # Repeat the above for lambda = 1.
        bridges1 = {}
        for ghost in ghosts1:
            for c in connectivity1.connections_to(ghost):
                if not _is_ghost(mol, [c], is_lambda1=True)[0]:
                    if c not in bridges1:
                        bridges1[c] = [ghost]
                    else:
                        bridges1[c].append(ghost)
        physical1 = {}
        for b in bridges1:
            physical1[b] = []
            for c in connectivity1.connections_to(b):
                if not _is_ghost(mol, [c], is_lambda1=True)[0]:
                    physical1[b].append(c)
        for b in physical1:
            physical1[b].sort(key=lambda x: x.value())

        # Log the results for each end state.

        if len(bridges0) > 0:
            _logger.debug("Ghost atom bridges at lambda = 0")
            for i, b in enumerate(bridges0):
                _logger.debug(f"  Bridge {i}: {b.value()}")
                _logger.debug(
                    f"  ghosts: [{','.join([str(x.value()) for x in bridges0[b]])}]"
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
                    f"  ghosts: [{','.join([str(x.value()) for x in bridges1[b]])}]"
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
                mol = _dual(mol, b, bridges0[b], physical0[b], k_hard=k_hard)

            # Triple junction.
            elif junction == 3:
                mol = _triple(
                    mol,
                    b,
                    bridges0[b],
                    physical0[b],
                    k_hard=k_hard,
                    k_soft=k_soft,
                    optimise_angles=optimise_angles,
                    num_optimise=num_optimise,
                )

            # Higher order junction.
            else:
                mol = _higher(
                    mol,
                    b,
                    bridges0[b],
                    physical0[b],
                    k_hard=k_hard,
                    k_soft=k_soft,
                    optimise_angles=optimise_angles,
                    num_optimise=num_optimise,
                )

        # Now lambda = 1.
        for b in bridges1:
            junction = len(physical1[b])

            if junction == 1:
                mol = _terminal(mol, b, bridges1[b], physical1[b], is_lambda1=True)

            elif junction == 2:
                mol = _dual(
                    mol, b, bridges1[b], physical1[b], k_hard=k_hard, is_lambda1=True
                )

            elif junction == 3:
                mol = _triple(
                    mol,
                    b,
                    bridges1[b],
                    physical1[b],
                    k_hard=k_hard,
                    k_soft=k_soft,
                    optimise_angles=optimise_angles,
                    num_optimise=num_optimise,
                    is_lambda1=True,
                )

            # Higher order junction.
            else:
                mol = _higher(
                    mol,
                    b,
                    bridges1[b],
                    physical1[b],
                    k_hard=k_hard,
                    k_soft=k_soft,
                    optimise_angles=optimise_angles,
                    num_optimise=num_optimise,
                    is_lambda1=True,
                )

        # Update the molecule in the system.
        system.update(mol)

    # Return the updated system.
    return system


def _terminal(mol, bridge, ghosts, physical, is_lambda1=False):
    r"""
    Apply Boresch modifications to a terminal junction.

    An example terminal junction with three ghost branches. Here X is the
    physical bridge atom.

               DR1
              /
             /
        R---X---DR2
             \
              \
               DR3

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    bridge : sire.legacy.Mol.AtomIdx
        The physical bridge atom.

    ghosts : List[sire.legacy.Mol.AtomIdx]
        The list of ghost atoms connected to the bridge atom.

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
        f"Applying Boresch modifications to terminal ghost junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Get the end state connectivity property.
    if is_lambda1:
        connectivity = _create_connectivity(_morph.link_to_perturbed(mol))
    else:
        connectivity = _create_connectivity(_morph.link_to_reference(mol))

    # First, we need to work out the physical atoms two atoms away from the
    # bridge atom.
    physical2 = []
    # Loop over the physical atoms connected to the bridge atom.
    for p in physical:
        # Loop over the atoms connected to the physical atom.
        for c in connectivity.connections_to(p):
            # If the atom is not a ghost atom or the bridge atom itself, we have
            # found a physical atom two atoms away from the bridge atom.
            if not _is_ghost(mol, [c], is_lambda1)[0] and c != bridge:
                if c not in physical2:
                    physical2.append(c)

    # Sort based on the atom indices.
    physical2.sort(key=lambda x: x.value())

    # Get the end state dihedral functions.
    prop = "dihedral0" if not is_lambda1 else "dihedral1"
    dihedrals = mol.property(prop)

    # Initialise a container to store the updated dihedrals.
    new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

    # Remove all dihedral terms for all but one of the physical atoms two atoms
    # from the physical bridge atom.
    physical2.pop(0)
    for p in dihedrals.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())
        idx3 = info.atom_idx(p.atom3())
        if (idx0 in physical2 and idx3 in ghosts) or (
            idx3 in physical2 and idx0 in ghosts
        ):
            _logger.debug(
                f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
            )
        else:
            new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

    # Set the updated dihedrals.
    mol = mol.edit().set_property(prop, new_dihedrals).molecule().commit()

    # Return the updated molecule.
    return mol


def _dual(mol, bridge, ghosts, physical, k_hard=100, is_lambda1=False):
    r"""
    Apply Boresch modifications to a dual junction.

    An example dual junction with two ghost branches. Here X is the physical
    bridge atom.

         R1     DR1
           \   /
            \ /
             X
            / \
           /   \
         R2     DR2

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    bridge : sire.legacy.Mol.AtomIdx
        The physical bridge atom.

    ghosts : List[sire.legacy.Mol.AtomIdx]
        The list of ghost atoms connected to the bridge atom.

    physical : List[sire.legacy.Mol.AtomIdx]
        The list of physical atoms connected to the bridge atom.

    k_hard : float, optional
        The force constant to use when setting angle terms involving ghost
        atoms to 90 degrees to avoid flapping. (In kcal/mol/rad^2)

    is_lambda1 : bool, optional
        Whether the junction is at lambda = 1.

    Returns
    -------

    mol : sire.mol.Molecule
        The updated molecule.
    """

    _logger.debug(
        f"Applying Boresch modifications to dual ghost junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Property suffix based on the end state.
    suffix = "0" if not is_lambda1 else "1"

    # Get the end state connectivity property.
    try:
        connectivity = mol.property("connectivity" + suffix)
    except:
        connectivity = mol.property("connectivity")

    # Single branch.
    if len(ghosts) == 1:
        _logger.debug("  Single branch:")

        # First remove all dihedrals starting from the ghost atom and ending in
        # physical system.

        # Get the end state bond functions.
        angles = mol.property("angle" + suffix)
        dihedrals = mol.property("dihedral" + suffix)

        # Initialise a container to store the updated bonded terms.
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

        # Dihedrals.
        for p in dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())

            # Dihedral terminates at the ghost bridge.
            if (
                not _is_ghost(mol, [idx0], is_lambda1)[0]
                and idx3 in ghosts
                or not _is_ghost(mol, [idx3], is_lambda1)[0]
                and idx0 in ghosts
            ):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            # Dihedral terminates at the second physical atom.
            elif (_is_ghost(mol, [idx0], is_lambda1)[0] and idx3 == physical[1]) or (
                _is_ghost(mol, [idx3], is_lambda1)[0] and idx0 == physical[1]
            ):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Next we modify the angle terms between the physical and
        # ghost atom so that the equilibrium angle is 90 degrees.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if (
                idx0 in ghosts
                and idx2 in physical
                or idx0 in physical
                and idx2 in ghosts
            ):
                from math import pi
                from sire.legacy.CAS import Symbol

                theta0 = pi / 2.0

                # Create the new angle function.
                amber_angle = _SireMM.AmberAngle(k_hard, theta0)

                # Generate the new angle expression.
                expression = amber_angle.to_expression(Symbol("theta"))

                # Set the equilibrium angle to 90 degrees.
                new_angles.set(idx0, idx1, idx2, expression)

                _logger.debug(
                    f"  Stiffening angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], "
                    f"{p.function()} --> {expression}"
                )

            else:
                new_angles.set(idx0, idx1, idx2, p.function())

        # Update the molecule.
        mol = mol.edit().set_property("angle" + suffix, new_angles).molecule().commit()
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit()
        )

    # Dual branch.
    else:
        _logger.debug("  Dual branch:")

        # First, delete all bonded terms between atoms in two ghost branches.

        # Get the end state bond functions.
        angles = mol.property("angle" + suffix)
        dihedrals = mol.property("dihedral" + suffix)

        # Initialise containers to store the updated bonded terms.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

        # Angles.
        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if idx0 in ghosts and idx2 in ghosts:
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

            # Work out the number of ghosts in the dihedral.
            num_ghosts = len([idx for idx in [idx0, idx1, idx2, idx3] if idx in ghosts])

            # If there is more than one ghost, then this dihedral must bridge the
            # ghost branches.
            if num_ghosts > 1:
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Set the updated bonded terms.
        mol = mol.edit().set_property("angle" + suffix, new_angles).molecule().commit()
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit()
        )

        # Now treat the ghost branches individually.
        for d in ghosts:
            mol = _dual(
                mol, bridge, [d], physical, k_hard=k_hard, is_lambda1=is_lambda1
            )

    # Return the updated molecule.
    return mol


def _triple(
    mol,
    bridge,
    ghosts,
    physical,
    k_hard=100,
    k_soft=5,
    optimise_angles=True,
    num_optimise=10,
    is_lambda1=False,
):
    r"""
    Apply Boresch modifications to a triple junction.

    An example triple junction. Here X is the physical bridge atom.

         R1
           \
            \
        R2---X---DR
            /
           /
         R3

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    bridge : sire.legacy.Mol.AtomIdx
        The physical bridge atom.

    ghosts : List[sire.legacy.Mol.AtomIdx]
        The list of ghost atoms connected to the bridge atom.

    physical : List[sire.legacy.Mol.AtomIdx]
        The list of physical atoms connected to the bridge atom.

    k_hard : float, optional
        The force constant to use when setting angle terms involving ghost
        atoms to 90 degrees to avoid flapping. (In kcal/mol/rad^2)

    k_soft : float, optional
        The force constant to use when setting angle terms involving ghost
        atoms for non-planar triple junctions. (In kcal/mol/rad^2)

    optimise_angles : bool, optional
        Whether to optimise the equilibrium value of the angle terms involving
        ghost atoms for non-planar triple junctions.

    num_optimise : int, optional
        The number of repeats to use when optimising the angle terms involving
        ghost atoms for non-planar triple junctions.

    is_lambda1 : bool, optional
        Whether the junction is at lambda = 1.

    Returns
    -------

    mol : sire.mol.Molecule
        The updated molecule.
    """

    _logger.debug(
        f"Applying Boresch modifications to triple ghost junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Property suffix based on the end state.
    suffix = "0" if not is_lambda1 else "1"

    # Get the end state connectivity property.
    try:
        connectivity = mol.property("connectivity" + suffix)
    except:
        connectivity = mol.property("connectivity")

    # Store the element of the bridge atom.
    element = mol.atom(bridge).property("element" + suffix)

    # Planar junction.
    if element == _SireMol.Element("C"):
        _logger.debug("  Planar junction.")

        # First remove all bonded terms between one of the physical atoms
        # and the ghost group.

        # Store the index of the first physical atom.
        idx = physical[0]

        # Get the end state bond functions.
        angles = mol.property("angle" + suffix)
        dihedrals = mol.property("dihedral" + suffix)

        # Initialise a container to store the updated bonded terms.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

        # Angles.
        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idxs = [idx0, idx1, idx2]

            if idx in idxs and any([x in ghosts for x in idxs]):
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

            if idx in idxs and any([x in ghosts for x in idxs]):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )

            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Next we modify the angle terms between the remaining physical and
        # ghost atoms so that the equilibrium angle is 90 degrees.
        new_new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        for p in new_angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if (
                idx0 in ghosts
                and idx2 in physical[1:]
                or idx0 in physical[1:]
                and idx2 in ghosts
            ):
                from math import pi
                from sire.legacy.CAS import Symbol

                theta0 = pi / 2.0

                # Create the new angle function.
                amber_angle = _SireMM.AmberAngle(k_hard, theta0)

                # Generate the new angle expression.
                expression = amber_angle.to_expression(Symbol("theta"))

                # Set the equilibrium angle to 90 degrees.
                new_new_angles.set(idx0, idx1, idx2, expression)

                _logger.debug(
                    f"  Stiffening angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], "
                    f"{p.function()} --> {expression}"
                )

            else:
                new_new_angles.set(idx0, idx1, idx2, p.function())

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
            .commit()
        )

    # Non-planar junction.
    elif element == _SireMol.Element("N"):
        _logger.debug("  Non-planar junction.")

        # First, modify the force constants of the angle terms between the ghost
        # atoms and the physical system to be very low.

        # Get the end state angle functions.
        angles = mol.property("angle" + suffix)

        # Initialise a container to store the updated angle functions.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())

        # Indices for the softened angle terms.
        angle_idxs = []

        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if (
                idx0 in ghosts
                and idx2 in physical
                or idx2 in ghosts
                and idx0 in physical
            ):
                from sire.legacy.CAS import Symbol

                theta = Symbol("theta")

                # Cast the angle to an Amber angle to get the expression.
                amber_angle = _SireMM.AmberAngle(p.function(), theta)

                # Create a new Amber angle with a modified force constant.

                # We'll optimise the equilibrium angle for the softened angle term.
                if optimise_angles:
                    amber_angle = _SireMM.AmberAngle(0.05, amber_angle.theta0())
                    angle_idxs.append((idx0, idx1, idx2))
                # Use the existing equilibrium angle.
                else:
                    amber_angle = _SireMM.AmberAngle(k_soft, amber_angle.theta0())

                # Generate the new angle expression.
                expression = amber_angle.to_expression(theta)

                # Set the force constant to a very low value.
                new_angles.set(idx0, idx1, idx2, expression)

                _logger.debug(
                    f"  Softening angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], "
                    f"{p.function()} --> {expression}"
                )

            else:
                new_angles.set(idx0, idx1, idx2, p.function())

        # Next, remove all dihedral starting from the ghost atoms and ending in
        # the physical system. Also, only preserve dihedrals terminating at one
        # of the physical atoms.

        # Get the end state dihedral functions.
        dihedrals = mol.property("dihedral" + suffix)

        # Initialise containers to store the updated dihedral functions.
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

        for p in dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            idxs = [idx0, idx1, idx2, idx3]

            # If there is one ghost atom, then this dihedral must begin or terminate
            # at the ghost atom.
            num_ghosts = len([x for x in idxs if x in ghosts])
            if num_ghosts == 1:
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            # Remove the dihedral if includes a ghost and doesn't terminate at the first
            # physical atom.
            elif (_is_ghost(mol, [idx0], is_lambda1)[0] and idx3 in physical[1:]) or (
                _is_ghost(mol, [idx3], is_lambda1)[0] and idx0 in physical[1:]
            ):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Update the molecule.
        mol = mol.edit().set_property("angle" + suffix, new_angles).molecule().commit()
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit()
        )

        # Optimise the equilibrium value of theta0 for the softened angle terms.
        if optimise_angles:
            _logger.debug("  Optimising equilibrium values for softened angles.")

            import sire.morph as _morph
            from sire.units import radian as _radian

            # Initialise the equilibrium angle values.
            theta0s = {}
            for idx in angle_idxs:
                theta0s[idx] = []

            # Perform multiple minimisations to get an average for the theta0 values.
            for _ in range(num_optimise):
                # Minimise the molecule.
                min_mol = _morph.link_to_reference(mol)
                minimiser = min_mol.minimisation(
                    lambda_value=1.0 if is_lambda1 else 0.0,
                    constraint="none",
                    platform="cpu",
                )
                minimiser.run()

                # Commit the changes.
                min_mol = minimiser.commit()

                # Get the equilibrium angle values.
                for idx in angle_idxs:
                    try:
                        theta0s[idx].append(min_mol.angles(*idx).sizes()[0].to(_radian))
                    except:
                        raise ValueError(f"Could not find optimised angle term: {idx}")

            # Compute the mean and standard error.
            import numpy as _np

            theta0_means = {}
            theta0_stds = {}
            for idx in theta0s:
                theta0_means[idx] = _np.mean(theta0s[idx])
                theta0_stds[idx] = _np.std(theta0s[idx]) / _np.sqrt(len(theta0s[idx]))

            # Get the existing angles.
            angles = mol.property("angle" + suffix)

            # Initialise a container to store the updated angle functions.
            new_angles = _SireMM.ThreeAtomFunctions(mol.info())

            # Update the angle potentials.
            for p in angles.potentials():
                idx0 = info.atom_idx(p.atom0())
                idx1 = info.atom_idx(p.atom1())
                idx2 = info.atom_idx(p.atom2())
                idx = (idx0, idx1, idx2)

                # This is the softened angle term.
                if idx in angle_idxs:
                    # Get the optimised equilibrium angle.
                    theta0 = theta0_means[idx]
                    std = theta0_stds[idx]

                    # Create the new angle function.
                    amber_angle = _SireMM.AmberAngle(k_soft, theta0)

                    # Generate the new angle expression.
                    expression = amber_angle.to_expression(Symbol("theta"))

                    # Set the equilibrium angle to 90 degrees.
                    new_angles.set(idx0, idx1, idx2, expression)

                    _logger.debug(
                        f"  Optimising angle: [{idx0.value()}-{idx1.value()}-{idx2.value()}], "
                        f"{p.function()} --> {expression} (std err: {std:.3f} radian)"
                    )

                else:
                    new_angles.set(idx0, idx1, idx2, p.function())

            # Update the molecule.
            mol = (
                mol.edit()
                .set_property("angle" + suffix, new_angles)
                .molecule()
                .commit()
            )

    # Return the updated molecule.
    return mol


def _higher(
    mol,
    bridge,
    ghosts,
    physical,
    k_hard=100,
    k_soft=5,
    optimise_angles=True,
    num_optimise=10,
    is_lambda1=False,
):
    r"""
    Apply Boresch modifications to higher order junctions.

    Parameters
    ----------

    mol : sire.mol.Molecule
        The perturbable molecule.

    bridge : sire.legacy.Mol.AtomIdx
        The physical bridge atom.

    ghosts : List[sire.legacy.Mol.AtomIdx]
        The list of ghost atoms connected to the bridge atom.

    physical : List[sire.legacy.Mol.AtomIdx]
        The list of physical atoms connected to the bridge atom.

    k_hard : float, optional
        The force constant to use when setting angle terms involving ghost
        atoms to 90 degrees to avoid flapping. (In kcal/mol/rad^2)

    k_soft : float, optional
        The force constant to use when setting angle terms involving ghost
        atoms for non-planar triple junctions. (In kcal/mol/rad^2)

    optimise_angles : bool, optional
        Whether to optimise the equilibrium value of the angle terms involving
        ghost atoms for non-planar triple junctions.

    num_optimise : int, optional
        The number of repeats to use when optimising the angle terms involving
        ghost atoms for non-planar triple junctions.

    is_lambda1 : bool, optional
        Whether the junction is at lambda = 1.

    Returns
    -------

    mol : sire.mol.Molecule
        The updated molecule.
    """

    _logger.debug(
        f"Applying Boresch modifications to higher order junction at "
        f"{_lam_sym} = {int(is_lambda1)}:"
    )

    # Store the molecular info.
    info = mol.info()

    # Property suffix based on the end state.
    suffix = "0" if not is_lambda1 else "1"

    # Get the end state connectivity property.
    try:
        connectivity = mol.property("connectivity" + suffix)
    except:
        connectivity = mol.property("connectivity")

    # Now remove all bonded interactions between the ghost atoms and one of the
    # physical atoms connected to the bridge atom, hence reducing the problem to
    # that of a triple junction.
    while len(physical) > 3:
        # Pop the first physical atom index from the list.
        idx = physical.pop(0)

        # Get the end state bond functions.
        angles = mol.property("angle" + suffix)
        dihedrals = mol.property("dihedral" + suffix)

        # Initialise containers to store the updated bonded terms.
        new_angles = _SireMM.ThreeAtomFunctions(mol.info())
        new_dihedrals = _SireMM.FourAtomFunctions(mol.info())

        # Angles.
        for p in angles.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())

            if idx == idx0 and idx2 in ghosts or idx == idx2 and idx0 in ghosts:
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

            if idx in idxs and any([x in ghosts for x in idxs]):
                _logger.debug(
                    f"  Removing dihedral: [{idx0.value()}-{idx1.value()}-{idx2.value()}-{idx3.value()}], {p.function()}"
                )
            else:
                new_dihedrals.set(idx0, idx1, idx2, idx3, p.function())

        # Update the molecule.
        mol = mol.edit().set_property("angle" + suffix, new_angles).molecule().commit()
        mol = (
            mol.edit()
            .set_property("dihedral" + suffix, new_dihedrals)
            .molecule()
            .commit()
        )

    # Now treat the triple junction.
    return _triple(
        mol,
        bridge,
        ghosts,
        physical,
        k_hard=k_hard,
        k_soft=k_soft,
        optimise_angles=optimise_angles,
        num_optimise=num_optimise,
        is_lambda1=is_lambda1,
    )


def _create_connectivity(mol):
    """
    Create a connectivity object for an end state molecule.

    Parameters
    ----------

    mol : sire.mol.Molecule
        The molecule at the end state.

    Returns

    connectivity : sire.legacy.Mol.Connectivity
        The connectivity object.
    """

    # Create an editable connectivity object.
    connectivity = _SireMol.Connectivity(mol.info()).edit()

    # Loop over the bonds in the molecule and connect the atoms.
    for bond in mol.bonds():
        connectivity.connect(bond.atom0().index(), bond.atom1().index())

    # Commit the changes and return the connectivity object.
    return connectivity.commit()
