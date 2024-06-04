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


def _make_compatible(system):
    """
    Makes a perturbation SOMD1 compatible.

    Parameters
    ----------

    system : sire.system.System, sire.legacy.System.System
        The system containing the molecules to be perturbed.

    Returns
    -------

    system : sire.legacy.System.System
        The updated system.
    """

    # Check the system is a Sire system.
    if not isinstance(system, (_System, _LegacySystem)):
        raise TypeError(
            "'system' must of type 'sire.system.System' or 'sire.legacy.System.System'"
        )

    # Extract the legacy system.
    if isinstance(system, _LegacySystem):
        system = _System(system)

    # Search for perturbable molecules.
    try:
        pert_mols = system.molecules("property is_perturbable")
    except KeyError:
        raise KeyError("No perturbable molecules in the system")

    for mol in pert_mols:
        # Store the molecule info.
        info = mol.info()

        # Get an editable version of the molecule.
        edit_mol = mol.edit()

        ##################################
        # First fix zero LJ sigmas values.
        ##################################

        # Tolerance for zero sigma values.
        null_lj_sigma = 1e-9

        for atom in mol.atoms():
            # Get the end state LJ sigma values.
            lj0 = atom.property("LJ0")
            lj1 = atom.property("LJ1")

            # Lambda = 0 state has a zero sigma value.
            if abs(lj0.sigma().value()) <= null_lj_sigma:
                # Use the sigma value from the lambda = 1 state.
                edit_mol = (
                    edit_mol.atom(atom.index())
                    .set_property(
                        "LJ0", _SireMM.LJParameter(lj1.sigma(), lj0.epsilon())
                    )
                    .molecule()
                )

            # Lambda = 1 state has a zero sigma value.
            if abs(lj1.sigma().value()) <= null_lj_sigma:
                # Use the sigma value from the lambda = 0 state.
                edit_mol = (
                    edit_mol.atom(atom.index())
                    .set_property(
                        "LJ1", _SireMM.LJParameter(lj0.sigma(), lj1.epsilon())
                    )
                    .molecule()
                )

        ########################
        # Now process the bonds.
        ########################

        new_bonds0 = _SireMM.TwoAtomFunctions(mol.info())
        new_bonds1 = _SireMM.TwoAtomFunctions(mol.info())

        # Extract the bonds at lambda = 0 and 1.
        bonds0 = mol.property("bond0").potentials()
        bonds1 = mol.property("bond1").potentials()

        # Dictionaries to store the BondIDs at lambda = 0 and 1.
        bonds0_idx = {}
        bonds1_idx = {}

        # Loop over all bonds at lambda = 0.
        for idx, bond in enumerate(bonds0):
            # Get the AtomIdx for the atoms in the bond.
            idx0 = info.atom_idx(bond.atom0())
            idx1 = info.atom_idx(bond.atom1())

            # Create the BondID.
            bond_id = _SireMol.BondID(idx0, idx1)

            # Add to the list of ids.
            bonds0_idx[bond_id] = idx

        # Loop over all bonds at lambda = 1.
        for idx, bond in enumerate(bonds1):
            # Get the AtomIdx for the atoms in the bond.
            idx0 = info.atom_idx(bond.atom0())
            idx1 = info.atom_idx(bond.atom1())

            # Create the BondID.
            bond_id = _SireMol.BondID(idx0, idx1)

            # Add to the list of ids.
            if bond_id.mirror() in bonds0_idx:
                bonds1_idx[bond_id.mirror()] = idx
            else:
                bonds1_idx[bond_id] = idx

        # Now work out the BondIDs that are unique at lambda = 0 and 1
        # as well as those that are shared.
        bonds0_unique_idx = {}
        bonds1_unique_idx = {}
        bonds_shared_idx = {}

        # lambda = 0.
        for idx in bonds0_idx.keys():
            if idx not in bonds1_idx.keys():
                bonds0_unique_idx[idx] = bonds0_idx[idx]
            else:
                bonds_shared_idx[idx] = (bonds0_idx[idx], bonds1_idx[idx])

        # lambda = 1.
        for idx in bonds1_idx.keys():
            if idx not in bonds0_idx.keys():
                bonds1_unique_idx[idx] = bonds1_idx[idx]
            elif idx not in bonds_shared_idx.keys():
                bonds_shared_idx[idx] = (bonds0_idx[idx], bonds1_idx[idx])

        # Loop over the shared bonds.
        for idx0, idx1 in bonds_shared_idx.values():
            # Get the bond potentials.
            p0 = bonds0[idx0]
            p1 = bonds1[idx1]

            # Get the AtomIdx for the atoms in the angle.
            idx0 = p0.atom0()
            idx1 = p0.atom1()

            # Check whether a dummy atoms are present in the lambda = 0
            # and lambda = 1 states.
            initial_dummy = _has_dummy(mol, [idx0, idx1])
            final_dummy = _has_dummy(mol, [idx0, idx1], True)

            # If there is a dummy, then set the potential to the opposite state.
            # This should already be the case, but we explicitly set it here.

            if initial_dummy:
                new_bonds0.set(idx0, idx1, p1.function())
                new_bonds1.set(idx0, idx1, p1.function())
            elif final_dummy:
                new_bonds0.set(idx0, idx1, p0.function())
                new_bonds1.set(idx0, idx1, p0.function())
            else:
                new_bonds0.set(idx0, idx1, p0.function())
                new_bonds1.set(idx0, idx1, p1.function())

        # Set the new bonded terms.
        edit_mol = edit_mol.set_property("bond0", new_bonds0).molecule()
        edit_mol = edit_mol.set_property("bond1", new_bonds1).molecule()

        #########################
        # Now process the angles.
        #########################

        new_angles0 = _SireMM.ThreeAtomFunctions(mol.info())
        new_angles1 = _SireMM.ThreeAtomFunctions(mol.info())

        # Extract the angles at lambda = 0 and 1.
        angles0 = mol.property("angle0").potentials()
        angles1 = mol.property("angle1").potentials()

        # Dictionaries to store the AngleIDs at lambda = 0 and 1.
        angles0_idx = {}
        angles1_idx = {}

        # Loop over all angles at lambda = 0.
        for idx, angle in enumerate(angles0):
            # Get the AtomIdx for the atoms in the angle.
            idx0 = info.atom_idx(angle.atom0())
            idx1 = info.atom_idx(angle.atom1())
            idx2 = info.atom_idx(angle.atom2())

            # Create the AngleID.
            angle_id = _SireMol.AngleID(idx0, idx1, idx2)

            # Add to the list of ids.
            angles0_idx[angle_id] = idx

        # Loop over all angles at lambda = 1.
        for idx, angle in enumerate(angles1):
            # Get the AtomIdx for the atoms in the angle.
            idx0 = info.atom_idx(angle.atom0())
            idx1 = info.atom_idx(angle.atom1())
            idx2 = info.atom_idx(angle.atom2())

            # Create the AngleID.
            angle_id = _SireMol.AngleID(idx0, idx1, idx2)

            # Add to the list of ids.
            if angle_id.mirror() in angles0_idx:
                angles1_idx[angle_id.mirror()] = idx
            else:
                angles1_idx[angle_id] = idx

        # Now work out the AngleIDs that are unique at lambda = 0 and 1
        # as well as those that are shared.
        angles0_unique_idx = {}
        angles1_unique_idx = {}
        angles_shared_idx = {}

        # lambda = 0.
        for idx in angles0_idx.keys():
            if idx not in angles1_idx.keys():
                angles0_unique_idx[idx] = angles0_idx[idx]
            else:
                angles_shared_idx[idx] = (angles0_idx[idx], angles1_idx[idx])

        # lambda = 1.
        for idx in angles1_idx.keys():
            if idx not in angles0_idx.keys():
                angles1_unique_idx[idx] = angles1_idx[idx]
            elif idx not in angles_shared_idx.keys():
                angles_shared_idx[idx] = (angles0_idx[idx], angles1_idx[idx])

        # Loop over the angles.
        for idx0, idx1 in angles_shared_idx.values():
            # Get the angle potentials.
            p0 = angles0[idx0]
            p1 = angles1[idx1]

            # Get the AtomIdx for the atoms in the angle.
            idx0 = p0.atom0()
            idx1 = p0.atom1()
            idx2 = p0.atom2()

            # Check whether a dummy atoms are present in the lambda = 0
            # and lambda = 1 states.
            initial_dummy = _has_dummy(mol, [idx0, idx1, idx2])
            final_dummy = _has_dummy(mol, [idx0, idx1, idx2], True)

            # If both end states contain a dummy, then don't add the potentials.
            if initial_dummy and final_dummy:
                continue
            # If the initial state contains a dummy, then use the potential from the final state.
            # This should already be the case, but we explicitly set it here.
            elif initial_dummy:
                new_angles0.set(idx0, idx1, idx2, p1.function())
                new_angles1.set(idx0, idx1, idx2, p1.function())
            # If the final state contains a dummy, then use the potential from the initial state.
            # This should already be the case, but we explicitly set it here.
            elif final_dummy:
                new_angles0.set(idx0, idx1, idx2, p0.function())
                new_angles1.set(idx0, idx1, idx2, p0.function())
            # Otherwise, use the potentials from the initial and final states.
            else:
                new_angles0.set(idx0, idx1, idx2, p0.function())
                new_angles1.set(idx0, idx1, idx2, p1.function())

        # Set the new angle terms.
        edit_mol = edit_mol.set_property("angle0", new_angles0).molecule()
        edit_mol = edit_mol.set_property("angle1", new_angles1).molecule()

        ############################
        # Now process the dihedrals.
        ############################

        new_dihedrals0 = _SireMM.FourAtomFunctions(mol.info())
        new_dihedrals1 = _SireMM.FourAtomFunctions(mol.info())

        # Extract the dihedrals at lambda = 0 and 1.
        dihedrals0 = mol.property("dihedral0").potentials()
        dihedrals1 = mol.property("dihedral1").potentials()

        # Dictionaries to store the DihedralIDs at lambda = 0 and 1.
        dihedrals0_idx = {}
        dihedrals1_idx = {}

        # Loop over all dihedrals at lambda = 0.
        for idx, dihedral in enumerate(dihedrals0):
            # Get the AtomIdx for the atoms in the dihedral.
            idx0 = info.atom_idx(dihedral.atom0())
            idx1 = info.atom_idx(dihedral.atom1())
            idx2 = info.atom_idx(dihedral.atom2())
            idx3 = info.atom_idx(dihedral.atom3())

            # Create the DihedralID.
            dihedral_id = _SireMol.DihedralID(idx0, idx1, idx2, idx3)

            # Add to the list of ids.
            dihedrals0_idx[dihedral_id] = idx

        # Loop over all dihedrals at lambda = 1.
        for idx, dihedral in enumerate(dihedrals1):
            # Get the AtomIdx for the atoms in the dihedral.
            idx0 = info.atom_idx(dihedral.atom0())
            idx1 = info.atom_idx(dihedral.atom1())
            idx2 = info.atom_idx(dihedral.atom2())
            idx3 = info.atom_idx(dihedral.atom3())

            # Create the DihedralID.
            dihedral_id = _SireMol.DihedralID(idx0, idx1, idx2, idx3)

            # Add to the list of ids.
            if dihedral_id.mirror() in dihedrals0_idx:
                dihedrals1_idx[dihedral_id.mirror()] = idx
            else:
                dihedrals1_idx[dihedral_id] = idx

        # Now work out the DihedralIDs that are unique at lambda = 0 and 1
        # as well as those that are shared.
        dihedrals0_unique_idx = {}
        dihedrals1_unique_idx = {}
        dihedrals_shared_idx = {}

        # lambda = 0.
        for idx in dihedrals0_idx.keys():
            if idx not in dihedrals1_idx.keys():
                dihedrals0_unique_idx[idx] = dihedrals0_idx[idx]
            else:
                dihedrals_shared_idx[idx] = (dihedrals0_idx[idx], dihedrals1_idx[idx])

        # lambda = 1.
        for idx in dihedrals1_idx.keys():
            if idx not in dihedrals0_idx.keys():
                dihedrals1_unique_idx[idx] = dihedrals1_idx[idx]
            elif idx not in dihedrals_shared_idx.keys():
                dihedrals_shared_idx[idx] = (dihedrals0_idx[idx], dihedrals1_idx[idx])

        # Loop over the dihedrals.
        for idx0, idx1 in dihedrals_shared_idx.values():
            # Get the dihedral potentials.
            p0 = dihedrals0[idx0]
            p1 = dihedrals1[idx1]

            # Get the AtomIdx for the atoms in the dihedral.
            idx0 = info.atom_idx(p0.atom0())
            idx1 = info.atom_idx(p0.atom1())
            idx2 = info.atom_idx(p0.atom2())
            idx3 = info.atom_idx(p0.atom3())

            # Whether any atom in each end state is a dummy.
            has_dummy_initial = _has_dummy(mol, [idx0, idx1, idx2, idx3])
            has_dummy_final = _has_dummy(mol, [idx0, idx1, idx2, idx3], True)

            # Whether all atoms in each state are dummies.
            all_dummy_initial = all(_is_dummy(mol, [idx0, idx1, idx2, idx3]))
            all_dummy_final = all(_is_dummy(mol, [idx0, idx1, idx2, idx3], True))

            # If both end states contain a dummy, then don't add the potentials.
            if has_dummy_initial and has_dummy_final:
                continue
            elif has_dummy_initial:
                # If all the atoms are dummy, then use the potential from the final state.
                if all_dummy_initial:
                    new_dihedrals0.set(idx0, idx1, idx2, idx3, p1.function())
                    new_dihedrals1.set(idx0, idx1, idx2, idx3, p1.function())
                # Otherwise, remove the potential from the initial state.
                else:
                    new_dihedrals1.set(idx0, idx1, idx2, idx3, p1.function())
            elif has_dummy_final:
                # If all the atoms are dummy, then use the potential from the initial state.
                if all_dummy_final:
                    new_dihedrals0.set(idx0, idx1, idx2, idx3, p0.function())
                    new_dihedrals1.set(idx0, idx1, idx2, idx3, p0.function())
                # Otherwise, remove the potential from the final state.
                else:
                    new_dihedrals0.set(idx0, idx1, idx2, idx3, p0.function())
            else:
                new_dihedrals0.set(idx0, idx1, idx2, idx3, p0.function())
                new_dihedrals1.set(idx0, idx1, idx2, idx3, p1.function())

        # Set the new dihedral terms.
        edit_mol = edit_mol.set_property("dihedral0", new_dihedrals0).molecule()
        edit_mol = edit_mol.set_property("dihedral1", new_dihedrals1).molecule()

        ############################
        # Now process the impropers.
        ############################

        new_impropers0 = _SireMM.FourAtomFunctions(mol.info())
        new_impropers1 = _SireMM.FourAtomFunctions(mol.info())

        # Extract the impropers at lambda = 0 and 1.
        impropers0 = mol.property("improper0").potentials()
        impropers1 = mol.property("improper1").potentials()

        # Dictionaries to store the ImproperIDs at lambda = 0 and 1.
        impropers0_idx = {}
        impropers1_idx = {}

        # Loop over all impropers at lambda = 0.
        for idx, improper in enumerate(impropers0):
            # Get the AtomIdx for the atoms in the improper.
            idx0 = info.atom_idx(improper.atom0())
            idx1 = info.atom_idx(improper.atom1())
            idx2 = info.atom_idx(improper.atom2())
            idx3 = info.atom_idx(improper.atom3())

            # Create the ImproperID.
            improper_id = _SireMol.ImproperID(idx0, idx1, idx2, idx3)

            # Add to the list of ids.
            impropers0_idx[improper_id] = idx

        # Loop over all impropers at lambda = 1.
        for idx, improper in enumerate(impropers1):
            # Get the AtomIdx for the atoms in the improper.
            idx0 = info.atom_idx(improper.atom0())
            idx1 = info.atom_idx(improper.atom1())
            idx2 = info.atom_idx(improper.atom2())
            idx3 = info.atom_idx(improper.atom3())

            # Create the ImproperID.
            improper_id = _SireMol.ImproperID(idx0, idx1, idx2, idx3)

            # Add to the list of ids.
            # You cannot mirror an improper!
            impropers1_idx[improper_id] = idx

        # Now work out the ImproperIDs that are unique at lambda = 0 and 1
        # as well as those that are shared. Note that the ordering of
        # impropers is inconsistent between molecular topology formats so
        # we test all permutations of atom ordering to find matches. This
        # is achieved using the ImproperID.equivalent() method.
        impropers0_unique_idx = {}
        impropers1_unique_idx = {}
        impropers_shared_idx = {}

        # lambda = 0.
        for idx0 in impropers0_idx.keys():
            for idx1 in impropers1_idx.keys():
                if idx0.equivalent(idx1):
                    impropers_shared_idx[idx0] = (
                        impropers0_idx[idx0],
                        impropers1_idx[idx1],
                    )
                    break
            else:
                impropers0_unique_idx[idx0] = impropers0_idx[idx0]

        # lambda = 1.
        for idx1 in impropers1_idx.keys():
            for idx0 in impropers0_idx.keys():
                if idx1.equivalent(idx0):
                    # Don't store duplicates.
                    if not idx0 in impropers_shared_idx.keys():
                        impropers_shared_idx[idx1] = (
                            impropers0_idx[idx0],
                            impropers1_idx[idx1],
                        )
                    break
            else:
                impropers1_unique_idx[idx1] = impropers1_idx[idx1]

        # Loop over the impropers.
        for idx0, idx1 in impropers_shared_idx.values():
            # Get the improper potentials.
            p0 = impropers0[idx0]
            p1 = impropers1[idx1]

            # Get the AtomIdx for the atoms in the dihedral.
            idx0 = info.atom_idx(p0.atom0())
            idx1 = info.atom_idx(p0.atom1())
            idx2 = info.atom_idx(p0.atom2())
            idx3 = info.atom_idx(p0.atom3())

            # Whether any atom in each end state is a dummy.
            has_dummy_initial = _has_dummy(mol, [idx0, idx1, idx2, idx3])
            has_dummy_final = _has_dummy(mol, [idx0, idx1, idx2, idx3], True)

            # Whether all atoms in each state are dummies.
            all_dummy_initial = all(_is_dummy(mol, [idx0, idx1, idx2, idx3]))
            all_dummy_final = all(_is_dummy(mol, [idx0, idx1, idx2, idx3], True))

            # If both end states contain a dummy, then don't add the potentials.
            if has_dummy_initial and has_dummy_final:
                continue
            elif has_dummy_initial:
                # If all the atoms are dummy, then use the potential from the final state.
                if all_dummy_initial:
                    new_impropers0.set(idx0, idx1, idx2, idx3, p1.function())
                    new_impropers1.set(idx0, idx1, idx2, idx3, p1.function())
                # Otherwise, remove the potential from the initial state.
                else:
                    new_impropers1.set(idx0, idx1, idx2, idx3, p1.function())
            elif has_dummy_final:
                # If all the atoms are dummy, then use the potential from the initial state.
                if all_dummy_final:
                    new_impropers0.set(idx0, idx1, idx2, idx3, p0.function())
                    new_impropers1.set(idx0, idx1, idx2, idx3, p0.function())
                # Otherwise, remove the potential from the final state.
                else:
                    new_impropers0.set(idx0, idx1, idx2, idx3, p0.function())
            else:
                new_impropers0.set(idx0, idx1, idx2, idx3, p0.function())
                new_impropers1.set(idx0, idx1, idx2, idx3, p1.function())

        # Set the new improper terms.
        edit_mol = edit_mol.set_property("improper0", new_impropers0).molecule()
        edit_mol = edit_mol.set_property("improper1", new_impropers1).molecule()

        # Commit the changes and update the molecule in the system.
        system.update(edit_mol.commit())

    # Return the updated system.
    return system


def _has_dummy(mol, idxs, is_lambda1=False):
    """
    Internal function to check whether any atom is a dummy.

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

    has_dummy : bool
        Whether a dummy atom is present.
    """

    # We need to check by ambertype too since this molecule may have been
    # created via sire.morph.create_from_pertfile, in which case the element
    # property will have been set to the end state with the largest mass, i.e.
    # may no longer by a dummy.
    if is_lambda1:
        element_prop = "element1"
        ambertype_prop = "ambertype1"
    else:
        element_prop = "element0"
        ambertype_prop = "ambertype0"

    element_dummy = _SireMol.Element(0)
    ambertype_dummy = "du"

    # Check whether an of the atoms is a dummy.
    for idx in idxs:
        if (
            mol.atom(idx).property(element_prop) == element_dummy
            or mol.atom(idx).property(ambertype_prop) == ambertype_dummy
        ):
            return True

    return False


def _is_dummy(mol, idxs, is_lambda1=False):
    """
    Internal function to return whether each atom is a dummy.

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

    is_dummy : [bool]
        Whether each atom is a dummy.
    """

    # We need to check by ambertype too since this molecule may have been
    # created via sire.morph.create_from_pertfile, in which case the element
    # property will have been set to the end state with the largest mass, i.e.
    # may no longer by a dummy.
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

    element_dummy = _SireMol.Element(0)
    ambertype_dummy = "du"

    # Initialise a list to store the state of each atom.
    is_dummy = []

    # Check whether each of the atoms is a dummy.
    for idx in idxs:
        is_dummy.append(
            mol.atom(idx).property(element_prop) == element_dummy
            or mol.atom(idx).property(ambertype_prop) == ambertype_dummy
        )

    return is_dummy


def _apply_pert(system, pert_file):
    """
    Helper function to apply a perturbation to a reference system.

    Parameters
    ----------

    system: sr.system.System
        The reference system.

    pert_file: str
        Path to a stream file containing the perturbation to apply to the
        reference system.

    Returns
    -------

    system: sire.system.System
        The perturbable system.
    """

    if not isinstance(system, _System):
        raise TypeError("'system' must be of type 'sr.system.System'.")

    if not isinstance(pert_file, str):
        raise TypeError("'pert_file' must be of type 'str'.")

    from sire import morph as _morph

    # Get the non-water molecules in the system.
    non_waters = system["not water"].molecules()

    # Try to apply the perturbation to each non-water molecule.
    is_pert = False
    for mol in non_waters:
        # Exclude ions.
        if mol.num_atoms() > 1:
            try:
                pert_mol = _morph.create_from_pertfile(mol, pert_file)
                is_pert = True
                break
            except:
                pass

    if not is_pert:
        raise ValueError(f"Failed to apply the perturbation in '{pert_file}'.")

    # Replace the reference molecule with the perturbed molecule.
    system.remove(mol)
    system.add(pert_mol)

    return system
