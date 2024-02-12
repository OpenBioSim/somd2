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

import sire.legacy.CAS as _SireCAS
import sire.legacy.MM as _SireMM
import sire.legacy.Mol as _SireMol


def _apply_somd1_pert(system):
    """
    Applies the somd1 perturbation to the system.

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

    # Store a dummy element.
    dummy = _SireMol.Element("Xx")

    for mol in pert_mols:
        # Get the end state bonded terms.

        # Lambda = 0.
        bonds0 = mol.property("bond0").potentials()
        angles0 = mol.property("angle0").potentials()
        dihedrals0 = mol.property("dihedral0").potentials()
        impropers0 = mol.property("improper0").potentials()

        # Lambda = 1.
        bonds1 = mol.property("bond1").potentials()
        angles1 = mol.property("angle1").potentials()
        dihedrals1 = mol.property("dihedral1").potentials()
        impropers1 = mol.property("improper1").potentials()

        # Get an editable version of the molecule.
        edit_mol = mol.edit()

        # First process the bonds.
        new_bonds0 = _SireMM.TwoAtomFunctions(mol.info())
        new_bonds1 = _SireMM.TwoAtomFunctions(mol.info())

        # Loop over the bonds.
        for p0, p1 in zip(bonds0, bonds1):
            # Get the atoms involved in the bond.
            a00 = p0.atom0()
            a01 = p0.atom1()
            a10 = p1.atom0()
            a11 = p1.atom1()

            # Get the elements of the atoms.
            e00 = mol[a00].property("element0")
            e01 = mol[a01].property("element0")
            e10 = mol[a10].property("element1")
            e11 = mol[a11].property("element1")

            initial_dummy = e00 == dummy or e01 == dummy
            final_dummy = e10 == dummy or e11 == dummy

            # If there is a dummy, then set the potential to the opposite state.
            # This should already be the case, but we explicitly set it here.

            if initial_dummy:
                new_bonds0.set(a00, a01, p1.function())
                new_bonds1.set(a10, a11, p1.function())
            elif final_dummy:
                new_bonds0.set(a00, a01, p0.function())
                new_bonds1.set(a10, a11, p0.function())
            else:
                new_bonds0.set(a00, a01, p0.function())
                new_bonds1.set(a10, a11, p1.function())

        # Set the new bonded terms.
        edit_mol = edit_mol.set_property("bond0", new_bonds0).molecule()
        edit_mol = edit_mol.set_property("bond1", new_bonds1).molecule()

        # Now process the angles.

        new_angles0 = _SireMM.ThreeAtomFunctions(mol.info())
        new_angles1 = _SireMM.ThreeAtomFunctions(mol.info())

        # Loop over the angles.
        for p0, p1 in zip(angles0, angles1):
            # Get the atoms involved in the angle.
            a00 = p0.atom0()
            a01 = p0.atom1()
            a02 = p0.atom2()
            a10 = p1.atom0()
            a11 = p1.atom1()
            a12 = p1.atom2()

            # Get the elements of the atoms.
            e00 = mol[a00].property("element0")
            e01 = mol[a01].property("element0")
            e02 = mol[a02].property("element0")
            e10 = mol[a10].property("element1")
            e11 = mol[a11].property("element1")
            e12 = mol[a12].property("element1")

            initial_dummy = e00 == dummy or e01 == dummy or e02 == dummy
            final_dummy = e10 == dummy or e11 == dummy or e12 == dummy

            # If both end states contain a dummy, the use null potentials.
            if initial_dummy and final_dummy:
                theta = _SireCAS.Symbol("theta")
                null_angle = _SireMM.AmberAngle(0.0, theta).to_expression(theta)
                new_angles0.set(a00, a01, a02, null_angle)
                new_angles1.set(a10, a11, a12, null_angle)
            # If the initial state contains a dummy, then use the potential from the final state.
            # This should already be the case, but we explicitly set it here.
            elif initial_dummy:
                new_angles0.set(a00, a01, a02, p1.function())
                new_angles1.set(a10, a11, a12, p1.function())
            # If the final state contains a dummy, then use the potential from the initial state.
            # This should already be the case, but we explicitly set it here.
            elif final_dummy:
                new_angles0.set(a00, a01, a02, p0.function())
                new_angles1.set(a10, a11, a12, p0.function())
            # Otherwise, use the potentials from the initial and final states.
            else:
                new_angles0.set(a00, a01, a02, p0.function())
                new_angles1.set(a10, a11, a12, p1.function())

        # Set the new angle terms.
        edit_mol = edit_mol.set_property("angle0", new_angles0).molecule()
        edit_mol = edit_mol.set_property("angle1", new_angles1).molecule()

        # Now process the dihedrals.

        new_dihedrals0 = _SireMM.FourAtomFunctions(mol.info())
        new_dihedrals1 = _SireMM.FourAtomFunctions(mol.info())

        # Loop over the dihedrals.
        for p0, p1 in zip(dihedrals0, dihedrals1):
            # Get the atoms involved in the dihedral.
            a00 = p0.atom0()
            a01 = p0.atom1()
            a02 = p0.atom2()
            a03 = p0.atom3()
            a10 = p1.atom0()
            a11 = p1.atom1()
            a12 = p1.atom2()
            a13 = p1.atom3()

            # Get the elements of the atoms.
            e00 = mol[a00].property("element0")
            e01 = mol[a01].property("element0")
            e02 = mol[a02].property("element0")
            e03 = mol[a03].property("element0")
            e10 = mol[a10].property("element1")
            e11 = mol[a11].property("element1")
            e12 = mol[a12].property("element1")
            e13 = mol[a13].property("element1")

            initial_dummy = e00 == dummy or e01 == dummy or e02 == dummy or e03 == dummy
            final_dummy = e10 == dummy or e11 == dummy or e12 == dummy or e13 == dummy

            # If both end states contain a dummy, the use null potentials.
            if initial_dummy and final_dummy:
                phi = _SireCAS.Symbol("phi")
                null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                new_dihedrals0.set(a00, a01, a02, a03, null_dihedral)
                new_dihedrals1.set(a10, a11, a12, a13, null_dihedral)
            elif initial_dummy:
                # If all the atoms are dummy, then use the potential from the final state.
                if e10 == dummy and e11 == dummy and e12 == dummy and e13 == dummy:
                    new_dihedrals0.set(a00, a01, a02, a03, p1.function())
                    new_dihedrals1.set(a10, a11, a12, a13, p1.function())
                # Otherwise, zero the potential.
                else:
                    phi = _SireCAS.Symbol("phi")
                    null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                    new_dihedrals0.set(a00, a01, a02, a03, null_dihedral)
                    new_dihedrals1.set(a10, a11, a12, a13, p1.function())
            elif final_dummy:
                # If all the atoms are dummy, then use the potential from the initial state.
                if e00 == dummy and e01 == dummy and e02 == dummy and e03 == dummy:
                    new_dihedrals0.set(a00, a01, a02, a03, p0.function())
                    new_dihedrals1.set(a10, a11, a12, a13, p0.function())
                # Otherwise, zero the potential.
                else:
                    phi = _SireCAS.Symbol("phi")
                    null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                    new_dihedrals0.set(a00, a01, a02, a03, p0.function())
                    new_dihedrals1.set(a10, a11, a12, a13, null_dihedral)
            else:
                new_dihedrals0.set(a00, a01, a02, a03, p0.function())
                new_dihedrals1.set(a10, a11, a12, a13, p1.function())

        # Set the new dihedral terms.
        edit_mol = edit_mol.set_property("dihedral0", new_dihedrals0).molecule()
        edit_mol = edit_mol.set_property("dihedral1", new_dihedrals1).molecule()

        # Now process the impropers.

        new_impropers0 = _SireMM.FourAtomFunctions(mol.info())
        new_impropers1 = _SireMM.FourAtomFunctions(mol.info())

        # Loop over the impropers.
        for p0, p1 in zip(impropers0, impropers1):
            # Get the atoms involved in the improper.
            a00 = p0.atom0()
            a01 = p0.atom1()
            a02 = p0.atom2()
            a03 = p0.atom3()
            a10 = p1.atom0()
            a11 = p1.atom1()
            a12 = p1.atom2()
            a13 = p1.atom3()

            # Get the elements of the atoms.
            e00 = mol[a00].property("element0")
            e01 = mol[a01].property("element0")
            e02 = mol[a02].property("element0")
            e03 = mol[a03].property("element0")
            e10 = mol[a10].property("element1")
            e11 = mol[a11].property("element1")
            e12 = mol[a12].property("element1")
            e13 = mol[a13].property("element1")

            initial_dummy = e00 == dummy or e01 == dummy or e02 == dummy or e03 == dummy
            final_dummy = e10 == dummy or e11 == dummy or e12 == dummy or e13 == dummy

            if initial_dummy and final_dummy:
                phi = _SireCAS.Symbol("phi")
                null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                new_impropers0.set(a00, a01, a02, a03, null_dihedral)
                new_impropers1.set(a10, a11, a12, a13, null_dihedral)
            elif initial_dummy:
                # If all the atoms are dummy, then use the potential from the final state.
                if e10 == dummy and e11 == dummy and e12 == dummy and e13 == dummy:
                    new_impropers0.set(a00, a01, a02, a03, p1.function())
                    new_impropers1.set(a10, a11, a12, a13, p1.function())
                # Otherwise, zero the potential.
                else:
                    phi = _SireCAS.Symbol("phi")
                    null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                    new_impropers0.set(a00, a01, a02, a03, null_dihedral)
                    new_impropers1.set(a10, a11, a12, a13, p1.function())
            elif final_dummy:
                # If all the atoms are dummy, then use the potential from the initial state.
                if e00 == dummy and e01 == dummy and e02 == dummy and e03 == dummy:
                    new_impropers0.set(a00, a01, a02, a03, p0.function())
                    new_impropers1.set(a10, a11, a12, a13, p0.function())
                # Otherwise, zero the potential.
                else:
                    phi = _SireCAS.Symbol("phi")
                    null_dihedral = _SireMM.AmberDihedral(0.0, phi).to_expression(phi)
                    new_impropers0.set(a00, a01, a02, a03, p0.function())
                    new_impropers1.set(a10, a11, a12, a13, null_dihedral)
            else:
                new_impropers0.set(a00, a01, a02, a03, p0.function())
                new_impropers1.set(a10, a11, a12, a13, p1.function())

        # Set the new improper terms.
        edit_mol = edit_mol.set_property("improper0", new_impropers0).molecule()
        edit_mol = edit_mol.set_property("improper1", new_impropers1).molecule()

        # Commit the changes and update the molecule in the system.
        system.update(edit_mol.commit())

    # Return the updated system.
    return system
