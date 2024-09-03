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

    from ._somd1 import _is_dummy

    for mol in pert_mols:
        # Store the molecule info.
        info = mol.info()

        # Extract the connectivity property. This is currently the same
        # for both end states.
        connectivity = mol.property("connectivity")

        # Find the indices of the dummy atoms.
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

        # Print out the results for each end state.

        if len(bridges0) > 0:
            print("Bridges at lambda = 0")
            for i, b in enumerate(bridges0):
                print(f"  Bridge {i}: {b}")
                print(f"  dummies: {bridges0[b]}")
                print(f"  physical: {physical0[b]}")
                print(f"  type: {len(physical0[b])}")
            print()

        if len(bridges1) > 0:
            print("Bridges at lambda = 1")
            for i, b in enumerate(bridges1):
                print(f"  Bridge {i}: {b}")
                print(f"  dummies: {bridges1[b]}")
                print(f"  physical: {physical1[b]}")
                print(f"  type: {len(physical1[b])}")
