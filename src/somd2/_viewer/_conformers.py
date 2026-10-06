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

"""
3D conformers of the perturbed molecules at each end state, for the viewer.
"""

__all__ = ["conformers"]

# A fixed seed, so that the same molecule always gives the same conformer.
_SEED = 42

# Changed whenever the conformers do, so that cached ones are regenerated.
_VERSION = 1


def conformers(topology0, topology1, has_positions, load_positions):
    """
    Generate a 3D conformer of each end state of each perturbed molecule,
    reusing one cached on disk for the same topologies if there is one.

    Parameters
    ----------

    topology0, topology1: str
        The end-state topologies.

    has_positions: bool
        Whether coordinates have been saved for the system.

    load_positions: callable
        Returns the saved coordinates as SavedCoordinates, or None if there
        are none. Only called if a molecule has stereochemistry to assign.

    Returns
    -------

    molecules: [dict]
        Each molecule's name, a mol block for each end state, the indices of
        the atoms that are unique to each end state or change element, and
        its stereochemistry: from the simulation, unknown until coordinates
        are saved, failed if they couldn't be used, or none.
    """
    from ._cache import read_json, write_json
    from ._summary import _file_key

    def name(final):
        stereo = "known" if final else "unknown"
        return (
            f"conformers/v{_VERSION}-{_file_key(topology0)}-"
            f"{_file_key(topology1)}-{stereo}.json"
        )

    # A result is final once its stereochemistry no longer waits on
    # coordinates, which is preferred.
    cached = read_json(name(True))
    if cached is not None:
        return cached
    if not has_positions:
        cached = read_json(name(False))
        if cached is not None:
            return cached

    result = _generate(topology0, topology1, load_positions if has_positions else None)
    write_json(name(all(m["stereo"] != "unknown" for m in result)), result)
    return result


def _generate(topology0, topology1, load_positions):
    from ._data import sire_lock
    from ._depict import _perturbed_molecules

    # Only reading the topologies needs Sire.
    with sire_lock:
        molecules = list(_perturbed_molecules(topology0, topology1))

    positions = []

    def system_positions():
        # Read once, and only if a molecule has stereochemistry to assign.
        if not positions:
            positions.append(load_positions() if load_positions else None)
        return positions[0]

    results = []
    for p in molecules:
        n = len(p.dummy0)

        def coordinates(p=p, n=n):
            saved = system_positions()
            if saved is None:
                return None
            return saved.molecule(p.offset, n, p.num_system_atoms)

        mol0, stereo0 = _with_stereo(p.rdmol0, p.map0, coordinates)
        mol1, stereo1 = _with_stereo(p.rdmol1, p.map1, coordinates)
        _embed(mol0)
        _embed(mol1, mol0, p.mapping)

        index0 = {merged: i for i, merged in enumerate(p.map0)}
        index1 = {merged: i for i, merged in enumerate(p.map1)}
        results.append(
            {
                "name": p.name,
                "mol0": _mol_block(mol0),
                "mol1": _mol_block(mol1),
                "unique0": sorted(index0[i] for i in p.unique0),
                "unique1": sorted(index1[i] for i in p.unique1),
                "changed0": sorted(index0[i] for i in p.changed),
                "changed1": sorted(index1[i] for i in p.changed),
                "stereo": _combine(stereo0, stereo1),
            }
        )
    return results


def _with_stereo(rdmol, atoms, coordinates):
    """
    A copy of an end state with its stereochemistry assigned from the
    simulation's coordinates, which are only fetched, by calling coordinates(),
    if it has any. Also returns whether the stereochemistry is from the
    simulation, unknown, failed, or none.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from ._coordinates import Mismatch

    mol = Chem.Mol(rdmol)
    mol.RemoveAllConformers()
    try:
        num_stereo = len(Chem.FindPotentialStereo(mol))
    except Exception:
        num_stereo = 0
    if num_stereo == 0:
        return mol, "none"

    # Errors reading the coordinates are raised, so that they are retried.
    try:
        positions = coordinates()
    except Mismatch:
        return mol, "failed"
    if positions is None:
        return mol, "unknown"

    try:
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i, atom in enumerate(atoms):
            # The coordinates are in nm.
            x, y, z = (10.0 * positions[atom]).tolist()
            conformer.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conformer, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        stereo = "simulation"
    except Exception:
        stereo = "failed"
    mol.RemoveAllConformers()
    return mol, stereo


def _combine(stereo0, stereo1):
    for stereo in ("unknown", "failed", "simulation"):
        if stereo in (stereo0, stereo1):
            return stereo
    return "none"


def _embed(mol, template=None, mapping=None):
    """
    Embed a molecule in 3D and relax it. If a template is given, the atoms
    mapped onto it are placed and kept at its coordinates, so that the two
    overlay.
    """
    from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

    coord_map = None
    if template is not None:
        positions = template.GetConformer()
        coord_map = {j: positions.GetAtomPosition(i) for i, j in mapping.items()}

    kwargs = {"randomSeed": _SEED}
    if coord_map:
        kwargs["coordMap"] = coord_map
    if AllChem.EmbedMolecule(mol, **kwargs) < 0:
        kwargs.pop("coordMap", None)
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, **kwargs) < 0:
            raise RuntimeError("Couldn't generate 3D coordinates.")
        coord_map = None

    _relax(mol, coord_map or {})

    if template is None:
        # Lay the molecule's long axis across the wide viewer.
        rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
    else:
        rdMolAlign.AlignMol(mol, template, atomMap=[(j, i) for i, j in mapping.items()])


def _relax(mol, fixed):
    """
    Relax an embedded molecule with MMFF, or UFF if that fails. The embedded
    coordinates are kept if neither works, since they are only for viewing.
    """
    from rdkit.Chem import AllChem

    # There is nothing to relax, e.g. if both end states have the same atoms.
    if len(fixed) >= mol.GetNumAtoms():
        return

    def mmff():
        if AllChem.MMFFHasAllMoleculeParams(mol):
            properties = AllChem.MMFFGetMoleculeProperties(mol)
            return AllChem.MMFFGetMoleculeForceField(mol, properties)

    def uff():
        if AllChem.UFFHasAllMoleculeParams(mol):
            return AllChem.UFFGetMoleculeForceField(mol)

    start = mol.GetConformer().GetPositions()
    for make in (mmff, uff):
        try:
            forcefield = make()
            if forcefield is None:
                continue
            for i in fixed:
                forcefield.AddFixedPoint(i)
            forcefield.Minimize(maxIts=500)
            return
        except Exception:
            # A failed minimisation can leave the coordinates part way.
            conformer = mol.GetConformer()
            for i, position in enumerate(start):
                conformer.SetAtomPosition(i, position.tolist())


def _mol_block(mol):
    from rdkit import Chem

    try:
        return Chem.MolToMolBlock(mol)
    except Exception:
        # Molecules without bond orders can't be kekulised.
        return Chem.MolToMolBlock(mol, kekulize=False)
