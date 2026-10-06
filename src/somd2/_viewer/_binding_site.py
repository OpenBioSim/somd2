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
The protein and perturbed molecules from a run's saved coordinates, for the
viewer.
"""

__all__ = ["binding_site", "has_binding_site"]

import threading as _threading
from collections import OrderedDict as _OrderedDict

import numpy as _np

# Protein residues with an atom within this distance, in nm, of a perturbed
# molecule are shown in detail.
_CUTOFF = 0.6

# Residue numbers for the perturbed molecules, which are given their own chain.
_LIGAND_CHAIN = "L"
_LIGAND_RESIDUE = 9999

# The topologies read most recently, which are kept to a limited number.
_topologies = _OrderedDict()
_topologies_lock = _threading.Lock()
_MAX_TOPOLOGIES = 16


def binding_site(topology0, topology1, saved):
    """
    The protein and perturbed molecules at the saved coordinates.

    Parameters
    ----------

    topology0, topology1: str
        The end-state topologies.

    saved: SavedCoordinates
        The coordinates, from the λ = 0 window.

    Returns
    -------

    site: dict
        The protein, without hydrogens, and the λ = 0 end state of each
        perturbed molecule, as PDB text, with the indices of the atoms of the
        protein residues near them, and of the perturbed molecules, in the
        order they are written. None if there isn't a protein and a perturbed
        molecule to show.
    """
    topology = _topology(topology0, topology1)
    if topology is None:
        return None

    def coordinates(molecule):
        positions = saved.molecule(
            molecule["first"], molecule["num_atoms"], topology["num_system_atoms"]
        )
        return positions[molecule["atoms"]]

    proteins = [coordinates(m) for m in topology["proteins"]]
    ligands = [coordinates(m) for m in topology["ligands"]]

    # Coordinates aren't wrapped into the box, so each perturbed molecule is
    # moved into the periodic image nearest to the protein.
    if _is_periodic(saved.box):
        box = _np.asarray(saved.box, dtype=float)
        inverse = _np.linalg.inv(box)
        centre = _np.concatenate(proteins).mean(axis=0)
        for positions in ligands:
            shift = _np.round((positions.mean(axis=0) - centre) @ inverse) @ box
            positions -= shift

    near = _np.concatenate(ligands)
    lines = []
    pocket = []
    index = 0
    for molecule, positions in zip(topology["proteins"], proteins):
        # A residue is in the pocket if any of its atoms is close enough.
        distances = _np.linalg.norm(
            positions[:, None, :] - near[None, :, :], axis=-1
        ).min(axis=1)
        records = molecule["records"]
        residues = {records[i][2] for i in _np.where(distances < _CUTOFF)[0]}
        for record, xyz in zip(records, positions):
            if record[2] in residues:
                pocket.append(index)
            lines.append(_atom("ATOM", index + 1, record, molecule["chain"], xyz))
            index += 1
        lines.append("TER")

    ligand_atoms = []
    for i, (molecule, positions) in enumerate(zip(topology["ligands"], ligands)):
        for record, xyz in zip(molecule["records"], positions):
            name, residue, _, element = record
            record = (name, residue, _LIGAND_RESIDUE - i, element)
            ligand_atoms.append(index)
            lines.append(_atom("HETATM", index + 1, record, _LIGAND_CHAIN, xyz))
            index += 1
    lines.append("END")

    return {
        "pdb": "\n".join(lines) + "\n",
        "pocket": pocket,
        "ligand": ligand_atoms,
        "time_ps": saved.time_ps,
    }


def has_binding_site(topology0, topology1):
    """
    Whether the end-state topologies have a protein and a perturbed molecule
    to show.
    """
    return _topology(topology0, topology1) is not None


def _is_periodic(box):
    """
    Whether box vectors are of a periodic system. OpenMM gives a 2 nm cube for
    a system that isn't periodic.
    """
    return box is not None and not _np.allclose(box, 2.0 * _np.eye(3))


def _atom(kind, serial, record, chain, xyz):
    """
    A PDB ATOM or HETATM record, with coordinates in nm.
    """
    name, residue, number, element = record
    # Names shorter than four characters start in the second column.
    name = name if len(name) == 4 else f" {name:<3}"
    x, y, z = (10.0 * _np.asarray(xyz)).tolist()
    return (
        f"{kind:<6}{serial % 100000:>5} {name:<4} {residue[:3]:>3} {chain}"
        f"{number % 10000:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
        f"          {element:>2}"
    )


def _topology(topology0, topology1):
    """
    The atoms of the protein, without hydrogens, and of the λ = 0 end state of
    each perturbed molecule, read once for each pair of topologies. None if
    there isn't both a protein and a perturbed molecule.
    """
    from ._summary import _file_key

    key = (_file_key(topology0), _file_key(topology1))
    with _topologies_lock:
        if key in _topologies:
            _topologies.move_to_end(key)
            return _topologies[key]

    import sire as _sr

    from ._data import sire_lock
    from ._depict import _perturbed_indices

    with sire_lock:
        system0 = _sr.load(str(topology0), show_warnings=False)
        system1 = _sr.load(str(topology1), show_warnings=False)
        atoms = system0.atoms()

        try:
            protein = list(system0.molecules("protein"))
        except KeyError:
            protein = []

        # Chains are only labels, but the perturbed molecules' is kept for them.
        chains = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != _LIGAND_CHAIN]

        topology = None
        if protein:
            proteins = []
            for i, mol in enumerate(protein):
                records = [
                    (
                        a.name().value(),
                        r.name().value(),
                        r.number().value(),
                        a.element().symbol(),
                    )
                    for r in mol.residues()
                    for a in r.atoms()
                ]
                # Hydrogens, and dummies, e.g. of a mutated residue, are left out.
                keep = [j for j, r in enumerate(records) if r[3] not in ("H", "Xx")]
                proteins.append(
                    {
                        "first": atoms.find(mol.atoms()[0]),
                        "num_atoms": mol.num_atoms(),
                        "atoms": keep,
                        "records": [records[j] for j in keep],
                        "chain": chains[i % len(chains)],
                    }
                )

            ligands = []
            mols0 = system0.molecules()
            for index in _perturbed_indices(system0, system1):
                mol = mols0[index]
                residue = mol.residues()[0].name().value()
                records = [
                    (a.name().value(), residue, 0, a.element().symbol())
                    for a in mol.atoms()
                ]
                # Ghost atoms are dummies at λ = 0.
                keep = [j for j, record in enumerate(records) if record[3] != "Xx"]
                ligands.append(
                    {
                        "first": atoms.find(mol.atoms()[0]),
                        "num_atoms": mol.num_atoms(),
                        "atoms": keep,
                        "records": [records[j] for j in keep],
                    }
                )

            # Nothing to centre on otherwise, e.g. for a protein mutation.
            if ligands:
                topology = {
                    "proteins": proteins,
                    "ligands": ligands,
                    "num_system_atoms": len(atoms),
                }

    with _topologies_lock:
        _topologies[key] = topology
        while len(_topologies) > _MAX_TOPOLOGIES:
            _topologies.popitem(last=False)
    return topology
