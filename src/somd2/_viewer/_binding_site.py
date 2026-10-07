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
The protein and perturbed molecules, or a mutated protein, from a run's saved
coordinates, for the viewer.
"""

__all__ = ["binding_site", "site_kind"]

import threading as _threading
from collections import OrderedDict as _OrderedDict

import numpy as _np

# Protein residues with an atom within this distance, in nm, of a perturbed
# molecule are shown in detail.
_CUTOFF = 0.6

# Residue numbers for the perturbed molecules, or the other molecules shown
# with a mutated protein, which are given their own chain.
_LIGAND_CHAIN = "L"
_LIGAND_RESIDUE = 9999

# With a mutated protein, other molecules, e.g. a ligand, are shown if they
# are within this distance, in nm, of a mutated residue.
_NEARBY = 1.2

# The topologies read most recently, which are kept to a limited number.
_topologies = _OrderedDict()
_topologies_lock = _threading.Lock()
_MAX_TOPOLOGIES = 16


def binding_site(topology0, topology1, saved, saved1=None):
    """
    The protein and perturbed molecules, or a mutated protein, at the saved
    coordinates.

    Parameters
    ----------

    topology0, topology1: str
        The end-state topologies.

    saved: SavedCoordinates
        The coordinates, from the λ = 0 window.

    saved1: SavedCoordinates
        The coordinates from the λ = 1 window, for that end state of a mutated
        protein. Those from the λ = 0 window are used if None.

    Returns
    -------

    site: dict
        For perturbed molecules, the protein, without hydrogens, and the λ = 0
        end state of each perturbed molecule, as PDB text, with the serial
        numbers of the atoms of the protein residues near them, and of the
        perturbed molecules. For a mutated protein, the protein, without
        hydrogens, and any other molecules, e.g. a ligand, at each end state,
        with the serial numbers of the atoms of the mutated residues, of the
        residues near them or the other molecules, and of the other molecules.
        None if there is nothing to show.
    """
    topology = _topology(topology0, topology1)
    if topology is None:
        return None
    if topology["kind"] == "mutation":
        return _mutation(topology, saved, saved1 or saved)

    proteins = [_coordinates(saved, topology, m) for m in topology["proteins"]]
    ligands = [_coordinates(saved, topology, m) for m in topology["others"]]
    _image(saved.box, proteins, ligands)

    near = _np.concatenate(ligands)
    lines = []
    pocket = []
    serial = 0
    for molecule, positions in zip(topology["proteins"], proteins):
        # A residue is in the pocket if any of its atoms is close enough.
        distances = _np.linalg.norm(
            positions[:, None, :] - near[None, :, :], axis=-1
        ).min(axis=1)
        state = molecule["states"][0]
        residues = {state["residues"][i] for i in _np.where(distances < _CUTOFF)[0]}
        for record, residue, xyz in zip(state["records"], state["residues"], positions):
            serial += 1
            if residue in residues:
                pocket.append(serial)
            lines.append(_atom("ATOM", serial, record, molecule["chain"], xyz))
        lines.append("TER")

    ligand_atoms = []
    for i, (molecule, positions) in enumerate(zip(topology["others"], ligands)):
        for record, xyz in zip(molecule["states"][0]["records"], positions):
            serial += 1
            ligand_atoms.append(serial)
            lines.append(_ligand_atom(serial, record, i, xyz))
    lines.append("END")

    return {
        "kind": "ligand",
        "pdb": "\n".join(lines) + "\n",
        "pocket": pocket,
        "ligand": ligand_atoms,
        "time_ps": saved.time_ps,
    }


def site_kind(topology0, topology1):
    """
    What the end-state topologies have to show: "ligand" for a protein and a
    perturbed molecule, "mutation" for a mutated protein, or None.
    """
    topology = _topology(topology0, topology1)
    return topology["kind"] if topology is not None else None


def _mutation(topology, saved0, saved1):
    """
    The mutated protein and any other molecules at each end state, each from
    its own window, with the λ = 1 window superposed on the λ = 0 window by
    the protein's Cα atoms.
    """

    def ca(saved):
        return _np.concatenate(
            [
                _coordinates(saved, topology, m, atoms=m["ca"])
                for m in topology["proteins"]
            ]
        )

    # At least three atoms are needed to superpose the windows.
    if saved1 is not saved0 and len(ca(saved0)) < 3:
        saved1 = saved0

    frames = []
    for state, saved in enumerate((saved0, saved1)):
        proteins = [
            _coordinates(saved, topology, m, state) for m in topology["proteins"]
        ]
        others = [_coordinates(saved, topology, m, state) for m in topology["others"]]
        _image(saved.box, proteins, others)
        frames.append((proteins, others))

    if saved1 is not saved0:
        fit = _superpose(ca(saved1), ca(saved0))
        frames[1] = tuple([fit(x) for x in group] for group in frames[1])

    def mutated_positions(state):
        proteins = frames[state][0]
        return _np.concatenate(
            [
                positions[[r in m["mutated"] for r in m["states"][state]["residues"]]]
                for m, positions in zip(topology["proteins"], proteins)
            ]
        )

    # Only the other molecules near the mutation are shown, e.g. so that a
    # membrane or a distant cofactor isn't.
    mutated = mutated_positions(0)
    shown = [
        i
        for i, positions in enumerate(frames[0][1])
        if _np.linalg.norm(positions[:, None, :] - mutated[None, :, :], axis=-1).min()
        < _NEARBY
    ]

    # Residues near the mutated residues or the other molecules shown at
    # either end state, so that the same ones are shown at both.
    nearby = [set() for _ in topology["proteins"]]
    for state, (proteins, others) in enumerate(frames):
        near = _np.concatenate([mutated_positions(state)] + [others[i] for i in shown])
        for residues, molecule, positions in zip(
            nearby, topology["proteins"], proteins
        ):
            distances = _np.linalg.norm(
                positions[:, None, :] - near[None, :, :], axis=-1
            ).min(axis=1)
            atoms = molecule["states"][state]
            residues.update(
                atoms["residues"][i] for i in _np.where(distances < _CUTOFF)[0]
            )
    for residues, molecule in zip(nearby, topology["proteins"]):
        residues -= molecule["mutated"]

    states = []
    for state, (proteins, others) in enumerate(frames):
        lines = []
        mutated = []
        neighbours = []
        serial = 0
        for residues, molecule, positions in zip(
            nearby, topology["proteins"], proteins
        ):
            atoms = molecule["states"][state]
            for record, residue, xyz in zip(
                atoms["records"], atoms["residues"], positions
            ):
                serial += 1
                if residue in molecule["mutated"]:
                    mutated.append(serial)
                elif residue in residues:
                    neighbours.append(serial)
                lines.append(_atom("ATOM", serial, record, molecule["chain"], xyz))
            lines.append("TER")
        ligand = []
        for i in shown:
            records = topology["others"][i]["states"][state]["records"]
            for record, xyz in zip(records, others[i]):
                serial += 1
                ligand.append(serial)
                lines.append(_ligand_atom(serial, record, i, xyz))
        lines.append("END")
        states.append(
            {
                "pdb": "\n".join(lines) + "\n",
                "mutated": mutated,
                "nearby": neighbours,
                "ligand": ligand,
            }
        )

    return {
        "kind": "mutation",
        "states": states,
        "residues": topology["labels"],
        "separate_frames": saved1 is not saved0,
        "time_ps": saved0.time_ps,
    }


def _coordinates(saved, topology, molecule, state=0, atoms=None):
    """
    The coordinates of the atoms of a molecule shown at an end state, or of
    the given atoms.
    """
    positions = saved.molecule(
        molecule["first"], molecule["num_atoms"], topology["num_system_atoms"]
    )
    if atoms is None:
        atoms = molecule["states"][state]["atoms"]
    return positions[atoms]


def _image(box, proteins, others):
    """
    Move each of the other molecules into the periodic image nearest to the
    protein, since coordinates aren't wrapped into the box.
    """
    if not _is_periodic(box) or not proteins:
        return
    box = _np.asarray(box, dtype=float)
    inverse = _np.linalg.inv(box)
    centre = _np.concatenate(proteins).mean(axis=0)
    for positions in others:
        shift = _np.round((positions.mean(axis=0) - centre) @ inverse) @ box
        positions -= shift


def _superpose(mobile, target):
    """
    The function that best fits a set of coordinates onto another, by
    rotation and translation.
    """
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    u, _, vt = _np.linalg.svd((mobile - mobile_centre).T @ (target - target_centre))
    # Avoids a reflection.
    d = _np.sign(_np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ _np.diag([1.0, 1.0, d]) @ u.T
    return lambda x: (x - mobile_centre) @ rotation.T + target_centre


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


def _ligand_atom(serial, record, index, xyz):
    """
    A HETATM record for an atom of a perturbed or other molecule, which are
    numbered as residues of their own chain.
    """
    name, residue, _, element = record
    record = (name, residue, _LIGAND_RESIDUE - index, element)
    return _atom("HETATM", serial, record, _LIGAND_CHAIN, xyz)


def _topology(topology0, topology1):
    """
    The atoms of each protein, without hydrogens, and of each perturbed
    molecule, or for a mutated protein, any other molecules, at each end
    state, read once for each pair of topologies. None if there is nothing to
    show.
    """
    from ._summary import _file_key

    key = (_file_key(topology0), _file_key(topology1))
    with _topologies_lock:
        if key in _topologies:
            _topologies.move_to_end(key)
            return _topologies[key]

    import sire as _sr

    from ._data import sire_lock
    from ._depict import (
        _MAX_ATOMS,
        _mutations,
        _non_water,
        _perturbed_indices,
        _protein_indices,
    )

    with sire_lock:
        system0 = _sr.load(str(topology0), show_warnings=False)
        system1 = _sr.load(str(topology1), show_warnings=False)
        atoms = system0.atoms()
        mols0 = system0.molecules()
        mols1 = system1.molecules()

        # Chains are only labels, but the other molecules' is kept for them.
        chains = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != _LIGAND_CHAIN]

        def entry(index, protein):
            mol0, mol1 = mols0[index], mols1[index]
            return {
                "first": atoms.find(mol0.atoms()[0]),
                "num_atoms": mol0.num_atoms(),
                "states": [_atoms(mol0, protein), _atoms(mol1, protein)],
            }

        proteins = sorted(_protein_indices(system0))
        # Mutated proteins and peptides, e.g. a peptide ligand, are shown with
        # any other protein, e.g. a receptor.
        mutated = _mutations(system0, system1, proteins)

        topology = None
        if mutated:
            # Other molecules, but not water or ions, e.g. a ligand.
            others = sorted(
                i
                for i in _non_water(system0) - set(proteins)
                if 3 < mols0[i].num_atoms() <= _MAX_ATOMS
            )
            topology = {
                "kind": "mutation",
                "proteins": [
                    dict(
                        entry(i, True),
                        chain=chains[n % len(chains)],
                        mutated=mutated.get(i, set()),
                        ca=_common_ca(mols0[i], mols1[i]),
                    )
                    for n, i in enumerate(proteins)
                ],
                "others": [entry(i, False) for i in others],
                "labels": [
                    _label(mols0[i], mols1[i], r)
                    for i in sorted(mutated)
                    for r in sorted(mutated[i])
                ],
            }
        else:
            ligands = _perturbed_indices(system0, system1, mutated)
            # E.g. a decoupled peptide isn't also drawn as part of the protein.
            receptors = [i for i in proteins if i not in ligands]
            # Nothing to centre on otherwise.
            if ligands and receptors:
                topology = {
                    "kind": "ligand",
                    "proteins": [
                        dict(entry(i, True), chain=chains[n % len(chains)])
                        for n, i in enumerate(receptors)
                    ],
                    "others": [entry(i, False) for i in ligands],
                }
        if topology is not None:
            topology["num_system_atoms"] = len(atoms)

    with _topologies_lock:
        _topologies[key] = topology
        while len(_topologies) > _MAX_TOPOLOGIES:
            _topologies.popitem(last=False)
    return topology


def _atoms(mol, protein):
    """
    The atoms of an end state of a molecule that are shown: their indices in
    the molecule, PDB records, and residue indices. Dummies are left out, as
    are the hydrogens of a protein.
    """
    excluded = ("H", "Xx") if protein else ("Xx",)
    indices, records, residues = [], [], []
    for r, residue in enumerate(mol.residues()):
        # A protein's residues keep their own names, but each other molecule
        # is labelled as a whole.
        name = (residue if protein else mol.residues()[0]).name().value()
        number = residue.number().value() if protein else 0
        for atom in residue.atoms():
            element = atom.element().symbol()
            if element in excluded:
                continue
            indices.append(atom.index().value())
            records.append((atom.name().value(), name, number, element))
            residues.append(r)
    return {"atoms": indices, "records": records, "residues": residues}


def _common_ca(mol0, mol1):
    """
    The indices of a protein's Cα atoms that are present at both end states.
    """
    elements1 = mol1.property("element").to_list()
    return [
        atom.index().value()
        for atom in mol0.atoms()
        if atom.name().value() == "CA"
        and atom.element().symbol() == "C"
        and elements1[atom.index().value()].symbol() == "C"
    ]


def _label(mol0, mol1, residue):
    """
    A label for a mutated residue, with its name at each end state.
    """
    r0 = mol0.residues()[residue]
    r1 = mol1.residues()[residue]
    name0, name1 = r0.name().value(), r1.name().value()
    names = name0 if name0 == name1 else f"{name0} → {name1}"
    return f"{names} {r0.number().value()}"
