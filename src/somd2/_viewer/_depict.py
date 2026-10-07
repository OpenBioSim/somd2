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
RDKit depictions of the perturbed molecules at each end state.
"""

__all__ = ["depict"]

# Larger molecules, e.g. perturbable proteins, are too slow to assign bond
# orders for.
_MAX_ATOMS = 400

# Molecules with up to this many heavy atoms are drawn with their hydrogens,
# since a skeletal formula of e.g. methane is just a label.
_MAX_HEAVY_WITH_HYDROGENS = 1

# The number of bonds of a positively charged N, O, P or S, by atomic number.
_ONIUM_DEGREE = {7: 4, 8: 3, 15: 4, 16: 3}

# The bond length in pixels, so that small molecules aren't stretched to fill
# the panel. Larger molecules are scaled down to fit. The side-by-side mapping
# drawings are larger, since they show every atom with its index.
_BOND_LENGTH = 35
_MAPPING_BOND_LENGTH = 50


def _panel_size(num_atoms, scale=70, minimum=200, maximum=640):
    """
    The width of a depiction panel in pixels. Scaling with the square root of
    the number of atoms keeps atoms and labels a similar size across
    molecules.
    """
    return int(min(maximum, max(minimum, scale * num_atoms**0.5)))


def _mapping_panel_size(num_atoms):
    """
    The width of each panel of a side-by-side mapping drawing in pixels.
    """
    return _panel_size(num_atoms, scale=90, minimum=300, maximum=720)


def _set_options(drawer, bond_length=_BOND_LENGTH):
    """
    Drawing options shared by all depictions.
    """
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fixedBondLength = bond_length
    # Label terminal carbons, e.g. CH3-OH rather than a bare line to OH.
    options.explicitMethyl = True
    return options


def depict(topology0, topology1):
    """
    Depict the perturbed molecules in a pair of end-state topologies.

    The end-state topologies hold the merged molecules, with ghost atoms as
    dummies, so atoms correspond by index between the two.

    Parameters
    ----------

    topology0: str
        The topology of the system at lambda = 0.

    topology1: str
        The topology of the system at lambda = 1.

    Returns
    -------

    molecules: [dict]
        An SVG depiction and SMILES string for each end state of each
        perturbed molecule, along with a summary of the mapping.
    """
    from rdkit import Chem

    results = []
    for p in _perturbed_molecules(topology0, topology1):
        num_atoms = max(p.rdmol0.GetNumAtoms(), p.rdmol1.GetNumAtoms())
        pixels = _panel_size(num_atoms)
        mapping_pixels = _mapping_panel_size(num_atoms)
        results.append(
            {
                "name": p.name,
                "num_atoms": len(p.dummy0),
                "num_mapped": sum(
                    1 for a, b in zip(p.dummy0, p.dummy1) if not a and not b
                ),
                "num_unique0": len(p.unique0),
                "num_unique1": len(p.unique1),
                "num_changed": len(p.changed),
                "bond_orders0": p.bond_orders0,
                "bond_orders1": p.bond_orders1,
                "smiles0": Chem.MolToSmiles(Chem.RemoveHs(p.rdmol0, sanitize=False)),
                "smiles1": Chem.MolToSmiles(Chem.RemoveHs(p.rdmol1, sanitize=False)),
                "pixels": pixels,
                "mapping_pixels": mapping_pixels,
                "svg_mapping": _draw_mapping(
                    p.rdmol0, p.rdmol1, p.mapping, p.map0, p.map1, mapping_pixels
                ),
                "svg0": _draw(p.rdmol0, pixels),
                "svg1": _draw(p.rdmol1, pixels),
            }
        )

    return results


def _perturbed_molecules(topology0, topology1):
    """
    The perturbed molecules in a pair of end-state topologies, with each end
    state converted to RDKit.

    Each is a namespace holding its name, its index and first atom index in
    the system, the number of atoms in the system, which atoms are dummies,
    unique or change element, the RDKit molecule of each end state with the
    merged molecule index of each of its atoms (map0, map1), the mapping
    between the two, and whether bond orders could be assigned.
    """
    from types import SimpleNamespace

    import sire as _sr

    system0 = _sr.load(str(topology0), show_warnings=False)
    system1 = _sr.load(str(topology1), show_warnings=False)

    mols0 = system0.molecules()
    mols1 = system1.molecules()

    offsets = [0]
    for mol in mols0:
        offsets.append(offsets[-1] + mol.num_atoms())

    for index in _perturbed_indices(system0, system1):
        mol0 = mols0[index]
        mol1 = mols1[index]
        charges0 = [q.value() for q in mol0.property("charge").to_list()]
        charges1 = [q.value() for q in mol1.property("charge").to_list()]

        dummy0 = [e.num_protons() == 0 for e in mol0.property("element").to_list()]
        dummy1 = [e.num_protons() == 0 for e in mol1.property("element").to_list()]
        elements0 = [e.num_protons() for e in mol0.property("element").to_list()]
        elements1 = [e.num_protons() for e in mol1.property("element").to_list()]

        # The total charge is needed to assign bond orders. Partial charges can
        # be zeroed at a decoupled end state, so also try the other end state.
        charge0 = round(sum(charges0))
        charge1 = round(sum(charges1))
        rdmol0, map0, bond_orders0 = _to_rdkit(mol0, dummy0, [charge0, charge1])
        rdmol1, map1, bond_orders1 = _to_rdkit(mol1, dummy1, [charge1, charge0])

        # Map between the atoms of the two end states via the merged molecule.
        index1 = {orig: i for i, orig in enumerate(map1)}

        yield SimpleNamespace(
            name=f"{mol0.residues()[0].name().value()} (molecule {index})",
            index=index,
            offset=offsets[index],
            num_system_atoms=offsets[-1],
            dummy0=dummy0,
            dummy1=dummy1,
            unique0={i for i, d in enumerate(dummy0) if not d and dummy1[i]},
            unique1={i for i, d in enumerate(dummy1) if not d and dummy0[i]},
            changed={
                i
                for i in range(len(elements0))
                if not dummy0[i] and not dummy1[i] and elements0[i] != elements1[i]
            },
            rdmol0=rdmol0,
            rdmol1=rdmol1,
            map0=map0,
            map1=map1,
            mapping={i: index1[o] for i, o in enumerate(map0) if o in index1},
            bond_orders0=bond_orders0,
            bond_orders1=bond_orders1,
        )


def _perturbed_indices(system0, system1):
    """
    The indices of the molecules that change between the end-state systems,
    leaving out ions and molecules too large to depict, e.g. proteins, whose
    mutations are shown in 3D instead.
    """
    # Molecules are paired by index, since alchemical ions are only water at
    # one end state.
    candidates = sorted(_non_water(system0) & _non_water(system1))
    mols0 = system0.molecules()
    mols1 = system1.molecules()

    indices = []
    for index in candidates:
        mol0 = mols0[index]
        mol1 = mols1[index]
        if mol0.num_atoms() <= 3 or mol0.num_atoms() > _MAX_ATOMS:
            continue
        types0 = mol0.property("ambertype").to_list()
        types1 = mol1.property("ambertype").to_list()
        charges0 = [q.value() for q in mol0.property("charge").to_list()]
        charges1 = [q.value() for q in mol1.property("charge").to_list()]
        if types0 == types1 and all(
            abs(a - b) < 1e-6 for a, b in zip(charges0, charges1)
        ):
            continue
        indices.append(index)
    return indices


def _protein_indices(system):
    """
    Return the indices of the molecules in a system that are proteins.
    """
    try:
        numbers = {mol.number() for mol in system.molecules("protein")}
    except KeyError:
        return set()
    return {i for i, mol in enumerate(system.molecules()) if mol.number() in numbers}


def _non_water(system):
    """
    Return the indices of the molecules in a system that aren't water.
    """
    numbers = {mol.number() for mol in system.molecules("not water")}
    return {i for i, mol in enumerate(system.molecules()) if mol.number() in numbers}


def _assign_bond_orders(rdmol, charges):
    """
    Assign bond orders from connectivity alone, trying each candidate total
    charge until a chemically sensible structure is found.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdDetermineBonds

    # Failed attempts are expected, so don't report them.
    RDLogger.DisableLog("rdApp.*")

    template = Chem.RWMol(rdmol)
    for bond in template.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in template.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumRadicalElectrons(0)
        if atom.GetAtomicNum() > 1:
            atom.SetNoImplicit(True)

    for charge in dict.fromkeys(charges + [0, 1, -1, 2, -2]):
        mol = Chem.Mol(template)
        try:
            rdDetermineBonds.DetermineBondOrders(
                mol, charge=charge, allowChargedFragments=True, embedChiral=False
            )
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        if all(
            atom.GetNumRadicalElectrons() == 0
            and (atom.GetFormalCharge() == 0 or atom.GetAtomicNum() not in (6, 1))
            for atom in mol.GetAtoms()
        ):
            return mol, True

    # Fall back to the connectivity alone, charging atoms with an extra bond,
    # e.g. a protonated amine, which RDKit would otherwise reject.
    mol = template.GetMol()
    for atom in mol.GetAtoms():
        if atom.GetDegree() == _ONIUM_DEGREE.get(atom.GetAtomicNum()):
            atom.SetFormalCharge(1)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol, False


def _to_rdkit(mol, dummy, charges):
    """
    Convert the real atoms of an end state to RDKit.

    Returns the RDKit molecule, the merged molecule index of each atom, and
    whether bond orders could be assigned.
    """
    import sire as _sr

    real = [i for i, d in enumerate(dummy) if not d]

    # Drop the parameters, which can't be carried over to a subset of atoms.
    mol = mol.edit().remove_property("parameters").commit()
    if len(real) < len(dummy):
        mol = mol.atoms("not element Xx").extract()

    rdmol = _sr.convert.to_rdkit(mol, determine_bond_orders=False)
    rdmol, has_bond_orders = _assign_bond_orders(rdmol, charges)
    return rdmol, real, has_bond_orders


def _draw_mapping(rdmol0, rdmol1, mapping, map0, map1, pixels):
    """
    Draw the two end states side by side as SVG, using the same highlighting
    scheme as BioSimSpace.Align.viewMapping. Atoms are labelled with their
    index in the merged molecule, so the same atom has the same label in both.
    """
    from BioSimSpace.Align._align import _get_unique_bonds_and_atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    red = (220 / 255, 50 / 255, 32 / 255, 1.0)
    blue = (0.0, 90 / 255, 181 / 255, 1.0)

    uniques0 = _get_unique_bonds_and_atoms(mapping, rdmol0, rdmol1)
    uniques1 = _get_unique_bonds_and_atoms(
        {v: k for k, v in mapping.items()}, rdmol1, rdmol0
    )

    mols = []
    for rdmol, labels in ((rdmol0, map0), (rdmol1, map1)):
        mol = Chem.Mol(rdmol)
        for atom, label in zip(mol.GetAtoms(), labels):
            atom.SetProp("atomNote", str(label))
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
    AllChem.AlignMol(mols[1], mols[0], atomMap=[(v, k) for k, v in mapping.items()])

    height = int(0.75 * pixels)
    drawer = rdMolDraw2D.MolDraw2DSVG(2 * pixels, height, pixels, height)
    options = _set_options(drawer, _MAPPING_BOND_LENGTH)
    options.continuousHighlight = False
    options.setHighlightColour(red)
    drawer.DrawMolecules(
        mols,
        highlightAtoms=[
            uniques0["atoms"] | uniques0["elements"],
            uniques1["atoms"] | uniques1["elements"],
        ],
        highlightBonds=[
            uniques0["bond_deletions"] | uniques0["bond_changes"],
            uniques1["bond_deletions"] | uniques1["bond_changes"],
        ],
        highlightAtomColors=[
            {i: blue for i in uniques0["elements"]},
            {i: blue for i in uniques1["elements"]},
        ],
        highlightBondColors=[
            {i: blue for i in uniques0["bond_changes"]},
            {i: blue for i in uniques1["bond_changes"]},
        ],
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg") :]


def _draw(rdmol, pixels):
    """
    Draw a single end state as SVG, without hydrogens unless the molecule is
    very small.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    if rdmol.GetNumHeavyAtoms() > _MAX_HEAVY_WITH_HYDROGENS:
        try:
            rdmol = Chem.RemoveHs(rdmol)
        except Exception:
            rdmol = Chem.RemoveHs(rdmol, sanitize=False)

    drawer = rdMolDraw2D.MolDraw2DSVG(pixels, int(0.75 * pixels))
    _set_options(drawer)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, rdmol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg") :]
