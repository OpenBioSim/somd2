import sire as sr

from somd2._utils._perturbed import perturbed_atoms, perturbed_selection


def test_perturbed_atoms(ethane_methanol):
    """
    Make sure that atoms whose element changes, and heavy atoms whose LJ
    parameters change, are perturbed, but not hydrogens whose LJ parameters
    change.
    """
    mol = ethane_methanol.molecules("property is_perturbable")[0]
    indices = perturbed_atoms(mol)

    hydrogens = []
    for i, atom in enumerate(mol.atoms()):
        element0 = atom.property("element0").symbol()
        element1 = atom.property("element1").symbol()
        lj_changes = atom.property("LJ0") != atom.property("LJ1")
        changed = element0 != element1 or (element0 != "H" and lj_changes)
        assert (i in indices) == changed
        if element0 == element1 == "H" and lj_changes:
            hydrogens.append(i)

    # Some hydrogens change LJ parameters without being perturbed.
    assert hydrogens
    assert 0 < len(indices) < mol.num_atoms()


def test_perturbed_atoms_decoupled(abfe_charge_change_mols):
    """
    Make sure that every heavy atom of a decoupled ligand is perturbed.
    """
    mol = abfe_charge_change_mols.molecules("property is_perturbable")[0]
    heavy = [
        i
        for i, atom in enumerate(mol.atoms())
        if atom.property("element0").symbol() != "H"
    ]
    assert perturbed_atoms(mol) == heavy


def test_perturbed_selection_reloaded(ethane_methanol, tmp_path):
    """
    Make sure that the selection picks the same atoms from the system reloaded
    from file, as the viewer does, which has new molecule numbers.
    """
    selection = perturbed_selection(ethane_methanol)
    indices = ethane_methanol.atoms().find(ethane_methanol[selection].atoms())

    mol = ethane_methanol.molecules("property is_perturbable")[0]
    offset = ethane_methanol.atoms().find(mol.atoms()[0])
    assert indices == [offset + i for i in perturbed_atoms(mol)]

    reference = sr.morph.extract_reference(ethane_methanol)
    files = sr.save(reference, str(tmp_path / "system"), format=["prm7", "rst7"])
    reloaded = sr.load(files)
    assert reloaded.atoms().find(reloaded[selection].atoms()) == indices


def test_perturbed_selection_skips_perturbable_waters(ethane_methanol):
    """
    Make sure that perturbable waters and single-atom ions, e.g. co-alchemical
    ions, aren't selected alongside the perturbable solute.
    """
    water = ethane_methanol["water"].molecules()[0]
    system = ethane_methanol.clone()
    system.update(water.edit().set_property("is_perturbable", True).commit())

    assert perturbed_selection(system) == perturbed_selection(ethane_methanol)

    # A single-atom molecule, standing in for a perturbable ion.
    ion = water.edit().remove(sr.legacy.Mol.AtomIdx(2)).remove(sr.legacy.Mol.AtomIdx(1))
    ion = ion.commit().edit().renumber().set_property("is_perturbable", True).commit()
    system = ethane_methanol.clone()
    system.add(ion)

    assert perturbed_selection(system) == perturbed_selection(ethane_methanol)


def test_perturbed_selection_bonds(ethane_methanol):
    """
    Make sure the selection extends by the number of bonds, and that two bonds
    include every proper torsion about a bond to a perturbed atom.
    """
    mol = ethane_methanol.molecules("property is_perturbable")[0]
    perturbed = set(perturbed_atoms(mol))

    def selected(bonds):
        selection = perturbed_selection(ethane_methanol, bonds)
        offset = ethane_methanol.atoms().find(mol.atoms()[0])
        atoms = ethane_methanol[selection].atoms()
        return {i - offset for i in ethane_methanol.atoms().find(atoms)}

    assert selected(0) == perturbed
    assert selected(0) <= selected(1) <= selected(2)
    assert selected(2) > perturbed

    region = selected(2)
    info = mol.info()
    for dihedral in mol.property("dihedral0").potentials():
        idx = [
            info.atom_idx(a).value()
            for a in (
                dihedral.atom0(),
                dihedral.atom1(),
                dihedral.atom2(),
                dihedral.atom3(),
            )
        ]
        if {idx[1], idx[2]} & perturbed:
            assert set(idx) <= region


def test_perturbed_selection_ring_break(ring_break_mols):
    """
    Make sure that bonds in either end state are followed, so that a bond only
    present in one end state links its atoms.
    """
    mol = ring_break_mols.molecules("property is_perturbable")[0]
    c0 = mol.property("connectivity0")
    c1 = mol.property("connectivity1")
    from sire.legacy.Mol import AtomIdx

    # A bond that only exists in one end state.
    only = [
        (i, j.value())
        for a, b in ((c0, c1), (c1, c0))
        for i in range(mol.num_atoms())
        for j in a.connections_to(AtomIdx(i))
        if j not in b.connections_to(AtomIdx(i))
    ]
    assert only

    from somd2._utils._perturbed import _neighbours

    # Make the default connectivity one that lacks the bond, so that it can
    # only be found through the end-state connectivities.
    for i, j in only:
        default = c1 if j not in c1.connections_to(AtomIdx(i)) else c0
        edited = mol.edit().set_property("connectivity", default).commit()
        assert j not in [n.value() for n in default.connections_to(AtomIdx(i))]
        assert j in _neighbours(edited, [i], 1)
