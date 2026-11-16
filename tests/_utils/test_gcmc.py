import pytest
import sire as sr

from somd2._utils._gcmc import (
    _minimum_image,
    _positions,
    auto_selection,
    ghost_waters,
    perturbed_atoms,
    selection_geometry,
    selection_sites,
    vanishing_atoms,
)


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


def test_auto_selection_reloaded(ethane_methanol, tmp_path):
    """
    Make sure that the selection picks the same atoms from the system reloaded
    from file, as the viewer does, which has new molecule numbers.
    """
    selection = auto_selection(ethane_methanol)
    indices = ethane_methanol.atoms().find(ethane_methanol[selection].atoms())

    mol = ethane_methanol.molecules("property is_perturbable")[0]
    offset = ethane_methanol.atoms().find(mol.atoms()[0])
    assert indices == [offset + i for i in perturbed_atoms(mol)]

    reference = sr.morph.extract_reference(ethane_methanol)
    files = sr.save(reference, str(tmp_path / "system"), format=["prm7", "rst7"])
    reloaded = sr.load(files)
    assert reloaded.atoms().find(reloaded[selection].atoms()) == indices


def test_auto_selection_skips_perturbable_waters(ethane_methanol):
    """
    Make sure that perturbable waters and single-atom ions, e.g. co-alchemical
    ions, don't move the GCMC sphere away from the perturbable solute.
    """
    water = ethane_methanol["water"].molecules()[0]
    system = ethane_methanol.clone()
    system.update(water.edit().set_property("is_perturbable", True).commit())

    assert auto_selection(system) == auto_selection(ethane_methanol)

    # A single-atom molecule, standing in for a perturbable ion.
    ion = water.edit().remove(sr.legacy.Mol.AtomIdx(2)).remove(sr.legacy.Mol.AtomIdx(1))
    ion = ion.commit().edit().renumber().set_property("is_perturbable", True).commit()
    system = ethane_methanol.clone()
    system.add(ion)

    assert auto_selection(system) == auto_selection(ethane_methanol)


def test_vanishing_atoms(ethane_methanol, abfe_charge_change_mols):
    """
    Make sure that ghost atoms vanish, as do the atoms of a decoupled ligand.
    """
    mol = ethane_methanol.molecules("property is_perturbable")[0]
    ghosts = [
        i
        for i, atom in enumerate(mol.atoms())
        if "Xx"
        in (atom.property("element0").symbol(), atom.property("element1").symbol())
    ]
    assert ghosts
    assert vanishing_atoms(mol) == ghosts

    mol = abfe_charge_change_mols.molecules("property is_perturbable")[0]
    assert vanishing_atoms(mol) == list(range(mol.num_atoms()))


def test_selection_geometry(ethane_methanol):
    """
    Make sure the extent covers the selected atoms in both end states.
    """
    selection = auto_selection(ethane_methanol)
    centre, extent = selection_geometry(ethane_methanol, selection)
    assert 0 < extent < 5
    assert centre.shape == (3,)

    # A single atom has no extent.
    _, extent = selection_geometry(ethane_methanol, "molidx 0 and atomidx 1")
    assert extent == pytest.approx(0)


def test_selection_geometry_periodic(ethane_methanol):
    """
    Make sure that a molecule split across a periodic boundary has the same
    centre and extent as when it is whole.
    """
    import numpy as np

    selection = auto_selection(ethane_methanol)
    centre, extent = selection_geometry(ethane_methanol, selection)

    # Move one atom of the molecule by a box vector in both end states.
    box = ethane_methanol.property("space")
    shift = sr.maths.Vector(box.dimensions().x().value(), 0, 0)
    mol = ethane_methanol.molecules("property is_perturbable")[0]
    cursor = mol.cursor()
    for name in ("coordinates0", "coordinates1"):
        cursor.atom(0)[name] = cursor.atom(0)[name] + shift
    system = ethane_methanol.clone()
    system.update(cursor.commit())

    # Without imaging, the molecule now spans the box.
    raw = _positions(system, selection, 0)
    assert np.ptp(raw[:, 0]) > box.dimensions().x().value() / 2

    split_centre, split_extent = selection_geometry(system, selection)
    assert split_extent == pytest.approx(extent)

    # The centres are periodic images of each other.
    delta = _minimum_image(np.array([split_centre - centre]), box)
    assert np.linalg.norm(delta) == pytest.approx(0, abs=1e-4)


def test_minimum_image_distances(ethane_methanol):
    """
    Make sure distances use the nearest periodic image.
    """
    import itertools

    import numpy as np

    space = ethane_methanol.property("space")
    d = space.dimensions()
    box = np.diag([d.x().value(), d.y().value(), d.z().value()])

    rng = np.random.default_rng(1)
    points = rng.uniform(-2 * box.max(), 2 * box.max(), (50, 3))
    centre = np.array([1.0, -2.0, 3.0])

    expected = [
        min(
            np.linalg.norm(p - centre + box @ np.array(k))
            for k in itertools.product(range(-3, 4), repeat=3)
        )
        for p in points
    ]
    images = _minimum_image(points - centre, space)
    assert np.allclose(np.linalg.norm(images, axis=1), expected)

    # Each image differs from its displacement by a whole box vector.
    shifts = np.linalg.solve(box, (images - (points - centre)).T)
    assert np.allclose(shifts, np.round(shifts))


def test_selection_sites(ethane_methanol):
    """
    Make sure that nearby atoms form one site, and distant atoms separate sites.
    """
    import numpy as np

    selection = auto_selection(ethane_methanol)
    assert len(selection_sites(ethane_methanol, selection)) == 1

    # Add the oxygen of a water far from the perturbable molecule.
    centre, _ = selection_geometry(ethane_methanol, selection)
    space = ethane_methanol.property("space")
    waters = ethane_methanol["water and element O"].atoms()
    xyz = _positions(ethane_methanol, "water and element O", 0)
    distances = np.linalg.norm(_minimum_image(xyz - centre, space), axis=1)
    far = waters[int(np.argmax(distances > 10))]
    molidx = ethane_methanol.molecules().find(far.molecule())
    two = f"({selection}) or (molidx {molidx} and atomidx {far.index().value()})"

    sites = selection_sites(ethane_methanol, two)
    assert len(sites) == 2
    separation = _minimum_image(np.array([sites[0] - sites[1]]), space)
    assert np.linalg.norm(separation) > 10


def test_ghost_waters(ethane_methanol):
    """
    Make sure the ghost waters cover the free volume of the sphere, which is
    small in water, plus a margin.
    """
    selection = auto_selection(ethane_methanol)
    centre, extent = selection_geometry(ethane_methanol, selection)

    volume = 30.543
    small = ghost_waters(ethane_methanol, centre, extent + 5, volume)
    large = ghost_waters(ethane_methanol, centre, extent + 10, volume)
    assert 10 <= small < large

    # Far fewer than the waters that fill the empty sphere.
    full = 4 / 3 * 3.14159 * (extent + 10) ** 3 / volume
    assert large < full / 2

    # A larger water model fills the sphere with fewer waters.
    assert ghost_waters(ethane_methanol, centre, extent + 10, 2 * volume) < large
