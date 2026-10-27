import tempfile

import numpy as np
import pytest
import sire as sr

from somd2.config import Config
from somd2.runner import RepexRunner, Runner
from somd2.runner._samplers._reservoir import DCDReader, rebuild_ligand


def _angle(a, b, c):
    u, v = a - b, c - b
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def _boresch(anchors, ligand):
    p1, p2, p3 = anchors
    l1, l2, l3 = ligand[:3]
    return np.array(
        [
            np.linalg.norm(l1 - p1),
            _angle(p2, p1, l1),
            _angle(p1, l1, l2),
            _dihedral(p3, p2, p1, l1),
            _dihedral(p2, p1, l1, l2),
            _dihedral(p1, l1, l2, l3),
        ]
    )


def test_rebuild_ligand_preserves_boresch():
    """The rebuilt ligand has the same Boresch coordinates and shape."""
    rng = np.random.default_rng(42)
    box = np.diag([5.0, 5.0, 5.0])

    anchors_old = rng.uniform(1.0, 2.0, (3, 3))
    ligand = anchors_old[0] + rng.uniform(0.3, 0.6, (6, 3))

    # The new anchors have a different internal geometry, in a new place.
    anchors_new = rng.uniform(2.5, 3.5, (3, 3))

    # Store the old ligand one periodic image away from the anchors.
    shifted = ligand + box[0]

    new_ligand = rebuild_ligand(shifted, anchors_old, box, anchors_new, box)

    assert np.allclose(_boresch(anchors_old, ligand), _boresch(anchors_new, new_ligand))

    old_dists = np.linalg.norm(ligand[:, None] - ligand[None], axis=-1)
    new_dists = np.linalg.norm(new_ligand[:, None] - new_ligand[None], axis=-1)
    assert np.allclose(old_dists, new_dists)


def test_read_sire_dcd(ethane_methanol):
    """Frames written by Sire are read in nanometers, with the box vectors."""
    mols = ethane_methanol.clone()
    mols.save_frame()

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/reservoir.dcd"
        sr.save(mols.trajectory(), filename, format=["DCD"])

        dcd = DCDReader(filename)
        assert dcd.num_frames == 1
        assert dcd.num_atoms == mols.num_atoms()
        coords, box = dcd.read(0)

    expected = np.array(
        [
            [c.x().value(), c.y().value(), c.z().value()]
            for c in (atom.coordinates() for atom in mols.atoms())
        ]
    )
    assert np.allclose(coords, expected / 10.0, atol=1e-3)

    dims = mols.space().dimensions()
    assert np.allclose(box, np.diag([d.value() / 10.0 for d in dims]), atol=1e-3)


def test_read_openmm_dcd():
    """Frames written by OpenMM, which stores the box angles as cosines."""
    import openmm.app as app
    from openmm import unit

    num_atoms = 10
    topology = app.Topology()
    residue = topology.addResidue("RES", topology.addChain())
    for i in range(num_atoms):
        topology.addAtom(f"A{i}", None, residue)

    rng = np.random.default_rng(42)
    frames = [rng.uniform(0.0, 3.0, (num_atoms, 3)) for _ in range(3)]
    box = np.array([[3.0, 0.0, 0.0], [1.0, 2.8, 0.0], [-1.0, 1.2, 2.5]])
    topology.setPeriodicBoxVectors(box * unit.nanometer)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/reservoir.dcd"
        with open(filename, "wb") as f:
            dcd = app.DCDFile(f, topology, 0.001)
            for positions in frames:
                dcd.writeModel(
                    positions * unit.nanometer,
                    periodicBoxVectors=box * unit.nanometer,
                )

        reader = DCDReader(filename)
        assert reader.num_frames == 3

        coords, read_box = reader.read(2)

    assert np.allclose(coords, frames[2], atol=1e-5)
    assert np.allclose(read_box, box, atol=1e-5)


def test_missing_reservoir_raises():
    with pytest.raises(ValueError, match="does not exist"):
        Config(reservoir="missing.dcd")


def test_runner_rejects_reservoir(ethane_methanol):
    with tempfile.NamedTemporaryFile(suffix=".dcd") as f:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(output_directory=tmpdir, platform="cpu", reservoir=f.name)
            with pytest.raises(ValueError, match="requires 'replica_exchange'"):
                Runner(ethane_methanol, config)


def test_reservoir_requires_abfe_bound(ethane_methanol):
    with tempfile.NamedTemporaryFile(suffix=".dcd") as f:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                output_directory=tmpdir,
                platform="cpu",
                replica_exchange=True,
                reservoir=f.name,
            )
            with pytest.raises(ValueError, match="bound leg of an ABFE"):
                RepexRunner(ethane_methanol, config)


def test_reservoir_mapping_with_alchemical_ions(
    abfe_charge_change_mols, ethane_methanol
):
    """
    Alchemical ions are appended to the reservoir in their decoupled state
    form, with their dummy atoms moving with them.
    """
    mols = ethane_methanol.clone()
    mols.save_frame()

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/reservoir.dcd"
        sr.save(mols.trajectory(), filename, format=["DCD"])

        config = Config(
            output_directory=f"{tmpdir}/output",
            platform="cpu",
            lambda_schedule="annihilate",
        )
        runner = Runner(abfe_charge_change_mols.clone(), config)
        runner._config.reservoir = filename

        # The trajectory has the wrong number of atoms, but the mapping is
        # worked out first.
        with pytest.raises(ValueError, match="atoms, but"):
            runner._check_reservoir()

    system = runner._system
    atoms = system.atoms()
    ions = system.molecules("property is_alchemical_ion")
    assert ions.num_molecules() > 0

    ligand = [
        mol
        for mol in system.molecules("property is_perturbable")
        if not mol.has_property("is_alchemical_ion")
    ][0]
    assert runner._reservoir_ligand_idxs == atoms.find(ligand.atoms())

    num_dummies = sum(len(d) for _, d in runner._reservoir_dummy_groups)
    assert len(runner._reservoir_idxs) == (
        system.num_atoms() - ligand.num_atoms() - num_dummies
    )
    assert len(set(runner._reservoir_idxs)) == len(runner._reservoir_idxs)

    # Each ion's real atoms are at the end, in order, and its dummies follow
    # its first real atom.
    num_appended = 0
    for mol in ions:
        idxs = atoms.find(mol.atoms())
        real = [
            i
            for i, atom in zip(idxs, mol.atoms())
            if atom.property("element1").num_protons() > 0
        ]
        num_appended += len(real)
        if len(real) < len(idxs):
            assert (real[0], [i for i in idxs if i not in real]) in [
                (p, list(d)) for p, d in runner._reservoir_dummy_groups
            ]
    appended = runner._reservoir_idxs[-num_appended:]
    assert appended == [
        i
        for mol in ions
        for i, atom in zip(atoms.find(mol.atoms()), mol.atoms())
        if atom.property("element1").num_protons() > 0
    ]
