import math
import pytest
import tempfile

from pathlib import Path

from somd2.config import Config
from somd2.runner import Runner


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_ions"])
def test_alchemical_ions(mols, request):
    """Ensure that alchemical ions are added correctly."""

    # Clone the system.
    mols = request.getfixturevalue(mols).clone()

    # Add 10 Cl- ions.
    new_mols, _, ion_indices, ion_mol_indices = Runner._create_alchemical_ions(mols, 10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), -10.0, rel_tol=1e-6)

    # Make sure there is one perturbable-molecule index per ion.
    assert len(ion_indices) == 10
    assert len(ion_mol_indices) == 10

    # Add 10 Na+ ions.
    new_mols, _, ion_indices, ion_mol_indices = Runner._create_alchemical_ions(
        mols, -10
    )

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), 10.0, rel_tol=1e-6)
    assert len(ion_indices) == 10
    assert len(ion_mol_indices) == 10


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_ions"])
def test_alchemical_ion_mol_indices_reproducible(mols, request):
    """
    Ensure that passing the molecule indices returned by a previous call to
    `_create_alchemical_ions` reproduces the exact same ion(s), bypassing the
    "furthest waters" search entirely. This is what a restart relies on.
    """
    mols = request.getfixturevalue(mols).clone()

    # Pick ions via the heuristic search, recording which molecule(s) were
    # converted.
    heuristic_mols, _, _, ion_mol_indices = Runner._create_alchemical_ions(mols, 3)
    heuristic_ion_numbers = {
        mol.number()
        for mol in heuristic_mols.molecules()["perturbable"].molecules()
        if mol.has_property("is_alchemical_ion")
    }

    # Reuse the stored indices directly - should convert the exact same
    # molecules, without running the search.
    replayed_mols, _, _, replayed_mol_indices = Runner._create_alchemical_ions(
        mols, 3, mol_indices=ion_mol_indices
    )
    replayed_ion_numbers = {
        mol.number()
        for mol in replayed_mols.molecules()["perturbable"].molecules()
        if mol.has_property("is_alchemical_ion")
    }

    assert replayed_ion_numbers == heuristic_ion_numbers
    assert replayed_mol_indices == ion_mol_indices
    assert math.isclose(
        Runner._get_charge_difference(replayed_mols), -3.0, rel_tol=1e-6
    )


@pytest.fixture(scope="module")
def ethane_methanol_sodium(ethane_methanol):
    """The ethane to methanol system with three waters replaced by Na+ ions."""
    from sire.legacy.IO import createSodiumIon

    mols = ethane_methanol.clone()
    for water in mols["water"].molecules()[:3]:
        mols.remove(water)
        mols.add(createSodiumIon(water["element O"].coordinates(), "tip3p"))
    return mols


@pytest.mark.parametrize("charge_diff", [2, 5])
def test_alchemical_ions_from_counter_ions(charge_diff, ethane_methanol_sodium):
    """
    With no free Cl- ion to use as a template, Na+ counter-ions are perturbed
    to water, each one only once, with waters perturbed to Cl- for any charge
    that remains.
    """
    mols = ethane_methanol_sodium.clone()
    sodium_numbers = {mol.number() for mol in mols["element Na"].molecules()}

    new_mols, _, ion_indices, ion_mol_indices = Runner._create_alchemical_ions(
        mols, charge_diff
    )

    assert math.isclose(
        Runner._get_charge_difference(new_mols), -charge_diff, rel_tol=1e-6
    )

    # Each alchemical ion is a distinct molecule.
    all_mols = mols.molecules()
    ion_numbers = [all_mols[idx].number() for idx in ion_mol_indices]
    assert len(set(ion_numbers)) == charge_diff

    # Counter-ions are used before waters.
    num_counter_ions = min(charge_diff, len(sodium_numbers))
    assert len(sodium_numbers & set(ion_numbers)) == num_counter_ions

    # The perturbable-molecule indices point at the alchemical ions.
    perturbable_mols = new_mols.molecules()["perturbable"].molecules()
    assert {perturbable_mols[idx].number() for idx in ion_indices} == set(ion_numbers)
    for idx in ion_indices:
        assert perturbable_mols[idx].has_property("is_alchemical_ion")

    # Each new water is built around its counter-ion.
    space = new_mols.space()
    for number in sodium_numbers & set(ion_numbers):
        centre = mols[number].coordinates()
        for atom in new_mols[number].atoms():
            assert float(space.calc_dist(centre, atom.coordinates())) < 1.5

    # Replaying the stored indices reproduces the same ions.
    _, _, _, replayed_mol_indices = Runner._create_alchemical_ions(
        mols, charge_diff, mol_indices=ion_mol_indices
    )
    assert replayed_mol_indices == ion_mol_indices


def test_alchemical_ion_mol_indices_mismatch_raises(ethane_methanol):
    """A stored index count that doesn't match the charge difference should
    raise a clear error, rather than silently converting the wrong number of
    waters."""
    mols = ethane_methanol.clone()

    with pytest.raises(ValueError, match="does not match the current charge"):
        Runner._create_alchemical_ions(mols, 3, mol_indices=[0, 1])


def test_alchemical_ion_restart_reuses_same_ion(ethane_methanol_ions):
    """
    Ensure that restarting a run picks the exact same alchemical ion as the
    original run, via the persisted `alchemical_ions.npz` file, rather than
    re-running the "furthest waters" search from scratch.
    """
    mols = ethane_methanol_ions.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = dict(
            output_directory=tmpdir,
            platform="cpu",
            charge_difference=1,
        )

        # Fresh run: picks an ion via the heuristic search and persists its
        # molecule index to alchemical_ions.npz.
        runner1 = Runner(mols.clone(), Config(restart=False, **base_config))
        ion_number_1 = next(
            mol.number()
            for mol in runner1._system.molecules()["perturbable"].molecules()
            if mol.has_property("is_alchemical_ion")
        )

        assert (Path(tmpdir) / "alchemical_ions.npz").exists()

        # "Restart": construct a new Runner against the same input and output
        # directory. It should reuse the stored ion index rather than
        # re-running the search.
        runner2 = Runner(mols.clone(), Config(restart=True, **base_config))
        ion_number_2 = next(
            mol.number()
            for mol in runner2._system.molecules()["perturbable"].molecules()
            if mol.has_property("is_alchemical_ion")
        )

        assert ion_number_1 == ion_number_2


@pytest.mark.parametrize("schedule_name", ["decouple", "annihilate"])
def test_alchemical_ion_abfe_schedule(schedule_name, ethane_methanol_ions):
    """
    Ensure that an alchemical ion added alongside an ABFE (decouple/annihilate)
    perturbable molecule gets its own plain morph schedule, rather than
    inheriting the ghost-atom decoupling/annihilation lever equations meant
    for the main perturbable molecule.
    """
    from sire.cas import LambdaSchedule

    # Clone the system (charge-neutral, but with plenty of waters and existing
    # free ions to match parameters against).
    mols = ethane_methanol_ions.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_directory=tmpdir,
            platform="cpu",
            lambda_schedule=schedule_name,
            # Force a charge difference so that an alchemical ion is added,
            # even though the fixture itself is charge neutral.
            charge_difference=1,
        )

        runner = Runner(mols, config)

        # There should be exactly two perturbable molecules now: the original
        # ligand and the single alchemical ion that was added.
        perturbable_mols = runner._system.molecules()["perturbable"].molecules()
        assert len(perturbable_mols) == 2

        ligand_idx = None
        ion_idx = None
        for i, mol in enumerate(perturbable_mols):
            if mol.has_property("is_alchemical_ion"):
                ion_idx = i
            else:
                ligand_idx = i

        assert ligand_idx is not None
        assert ion_idx is not None

        schedule = runner._config.lambda_schedule

        # The ligand follows the main ABFE schedule directly - no override.
        assert not schedule.has_molecule_schedule(ligand_idx)

        # The ion has its own molecule-specific schedule, and it is a plain
        # morph (not the ligand's decharge/decouple or decharge/annihilate
        # staging).
        assert schedule.has_molecule_schedule(ion_idx)
        ion_schedule = schedule.get_molecule_schedule(ion_idx)
        assert ion_schedule.get_stages() == ["morph"]
        assert ion_schedule.to_string() == LambdaSchedule.standard_morph().to_string()


def test_charge_difference_with_virtual_sites(ethane_methanol):
    """
    Off-site charges are held as a molecule property rather than on the atoms,
    so they are invisible to charge() and must be added separately.
    """

    mols = ethane_methanol.clone()

    # No virtual sites, so the end states have the same charge.
    assert math.isclose(Runner._get_charge_difference(mols), 0.0, abs_tol=1e-6)

    # Give the perturbable molecule off-site charges that differ by one unit
    # between the end states.
    mol = mols.molecules("property is_perturbable")[0]
    cursor = mol.cursor()
    cursor.set("vs_charges0", [0.5, 0.5])
    cursor.set("vs_charges1", [1.0, 1.0])
    mols.update(cursor.commit())

    assert math.isclose(Runner._get_charge_difference(mols), 1.0, abs_tol=1e-6)
