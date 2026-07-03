import math
import tempfile

import pytest

from somd2.config import Config
from somd2.runner import Runner


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_ions"])
def test_alchemical_ions(mols, request):
    """Ensure that alchemical ions are added correctly."""

    # Clone the system.
    mols = request.getfixturevalue(mols).clone()

    # Add 10 Cl- ions.
    new_mols, _, ion_indices = Runner._create_alchemical_ions(mols, 10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), -10.0, rel_tol=1e-6)

    # Make sure there is one perturbable-molecule index per ion.
    assert len(ion_indices) == 10

    # Add 10 Na+ ions.
    new_mols, _, ion_indices = Runner._create_alchemical_ions(mols, -10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), 10.0, rel_tol=1e-6)
    assert len(ion_indices) == 10


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
