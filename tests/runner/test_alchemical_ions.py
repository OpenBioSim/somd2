import math

from somd2.runner import Runner


def test_alchemical_ions(ethane_methanol):
    """Ensure that alchemical ions are added correctly."""

    # Clone the system.
    mols = ethane_methanol.clone()

    # Add 10 Cl- ions.
    new_mols = Runner._create_alchemical_ions(mols, 10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), -10.0, rel_tol=1e-6)

    # Add 10 Na+ ions.
    new_mols = Runner._create_alchemical_ions(mols, -10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), 10.0, rel_tol=1e-6)
