import math
import pytest

from somd2.runner import Runner


@pytest.mark.parametrize("mols", ["ethane_methanol", "ethane_methanol_ions"])
def test_alchemical_ions(mols, request):
    """Ensure that alchemical ions are added correctly."""

    # Clone the system.
    mols = request.getfixturevalue(mols).clone()

    # Add 10 Cl- ions.
    new_mols = Runner._create_alchemical_ions(mols, 10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), -10.0, rel_tol=1e-6)

    # Add 10 Na+ ions.
    new_mols = Runner._create_alchemical_ions(mols, -10)

    # Make sure the charge difference is correct.
    assert math.isclose(Runner._get_charge_difference(new_mols), 10.0, rel_tol=1e-6)
