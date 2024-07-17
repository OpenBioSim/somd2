import math

from somd2.runner import Runner


def test_alchemical_ions(ethane_methanol):
    """Ensure that alchemical ions are added correctly."""

    # Clone the system.
    mols = ethane_methanol.clone()

    # Helper function to return the charge difference between the end states.
    def charge_difference(mols):
        from sire import morph

        reference = morph.link_to_reference(mols)
        perturbed = morph.link_to_perturbed(mols)

        return (perturbed.charge() - reference.charge()).value()

    # Add 10 Cl- ions.
    new_mols = Runner._create_alchemical_ions(mols, 10)

    # Make sure the charge difference is correct.
    assert math.isclose(charge_difference(new_mols), -10.0, rel_tol=1e-6)

    # Add 10 Na+ ions.
    new_mols = Runner._create_alchemical_ions(mols, -10)

    # Make sure the charge difference is correct.
    assert math.isclose(charge_difference(new_mols), 10.0, rel_tol=1e-6)
