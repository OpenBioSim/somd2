import math

from somd2.runner import Runner


def test_hmr(ethane_methanol, ethane_methanol_hmr):
    """Ensure that we can handle systems that have already been repartioned."""

    hmr_factor0 = Runner._get_h_mass_factor(ethane_methanol)
    hmr_factor1 = Runner._get_h_mass_factor(ethane_methanol_hmr)

    # Make sure that the HMR factor is as expected.
    assert math.isclose(hmr_factor0, 1.0, abs_tol=1e-4)
    assert math.isclose(hmr_factor1, 3.0, abs_tol=1e-4)

    # Invert the HMR of the second system.
    new_system = Runner._repartition_h_mass(ethane_methanol_hmr, 1.0 / hmr_factor1)

    hmr_factor2 = Runner._get_h_mass_factor(new_system)

    # Make sure the HMR factor is 1.0 again.
    assert math.isclose(hmr_factor2, 1.0, abs_tol=1e-4)



