import math

from somd2.runner import Runner


def test_hmr(ethane_methanol, ethane_methanol_hmr):
    """Ensure that we can handle systems that have already been repartioned."""

    hmr_factor0, _ = Runner._get_h_mass_factor(ethane_methanol)
    hmr_factor1, _ = Runner._get_h_mass_factor(ethane_methanol_hmr)

    # Make sure that the HMR factor is as expected.
    assert math.isclose(hmr_factor0, 1.0, abs_tol=1e-4)
    assert math.isclose(hmr_factor1, 3.0, abs_tol=1e-4)

    # Invert the HMR of the second system.
    new_system = Runner._repartition_h_mass(ethane_methanol_hmr, 1.0 / hmr_factor1)
    hmr_factor2, _ = Runner._get_h_mass_factor(new_system)

    # Make sure the HMR factor is 1.0 again.
    assert math.isclose(hmr_factor2, 1.0, abs_tol=1e-4)

    # Work out the partial HMR factor if we actually want to use
    # 1.5 as the HMR factor.
    hmr_factor3 = 1.5 / hmr_factor1

    # Now apply the partial HMR factor.
    new_system = Runner._repartition_h_mass(ethane_methanol_hmr, hmr_factor3)
    hmr_factor4, _ = Runner._get_h_mass_factor(new_system)

    # Make sure the HMR factor is 1.5.
    assert math.isclose(hmr_factor4, 1.5, abs_tol=1e-4)
