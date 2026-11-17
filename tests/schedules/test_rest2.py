import math

import pytest

from somd2._utils._schedules import rest2_scale_factors


@pytest.mark.parametrize("profile", ["linear", "exponential"])
def test_rest2_scale_factors_ends(profile):
    """Both profiles are 1.0 at the end states and the scale at lambda = 0.5."""
    factors = rest2_scale_factors(4.0, [0.0, 0.25, 0.5, 0.75, 1.0], profile)
    assert factors[0] == pytest.approx(1.0)
    assert factors[2] == pytest.approx(4.0)
    assert factors[4] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(factors[3])


def test_rest2_scale_factors_profiles():
    """The linear profile rises in equal steps, the exponential one by equal ratios."""
    lambdas = [i / 10 for i in range(6)]

    linear = rest2_scale_factors(4.0, lambdas, "linear")
    steps = [b - a for a, b in zip(linear, linear[1:])]
    assert steps == pytest.approx([steps[0]] * len(steps))

    exponential = rest2_scale_factors(4.0, lambdas, "exponential")
    ratios = [b / a for a, b in zip(exponential, exponential[1:])]
    assert ratios == pytest.approx([4.0 ** (1 / 5)] * len(ratios))
    assert math.prod(ratios) == pytest.approx(4.0)


def test_rest2_scale_factors_unknown_profile():
    with pytest.raises(ValueError):
        rest2_scale_factors(2.0, [0.0, 1.0], "cubic")
