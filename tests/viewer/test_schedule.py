import pytest

from somd2._viewer._data import Simulation


def test_schedule_torsion_window(tmp_path):
    """
    Make sure that the torsion windows recorded in a config are shown as
    per-molecule torsion levers.
    """
    simulation = Simulation(tmp_path, "test")

    plain = simulation._schedule({"lambda_schedule": "standard_morph"}, [0.0, 1.0])
    assert plain["name"] == "standard_morph"
    assert not any("molecule" in lever for lever in plain["levers"])

    config = {"lambda_schedule": "standard_morph", "torsion_windows": "0:0.0-0.5"}
    result = simulation._schedule(config, [0.0, 1.0])
    assert result["name"] == "standard_morph + torsion window"
    assert "molecule 0 [0.0, 0.5]" in result["description"]

    torsion_k = result["levers"]["torsion_k (molecule 0)"]
    lam = result["lambda"]
    values = {round(x, 2): y for x, y in zip(lam, torsion_k)}
    assert values[0.25] == pytest.approx(1.5)
    assert values[0.5] == pytest.approx(2.0)
    assert values[0.75] == pytest.approx(2.0)
    assert result["levers"]["default"][lam.index(0.25)] == pytest.approx(1.25)
