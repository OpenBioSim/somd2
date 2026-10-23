import sire as sr

from somd2._viewer._data import Simulation
from somd2._viewer._summary import leg_settings_key, settings_key
from somd2.config import Config


def test_restraints_compared_by_value(tmp_path, ethane_methanol):
    """
    Make sure that runs with the same restraints have the same settings key,
    even though each serialisation of the restraints differs.
    """
    mols = ethane_methanol.clone()
    ligand = mols[0]

    def config(r0, use_pbc):
        restraint = sr.restraints.bond(
            mols, ligand[0], ligand[1], k="100 kcal mol-1 A-2", r0=r0, use_pbc=use_pbc
        )
        return {"restraints": [Config._to_hex(restraint)]}

    simulation = Simulation(tmp_path, "test")
    keys = [
        settings_key(simulation._comparable(config(r0, use_pbc)), set())
        for r0, use_pbc in (
            ("1.5 A", False),
            ("1.5 A", False),
            ("2.0 A", False),
            ("1.5 A", True),
        )
    ]

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[0] != keys[3]


def test_lambda_schedule_compared_by_value(tmp_path):
    """
    Make sure that runs with the same custom λ schedule have the same settings
    and leg settings keys, and that named schedules are left alone.
    """

    def config(scale):
        schedule = sr.cas.LambdaSchedule.standard_morph()
        schedule.set_equation(
            stage="morph", lever="charge", equation=scale * schedule.lam()
        )
        return {"lambda_schedule": Config._to_hex(schedule)}

    simulation = Simulation(tmp_path, "test")
    configs = [simulation._comparable(config(scale)) for scale in (0.5, 0.5, 0.8)]

    for key in (settings_key, leg_settings_key):
        keys = [key(c, set()) for c in configs]
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]

    named = {"lambda_schedule": "standard_morph"}
    assert simulation._comparable(named) is named


def test_unreadable_values_compared_as_stored(tmp_path):
    """
    Make sure that restraints and schedules that can't be read are compared
    as stored, rather than failing.
    """
    config = {"restraints": ["not a restraint"], "lambda_schedule": "ab" * 32}
    assert Simulation(tmp_path, "test")._comparable(config) == config
