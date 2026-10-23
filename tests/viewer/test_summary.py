import sire as sr

from somd2._viewer._data import Simulation
from somd2._viewer._summary import settings_key
from somd2.config import Config


def test_restraints_compared_by_value(tmp_path, ethane_methanol):
    """
    Make sure that runs with the same restraints have the same settings key,
    even though each serialisation of the restraints differs.
    """
    mols = ethane_methanol.clone()
    ligand = mols[0]

    def config(r0):
        restraint = sr.restraints.bond(
            mols, ligand[0], ligand[1], k="100 kcal mol-1 A-2", r0=r0
        )
        return {"restraints": [Config._to_hex(restraint)]}

    simulation = Simulation(tmp_path, "test")
    keys = [
        settings_key(simulation._comparable(config(r0)), set())
        for r0 in ("1.5 A", "1.5 A", "2.0 A")
    ]

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
