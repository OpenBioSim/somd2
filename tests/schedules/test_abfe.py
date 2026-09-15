import pytest

from somd2._utils._schedules import annihilate, decouple

# Lambda schedules are always symmetric between the two stages (decharge is
# the first stage, annihilate/decouple is the second), so both builders share
# identical expected lever values.
BUILDERS = [annihilate, decouple]

_LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Expected "restraint" lever values (restraint_lever="combined"): ramps
# linearly 0 -> 1 across the whole first stage (decharge), then held at 1.
_COMBINED_RESTRAINT = [0.0, 0.4, 0.8, 1.0, 1.0, 1.0]

# Expected "restraint_dihedral"/"restraint_distance_angle" lever values
# (restraint_lever="split"): a geometric progression from ~0.01 to 1.0 across
# each stage in turn, reproducing the ratio in Table S1 of the RXRX paper's
# SI. dihedral ramps during the first stage then holds at 1; distance_angle
# holds at 0 during the first stage then ramps during the second.
_SPLIT_DIHEDRAL = [0.01, 0.06309573444801934, 0.39810717055349737, 1.0, 1.0, 1.0]
_SPLIT_DISTANCE_ANGLE = [0.0, 0.0, 0.0, 0.025118864315095805, 0.15848931924611143, 1.0]


def _morph(schedule, lever, lambda_value):
    """
    Query a lever's value exactly as SireOpenMM's LambdaLever does at
    runtime: lambda_schedule.morph("*", restraint_name, 1.0, 1.0, lambda_value).
    """
    return schedule.morph("*", lever, 1.0, 1.0, lambda_value)


@pytest.mark.parametrize("builder", BUILDERS)
def test_restraint_lever_defaults_to_split(builder):
    """
    annihilate()/decouple() default to restraint_lever="split", matching
    boresch_search()'s own default of restraint_lever="split" for its
    default protocol="rxrx"
    """
    schedule = builder()
    assert "restraint_dihedral" in schedule.get_levers()
    assert "restraint_distance_angle" in schedule.get_levers()
    assert "restraint" not in schedule.get_levers()


@pytest.mark.parametrize("builder", BUILDERS)
def test_restraint_lever_invalid_raises(builder):
    with pytest.raises(ValueError, match="restraint_lever"):
        builder(restraint_lever="not_a_real_lever")


@pytest.mark.parametrize("builder", BUILDERS)
def test_restraint_lever_combined(builder):
    """
    restraint_lever="combined" sets a single "restraint" lever that ramps
    0 -> 1 across the first stage (decharge), then holds at 1.
    """
    schedule = builder(restraint_lever="combined")
    assert "restraint" in schedule.get_levers()
    assert "restraint_dihedral" not in schedule.get_levers()
    assert "restraint_distance_angle" not in schedule.get_levers()

    values = [_morph(schedule, "restraint", lv) for lv in _LAMBDA_VALUES]
    assert values == pytest.approx(_COMBINED_RESTRAINT, abs=1e-6)


@pytest.mark.parametrize("builder", BUILDERS)
def test_restraint_lever_split(builder):
    """
    restraint_lever="split" sets two independent levers ("restraint_dihedral"
    and "restraint_distance_angle"), each following a geometric progression
    across its own stage while the other is held fixed, reproducing the RXRX
    protocol's staged restraint turn-on.
    """
    schedule = builder(restraint_lever="split")

    dihedral_values = [
        _morph(schedule, "restraint_dihedral", lv) for lv in _LAMBDA_VALUES
    ]
    distance_angle_values = [
        _morph(schedule, "restraint_distance_angle", lv) for lv in _LAMBDA_VALUES
    ]

    assert dihedral_values == pytest.approx(_SPLIT_DIHEDRAL, abs=1e-6)
    assert distance_angle_values == pytest.approx(_SPLIT_DISTANCE_ANGLE, abs=1e-6)


@pytest.mark.skipif(
    "openmm" not in __import__("sire").convert.supported_formats(),
    reason="openmm support is not available",
)
def test_restraint_lever_split_openmm_system():
    """
    End-to-end regression test: build a real dynamics object with a "split"
    Boresch restraint and confirm the two independently-lambda-addressable
    OpenMM Forces (distance/angle and dihedral) exist, with 'rho' values
    that follow the expected geometric progression as lambda changes,
    matching what test_restraint_lever_split checks at the schedule level.
    """
    import xml.etree.ElementTree as ET

    import sire as sr

    mols = sr.load_test_files("boresch_restraints.prm7", "boresch_restraints.dcd")
    mols.update(sr.morph.decouple(mols.molecule(1), as_new_molecule=False))

    restraints = sr.restraints.boresch(
        mols,
        receptor=[692, 702, 704],
        ligand=[1496, 1498, 1499],
        kr="1 kcal mol-1 A-2",
        ktheta=["80 kcal mol-1 rad-2"] * 2,
        kphi=["80 kcal mol-1 rad-2"] * 3,
        r0="4.56908 A",
        theta0=["82.5581 degrees", "94.9595 degrees"],
        phi0=["27.9429 degrees", "125.68 degrees", "-107.008 degrees"],
        angle_potential="restricted_bending",
        restraint_lever="split",
    )

    schedule = decouple(restraint_lever="split")

    d = mols.dynamics(
        timestep="2fs",
        temperature="298 K",
        schedule=schedule,
        lambda_value=0.0,
        map={"restraints": restraints},
    )

    expected_rho = {
        0.0: {"distance_angle": 0.0, "dihedral": 0.01},
        0.5: {"distance_angle": 0.01, "dihedral": 1.0},
        1.0: {"distance_angle": 1.0, "dihedral": 1.0},
    }

    for lam, expected in expected_rho.items():
        d.set_lambda(lam)
        root = ET.fromstring(d.to_xml())

        rhos = {}
        for force in root.iter("Force"):
            if force.get("name") == "BoreschRestraintForce":
                kind = (
                    "distance_angle" if "e_bond" in force.get("energy") else "dihedral"
                )
                rhos[kind] = float(force.find("Bonds")[0].get("param1"))

        assert set(rhos.keys()) == {"distance_angle", "dihedral"}
        assert rhos["distance_angle"] == pytest.approx(
            expected["distance_angle"], abs=1e-6
        )
        assert rhos["dihedral"] == pytest.approx(expected["dihedral"], abs=1e-6)
