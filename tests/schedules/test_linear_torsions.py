import math
import tempfile

import pytest
from sire.cas import LambdaSchedule
from sire.legacy.CAS import Symbol
from sire.legacy.MM import AmberAngle, FourAtomFunctions, ThreeAtomFunctions

from somd2._utils._linear_angles import find_linear_angles, torsion_window
from somd2._utils._schedules import linear_torsions, ring_break_morph


def _torsion_k(schedule, lambda_values):
    return [schedule.morph("*", "torsion_k", 4.0, 0.0, x) for x in lambda_values]


@pytest.mark.parametrize(
    "window, expected",
    [
        ((0.0, 0.5), [4.0, 2.0, 0.0, 0.0, 0.0]),
        ((0.5, 1.0), [4.0, 4.0, 4.0, 2.0, 0.0]),
        ((0.25, 0.75), [4.0, 4.0, 2.0, 0.0, 0.0]),
    ],
)
def test_linear_torsions_window(window, expected):
    """
    Torsions change over the window only, and other levers are unchanged.
    """
    schedule = linear_torsions(LambdaSchedule.standard_morph(), *window)
    assert _torsion_k(schedule, [0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(expected)
    assert schedule.morph("*", "angle_size", 2.0, 4.0, 0.25) == pytest.approx(2.5)


def test_linear_torsions_stages():
    """
    The window applies within each stage in which the torsions change linearly.
    """
    schedule = linear_torsions(ring_break_morph(), 0.0, 0.5)
    values = _torsion_k(schedule, [0.2, 0.4, 0.5, 0.6])
    assert values == pytest.approx([4.0, 2.4, 0.0, 0.0])


def test_linear_torsions_idempotent():
    schedule = linear_torsions(LambdaSchedule.standard_morph(), 0.0, 0.5)
    assert linear_torsions(schedule, 0.0, 0.5) == schedule


def test_linear_torsions_invalid():
    with pytest.raises(ValueError):
        linear_torsions(LambdaSchedule.standard_morph(), 0.5, 0.5)

    schedule = LambdaSchedule.standard_morph()
    schedule.set_equation(
        stage="morph", lever="torsion_k", equation=schedule.lam() ** 2
    )
    with pytest.raises(ValueError):
        linear_torsions(schedule, 0.0, 0.5)


def test_torsion_window():
    assert torsion_window({1}) == (0.0, 0.5)
    assert torsion_window({0}) == (0.5, 1.0)
    assert torsion_window({0, 1}) == (0.25, 0.75)
    with pytest.raises(ValueError):
        torsion_window(set())


def _linearise(mol, angle, state, remove_terms=False):
    """
    Set the equilibrium value of an angle to 180 degrees at an end state, and
    optionally remove the dihedrals and impropers that pass through it.
    """
    info = mol.info()
    theta = Symbol("theta")
    atoms = set(angle)

    angles = ThreeAtomFunctions(info)
    for p in mol.property(f"angle{state}").potentials():
        idx = [info.atom_idx(a) for a in (p.atom0(), p.atom1(), p.atom2())]
        function = p.function()
        if {i.value() for i in idx} == atoms and idx[1].value() == angle[1]:
            k = AmberAngle(function, theta).k()
            function = AmberAngle(k, math.pi).to_expression(theta)
        angles.set(*idx, function)

    cursor = mol.edit().set_property(f"angle{state}", angles).molecule()

    if remove_terms:
        for name in ("dihedral", "improper"):
            for s in (0, 1):
                terms = FourAtomFunctions(info)
                for p in mol.property(f"{name}{s}").potentials():
                    idx = [
                        info.atom_idx(a)
                        for a in (p.atom0(), p.atom1(), p.atom2(), p.atom3())
                    ]
                    values = [i.value() for i in idx]
                    if atoms in (set(values[:3]), set(values[1:])):
                        continue
                    terms.set(*idx, p.function())
                cursor = cursor.set_property(f"{name}{s}", terms).molecule()

    return cursor.commit()


def _spanned_angle(mol):
    """
    Return an angle of the molecule that a non-zero dihedral passes through at
    both end states.
    """
    info = mol.info()
    for p in mol.property("dihedral0").potentials():
        if p.function().is_zero():
            continue
        idx = [info.atom_idx(a).value() for a in (p.atom0(), p.atom1(), p.atom2())]
        return tuple(idx)
    raise ValueError("No non-zero dihedrals")


def test_find_linear_angles(diphenylethane_mols):
    mol = diphenylethane_mols.molecules("property is_perturbable")[0]
    assert find_linear_angles(mol) == []

    angle = _spanned_angle(mol)
    key = (min(angle[0], angle[2]), angle[1], max(angle[0], angle[2]))

    linear = _linearise(mol, angle, 1)
    assert find_linear_angles(linear) == [(key, 1)]

    bare = _linearise(mol, angle, 0, remove_terms=True)
    assert find_linear_angles(bare) == []
    assert find_linear_angles(bare, require_terms=False) == [(key, 0)]


@pytest.mark.parametrize("mode", ["auto", "false"])
def test_runner_linear_torsions(diphenylethane_mols, mode):
    """
    The runner changes the schedule only when the option allows it.
    """
    from somd2.config import Config
    from somd2.runner import Runner

    mols = diphenylethane_mols.clone()
    mol = mols.molecules("property is_perturbable")[0]
    mols.update(_linearise(mol, _spanned_angle(mol), 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cpu",
            output_directory=tmpdir,
            linear_torsions=mode,
            ghost_modifications=False,
        )
        runner = Runner(mols, config)

        expected = [4.0, 0.0] if mode == "auto" else [4.0, 2.0]
        assert _torsion_k(runner._config.lambda_schedule, [0.0, 0.5]) == (
            pytest.approx(expected)
        )
