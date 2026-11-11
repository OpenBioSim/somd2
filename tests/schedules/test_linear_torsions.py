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


def _linearise(mol, angle, state, remove_terms=()):
    """
    Set the equilibrium value of an angle to 180 degrees at an end state, and
    remove the dihedrals and impropers that pass through it at the given end
    states.
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

    for name in ("dihedral", "improper"):
        for s in remove_terms:
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


def _linear_system(mols, remove_terms=(1,)):
    """
    Return a copy of the system with an angle of the perturbable molecule made
    linear at lambda = 1.
    """
    mols = mols.clone()
    mol = mols.molecules("property is_perturbable")[0]
    mols.update(_linearise(mol, _spanned_angle(mol), 1, remove_terms))
    return mols


def test_find_linear_angles(diphenylethane_mols):
    mol = diphenylethane_mols.molecules("property is_perturbable")[0]
    assert find_linear_angles(mol) == []

    angle = _spanned_angle(mol)
    key = (min(angle[0], angle[2]), angle[1], max(angle[0], angle[2]))

    assert find_linear_angles(_linearise(mol, angle, 1)) == [(key, 1, True, True)]

    linear = _linearise(mol, angle, 1, remove_terms=(1,))
    assert find_linear_angles(linear) == [(key, 1, True, False)]

    linear = _linearise(mol, angle, 1, remove_terms=(0,))
    assert find_linear_angles(linear) == [(key, 1, False, True)]

    bare = _linearise(mol, angle, 0, remove_terms=(0, 1))
    assert find_linear_angles(bare) == [(key, 0, False, False)]


def test_find_linear_angles_no_angles(diphenylethane_mols):
    mol = diphenylethane_mols.molecules("property is_perturbable")[0]
    mol = mol.edit().remove_property("angle1").molecule().commit()
    assert find_linear_angles(mol) == []


def test_linear_torsions_unsupported():
    schedule = LambdaSchedule.standard_morph()
    schedule.set_equation(
        stage="morph", force="torsion", lever="torsion_k", equation=schedule.final()
    )
    with pytest.raises(ValueError, match="force-specific"):
        linear_torsions(schedule, 0.0, 0.5)

    schedule = LambdaSchedule.standard_morph()
    schedule.set_constant("scale", 0.5)
    schedule.set_equation(
        stage="morph",
        lever="torsion_k",
        equation=schedule.get_constant_symbol("scale") * schedule.final(),
    )
    with pytest.raises(ValueError, match="symbols"):
        linear_torsions(schedule, 0.0, 0.5)


def _runner(mols, tmpdir, **kwargs):
    from somd2.config import Config
    from somd2.runner import Runner

    config = Config(
        platform="cpu", output_directory=tmpdir, ghost_modifications=False, **kwargs
    )
    return config, Runner(mols, config)


@pytest.mark.parametrize("mode", ["auto", "false"])
def test_runner_linear_torsions(diphenylethane_mols, mode):
    """
    The runner gives the affected molecule its own schedule only when the
    option allows it, records the window, and leaves the user's config
    unchanged.
    """
    from somd2.config import Config

    mols = _linear_system(diphenylethane_mols)

    with tempfile.TemporaryDirectory() as tmpdir:
        config, runner = _runner(mols, tmpdir, linear_torsions=mode)
        schedule = runner._config.lambda_schedule

        assert _torsion_k(schedule, [0.0, 0.5]) == pytest.approx([4.0, 2.0])
        assert schedule.has_molecule_schedule(0) == (mode == "auto")
        if mode == "auto":
            assert _torsion_k(schedule.get_molecule_schedule(0), [0.0, 0.5]) == (
                pytest.approx([4.0, 0.0])
            )

        assert not config.lambda_schedule.has_molecule_schedule(0)

        d = runner._config.as_dict()
        assert d["linear_torsions"] == mode
        assert d["linear_torsion_windows"] == ("0:0.0-0.5" if mode == "auto" else "")

        from somd2.io import dict_to_yaml

        dict_to_yaml(d, "saved.yaml", tmpdir)
        assert Config.from_yaml(f"{tmpdir}/saved.yaml").linear_torsions == mode


def test_runner_linear_torsions_constant(diphenylethane_mols):
    """
    No window is applied when the schedule doesn't change the torsions.
    """
    schedule = LambdaSchedule.standard_morph()
    for lever in ("torsion_k", "torsion_phase"):
        schedule.set_equation(stage="morph", lever=lever, equation=schedule.initial())
    mols = _linear_system(diphenylethane_mols)

    with tempfile.TemporaryDirectory() as tmpdir:
        _, runner = _runner(mols, tmpdir, lambda_schedule=schedule)
        assert not runner._config.lambda_schedule.has_molecule_schedule(0)
        assert runner._config.as_dict()["linear_torsion_windows"] == ""


def test_runner_linear_torsions_true_fails(diphenylethane_mols):
    """
    A window that can't be applied is an error when the option is forced on.
    """
    schedule = LambdaSchedule.standard_morph()
    schedule.set_equation(
        stage="morph", lever="torsion_k", equation=schedule.lam() ** 2
    )
    mols = _linear_system(diphenylethane_mols)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="linear_torsions"):
            _runner(mols, tmpdir, linear_torsions="true", lambda_schedule=schedule)


def test_restart_legacy_linear_torsions():
    """
    A simulation saved before the option existed can only be restarted without
    torsion windows.
    """
    from somd2.runner._base import RunnerBase

    RunnerBase._compare_configs({}, {"linear_torsion_windows": ""})
    with pytest.raises(ValueError, match="linear_torsions"):
        RunnerBase._compare_configs({}, {"linear_torsion_windows": "0:0.0-0.5"})
