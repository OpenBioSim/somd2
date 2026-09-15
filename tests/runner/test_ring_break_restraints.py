import tempfile

import pytest

from somd2.config import Config
from somd2.runner import Runner


def _config(tmpdir, **kwargs):
    """Return a minimal ring-breaking config rooted at 'tmpdir'."""
    options = {
        "output_directory": tmpdir,
        "lambda_schedule": "ring_break_morph",
        "num_lambda": 3,
        "runtime": "12fs",
        "energy_frequency": "4fs",
        "frame_frequency": "4fs",
        "checkpoint_frequency": "4fs",
        "equilibration_time": "0fs",
        "minimise": False,
        "platform": "CPU",
        "max_threads": 1,
    }
    options.update(kwargs)
    return Config(**options)


def _restraint(restraints, name):
    """Return the single restraint from the set called 'name'."""
    for restraint_set in restraints:
        if str(restraint_set.name()) == name:
            assert len(restraint_set) == 1
            return restraint_set[0]
    raise AssertionError(f"No restraint set named {name!r} in {restraints}")


def test_restraints_are_generated(syk_ring_break_mols):
    """
    Ensure that a pair of Morse restraints is automatically generated for a
    ring-breaking perturbation when no restraint is supplied, and that they act
    on the same pair of atoms, at the same equilibrium distance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Runner(syk_ring_break_mols.clone(), _config(tmpdir))

        restraints = runner._config.restraints
        assert restraints is not None
        assert len(restraints) == 2

        hard = _restraint(restraints, "morse_hard")
        soft = _restraint(restraints, "morse_soft")

        # Both restraints act on the bond that is broken.
        assert hard.atom0() == soft.atom0()
        assert hard.atom1() == soft.atom1()
        assert hard.r0() == soft.r0()

        # The restraints are passed through to the dynamics.
        assert runner._dynamics_kwargs["restraints"] is restraints


def test_restraints_match_config(syk_ring_break_mols):
    """
    Ensure that the generated restraints use the well depths and force constant
    from the config, and that the hard restraint inherits the force constant of
    the bond that it replaces.
    """
    import sire as sr

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(
            tmpdir,
            morse_hard_well_depth="123 kcal mol-1",
            morse_soft_well_depth="45 kcal mol-1",
            morse_soft_force_constant="67 kcal mol-1 A-2",
        )
        runner = Runner(syk_ring_break_mols.clone(), config)

        hard = _restraint(runner._config.restraints, "morse_hard")
        soft = _restraint(runner._config.restraints, "morse_soft")

        assert hard.de() == sr.u("123 kcal mol-1")
        assert soft.de() == sr.u("45 kcal mol-1")
        assert soft.k() == sr.u("67 kcal mol-1 A-2")

        # The hard restraint is auto-parametrised from the broken bond, so its
        # force constant comes from the bond, not the config.
        assert hard.k() != soft.k()
        assert hard.k().value() > 0


def test_broken_bond_is_replaced(syk_ring_break_mols):
    """
    Ensure that the hard restraint replaces the harmonic bond that is broken by
    the perturbation, i.e. that the bond is removed from the runner's system.
    Leaving both in place would double count the interaction.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mols = syk_ring_break_mols.clone()
        runner = Runner(mols, _config(tmpdir))

        hard = _restraint(runner._config.restraints, "morse_hard")

        def num_bonds(system, idx0, idx1):
            """Count the bond potentials between a pair of atom indices."""
            atoms = system.atoms()
            atom0 = atoms[idx0]
            atom1 = atoms[idx1]
            mol = system[atom0.molecule().number()]
            info = mol.info()
            expected = {atom0.index().value(), atom1.index().value()}

            count = 0
            for bond_prop in ("bond0", "bond1"):
                for potential in mol.property(bond_prop).potentials():
                    idxs = {
                        info.atom_idx(potential.atom0()).value(),
                        info.atom_idx(potential.atom1()).value(),
                    }
                    if idxs == expected:
                        count += 1
            return count

        # The unmodified input still has the bond, in the reference end state
        # only, since it is broken by the perturbation.
        assert num_bonds(syk_ring_break_mols, hard.atom0(), hard.atom1()) == 1

        # The runner's system has it removed, replaced by the Morse restraint.
        assert num_bonds(runner._system, hard.atom0(), hard.atom1()) == 0


def test_restraints_are_deterministic(syk_ring_break_mols):
    """
    Ensure that generating the restraints twice from the same input gives
    identical restraints. A restart regenerates them from the input system
    rather than reloading them, so they must not drift between runs, otherwise
    the accumulated free energy would be invalidated.
    """
    with tempfile.TemporaryDirectory() as tmpdir0:
        runner0 = Runner(syk_ring_break_mols.clone(), _config(tmpdir0))

    with tempfile.TemporaryDirectory() as tmpdir1:
        runner1 = Runner(syk_ring_break_mols.clone(), _config(tmpdir1))

    for name in ("morse_hard", "morse_soft"):
        assert _restraint(runner0._config.restraints, name) == _restraint(
            runner1._config.restraints, name
        )


def test_reverse_schedule_generates_restraints(syk_ring_break_mols):
    """
    Ensure that restraints are also generated for the ring-making direction,
    which uses the reversed schedule.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir, lambda_schedule="reverse_ring_break_morph")
        runner = Runner(syk_ring_break_mols.clone(), config)

        assert runner._config.restraints is not None
        assert len(runner._config.restraints) == 2


def test_user_restraints_are_not_overridden(syk_ring_break_mols):
    """
    Ensure that a user-supplied restraint is left alone, and that the system is
    not modified behind their back.
    """
    import sire as sr

    mols = syk_ring_break_mols.clone()

    restraints = sr.restraints.distance(
        mols,
        atoms0=0,
        atoms1=1,
        k="10 kcal mol-1 A-2",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Runner(mols, _config(tmpdir, restraints=restraints))

        assert len(runner._config.restraints) == 1
        assert runner._config.restraints[0] == restraints


@pytest.mark.parametrize("schedule", ["standard_morph", "charge_scaled_morph"])
def test_no_restraints_for_other_schedules(schedule, ethane_methanol):
    """
    Ensure that Morse restraints are only generated for ring-breaking
    schedules.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(tmpdir, lambda_schedule=schedule)
        runner = Runner(ethane_methanol.clone(), config)

        assert runner._config.restraints is None


def test_no_broken_bond_raises(ethane_methanol):
    """
    Ensure that a clear error is raised when a ring-breaking schedule is used
    for a perturbation that doesn't break (or form) a bond.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="Unable to generate Morse restraints"):
            Runner(ethane_methanol.clone(), _config(tmpdir))


def test_already_applied_raises(syk_ring_break_mols):
    """
    Ensure that a helpful error is raised if the Morse potential has already
    been applied to the input system, but the corresponding restraints were not
    passed via the config. The replacement must not be applied twice.
    """
    import sire as sr

    mols = syk_ring_break_mols.clone()

    # Apply the Morse replacement, as a user following the existing workflow
    # would, but don't pass the restraints to the config.
    _, mols = sr.restraints.morse_potential(
        mols,
        de="150 kcal mol-1",
        auto_parametrise=True,
        direct_morse_replacement=True,
        name="morse_hard",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="already been applied"):
            Runner(mols, _config(tmpdir))
