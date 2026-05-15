import pytest
import sire as sr

pytestmark = pytest.mark.skipif(
    "openmm" not in sr.convert.supported_formats(),
    reason="openmm support is not available",
)

# Energy threshold (kcal/mol) for the "active" state: kappa=1 should give
# clearly non-zero CustomBondForce energy.
_ACTIVE_THRESHOLD = 0.1


def _build_dynamics(mols, schedule, swap_end_states):
    """
    Construct a dynamics context for the ring-break system with Morse restraints.
    """
    from somd2._utils._somd1 import make_compatible

    mols = mols.clone()

    hard_restraints, mols = sr.restraints.morse_potential(
        mols,
        de="150 kcal/mol",
        auto_parametrise=True,
        direct_morse_replacement=True,
        name="morse_hard",
    )
    soft_restraints, _ = sr.restraints.morse_potential(
        mols,
        atoms0=hard_restraints[0].atom0(),
        atoms1=hard_restraints[0].atom1(),
        r0=hard_restraints[0].r0(),
        k="125 kcal mol-1 A-2",
        auto_parametrise=False,
        de="50 kcal mol-1",
        name="morse_soft",
    )
    mols = make_compatible(mols)

    return mols.dynamics(
        constraint="h_bonds",
        perturbable_constraint="h_bonds_not_heavy_perturbed",
        cutoff="10A",
        cutoff_type="rf",
        dynamic_constraints=True,
        include_constrained_energies=False,
        swap_end_states=swap_end_states,
        map={
            "ghosts_are_light": True,
            "check_for_h_by_max_mass": True,
            "check_for_h_by_mass": False,
            "check_for_h_by_element": False,
            "check_for_h_by_ambertype": False,
            "fix_perturbable_zero_sigmas": True,
            "lambda_schedule": schedule,
            "restraints": [hard_restraints, soft_restraints],
        },
    )


def _force_energy_kcal(d, lam, force_name):
    """Return the energy (kcal/mol) for *force_name* at *lam*."""
    import openmm

    context = d.context()
    d.set_lambda(lam, update_constraints=True)
    grp = context._force_group_map[force_name]
    state = context.getState(getEnergy=True, groups=(1 << grp))
    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalories_per_mole)


@pytest.fixture(scope="module")
def forward_dynamics(syk_ring_break_mols):
    """Forward ring-breaking dynamics: swap_end_states=False, ring_break_morph."""
    from somd2._utils._schedules import ring_break_morph

    return _build_dynamics(
        syk_ring_break_mols, ring_break_morph(), swap_end_states=False
    )


@pytest.fixture(scope="module")
def reverse_dynamics(syk_ring_break_mols):
    """
    Reverse ring-making dynamics: swap_end_states=True, reverse_ring_break_morph.

    reverse_ring_break_morph() == ring_break_morph().reverse(), so this fixture
    also implicitly tests the reversed schedule path used by the runner when a
    ring-breaking perturbation is run with swap_end_states=True.
    """
    from somd2._utils._schedules import reverse_ring_break_morph

    return _build_dynamics(
        syk_ring_break_mols, reverse_ring_break_morph(), swap_end_states=True
    )


# ── force-presence tests ──────────────────────────────────────────────────────


def test_forward_has_ring_break_not_ring_make(forward_dynamics):
    """
    Forward direction: ring-break CustomBondForce is registered; ring-make is absent.
    """
    fmap = forward_dynamics.context()._force_group_map
    assert "ring-break" in fmap, "ring-break force group missing for forward direction"
    assert "ring-make" not in fmap, (
        "ring-make force group should not exist for forward direction"
    )


def test_reverse_has_ring_make_not_ring_break(reverse_dynamics):
    """
    Reverse direction (swap_end_states=True): ring-make CustomBondForce is
    registered; ring-break is absent.
    """
    fmap = reverse_dynamics.context()._force_group_map
    assert "ring-make" in fmap, "ring-make force group missing for reverse direction"
    assert "ring-break" not in fmap, (
        "ring-break force group should not exist for reverse direction"
    )


# ── schedule kappa/alpha tests ────────────────────────────────────────────────
#
# These tests verify kappa and alpha values by calling schedule.morph() directly,
# using the same initial/final values that lambdalever passes in production.
# They are completely independent of Sire's energy formula and will continue to
# work correctly regardless of changes to the softcore implementation.

# ring_break_morph kappa/alpha points:
#   λ=0.00  potential_swap start  kappa=0, alpha=1
#   λ=0.15  potential_swap mid    kappa=0, alpha=1
#   λ=1/3   restraints_off start  kappa=0, alpha=1
#   λ=0.45  restraints_off mid    kappa=0, alpha=1
#   λ=0.50  ring_open start       kappa=0, alpha=1  (within-stage lam=0)
#   λ=0.55  ring_open mid         kappa=0.3, alpha=0.7
#   λ=0.60  ring_open mid         kappa=0.6, alpha=0.4
#   λ=2/3   morph start           kappa=1, alpha=0
#   λ=0.85  morph mid             kappa=1, alpha=0
#   λ=1.00  morph end             kappa=1, alpha=0
_FWD_KAPPA_ALPHA = [
    (0.00, 0.0, 1.0),
    (0.15, 0.0, 1.0),
    (1 / 3, 0.0, 1.0),
    (0.45, 0.0, 1.0),
    (0.50, 0.0, 1.0),
    (0.55, 0.3, 0.7),
    (0.60, 0.6, 0.4),
    (2 / 3, 1.0, 0.0),
    (0.85, 1.0, 0.0),
    (1.00, 1.0, 0.0),
]

# reverse_ring_break_morph ring-make kappa/alpha points (mirror of forward):
#   λ=0.00  morph start           kappa=1, alpha=0
#   λ=0.15  morph mid             kappa=1, alpha=0
#   λ=1/3   ring_open start       kappa=1, alpha=0  (within-stage lam=0)
#   λ=0.45  ring_open mid         kappa=0.3, alpha=0.7  (within-stage lam=0.7)
#   λ=0.50  restraints_off start  kappa=0, alpha=1
#   λ=0.60  restraints_off mid    kappa=0, alpha=1
#   λ=2/3   potential_swap start  kappa=0, alpha=1
#   λ=0.85  potential_swap mid    kappa=0, alpha=1
#   λ=1.00  potential_swap end    kappa=0, alpha=1
_REV_KAPPA_ALPHA = [
    (0.00, 1.0, 0.0),
    (0.15, 1.0, 0.0),
    (1 / 3, 1.0, 0.0),
    (0.45, 0.3, 0.7),
    (0.50, 0.0, 1.0),
    (0.60, 0.0, 1.0),
    (2 / 3, 0.0, 1.0),
    (0.85, 0.0, 1.0),
    (1.00, 0.0, 1.0),
]

# ring_break_morph coul_kappa points (initial=0, final=1):
#   λ=0.00–2/3  all pre-morph stages   coul_kappa=0
#   λ=2/3       morph start            coul_kappa=0  (within-stage lam=0)
#   λ=0.85      morph mid              coul_kappa=0.55  ((0.85-2/3)*3)
#   λ=1.00      morph end              coul_kappa=1.0
_FWD_COUL_KAPPA = [
    (0.00, 0.0),
    (0.15, 0.0),
    (1 / 3, 0.0),
    (0.45, 0.0),
    (0.50, 0.0),
    (0.55, 0.0),
    (0.60, 0.0),
    (2 / 3, 0.0),
    (0.85, 0.55),
    (1.00, 1.0),
]

# reverse_ring_break_morph ring-make coul_kappa points (initial=1, final=0):
#   λ=0.00      morph start (reversed)  coul_kappa=1.0
#   λ=0.15      morph mid               coul_kappa=0.55  (1 - 0.15*3)
#   λ=1/3       morph end / ring_open   coul_kappa=0.0
#   λ=0.45–1.0  ring_open/restraints_off/potential_swap  coul_kappa=0
_REV_COUL_KAPPA = [
    (0.00, 1.0),
    (0.15, 0.55),
    (1 / 3, 0.0),
    (0.45, 0.0),
    (0.50, 0.0),
    (0.60, 0.0),
    (2 / 3, 0.0),
    (0.85, 0.0),
    (1.00, 0.0),
]


@pytest.mark.parametrize("lam,expected_kappa,expected_alpha", _FWD_KAPPA_ALPHA)
def test_ring_break_morph_schedule(lam, expected_kappa, expected_alpha):
    """
    ring_break_morph() produces the correct ring-break kappa and alpha at each λ.

    Uses lambdalever's initial/final values (kappa: 0→1, alpha: 1→0) to ensure
    the test matches production behaviour exactly.
    """
    from somd2._utils._schedules import ring_break_morph

    s = ring_break_morph()
    kappa = s.morph("ring-break", "kappa", 0.0, 1.0, lam)
    alpha = s.morph("ring-break", "alpha", 1.0, 0.0, lam)
    assert abs(kappa - expected_kappa) < 1e-10, (
        f"ring-break kappa={kappa:.8f} at λ={lam:.4f}, expected {expected_kappa}"
    )
    assert abs(alpha - expected_alpha) < 1e-10, (
        f"ring-break alpha={alpha:.8f} at λ={lam:.4f}, expected {expected_alpha}"
    )


@pytest.mark.parametrize("lam,expected_kappa,expected_alpha", _REV_KAPPA_ALPHA)
def test_reverse_ring_break_morph_schedule(lam, expected_kappa, expected_alpha):
    """
    reverse_ring_break_morph() produces the correct ring-make kappa and alpha at each λ.

    Uses lambdalever's initial/final values (kappa: 1→0, alpha: 0→1) to ensure
    the test matches production behaviour exactly.
    """
    from somd2._utils._schedules import reverse_ring_break_morph

    s = reverse_ring_break_morph()
    kappa = s.morph("ring-make", "kappa", 1.0, 0.0, lam)
    alpha = s.morph("ring-make", "alpha", 0.0, 1.0, lam)
    assert abs(kappa - expected_kappa) < 1e-10, (
        f"ring-make kappa={kappa:.8f} at λ={lam:.4f}, expected {expected_kappa}"
    )
    assert abs(alpha - expected_alpha) < 1e-10, (
        f"ring-make alpha={alpha:.8f} at λ={lam:.4f}, expected {expected_alpha}"
    )


@pytest.mark.parametrize("lam,expected_coul_kappa", _FWD_COUL_KAPPA)
def test_ring_break_morph_coul_kappa(lam, expected_coul_kappa):
    """
    ring_break_morph() pins coul_kappa=0 through all pre-morph stages and ramps
    it 0→1 during morph only, so Coulomb only activates once atoms are separated.

    Uses lambdalever's initial/final values (coul_kappa: 0→1).
    """
    from somd2._utils._schedules import ring_break_morph

    s = ring_break_morph()
    coul_kappa = s.morph("ring-break", "coul_kappa", 0.0, 1.0, lam)
    assert abs(coul_kappa - expected_coul_kappa) < 1e-10, (
        f"ring-break coul_kappa={coul_kappa:.8f} at λ={lam:.4f}, "
        f"expected {expected_coul_kappa}"
    )


@pytest.mark.parametrize("lam,expected_coul_kappa", _REV_COUL_KAPPA)
def test_reverse_ring_break_morph_coul_kappa(lam, expected_coul_kappa):
    """
    reverse_ring_break_morph() ramps ring-make coul_kappa 1→0 through the
    reversed morph stage and pins it to 0 in all subsequent stages.

    Uses lambdalever's initial/final values (coul_kappa: 1→0).
    """
    from somd2._utils._schedules import reverse_ring_break_morph

    s = reverse_ring_break_morph()
    coul_kappa = s.morph("ring-make", "coul_kappa", 1.0, 0.0, lam)
    assert abs(coul_kappa - expected_coul_kappa) < 1e-10, (
        f"ring-make coul_kappa={coul_kappa:.8f} at λ={lam:.4f}, "
        f"expected {expected_coul_kappa}"
    )


# ── energy magnitude tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("lam", [2 / 3, 1.0])
def test_ring_break_active_after_ring_open(forward_dynamics, lam):
    """
    Ring-break energy is clearly non-zero (kappa=1) at the end of ring_open
    and throughout morph.
    """
    e = _force_energy_kcal(forward_dynamics, lam, "ring-break")
    assert abs(e) > _ACTIVE_THRESHOLD, (
        f"ring-break energy {e:.4f} kcal/mol at λ={lam:.4f} is below active "
        f"threshold {_ACTIVE_THRESHOLD} kcal/mol (kappa should be 1)"
    )


def test_ring_make_active_at_lambda_zero(reverse_dynamics):
    """
    Ring-make energy is non-zero at λ=0: the morph stage fixes kappa=1
    so the ring-make interaction is fully on from the start.
    """
    e = _force_energy_kcal(reverse_dynamics, 0.0, "ring-make")
    assert abs(e) > _ACTIVE_THRESHOLD, (
        f"ring-make energy {e:.4f} kcal/mol at λ=0 is below active threshold "
        f"{_ACTIVE_THRESHOLD} kcal/mol (kappa should be 1 in morph stage)"
    )


def test_ring_make_inactive_at_lambda_one(reverse_dynamics):
    """
    Ring-make energy is near-zero at λ=1 (potential_swap end, kappa=0).

    At λ=1 the system is at the ring-open end state; the hard-hard correction
    term in the CustomBondForce is small because the pair is at nonbonded
    separation, so the absolute energy remains below the active threshold.
    """
    e = _force_energy_kcal(reverse_dynamics, 1.0, "ring-make")
    assert abs(e) < _ACTIVE_THRESHOLD, (
        f"ring-make energy {e:.4f} kcal/mol at λ=1 exceeds threshold "
        f"{_ACTIVE_THRESHOLD} kcal/mol (kappa should be 0)"
    )


# ── energy symmetry tests ─────────────────────────────────────────────────────
#
# The invariant ring_break_morph().reverse() == reverse_ring_break_morph() means
# that the softcore kappa/alpha values at (forward, λ) and (reverse, 1-λ) are
# equal.  Both forces act on the same bond (the original ring_breaking_bond,
# which swap_end_states=True maps to ring_making_pairs), so the energies must
# also match.  The hard-hard correction appears identically on both sides and
# cancels in the comparison, making this test robust to formula changes.
#
# Test points span zero and non-zero energy regions:
#   λ=0.0  → forward kappa=0, reverse at 1-λ=1.0 kappa=0 (both ≈0)
#   λ=0.55 → forward ring_open (kappa=0.3), reverse ring_open at 0.45 (kappa=0.3)
#   λ=2/3  → forward morph start (kappa=1), reverse ring_open start at 1/3 (kappa=1)
#   λ=0.85 → forward morph (kappa=1), reverse reversed-morph at 0.15 (kappa=1)
#   λ=1.0  → forward morph end (kappa=1), reverse at 0.0 reversed-morph (kappa=1)


@pytest.mark.parametrize("lam", [0.0, 0.55, 2 / 3, 0.85, 1.0])
def test_energy_symmetry_forward_reverse(forward_dynamics, reverse_dynamics, lam):
    """
    Single-point energy symmetry: E_ring_break_forward(λ) == E_ring_make_reverse(1-λ).

    Verifies that reverse_ring_break_morph() == ring_break_morph().reverse() and
    that the mirrored kappa/alpha produce identical corrections on the same bond.
    """
    e_fwd = _force_energy_kcal(forward_dynamics, lam, "ring-break")
    e_rev = _force_energy_kcal(reverse_dynamics, 1.0 - lam, "ring-make")
    assert abs(e_fwd - e_rev) < 1e-4, (
        f"Energy symmetry broken at λ={lam:.4f}: "
        f"ring-break forward = {e_fwd:.6f} kcal/mol, "
        f"ring-make reverse(1-λ={1 - lam:.4f}) = {e_rev:.6f} kcal/mol, "
        f"difference = {abs(e_fwd - e_rev):.2e} kcal/mol"
    )


def test_schedule_symmetry():
    """
    reverse_ring_break_morph() must equal ring_break_morph().reverse().

    Checks that the simplified implementation produces identical schedules by
    comparing kappa values at a dense grid of lambda points using the default
    initial/final values that lambdalever passes for ring-break kappa.
    """
    from somd2._utils._schedules import ring_break_morph, reverse_ring_break_morph

    fwd = ring_break_morph()
    rev = reverse_ring_break_morph()
    rev_via_reverse = fwd.reverse()

    test_lambdas = [i / 20 for i in range(21)]
    for lam in test_lambdas:
        for force, lever, init, fin in [
            ("ring-break", "kappa", 0.0, 1.0),
            ("ring-break", "alpha", 1.0, 0.0),
            ("ring-break", "coul_kappa", 0.0, 1.0),
            ("ring-make", "kappa", 1.0, 0.0),
            ("ring-make", "alpha", 0.0, 1.0),
            ("ring-make", "coul_kappa", 1.0, 0.0),
        ]:
            v_rev = rev.morph(force, lever, init, fin, lam)
            v_rev2 = rev_via_reverse.morph(force, lever, init, fin, lam)
            assert abs(v_rev - v_rev2) < 1e-12, (
                f"Mismatch for {force}/{lever} at λ={lam:.2f}: "
                f"reverse_ring_break_morph={v_rev}, ring_break_morph().reverse()={v_rev2}"
            )
