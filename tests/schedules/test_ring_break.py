"""
Tests for ring-break / ring-make CustomBondForce setup and schedule behaviour.

Checks:
  - ``ring-break`` force is registered (not ``ring-make``) for the forward
    direction (swap_end_states=False, ring_break_morph schedule).
  - ``ring-make`` force is registered (not ``ring-break``) for the reverse
    direction (swap_end_states=True, reverse_ring_break_morph schedule).
  - Ring-break energy is near-zero when kappa=0 (potential_swap and
    restraints_off stages) and clearly non-zero when kappa=1 (morph stage).
  - Ring-make energy follows the symmetric pattern under reverse_ring_break_morph:
    non-zero at λ=0 (morph, kappa=1) and near-zero at λ=1 (potential_swap, kappa=0).

Stage boundaries for ring_break_morph (weights 2:1:1:2, total 6):
  potential_swap   λ ∈ [0,   1/3)   ring-break kappa = 0
  restraints_off   λ ∈ [1/3, 1/2)   ring-break kappa = 0
  ring_open        λ ∈ [1/2, 2/3)   ring-break kappa ramps 0 → 1
  morph            λ ∈ [2/3, 1]     ring-break kappa = 1 (fixed)

Stage boundaries for reverse_ring_break_morph (weights 2:1:1:2, total 6):
  morph            λ ∈ [0,   1/3)   ring-make kappa = 1 (fixed)
  ring_close       λ ∈ [1/3, 1/2)   ring-make kappa ramps 1 → 0
  bonded_perturb   λ ∈ [1/2, 2/3)   ring-make kappa = 0
  potential_swap   λ ∈ [2/3, 1]     ring-make kappa = 0
"""

import pytest
import sire as sr

pytestmark = pytest.mark.skipif(
    "openmm" not in sr.convert.supported_formats(),
    reason="openmm support is not available",
)

# Energy threshold (kcal/mol) separating "inactive" from "active" force states.
# Chosen conservatively: sp_scan shows kappa=0 energies up to ~0.07 kcal/mol
# and kappa=1 energies of at least ~0.19 kcal/mol at the ring_open/morph boundary.
_INACTIVE_THRESHOLD = 0.1  # abs(E) must be below this when kappa=0
_ACTIVE_THRESHOLD = 0.1  # abs(E) must be above this when kappa=1


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_dynamics(mols, schedule, swap_end_states):
    """
    Construct a dynamics context for the ring-break system with Morse restraints.

    Mirrors the production setup in somd2_api_runner_dmr.py / sp_scan.py so
    that the force groups match what a real simulation would create.
    Uses reaction-field (rf) rather than PME since the test system is a
    vacuum/non-periodic input with no periodic box vectors.
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


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def forward_dynamics(syk_ring_break_mols):
    """Forward ring-breaking dynamics: swap_end_states=False, ring_break_morph."""
    from somd2._utils._schedules import ring_break_morph

    return _build_dynamics(
        syk_ring_break_mols, ring_break_morph(), swap_end_states=False
    )


@pytest.fixture(scope="module")
def reverse_dynamics(syk_ring_break_mols):
    """Reverse ring-making dynamics: swap_end_states=True, reverse_ring_break_morph."""
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


# ── forward schedule energy tests ─────────────────────────────────────────────

# ring_break_morph: kappa=0 throughout potential_swap (λ<1/3) and
# restraints_off (λ<1/2); kappa ramps 0→1 through ring_open (1/2→2/3);
# kappa=1 fixed throughout morph (λ>2/3).


@pytest.mark.parametrize("lam", [0.0, 1 / 3, 0.5])
def test_ring_break_inactive_before_ring_open(forward_dynamics, lam):
    """
    Ring-break energy is near-zero (kappa=0) in potential_swap and
    restraints_off stages and at the start of ring_open.
    """
    e = _force_energy_kcal(forward_dynamics, lam, "ring-break")
    assert abs(e) < _INACTIVE_THRESHOLD, (
        f"ring-break energy {e:.4f} kcal/mol at λ={lam:.4f} exceeds inactive "
        f"threshold {_INACTIVE_THRESHOLD} kcal/mol (kappa should be 0)"
    )


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


def test_ring_break_energy_increases_with_kappa(forward_dynamics):
    """
    Energy at kappa=1 (λ=1, morph) substantially exceeds energy at kappa=0 (λ=0).
    """
    e0 = _force_energy_kcal(forward_dynamics, 0.0, "ring-break")
    e1 = _force_energy_kcal(forward_dynamics, 1.0, "ring-break")
    assert abs(e1) > abs(e0) + 0.5, (
        f"ring-break energy at λ=1 ({e1:.4f} kcal/mol) should be much larger "
        f"than at λ=0 ({e0:.4f} kcal/mol)"
    )


# ── reverse schedule energy tests ─────────────────────────────────────────────

# reverse_ring_break_morph: ring-make kappa=1 fixed in morph (λ<1/3);
# kappa ramps 1→0 in ring_close (1/3→1/2); kappa=0 in bonded_perturb and
# potential_swap (λ>1/2).


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


@pytest.mark.parametrize("lam", [2 / 3, 1.0])
def test_ring_make_inactive_after_ring_close(reverse_dynamics, lam):
    """
    Ring-make energy is near-zero (kappa=0) in bonded_perturb and
    potential_swap stages of reverse_ring_break_morph.
    """
    e = _force_energy_kcal(reverse_dynamics, lam, "ring-make")
    assert abs(e) < _INACTIVE_THRESHOLD, (
        f"ring-make energy {e:.4f} kcal/mol at λ={lam:.4f} exceeds inactive "
        f"threshold {_INACTIVE_THRESHOLD} kcal/mol (kappa should be 0)"
    )
