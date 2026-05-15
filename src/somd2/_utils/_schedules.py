######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# SOMD2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SOMD2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SOMD2. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = [
    "annihilate",
    "decouple",
    "ring_break_morph",
    "reverse_ring_break_morph",
]


def annihilate(fix_epsilon=True):
    """
    Build the ABFE lambda schedule using decharge → annihilate.

    Annihilation removes ALL non-bonded interactions (including intramolecular LJ
    between non-bonded pairs).

    Parameters
    ----------
    fix_epsilon : bool, optional
        If True (default), epsilon is held constant at its real-atom value
        throughout the annihilate stage so that the (1-alpha) prefactor of the
        Beutler soft-core provides the sole LJ decay pathway.  The ghost-LRC
        force is then explicitly scaled to zero over the stage to compensate.
        If False, epsilon is scaled normally from initial to final and the LRC
        follows naturally.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    # Start with the standard decouple schedule and modify the stages and
    # equations as needed. This will be folded into Sire in future, but
    # we will use this approach for prototyping.
    s = _LambdaSchedule.standard_decouple()

    s.remove_stage("decouple")

    s.add_stage("decharge", equation=s.initial())
    s.set_equation(
        stage="decharge",
        lever="charge",
        equation=s.lam() * s.final() + s.initial() * (1 - s.lam()),
    )
    s.set_equation(stage="decharge", force="restraint", equation=s.lam() * s.final())

    s.add_stage(
        "annihilate",
        equation=(-s.lam() + 1) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(stage="annihilate", lever="charge", equation=s.final())
    s.set_equation(stage="annihilate", force="restraint", equation=s.final())

    if fix_epsilon:
        s.set_equation(stage="annihilate", lever="epsilon", equation=s.initial())
        s.set_equation(
            stage="annihilate",
            force="ghost-lrc",
            lever="lrc_scale",
            equation=1 - s.lam(),
        )

    return s


def decouple(fix_epsilon=True):
    """
    Build the ABFE lambda schedule using decharge → decouple.

    Decoupling removes only INTERMOLECULAR non-bonded interactions; intramolecular
    terms are preserved via kappa=0 on ghost/ghost and ghost-14 forces.

    Parameters
    ----------
    fix_epsilon : bool, optional
        If True (default), epsilon is held constant at its real-atom value
        throughout the decouple stage (see annihilate for rationale).  The
        ghost-LRC force is then explicitly scaled to zero over the stage.
        If False, epsilon is scaled normally and the LRC follows naturally.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    # Start with the standard decouple schedule and modify the stages and
    # equations as needed. This will be folded into Sire in future, but
    # we will use this approach for prototyping.
    s = _LambdaSchedule.standard_decouple()

    s.set_equation(stage="decouple", lever="restraint", equation=s.final())
    s.set_equation(stage="decouple", lever="kappa", force="ghost/ghost", equation=0)
    s.set_equation(stage="decouple", lever="kappa", force="ghost-14", equation=0)
    s.set_equation(stage="decouple", lever="charge", equation=s.final())

    if fix_epsilon:
        s.set_equation(stage="decouple", lever="epsilon", equation=s.initial())
        s.set_equation(
            stage="decouple",
            force="ghost-lrc",
            lever="lrc_scale",
            equation=1 - s.lam(),
        )

    s.prepend_stage("decharge", s.initial())
    s.set_equation(
        stage="decharge",
        lever="charge",
        equation=s.lam() * s.final() + s.initial() * (1 - s.lam()),
    )
    s.set_equation(stage="decharge", force="ghost/ghost", equation=s.initial())
    s.set_equation(stage="decharge", force="ghost-14", equation=s.initial())
    s.set_equation(
        stage="decharge", lever="kappa", force="ghost/ghost", equation=-s.lam() + 1
    )
    s.set_equation(
        stage="decharge", lever="kappa", force="ghost-14", equation=-s.lam() + 1
    )
    s.set_equation(stage="decharge", lever="restraint", equation=s.initial() * s.lam())

    return s


def ring_break_morph():
    """
    Build a lambda schedule for ring-breaking perturbations.

    Four stages: potential_swap → restraints_off → ring_open → morph.

    The ring-break softcore kappa ramps 0→1 through ring_open and is fixed at 1
    in morph.  The ring-make equations mirror ring-break so that
    ``ring_break_morph().reverse()`` is the correct schedule for the ring-making
    direction (used by :func:`reverse_ring_break_morph`).  Because ring_break_morph
    is only used for ring-breaking perturbations (no ring-make force present), the
    ring-make equations have no effect on forward simulations.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    s = _LambdaSchedule.standard_morph()
    s.set_stage_weight("morph", 2)

    # ring_open: Morse is already off; ring-break nonbonded interaction ramps
    # on (alpha: 1→0, kappa: 0→1) while non-bonded terms stay at initial and
    # bonded terms remain at final. The softcore interaction gently pushes the
    # atoms into the open-chain geometry before the full nonbonded morph begins,
    # improving HREX overlap at the ring-break boundary.
    #
    # ring-make equations mirror ring-break so the reversed schedule is correct:
    # after .reverse(), ring-make kappa ramps 1→0 through this stage, matching
    # what ring-break does here in the forward direction.
    s.prepend_stage("ring_open", s.initial(), weight=1)
    s.set_equation(stage="ring_open", lever="morse_hard", equation=0)
    s.set_equation(stage="ring_open", lever="morse_soft", equation=0)
    s.set_equation(stage="ring_open", lever="bond_k", equation=s.final())
    s.set_equation(stage="ring_open", lever="bond_length", equation=s.final())
    s.set_equation(stage="ring_open", lever="angle_k", equation=s.final())
    s.set_equation(stage="ring_open", lever="angle_size", equation=s.final())
    s.set_equation(stage="ring_open", lever="torsion_k", equation=s.final())
    s.set_equation(stage="ring_open", lever="torsion_phase", equation=s.final())
    s.set_equation(
        stage="ring_open", force="ring-break", lever="alpha", equation=1 - s.lam()
    )
    s.set_equation(
        stage="ring_open", force="ring-break", lever="kappa", equation=s.lam()
    )
    # ring-make mirrors ring-break so reversed schedule ramps ring-make 1→0 here.
    s.set_equation(
        stage="ring_open", force="ring-make", lever="alpha", equation=1 - s.lam()
    )
    s.set_equation(
        stage="ring_open", force="ring-make", lever="kappa", equation=s.lam()
    )

    s.prepend_stage("restraints_off", s.initial(), weight=1)
    s.set_equation(stage="restraints_off", lever="morse_soft", equation=1 - s.lam())
    s.set_equation(stage="restraints_off", lever="morse_hard", equation=0)
    s.set_equation(stage="restraints_off", lever="bond_k", equation=s.final())
    s.set_equation(stage="restraints_off", lever="bond_length", equation=s.final())
    s.set_equation(
        stage="restraints_off",
        lever="angle_k",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="restraints_off",
        lever="angle_size",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="restraints_off",
        lever="torsion_k",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="restraints_off",
        lever="torsion_phase",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )

    s.prepend_stage("potential_swap", s.initial(), weight=2)
    s.set_equation(stage="potential_swap", lever="morse_hard", equation=1 - s.lam())
    s.set_equation(stage="potential_swap", lever="morse_soft", equation=0 + s.lam())
    s.set_equation(
        stage="potential_swap",
        lever="bond_k",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="potential_swap",
        lever="bond_length",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(stage="potential_swap", lever="angle_k", equation=s.initial())
    s.set_equation(stage="potential_swap", lever="angle_size", equation=s.initial())
    s.set_equation(stage="potential_swap", lever="torsion_k", equation=s.initial())
    s.set_equation(stage="potential_swap", lever="torsion_phase", equation=s.initial())

    # morph: standard nonbonded morphing. Ring-break is fixed at fully open
    # (kappa=1, alpha=0) since geometry has already relaxed in ring_open.
    # ring-make mirrors ring-break: kappa=1, alpha=0 so that .reverse() gives
    # kappa=1 at lam=0 of the reversed morph stage (ring-making direction start).
    s.set_equation(stage="morph", lever="morse_hard", equation=0)
    s.set_equation(stage="morph", lever="morse_soft", equation=0)
    s.set_equation(stage="morph", lever="bond_k", equation=s.final())
    s.set_equation(stage="morph", lever="bond_length", equation=s.final())
    s.set_equation(stage="morph", lever="angle_k", equation=s.final())
    s.set_equation(stage="morph", lever="angle_size", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_k", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_phase", equation=s.final())
    s.set_equation(stage="morph", force="ring-break", lever="alpha", equation=0)
    s.set_equation(stage="morph", force="ring-break", lever="kappa", equation=1)
    s.set_equation(stage="morph", force="ring-make", lever="alpha", equation=0)
    s.set_equation(stage="morph", force="ring-make", lever="kappa", equation=1)

    return s


def reverse_ring_break_morph():
    """
    Build a lambda schedule for ring-making perturbations (reverse ring-break).

    Returns ``ring_break_morph().reverse()``: four stages in reversed order
    (morph → ring_open → restraints_off → potential_swap) with all equations
    reflected about λ=½ and initial/final end-states swapped.

    This schedule is correct for two equivalent use-cases:

    1. A ring-making perturbation run with ``swap_end_states=False``: the
       ring-make softcore force (kappa=1 at λ=0, ramping to 0) is controlled
       directly by the ring-make lever equations.
    2. A ring-breaking perturbation run with ``swap_end_states=True`` (the
       runner reverses the schedule automatically, yielding the same effective
       schedule): the ring-make softcore — which now controls the original
       ring-breaking bond after the end-state swap — is handled identically.

    The energy symmetry invariant holds for both cases:
    ``E_ring_make_reverse(λ) == E_ring_break_forward(1-λ)`` at any fixed
    geometry.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    return ring_break_morph().reverse()
