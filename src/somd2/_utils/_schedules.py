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


def _set_boresch_lever_equations(s, stage_dihedral, stage_distance_angle):
    """
    Set the equations for a "split" restraint_lever Boresch restraint (see
    sire.restraints.boresch's restraint_lever parameter), reproducing the
    RXRX protocol's staged restraint turn-on (Table S1 of the RXRX paper's
    SI): within each of the two named stages, the corresponding restraint
    group ramps from ~0 to 1 following a geometric progression, while the
    other group is held fixed. 'stage_dihedral' is the stage over which
    the dihedral restraint group ramps on (with the distance/angle group
    held at 0); 'stage_distance_angle' is the stage over which the
    distance/angle group ramps on (with the dihedral group held at 1,
    already fully on).

    Note: this aligns the restraint turn-on with SOMD2's own decharge/
    annihilate(or decouple) stage boundaries, rather than reproducing the
    RXRX paper's exact global 50/50 window split (which falls partway
    through the annihilate/decouple stage, since the paper's own decharge
    stage is only 21 of 64 total bound-leg windows) - the relative sizes of
    SOMD2's stages are controlled by lambda_values weighting, not fixed.
    """
    from sire.legacy.CAS import Exp as _Exp
    import math as _math

    # Geometric progression from ~0.01 (fully off) to 1.0 (fully on), matching
    # the ratio observed in the RXRX paper's published lambda schedule.
    ramp_on = _Exp((1 - s.lam()) * _math.log(0.01))

    s.set_equation(stage=stage_dihedral, lever="restraint_dihedral", equation=ramp_on)
    s.set_equation(stage=stage_dihedral, lever="restraint_distance_angle", equation=0)
    s.set_equation(stage=stage_distance_angle, lever="restraint_dihedral", equation=1)
    s.set_equation(
        stage=stage_distance_angle,
        lever="restraint_distance_angle",
        equation=ramp_on,
    )


def annihilate(fix_epsilon=True, restraint_lever="split"):
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

    restraint_lever : str, optional
        How the Boresch restraint is controlled by this schedule, matching
        sire.restraints.boresch's restraint_lever parameter. Either "split"
        (default), where the dihedral restraint terms are turned on during
        decharge and the distance/angle terms are turned on during
        annihilate, reproducing the RXRX protocol's staged restraint
        turn-on, or "combined", where the whole restraint is turned on
        together during the decharge stage. The Boresch restraint object
        passed to the simulation must have a matching restraint_lever value.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    if restraint_lever not in ("combined", "split"):
        raise ValueError(
            "'restraint_lever' must be either 'combined' or 'split', "
            f"got {restraint_lever!r}"
        )

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

    s.add_stage(
        "annihilate",
        equation=(-s.lam() + 1) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(stage="annihilate", lever="charge", equation=s.final())

    if restraint_lever == "split":
        _set_boresch_lever_equations(
            s, stage_dihedral="decharge", stage_distance_angle="annihilate"
        )
    else:
        s.set_equation(
            stage="decharge", lever="restraint", equation=s.lam() * s.final()
        )
        s.set_equation(stage="annihilate", lever="restraint", equation=s.final())

    if fix_epsilon:
        s.set_equation(stage="annihilate", lever="epsilon", equation=s.initial())
        s.set_equation(
            stage="annihilate",
            force="ghost-lrc",
            lever="lrc_scale",
            equation=1 - s.lam(),
        )

    return s


def decouple(fix_epsilon=True, restraint_lever="split"):
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

    restraint_lever : str, optional
        How the Boresch restraint is controlled by this schedule, matching
        sire.restraints.boresch's restraint_lever parameter. Either "split"
        (default), where the dihedral restraint terms are turned on during
        decharge and the distance/angle terms are turned on during decouple,
        reproducing the RXRX protocol's staged restraint turn-on, or
        "combined", where the whole restraint is turned on together during
        the decharge stage. The Boresch restraint object passed to the
        simulation must have a matching restraint_lever value.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    if restraint_lever not in ("combined", "split"):
        raise ValueError(
            "'restraint_lever' must be either 'combined' or 'split', "
            f"got {restraint_lever!r}"
        )

    from sire.cas import LambdaSchedule as _LambdaSchedule

    # Start with the standard decouple schedule and modify the stages and
    # equations as needed. This will be folded into Sire in future, but
    # we will use this approach for prototyping.
    s = _LambdaSchedule.standard_decouple()

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

    if restraint_lever == "split":
        _set_boresch_lever_equations(
            s, stage_dihedral="decharge", stage_distance_angle="decouple"
        )
    else:
        s.set_equation(stage="decouple", lever="restraint", equation=s.final())
        s.set_equation(
            stage="decharge", lever="restraint", equation=s.initial() * s.lam()
        )

    return s


def ring_break_morph():
    """
    Build a lambda schedule for ring-breaking perturbations.

    Three stages: potential_swap → restraints_off → morph.

    During restraints_off the Morse restraint ramps off (morse_soft: 1→0) while
    the ring-break softcore LJ simultaneously ramps on (alpha: 1→0), providing a
    smooth handover with no gap between the two forces.

    Coulomb is decoupled from the LJ and driven by its own coul_kappa lever,
    which is held at zero through both bonded stages and ramps 0→1 during morph
    only, once the softcore LJ has already separated the pair.

    The ring-make equations mirror ring-break so that
    ``ring_break_morph().reverse()`` is the correct schedule for the ring-making
    direction (used by :func:`reverse_ring_break_morph`). Because
    ring_break_morph is only used for ring-breaking perturbations (no ring-make
    force present), the ring-make equations have no effect on forward
    simulations.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    s = _LambdaSchedule.standard_morph()

    # restraints_off [1/3, 2/3): Morse ramps off while the ring-break softcore LJ
    # ramps on simultaneously (alpha: 1→0). Bonded terms (angles, torsions)
    # interpolate initial→final over the same stage. ring-make mirrors ring-break
    # so that after .reverse(), the ring-make softcore ramps off as morse_soft ramps
    # on in the reversed restraints_off stage, correct for ring-making perturbations.
    s.prepend_stage("restraints_off", s.initial())
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
    s.set_equation(
        stage="restraints_off", force="ring-break", lever="alpha", equation=1 - s.lam()
    )
    s.set_equation(
        stage="restraints_off", force="ring-make", lever="alpha", equation=1 - s.lam()
    )

    s.prepend_stage("potential_swap", s.initial())
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
    # Softcore off throughout potential_swap: explicit constants so the schedule
    # visualises correctly regardless of the initial/final values passed by the caller.
    s.set_equation(
        stage="potential_swap", force="ring-break", lever="alpha", equation=1
    )
    s.set_equation(stage="potential_swap", force="ring-make", lever="alpha", equation=1)

    # morph [2/3, 1]: standard nonbonded morphing with ring-break/ring-make fixed
    # at fully open (alpha=0). ring-make mirrors ring-break so .reverse() gives
    # alpha=0 at lam=0 of the reversed morph stage (ring-making start).
    s.set_equation(stage="morph", lever="morse_hard", equation=0)
    s.set_equation(stage="morph", lever="morse_soft", equation=0)
    s.set_equation(stage="morph", lever="bond_k", equation=s.final())
    s.set_equation(stage="morph", lever="bond_length", equation=s.final())
    s.set_equation(stage="morph", lever="angle_k", equation=s.final())
    s.set_equation(stage="morph", lever="angle_size", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_k", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_phase", equation=s.final())
    s.set_equation(stage="morph", force="ring-break", lever="alpha", equation=0)
    s.set_equation(stage="morph", force="ring-make", lever="alpha", equation=0)

    # coul_kappa: zero through both bonded stages so the CLJ exception carries no
    # charge while atoms are at covalent distances; ramps 0→1 in morph only once
    # the softcore has already separated the atoms. ring-make mirrors ring-break
    # so .reverse() gives coul_kappa ramps 1→0 through the reversed morph stage.
    s.set_equation(
        stage="potential_swap", force="ring-break", lever="coul_kappa", equation=0
    )
    s.set_equation(
        stage="restraints_off", force="ring-break", lever="coul_kappa", equation=0
    )
    s.set_equation(
        stage="morph", force="ring-break", lever="coul_kappa", equation=s.lam()
    )
    s.set_equation(
        stage="potential_swap", force="ring-make", lever="coul_kappa", equation=0
    )
    s.set_equation(
        stage="restraints_off", force="ring-make", lever="coul_kappa", equation=0
    )
    s.set_equation(
        stage="morph", force="ring-make", lever="coul_kappa", equation=s.lam()
    )

    return s


def reverse_ring_break_morph():
    """
    Build a lambda schedule for ring-making perturbations (reverse ring-break).

    Returns ``ring_break_morph().reverse()``: three stages in reversed order
    (morph → restraints_off → potential_swap) with all equations reflected about
    λ=½ and initial/final end-states swapped.

    This schedule is correct for two equivalent use-cases:

    1. A ring-making perturbation run with ``swap_end_states=False``: the
       ring-make softcore force (alpha=0 at λ=0, ramping to 1) is controlled
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
