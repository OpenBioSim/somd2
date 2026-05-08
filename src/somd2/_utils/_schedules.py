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

    Three stages: potential_swap → restraints_off → morph.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    s = _LambdaSchedule.standard_morph()

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

    s.set_equation(stage="morph", lever="morse_hard", equation=0)
    s.set_equation(stage="morph", lever="morse_soft", equation=0)
    s.set_equation(stage="morph", lever="bond_k", equation=s.final())
    s.set_equation(stage="morph", lever="bond_length", equation=s.final())
    s.set_equation(stage="morph", lever="angle_k", equation=s.final())
    s.set_equation(stage="morph", lever="angle_size", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_k", equation=s.final())
    s.set_equation(stage="morph", lever="torsion_phase", equation=s.final())

    # Ring-breaking bonds: softcore interaction grows from zero as the ring
    # opens during the morph stage (alpha: 1→0, kappa: 0→1).
    # Ring-making bonds: softcore interaction shrinks to zero as the ring
    # closes during the morph stage (alpha: 0→1, kappa: 1→0).
    # potential_swap and restraints_off stages use default (s.initial()):
    #   ring-break alpha=1.0/kappa=0.0 (no interaction, ring still bonded)
    #   ring-make  alpha=0.0/kappa=1.0 (full interaction, ring not yet formed)
    s.set_equation(
        stage="morph", force="ring-break", lever="alpha", equation=1 - s.lam()
    )
    s.set_equation(stage="morph", force="ring-break", lever="kappa", equation=s.lam())
    s.set_equation(stage="morph", force="ring-make", lever="alpha", equation=s.lam())
    s.set_equation(
        stage="morph", force="ring-make", lever="kappa", equation=1 - s.lam()
    )

    return s


def reverse_ring_break_morph():
    """
    Build a lambda schedule for reverse ring-breaking perturbations.

    Three stages: morph → bonded_perturb → potential_swap.

    Returns
    -------

    schedule : sire.legacy.CAS.LambdaSchedule
        The lambda schedule.
    """
    from sire.cas import LambdaSchedule as _LambdaSchedule

    s = _LambdaSchedule.standard_morph()

    s.set_equation(stage="morph", lever="morse_hard", equation=0)
    s.set_equation(stage="morph", lever="morse_soft", equation=0)
    s.set_equation(stage="morph", lever="bond_k", equation=s.initial())
    s.set_equation(stage="morph", lever="bond_length", equation=s.initial())
    s.set_equation(stage="morph", lever="angle_k", equation=s.initial())
    s.set_equation(stage="morph", lever="angle_size", equation=s.initial())
    s.set_equation(stage="morph", lever="torsion_k", equation=s.initial())
    s.set_equation(stage="morph", lever="torsion_phase", equation=s.initial())

    s.append_stage("bonded_perturb", s.final())
    s.set_equation(stage="bonded_perturb", lever="morse_soft", equation=0 + s.lam())
    s.set_equation(stage="bonded_perturb", lever="morse_hard", equation=0)
    s.set_equation(stage="bonded_perturb", lever="bond_k", equation=s.initial())
    s.set_equation(stage="bonded_perturb", lever="bond_length", equation=s.initial())
    s.set_equation(
        stage="bonded_perturb",
        lever="angle_k",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="bonded_perturb",
        lever="angle_size",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="bonded_perturb",
        lever="torsion_k",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )
    s.set_equation(
        stage="bonded_perturb",
        lever="torsion_phase",
        equation=(1 - s.lam()) * s.initial() + s.lam() * s.final(),
    )

    s.append_stage("potential_swap", s.final())
    s.set_equation(stage="potential_swap", lever="morse_hard", equation=0 + s.lam())
    s.set_equation(stage="potential_swap", lever="morse_soft", equation=1 - s.lam())
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
    s.set_equation(stage="potential_swap", lever="angle_k", equation=s.final())
    s.set_equation(stage="potential_swap", lever="angle_size", equation=s.final())
    s.set_equation(stage="potential_swap", lever="torsion_k", equation=s.final())
    s.set_equation(stage="potential_swap", lever="torsion_phase", equation=s.final())

    # morph stage (first): nonbonded-only changes; ring bonds still intact/absent.
    #   ring-break: alpha=1.0/kappa=0.0 throughout (no interaction, ring bonded at λ=0)
    #   ring-make:  alpha=0.0/kappa=1.0 throughout (full interaction, ring absent at λ=0)
    # bonded_perturb stage (second): ring bonds established/dissolved via Morse.
    #   ring-make softcore turns off as ring forms (alpha: 0→1, kappa: 1→0)
    #   ring-break softcore turns on as ring opens (alpha: 1→0, kappa: 0→1)
    # potential_swap stage (last): Morse→harmonic swap; ring fully transitioned.
    #   defaults (s.final()) give ring-break alpha=0/kappa=1, ring-make alpha=1/kappa=0.
    s.set_equation(stage="morph", force="ring-break", lever="alpha", equation=1)
    s.set_equation(stage="morph", force="ring-break", lever="kappa", equation=0)
    s.set_equation(stage="morph", force="ring-make", lever="alpha", equation=0)
    s.set_equation(stage="morph", force="ring-make", lever="kappa", equation=1)
    s.set_equation(
        stage="bonded_perturb", force="ring-break", lever="alpha", equation=1 - s.lam()
    )
    s.set_equation(
        stage="bonded_perturb", force="ring-break", lever="kappa", equation=s.lam()
    )
    s.set_equation(
        stage="bonded_perturb", force="ring-make", lever="alpha", equation=s.lam()
    )
    s.set_equation(
        stage="bonded_perturb", force="ring-make", lever="kappa", equation=1 - s.lam()
    )

    return s
