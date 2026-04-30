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


def annihilate():
    """
    Build the ABFE lambda schedule using decharge → annihilate with constant epsilon.

    Annihilation removes ALL non-bonded interactions (including intramolecular LJ
    between non-bonded pairs), matching the GROMACS ABFE protocol. Epsilon is held
    constant at its real-atom value throughout the annihilate stage so that the
    (1-alpha) prefactor for the Buetler soft-core provides the sole LJ decay pathway,
    giving true single (1-lambda) scaling consistent with GROMACS Beutler.

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
    s.set_equation(stage="annihilate", lever="epsilon", equation=s.initial())

    return s


def decouple():
    """
    Build the ABFE lambda schedule using decharge → decouple with constant epsilon.

    Decoupling removes only INTERMOLECULAR non-bonded interactions; intramolecular
    terms are preserved via kappa=0 on ghost/ghost and ghost-14 forces. Epsilon is
    held constant throughout the decouple stage (see annihilate for rationale).

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
    s.set_equation(stage="decouple", lever="epsilon", equation=s.initial())

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

    return s
