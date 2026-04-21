import math
from collections import defaultdict

import sire as sr

from somd2._utils import _lam_sym
from somd2.runner import Runner


def _masses_by_element(system):
    """
    Return atom masses for the perturbable ligand at both end states, grouped
    and sorted by element symbol. Dummy atoms (element Xx) are excluded.
    """
    result = {}
    for label, link_fn in (
        ("lam0", sr.morph.link_to_reference),
        ("lam1", sr.morph.link_to_perturbed),
    ):
        linked = link_fn(system)
        mol = next(m for m in linked["not water"].molecules() if m.num_atoms() > 1)
        by_elem = defaultdict(list)
        for atom in mol.atoms():
            elem = atom.element().symbol()
            if elem != "Xx":
                by_elem[elem].append(round(atom.mass().value(), 4))
        result[label] = {k: sorted(v) for k, v in by_elem.items()}
    return result


def test_hmr_pertfile(pert_fwd_mols, pert_rev_mols):
    """
    Verify HMR gives consistent masses for forward and reverse perturbations.

    Ligand A is the reference (lambda=0) in the forward perturbation and the
    perturbed state (lambda=1) in the reverse perturbation. After HMR, the
    same physical atoms must carry the same masses in both input paths.
    Likewise for Ligand B.
    """
    fwd = Runner._repartition_h_mass(pert_fwd_mols, 1.5)
    rev = Runner._repartition_h_mass(pert_rev_mols, 1.5)

    fwd_masses = _masses_by_element(fwd)
    rev_masses = _masses_by_element(rev)

    # Ligand A: forward lambda=0 must match reverse lambda=1
    assert fwd_masses["lam0"] == rev_masses["lam1"], (
        f"Ligand A masses differ between forward {_lam_sym}=0 and reverse {_lam_sym}=1 after HMR"
    )

    # Ligand B: forward lambda=1 must match reverse lambda=0
    assert fwd_masses["lam1"] == rev_masses["lam0"], (
        f"Ligand B masses differ between forward {_lam_sym}=1 and reverse {_lam_sym}=0 after HMR"
    )


def test_hmr(ethane_methanol, ethane_methanol_hmr):
    """Ensure that we can handle systems that have already been repartioned."""

    hmr_factor0, _ = Runner._get_h_mass_factor(ethane_methanol)
    hmr_factor1, _ = Runner._get_h_mass_factor(ethane_methanol_hmr)

    # Make sure that the HMR factor is as expected.
    assert math.isclose(hmr_factor0, 1.0, abs_tol=1e-4)
    assert math.isclose(hmr_factor1, 3.0, abs_tol=1e-4)

    # Invert the HMR of the second system.
    new_system = Runner._repartition_h_mass(ethane_methanol_hmr, 1.0 / hmr_factor1)
    hmr_factor2, _ = Runner._get_h_mass_factor(new_system)

    # Make sure the HMR factor is 1.0 again.
    assert math.isclose(hmr_factor2, 1.0, abs_tol=1e-4)

    # Work out the partial HMR factor if we actually want to use
    # 1.5 as the HMR factor.
    hmr_factor3 = 1.5 / hmr_factor1

    # Now apply the partial HMR factor.
    new_system = Runner._repartition_h_mass(ethane_methanol_hmr, hmr_factor3)
    hmr_factor4, _ = Runner._get_h_mass_factor(new_system)

    # Make sure the HMR factor is 1.5.
    assert math.isclose(hmr_factor4, 1.5, abs_tol=1e-4)
