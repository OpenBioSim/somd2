import pytest
import sire.legacy.Mol as _SireMol


def _unique_nonghost_terms(mol, term_type, n_atoms, final=False):
    """
    Return the set of atom-index tuples for terms in `term_type{0 or 1}` that
    are absent from the other end state and involve no ghost atoms in the state
    they exist in.
    """
    from somd2._utils import _has_ghost

    suffix_own = "1" if final else "0"
    suffix_other = "0" if final else "1"

    info = mol.info()

    def potentials(suffix):
        return mol.property(f"{term_type}{suffix}").potentials()

    def key(p):
        return tuple(
            info.atom_idx(getattr(p, f"atom{k}")()).value() for k in range(n_atoms)
        )

    own_keys = {key(p): p for p in potentials(suffix_own)}
    other_keys = {key(p) for p in potentials(suffix_other)}
    # also consider reversed keys for symmetric terms
    other_keys |= {k[::-1] for k in other_keys}

    unique = {}
    for k, p in own_keys.items():
        if k not in other_keys:
            atoms = [info.atom_idx(getattr(p, f"atom{i}")()) for i in range(n_atoms)]
            if not _has_ghost(mol, atoms, final):
                unique[k] = p.function()
    return unique


@pytest.fixture
def mols(request):
    return request.getfixturevalue(request.param)


def test_make_compatible_ring_break(ring_break_mols):
    """
    Verify that make_compatible preserves non-ghost bonded terms that are
    unique to one end state, rather than silently dropping them.

    The 6YNGD→intgd perturbation breaks an N-C ring bond. The cross-bond
    angles, dihedrals, and impropers that span this bond exist only in
    state0 (the ring is intact there) and involve no ghost atoms, so they
    must survive make_compatible unchanged.
    """
    from somd2._utils._somd1 import make_compatible

    mol_before = ring_break_mols.molecules("property is_perturbable")[0]

    # Collect unique non-ghost terms in state0 before the call.
    before = {
        term: _unique_nonghost_terms(mol_before, term, n)
        for term, n in [("angle", 3), ("dihedral", 4), ("improper", 4)]
    }

    # Require that there are actually unique non-ghost terms to test against.
    assert any(before[t] for t in before), (
        "No unique non-ghost terms found in state0 — test input may be wrong"
    )

    system_after = make_compatible(ring_break_mols)
    mol_after = system_after.molecules("property is_perturbable")[0]

    info = mol_after.info()

    for term, n in [("angle", 3), ("dihedral", 4), ("improper", 4)]:
        after_keys = {
            tuple(info.atom_idx(getattr(p, f"atom{k}")()).value() for k in range(n))
            for p in mol_after.property(f"{term}0").potentials()
        }
        for atom_key in before[term]:
            assert atom_key in after_keys or atom_key[::-1] in after_keys, (
                f"Unique non-ghost {term}0 term {atom_key} was incorrectly "
                f"removed by make_compatible"
            )


@pytest.mark.parametrize("mols", ["pert_fwd_mols", "pert_rev_mols"], indirect=True)
def test_reconstruct_intrascale(mols):
    """
    Verify that reconstruct_intrascale correctly rebuilds end-state connectivity
    and intrascale matrices from bond potentials.

    The forward perturbation has two hydrogen atoms that are real at lambda=0
    and become ghost atoms (du) at lambda=1; the reverse perturbation is the
    mirror image. In both cases the reconstructed connectivity objects must
    differ between the two end states.
    """
    from somd2._utils._somd1 import reconstruct_intrascale

    system = reconstruct_intrascale(mols)

    mol = system.molecules("property is_perturbable")[0]

    conn0 = mol.property("connectivity0")
    conn1 = mol.property("connectivity1")

    assert isinstance(conn0, _SireMol.Connectivity)
    assert isinstance(conn1, _SireMol.Connectivity)
    assert conn0 != conn1
