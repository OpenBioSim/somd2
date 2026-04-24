import pytest
import sire.legacy.Mol as _SireMol


@pytest.fixture
def mols(request):
    return request.getfixturevalue(request.param)


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
