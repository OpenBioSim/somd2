import sire as sr

from somd2._utils._ghosts import _boresch


def test_hexane_to_propane():
    """Test ghost atom modifications for hexane to propane."""

    # Load the system.
    mols = sr.load_test_files("hex2prp-1.s3")

    # Store the orginal angles and dihedrals at lambda = 1.
    angles = mols[0].property("angle1")
    dihedrals = mols[0].property("dihedral1")

    # Apply the ghost atom modifications.
    new_mols = _boresch(mols)

    # Get the new angles and dihedrals.
    new_angles = new_mols[0].property("angle1")
    new_dihedrals = new_mols[0].property("dihedral1")

    # No angles should be removed.
    assert angles.num_functions() == new_angles.num_functions()

    # Six dihedrals should be removed.
    assert dihedrals.num_functions() - 6 == new_dihedrals.num_functions()

    # Create dihedral IDs for the missing dihedrals.

    from sire.legacy.Mol import AtomIdx
    from sire.legacy.Mol import DihedralID

    missing_dihedrals = [
        DihedralID(AtomIdx(4), AtomIdx(3), AtomIdx(2), AtomIdx(11)),
        DihedralID(AtomIdx(4), AtomIdx(3), AtomIdx(2), AtomIdx(12)),
        DihedralID(AtomIdx(11), AtomIdx(2), AtomIdx(3), AtomIdx(13)),
        DihedralID(AtomIdx(11), AtomIdx(2), AtomIdx(3), AtomIdx(14)),
        DihedralID(AtomIdx(12), AtomIdx(2), AtomIdx(3), AtomIdx(14)),
        DihedralID(AtomIdx(12), AtomIdx(2), AtomIdx(3), AtomIdx(13)),
    ]

    # Store the molecular info.
    info = mols[0].info()

    # Check that the missing dihedrals are in the original dihedrals.
    for idx in missing_dihedrals:
        is_found = False
        for p in dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            dih = DihedralID(idx0, idx1, idx2, idx3)

            if dih == idx:
                is_found = True
                break

        assert is_found

    # Check that the missing dihedrals are not in the new dihedrals.
    for idx in missing_dihedrals:
        is_found = False
        for p in new_dihedrals.potentials():
            idx0 = info.atom_idx(p.atom0())
            idx1 = info.atom_idx(p.atom1())
            idx2 = info.atom_idx(p.atom2())
            idx3 = info.atom_idx(p.atom3())
            dih = DihedralID(idx0, idx1, idx2, idx3)

            if dih == idx:
                is_found = True
                break

        assert not is_found


def test_toluene_to_pyridine():
    """Test ghost atom modifications for toluene to pyridine."""

    mols = sr.load_test_files("tol2pyr-1.s3")


def test_acetone_to_propenol():
    """Test ghost atom modifications for acetone to propenol."""

    mols = sr.load_test_files("acepol-1.s3")
