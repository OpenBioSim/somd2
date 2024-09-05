import sire as sr

from somd2._utils._ghosts import _boresch


def test_hexane_to_propane():
    """
    Test ghost atom modifications for hexane to propane. This has a terminal
    junction at lambda = 1.
    """

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

    missing_dihedrals = [
        (AtomIdx(4), AtomIdx(3), AtomIdx(2), AtomIdx(11)),
        (AtomIdx(4), AtomIdx(3), AtomIdx(2), AtomIdx(12)),
        (AtomIdx(11), AtomIdx(2), AtomIdx(3), AtomIdx(13)),
        (AtomIdx(11), AtomIdx(2), AtomIdx(3), AtomIdx(14)),
        (AtomIdx(12), AtomIdx(2), AtomIdx(3), AtomIdx(14)),
        (AtomIdx(12), AtomIdx(2), AtomIdx(3), AtomIdx(13)),
    ]

    # Store the molecular info.
    info = mols[0].info()

    # Check that the missing dihedrals are in the original dihedrals.
    assert (
        all(
            check_dihedral(info, dihedrals.potentials(), *dihedral)
            for dihedral in missing_dihedrals
        )
        == True
    )

    # Check that the missing dihedrals are not the new dihedrals.
    assert (
        all(
            check_dihedral(info, new_dihedrals.potentials(), *dihedral)
            for dihedral in missing_dihedrals
        )
        == False
    )


def test_toluene_to_pyridine():
    """
    Test ghost atom modifications for toluene to pyridine. This has a dual
    junction with a single branch at lambda = 1.
    """

    # Load the system.
    mols = sr.load_test_files("tol2pyr-1.s3")

    # Store the orginal angles and dihedrals at lambda = 1.
    angles = mols[0].property("angle1")
    dihedrals = mols[0].property("dihedral1")

    # Apply the ghost atom modifications.
    new_mols = _boresch(mols)

    # Get the new angles and dihedrals.
    new_angles = new_mols[0].property("angle1")
    new_dihedrals = new_mols[0].property("dihedral1")

    # The number of angles should remain the same.
    assert angles.num_functions() == new_angles.num_functions()

    # There should be seven fewer dihedrals.
    assert dihedrals.num_functions() - 7 == new_dihedrals.num_functions()

    # Create dihedral IDs for the missing dihedrals.

    from sire.legacy.Mol import AtomIdx

    missing_dihedrals = [
        (AtomIdx(0), AtomIdx(1), AtomIdx(2), AtomIdx(3)),
        (AtomIdx(0), AtomIdx(1), AtomIdx(2), AtomIdx(10)),
        (AtomIdx(0), AtomIdx(1), AtomIdx(6), AtomIdx(5)),
        (AtomIdx(0), AtomIdx(1), AtomIdx(6), AtomIdx(14)),
        (AtomIdx(6), AtomIdx(1), AtomIdx(0), AtomIdx(7)),
        (AtomIdx(6), AtomIdx(1), AtomIdx(0), AtomIdx(8)),
        (AtomIdx(6), AtomIdx(1), AtomIdx(0), AtomIdx(9)),
    ]

    # Store the molecular info.
    info = mols[0].info()

    # Check that the missing dihedrals are in the original dihedrals.
    assert (
        all(
            check_dihedral(info, dihedrals.potentials(), *dihedral)
            for dihedral in missing_dihedrals
        )
        == True
    )

    # Check that the missing dihedrals are not in the new dihedrals.
    assert (
        all(
            check_dihedral(info, new_dihedrals.potentials(), *dihedral)
            for dihedral in missing_dihedrals
        )
        == False
    )

    # Create a list of angle IDs for the modified angles.
    modified_angles = [
        (AtomIdx(0), AtomIdx(1), AtomIdx(2)),
        (AtomIdx(0), AtomIdx(1), AtomIdx(6)),
    ]

    # Functional form of the modified angles.
    expression = "100 [theta - 1.5708]^2"

    # Check that the original angles don't have the modified functional form.
    for p in angles.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())

        if (idx0, idx1, idx2) in modified_angles:
            assert str(p.function()) != expression

    # Check that the modified angles have the correct functional form.
    for p in new_angles.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())

        if (idx0, idx1, idx2) in modified_angles:
            assert str(p.function()) == expression


def test_acetone_to_propenol():
    """
    Test ghost atom modifications for acetone to propenol. This is a more
    complex perturbation with a terminal junction at lambda = 0 and a planar
    triple junction at lambda = 1.
    """

    # Load the system.
    mols = sr.load_test_files("acepol-1.s3")

    # Store the orginal angles and dihedrals at lambda = 0 and lambda = 1.
    angles0 = mols[0].property("angle0")
    angles1 = mols[0].property("angle1")
    dihedrals0 = mols[0].property("dihedral0")
    dihedrals1 = mols[0].property("dihedral1")

    # Apply the ghost atom modifications.
    new_mols = _boresch(mols)

    # Get the new angles and dihedrals.
    new_angles0 = new_mols[0].property("angle0")
    new_angles1 = new_mols[0].property("angle1")
    new_dihedrals0 = new_mols[0].property("dihedral0")
    new_dihedrals1 = new_mols[0].property("dihedral1")

    # The number of angles should remain the same at lambda = 0.
    assert angles0.num_functions() == new_angles0.num_functions()

    # The number of dihedrals should be one fewer at lambda = 0.
    assert dihedrals0.num_functions() - 1 == new_dihedrals0.num_functions()

    # The number of angles should be one fewer at lambda = 1.
    assert angles1.num_functions() - 1 == new_angles1.num_functions()

    # The number of dihedrals should be two fewer at lambda = 1.
    assert dihedrals1.num_functions() - 2 == new_dihedrals1.num_functions()

    # Create dihedral IDs for the missing dihedrals at lambda = 0.

    from sire.legacy.Mol import AtomIdx
    from sire.legacy.Mol import DihedralID

    missing_dihedrals0 = [
        (AtomIdx(8), AtomIdx(3), AtomIdx(9), AtomIdx(10)),
    ]

    # Store the molecular info.
    info = mols[0].info()

    # Check that the missing dihedrals are in the original dihedrals at lambda = 0.
    assert (
        all(
            check_dihedral(info, dihedrals0.potentials(), *dihedral)
            for dihedral in missing_dihedrals0
        )
        == True
    )

    # Check that the missing dihedrals are not in the new dihedrals at lambda = 0.
    assert (
        all(
            check_dihedral(info, new_dihedrals0.potentials(), *dihedral)
            for dihedral in missing_dihedrals0
        )
        == False
    )

    # Create dihedral IDs for the missing dihedrals at lambda = 1.
    missing_dihedrals1 = [
        (AtomIdx(0), AtomIdx(1), AtomIdx(3), AtomIdx(7)),
        (AtomIdx(2), AtomIdx(1), AtomIdx(3), AtomIdx(7)),
    ]

    # Check that the missing dihedrals are in the original dihedrals at lambda = 1.
    assert (
        all(
            check_dihedral(info, dihedrals1.potentials(), *dihedral)
            for dihedral in missing_dihedrals1
        )
        == True
    )

    # Check that the missing dihedrals are not in the new dihedrals at lambda = 1.
    assert (
        all(
            check_dihedral(info, new_dihedrals1.potentials(), *dihedral)
            for dihedral in missing_dihedrals1
        )
        == False
    )

    # Create angle IDs for the removed angles at lambda = 1.
    removed_angles = [
        (AtomIdx(1), AtomIdx(3), AtomIdx(7)),
    ]

    # Check that the removed angles are in the original angles at lambda = 1.
    assert (
        all(check_angle(info, angles1.potentials(), *angle) for angle in removed_angles)
        == True
    )

    # Check that the removed angles are not in the new angles at lambda = 1.
    assert (
        all(
            check_angle(info, new_angles1.potentials(), *angle)
            for angle in removed_angles
        )
        == False
    )

    # Create angle IDs for the modified angles at lambda = 1.
    modified_angles = [
        (AtomIdx(7), AtomIdx(3), AtomIdx(8)),
        (AtomIdx(7), AtomIdx(3), AtomIdx(9)),
    ]

    # Functional form of the modified angles.
    expression = "100 [theta - 1.5708]^2"

    # Check that the original angles don't have the modified functional form.
    for p in angles1.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())

        if (idx0, idx1, idx2) in modified_angles:
            assert str(p.function()) != expression

    # Check that the modified angles have the correct functional form.
    for p in new_angles1.potentials():
        idx0 = info.atom_idx(p.atom0())
        idx1 = info.atom_idx(p.atom1())
        idx2 = info.atom_idx(p.atom2())

        if (idx0, idx1, idx2) in modified_angles:
            assert str(p.function()) == expression


def check_angle(info, potentials, idx0, idx1, idx2):
    """
    Check if an angle potential is in a list of potentials.
    """

    for p in potentials:
        if (
            idx0 == info.atom_idx(p.atom0())
            and idx1 == info.atom_idx(p.atom1())
            and idx2 == info.atom_idx(p.atom2())
        ):
            return True

    return False


def check_dihedral(info, potentials, idx0, idx1, idx2, idx3):
    """
    Check if a dihedral potential is in a list of potentials.
    """

    for p in potentials:
        if (
            idx0 == info.atom_idx(p.atom0())
            and idx1 == info.atom_idx(p.atom1())
            and idx2 == info.atom_idx(p.atom2())
            and idx3 == info.atom_idx(p.atom3())
        ):
            return True

    return False
