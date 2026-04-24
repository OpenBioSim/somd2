import os
from pathlib import Path

import pytest
import sire as sr

has_cuda = True if "CUDA_VISIBLE_DEVICES" in os.environ else False


@pytest.fixture(scope="session")
def diphenylethane_mols():
    """
    Load a merged perturbable system built from 1,2-diphenylethane (reference,
    lambda = 0) and 1,2-diphenylethanol (perturbed, lambda = 1).

    SMILES:
        reference : c1ccccc1CCc1ccccc1
        perturbed : OC(Cc1ccccc1)c1ccccc1

    Both phenyl rings are terminal, so two terminal ring groups should be
    detected.
    """
    mols = sr.load_test_files("12diphenylethane_12diphenylethanol.s3")
    return sr.morph.link_to_reference(mols)


@pytest.fixture(scope="session")
def phenethyl_mols():
    """
    Load a merged perturbable system built from phenethylamine (reference,
    lambda = 0) and 2-phenylethanol (perturbed, lambda = 1).

    SMILES:
        reference : NCCc1ccccc1
        perturbed : OCCc1ccccc1

    The phenyl ring is terminal — attached to the aliphatic chain by a single
    exocyclic bond — making it the only detectable terminal ring group.
    """
    mols = sr.load_test_files("phenethylamine_2phenylethanol.s3")
    return sr.morph.link_to_reference(mols)


@pytest.fixture(scope="session")
def ethane_methanol():
    mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))
    mols = sr.morph.link_to_reference(mols)
    return mols


@pytest.fixture(scope="session")
def ethane_methanol_hmr():
    mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule_hmr.s3"))
    mols = sr.morph.link_to_reference(mols)
    return mols


@pytest.fixture(scope="session")
def ethane_methanol_ions():
    mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule_ions.s3"))
    mols = sr.morph.link_to_reference(mols)
    return mols


@pytest.fixture(scope="session")
def pert_fwd_mols():
    """
    Load the forward perturbation system from AMBER files hosted on the sire
    test server and apply the local forward pert file.
    """
    from somd2._utils._somd1 import apply_pert

    mols = sr.load_test_files("somd1_forward.prm7", "somd1_forward.rst7")
    pert_file = str(Path(__file__).parent / "inputs" / "forward.pert")
    return apply_pert(mols, pert_file)


@pytest.fixture(scope="session")
def pert_rev_mols():
    """
    Load the reverse perturbation system from AMBER files hosted on the sire
    test server and apply the local backward pert file.
    """
    from somd2._utils._somd1 import apply_pert

    mols = sr.load_test_files("somd1_backward.prm7", "somd1_backward.rst7")
    pert_file = str(Path(__file__).parent / "inputs" / "backward.pert")
    return apply_pert(mols, pert_file)
