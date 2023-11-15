import pytest
import sire as sr


@pytest.fixture(scope="session")
def ethane_methanol():
    mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule.s3"))
    for mol in mols.molecules("molecule property is_perturbable"):
        mols.update(mol.perturbation().link_to_reference().commit())
    return mols


@pytest.fixture(scope="session")
def ethane_methanol_hmr():
    mols = sr.load(sr.expand(sr.tutorial_url, "merged_molecule_hmr.s3"))
    for mol in mols.molecules("molecule property is_perturbable"):
        mols.update(mol.perturbation().link_to_reference().commit())
    return mols
