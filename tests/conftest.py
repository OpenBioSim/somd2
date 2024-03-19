import pytest
import sire as sr


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
