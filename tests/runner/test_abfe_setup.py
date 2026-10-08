import tempfile

import pytest
import sire as sr

from somd2.config import Config
from somd2.runner import Runner


def _replace_ligand(mols, method):
    """Replace the perturbable molecule with one created by 'method'."""
    mols = mols.clone()
    ligand = mols.molecules("property is_perturbable")[0]
    mols.update(method(ligand, as_new_molecule=False))
    return mols


@pytest.mark.parametrize("schedule", ["annihilate", "decouple"])
def test_decoupled_ligand_accepted(schedule, ethane_methanol):
    mols = _replace_ligand(ethane_methanol, sr.morph.decouple)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_directory=tmpdir, platform="cpu", lambda_schedule=schedule
        )
        Runner(mols, config)


@pytest.mark.parametrize("schedule", ["annihilate", "decouple"])
def test_annihilated_ligand_raises(schedule, ethane_methanol):
    mols = _replace_ligand(ethane_methanol, sr.morph.annihilate)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_directory=tmpdir, platform="cpu", lambda_schedule=schedule
        )
        with pytest.raises(ValueError, match="sire.morph.decouple"):
            Runner(mols, config)
