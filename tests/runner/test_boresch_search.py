import tempfile

import pytest
import sire as sr

from somd2.config import Config
from somd2.runner import Runner


@pytest.mark.parametrize("schedule", ["annihilate", "decouple"])
def test_charged_bound_leg(schedule, abfe_charge_change_mols):
    """The restraint search ignores the alchemical ions added for a charged ligand."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_directory=tmpdir, platform="cpu", lambda_schedule=schedule
        )
        runner = Runner(abfe_charge_change_mols.clone(), config)

    pert_mols = runner._system.molecules("property is_perturbable")
    assert pert_mols.num_molecules() > 1

    assert Runner._boresch_search_protocol(runner._system) in ("rxrx", "aldeghi")


def test_rxrx_unsuitable_falls_back(abfe_charge_change_mols):
    """A ligand with no N/O atoms can't use RXRX, so Aldeghi is used instead."""
    mols = abfe_charge_change_mols.clone()
    cursor = mols.molecules("property is_perturbable")[0].cursor()
    for atom in cursor.atoms():
        if atom["element0"].symbol() in ("N", "O"):
            atom["element0"] = sr.mol.Element("C")
    mols.update(cursor.commit())

    assert Runner._boresch_search_protocol(mols) == "aldeghi"


def test_unsuitable_system_raises(abfe_charge_change_mols):
    """A failure that applies to both protocols is raised rather than falling back."""
    mols = abfe_charge_change_mols.clone()
    cursor = mols["water"].molecules()[0].cursor()
    cursor["is_perturbable"] = True
    mols.update(cursor.commit())

    with pytest.raises(ValueError, match="exactly one perturbable molecule"):
        Runner._boresch_search_protocol(mols)
