"""
Tests for terminal ring flip Monte Carlo functionality.

Two fixtures are used (both defined in conftest.py):

``phenethyl_mols``
    Merged system from phenethylamine (NCCc1ccccc1) and 2-phenylethanol
    (OCCc1ccccc1) via ``sr.load_test_files("phenethylamine_2phenylethanol.s3")``.
    Contains one terminal phenyl ring.

``diphenylethane_mols``
    Merged system from 1,2-diphenylethane (c1ccccc1CCc1ccccc1) and
    1,2-diphenylethanol (OC(Cc1ccccc1)c1ccccc1) via
    ``sr.load_test_files("12diphenylethane_12diphenylethanol.s3")``.
    Contains two terminal phenyl rings.
"""

import pytest
import tempfile

import numpy as np

from somd2.config import Config
from somd2.runner import Runner
from somd2.runner._samplers import TerminalFlipSampler, detect_terminal_groups

# ---------------------------------------------------------------------------
# detect_terminal_groups
# ---------------------------------------------------------------------------


def test_no_terminal_groups(ethane_methanol):
    """
    The ethane → methanol perturbation contains no rings, so no terminal
    ring groups should be detected.
    """
    groups = detect_terminal_groups(ethane_methanol)
    assert groups == []


def test_detect_one_terminal_group(phenethyl_mols):
    """
    The phenethyl system has exactly one terminal ring (the phenyl group
    attached via the –CH2– chain).  H atoms bonded to ring carbons must not
    be reported as separate groups.
    """
    groups = detect_terminal_groups(phenethyl_mols)
    assert len(groups) == 1


def test_terminal_group_flip_angle(phenethyl_mols):
    """
    The default flip angle should be 180°.
    """
    groups = detect_terminal_groups(phenethyl_mols)
    angle, _ = groups[0]
    assert angle == pytest.approx(180.0)


def test_terminal_group_atom_count(phenethyl_mols):
    """
    For a mono-substituted benzene ring:
      - 1 anchor atom (aliphatic C adjacent to ring)
      - 1 pivot atom  (ipso ring C)
      - 5 mobile ring carbons
      - 5 mobile ring hydrogens
    Total indices list length = 12.
    """
    groups = detect_terminal_groups(phenethyl_mols)
    _, indices = groups[0]
    # anchor + pivot + 10 mobile atoms
    assert len(indices) == 12


def test_anchor_not_in_mobile(phenethyl_mols):
    """
    The anchor index must not appear in the mobile atom list.
    """
    groups = detect_terminal_groups(phenethyl_mols)
    _, indices = groups[0]
    anchor_idx = indices[0]
    mobile_indices = indices[2:]
    assert anchor_idx not in mobile_indices


def test_pivot_not_in_mobile(phenethyl_mols):
    """
    The pivot index must not appear in the mobile atom list (the pivot is the
    fixed rotation centre).
    """
    groups = detect_terminal_groups(phenethyl_mols)
    _, indices = groups[0]
    pivot_idx = indices[1]
    mobile_indices = indices[2:]
    assert pivot_idx not in mobile_indices


def test_auto_flip_angle_phenethyl(phenethyl_mols):
    """
    With no flip_angle override, the angle for a monosubstituted benzene ring
    should be auto-detected as 180° (C2 symmetry).
    """
    groups = detect_terminal_groups(phenethyl_mols)
    angle, _ = groups[0]
    assert angle == pytest.approx(180.0)


def test_auto_flip_angle_diphenylethane(diphenylethane_mols):
    """
    Both terminal phenyl groups in the diphenylethane system should
    auto-detect as 180°.
    """
    groups = detect_terminal_groups(diphenylethane_mols)
    assert len(groups) == 2
    for angle, _ in groups:
        assert angle == pytest.approx(180.0)


def test_custom_flip_angle(phenethyl_mols):
    """
    An explicit flip_angle override should be stored and returned as-is,
    bypassing the geometric auto-detection.
    """
    groups = detect_terminal_groups(phenethyl_mols, flip_angle=90.0)
    angle, _ = groups[0]
    assert angle == pytest.approx(90.0)


def test_detect_two_terminal_groups(diphenylethane_mols):
    """
    1,2-diphenylethane → 1,2-diphenylethanol has two terminal phenyl rings,
    each attached via a non-ring CH2/CH anchor, so exactly two groups should
    be detected.
    """
    groups = detect_terminal_groups(diphenylethane_mols)
    assert len(groups) == 2


def test_multiple_groups_unique_pivots(diphenylethane_mols):
    """
    The two terminal groups must have distinct pivot atoms (each ring has its
    own ipso carbon).
    """
    groups = detect_terminal_groups(diphenylethane_mols)
    pivot_indices = [indices[1] for _, indices in groups]
    assert len(set(pivot_indices)) == 2


def test_multiple_groups_disjoint_mobile(diphenylethane_mols):
    """
    The mobile atom sets of the two terminal groups must be disjoint — each
    group owns its own ring atoms.
    """
    groups = detect_terminal_groups(diphenylethane_mols)
    mobile_0 = set(groups[0][1][2:])
    mobile_1 = set(groups[1][1][2:])
    assert mobile_0.isdisjoint(mobile_1)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_terminal_flip_frequency_none():
    """terminal_flip_frequency defaults to None (disabled)."""
    config = Config()
    assert config.terminal_flip_frequency is None


def test_config_terminal_flip_frequency_valid():
    """A valid time string is parsed to a Sire GeneralUnit."""
    config = Config(terminal_flip_frequency="1 ps")
    assert config.terminal_flip_frequency is not None
    assert str(config.terminal_flip_frequency).startswith("1")


def test_config_terminal_flip_frequency_bad_units():
    """Non-time units should raise ValueError."""
    with pytest.raises(ValueError, match="units are invalid"):
        Config(terminal_flip_frequency="5 A")


def test_config_terminal_flip_frequency_bad_type():
    """A non-string value should raise TypeError."""
    config = Config()
    with pytest.raises(TypeError, match="must be of type 'str'"):
        config.terminal_flip_frequency = 5


def test_config_terminal_flip_angle_none():
    """terminal_flip_angle defaults to None (auto-detect)."""
    config = Config()
    assert config.terminal_flip_angle is None


def test_config_terminal_flip_angle_valid():
    """A valid angle string is parsed to a Sire GeneralUnit."""
    config = Config(terminal_flip_angle="180 degrees")
    assert config.terminal_flip_angle is not None


def test_config_terminal_flip_angle_bad_units():
    """Non-angle units should raise ValueError."""
    with pytest.raises(ValueError, match="units are invalid"):
        Config(terminal_flip_angle="5 A")


def test_config_terminal_flip_angle_bad_type():
    """A non-string value should raise TypeError."""
    config = Config()
    with pytest.raises(TypeError, match="must be of type 'str'"):
        config.terminal_flip_angle = 180


# ---------------------------------------------------------------------------
# TerminalFlipSampler
# ---------------------------------------------------------------------------


def test_sampler_initial_state(phenethyl_mols):
    """
    A freshly constructed sampler should report zero attempts and zero
    accepted moves.
    """
    groups = detect_terminal_groups(phenethyl_mols)
    sampler = TerminalFlipSampler(groups, 300.0)
    assert sampler.num_attempted == 0
    assert sampler.num_accepted == 0
    assert sampler.acceptance_rate == 0.0


def test_sampler_move(phenethyl_mols):
    """
    After one call to move(), num_attempted should be 1 and the statistics
    should be internally consistent.  The outcome (accepted or rejected)
    depends on the torsional energy around the exocyclic bond and is not
    deterministic for an arbitrary starting configuration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cpu",
            output_directory=tmpdir,
            num_lambda=1,
            lambda_values=[0.0],
            terminal_flip_frequency="4fs",
            energy_frequency="4fs",
            checkpoint_frequency="4fs",
            frame_frequency="4fs",
        )
        runner = Runner(phenethyl_mols, config)

        # Create a dynamics object to obtain an OpenMM context.
        dynamics_kwargs = runner._dynamics_kwargs.copy()
        dynamics = runner._system.dynamics(**dynamics_kwargs)

        groups = detect_terminal_groups(phenethyl_mols)
        sampler = TerminalFlipSampler(groups, 300.0)

        sampler.move(dynamics.context())

        assert sampler.num_attempted == 1
        assert sampler.num_accepted in (0, 1)
        assert 0.0 <= sampler.acceptance_rate <= 1.0


def test_rotate(phenethyl_mols):
    """
    _rotate() must:
      - leave the anchor and pivot atoms stationary,
      - move all mobile atoms,
      - restore all mobile atom positions after two consecutive 180° flips.
    """
    from openmm import unit as omm_unit

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            platform="cpu",
            output_directory=tmpdir,
            num_lambda=1,
            lambda_values=[0.0],
            terminal_flip_frequency="4fs",
            energy_frequency="4fs",
            checkpoint_frequency="4fs",
            frame_frequency="4fs",
        )
        runner = Runner(phenethyl_mols, config)

        dynamics_kwargs = runner._dynamics_kwargs.copy()
        dynamics = runner._system.dynamics(**dynamics_kwargs)
        context = dynamics.context()

        groups = detect_terminal_groups(phenethyl_mols)
        sampler = TerminalFlipSampler(groups, 300.0)

        _, indices = groups[0]
        anchor_idx = indices[0]
        pivot_idx = indices[1]
        mobile_indices = indices[2:]

        pos_before = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(omm_unit.nanometer)
        )

        sampler._rotate(context, 0, 180.0)

        pos_after = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(omm_unit.nanometer)
        )

        # Anchor and pivot must not move.
        np.testing.assert_allclose(
            pos_after[anchor_idx], pos_before[anchor_idx], atol=1e-5
        )
        np.testing.assert_allclose(
            pos_after[pivot_idx], pos_before[pivot_idx], atol=1e-5
        )

        # All mobile atoms must have moved.
        for idx in mobile_indices:
            assert not np.allclose(pos_after[idx], pos_before[idx], atol=1e-5), (
                f"Mobile atom {idx} did not move after 180° rotation"
            )

        # A second 180° flip must restore all mobile atom positions.
        sampler._rotate(context, 0, 180.0)
        pos_restored = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(omm_unit.nanometer)
        )
        np.testing.assert_allclose(
            pos_restored[mobile_indices], pos_before[mobile_indices], atol=1e-5
        )


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_runner_no_terminal_groups(ethane_methanol):
    """
    Setting terminal_flip_frequency on a ring-free molecule should succeed
    (0 groups detected) and the simulation should complete normally.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            runtime="12fs",
            output_directory=tmpdir,
            energy_frequency="4fs",
            checkpoint_frequency="4fs",
            frame_frequency="4fs",
            platform="cpu",
            max_threads=1,
            num_lambda=2,
            terminal_flip_frequency="4fs",
        )
        runner = Runner(ethane_methanol, config)
        assert runner._terminal_groups == []
        runner.run()


def test_runner_with_terminal_flip(phenethyl_mols):
    """
    With terminal_flip_frequency set and a terminal ring present, the runner
    should detect one group and complete the simulation successfully.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            runtime="12fs",
            output_directory=tmpdir,
            energy_frequency="4fs",
            checkpoint_frequency="4fs",
            frame_frequency="4fs",
            platform="cpu",
            max_threads=1,
            num_lambda=2,
            terminal_flip_frequency="4fs",
        )
        runner = Runner(phenethyl_mols, config)
        assert len(runner._terminal_groups) == 1
        runner.run()


def test_runner_validation_frequency_multiple(ethane_methanol):
    """
    terminal_flip_frequency must be a multiple of energy_frequency.
    A non-multiple should raise ValueError during runner initialisation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            output_directory=tmpdir,
            platform="cpu",
            num_lambda=2,
            energy_frequency="4fs",
            terminal_flip_frequency="3fs",  # not a multiple of 4fs
        )
        with pytest.raises(
            ValueError, match="must be a multiple of 'energy_frequency'"
        ):
            Runner(ethane_methanol, config)
