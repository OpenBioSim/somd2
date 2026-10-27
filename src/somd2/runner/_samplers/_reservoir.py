######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# SOMD2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SOMD2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SOMD2. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = ["ReservoirSampler"]

import os as _os

import numpy as _np


class DCDReader:
    """
    Random access to the frames of a little-endian CHARMM-format DCD file,
    as written by OpenMM and Sire, using a memory map.
    """

    def __init__(self, filename):
        """
        Parameters
        ----------

        filename : str
            The path to the DCD file.
        """
        with open(filename, "rb") as f:
            size, magic = _np.fromfile(f, dtype="<i4", count=1)[0], f.read(4)
            if size != 84 or magic != b"CORD":
                raise ValueError("Not a little-endian CHARMM-format DCD file.")
            icntrl = _np.fromfile(f, dtype="<i4", count=21)[:20]

            title_size = _np.fromfile(f, dtype="<i4", count=1)[0]
            f.seek(title_size + 4, 1)

            self._num_atoms = int(_np.fromfile(f, dtype="<i4", count=3)[1])
            header_size = f.tell()

        if icntrl[8] != 0:
            raise ValueError("DCD files with fixed atoms aren't supported.")
        if icntrl[11] != 0:
            raise ValueError("DCD files with four dimensions aren't supported.")
        if icntrl[19] == 0 or icntrl[10] == 0:
            raise ValueError("The DCD file has no unit cell.")

        n = self._num_atoms
        dtype = [("cell", "<i4"), ("box", "<f8", 6), ("cell_end", "<i4")]
        for dim in "xyz":
            dtype += [(f"{dim}_start", "<i4"), (dim, "<f4", n), (f"{dim}_end", "<i4")]
        dtype = _np.dtype(dtype)

        num_frames = (_os.path.getsize(filename) - header_size) // dtype.itemsize
        self._frames = _np.memmap(
            filename, dtype=dtype, mode="r", offset=header_size, shape=(num_frames,)
        )

    @property
    def num_atoms(self):
        """The number of atoms in each frame."""
        return self._num_atoms

    @property
    def num_frames(self):
        """The number of frames."""
        return len(self._frames)

    def read(self, index):
        """
        Return the positions and box vectors, as rows, of a frame, in
        nanometers.
        """
        frame = self._frames[index]

        coords = _np.stack([frame["x"], frame["y"], frame["z"]], axis=1) / 10.0

        # The unit cell is stored as [A, gamma, B, beta, alpha, C], with the
        # angles as cosines (OpenMM) or in degrees (Sire).
        a, gamma, b, beta, alpha, c = frame["box"] / [10.0, 1, 10.0, 1, 1, 10.0]
        angles = _np.array([alpha, beta, gamma])
        if _np.all(_np.abs(angles) <= 1.0):
            alpha, beta, gamma = _np.arccos(angles)
        else:
            alpha, beta, gamma = _np.radians(angles)

        cx = c * _np.cos(beta)
        cy = c * (_np.cos(alpha) - _np.cos(beta) * _np.cos(gamma)) / _np.sin(gamma)
        box = _np.array(
            [
                [a, 0.0, 0.0],
                [b * _np.cos(gamma), b * _np.sin(gamma), 0.0],
                [cx, cy, _np.sqrt(c * c - cx * cx - cy * cy)],
            ]
        )
        box[_np.abs(box) < 1e-6] = 0.0

        return coords, box


def _min_image(d, box):
    """
    Wrap displacement vectors into the minimum image of a box whose rows are
    the box vectors.
    """
    return d - _np.round(d @ _np.linalg.inv(box)) @ box


def _anchor_frame(anchors, box):
    """
    Return an orthonormal frame, as rows, built from three anchor positions:
    the first axis points from the first anchor to the second, and the
    second axis lies in the plane of all three.
    """
    a = _min_image(anchors[1] - anchors[0], box)
    b = _min_image(anchors[2] - anchors[0], box)

    e1 = a / _np.linalg.norm(a)
    e2 = b - _np.dot(b, e1) * e1
    e2 /= _np.linalg.norm(e2)

    return _np.array([e1, e2, _np.cross(e1, e2)])


def rebuild_ligand(ligand, anchors_old, box_old, anchors_new, box_new):
    """
    Move the ligand rigidly so that its coordinates in the frame of the three
    receptor anchors are unchanged, which preserves every Boresch restraint
    coordinate.

    Parameters
    ----------

    ligand : np.ndarray
        The current ligand positions, shape (n, 3).

    anchors_old : np.ndarray
        The current receptor anchor positions, shape (3, 3).

    box_old : np.ndarray
        The current box vectors, as rows.

    anchors_new : np.ndarray
        The new receptor anchor positions, shape (3, 3).

    box_new : np.ndarray
        The new box vectors, as rows.

    Returns
    -------

    np.ndarray
        The new ligand positions, shape (n, 3).
    """
    # Displacements from the first anchor, shifted as a whole so that the
    # ligand stays intact.
    d = ligand - anchors_old[0]
    d -= d[0] - _min_image(d[0], box_old)

    local = d @ _anchor_frame(anchors_old, box_old).T

    return anchors_new[0] + local @ _anchor_frame(anchors_new, box_new)


class ReservoirSampler:
    """
    Monte Carlo sampler that replaces the configuration of the decoupled
    replica of an ABFE simulation with a frame drawn uniformly from a
    reservoir of Boltzmann samples of the ligand-free system.

    The ligand doesn't interact at the decoupled end state, so it is moved
    rigidly to keep its Boresch restraint coordinates fixed, and the move is
    always accepted. Dummy atoms that aren't in the frames, such as those of
    an alchemical ion, keep their displacement from a parent atom.
    """

    def __init__(
        self,
        filename,
        reservoir_idxs,
        ligand_idxs,
        dummy_groups,
        anchor_idxs,
        temperature,
    ):
        """
        Parameters
        ----------

        filename : str
            The path to the reservoir DCD trajectory.

        reservoir_idxs : list of int
            The index in the system of each atom in the frames.

        ligand_idxs : list of int
            The indices of the ligand atoms, which aren't in the frames.

        dummy_groups : list of tuple
            Each entry is (parent_idx, [dummy_idx, ...]) for dummy atoms that
            aren't in the frames and move with their parent.

        anchor_idxs : list of int
            The indices of the three Boresch receptor anchor atoms.

        temperature : float
            Simulation temperature in Kelvin.
        """
        self._dcd = DCDReader(filename)
        self._reservoir_idxs = _np.array(reservoir_idxs)
        self._ligand_idxs = _np.array(ligand_idxs)
        self._dummy_groups = [(p, _np.array(d)) for p, d in dummy_groups]
        self._anchor_idxs = _np.array(anchor_idxs)
        self._temperature = temperature

    def move(self, context):
        """
        Swap a randomly chosen reservoir frame into the context.

        Parameters
        ----------

        context : openmm.Context
            The active OpenMM context.

        Returns
        -------

        frame : int
            The index of the frame.

        delta_e : float
            The change in potential energy, in kJ/mol.
        """
        from openmm import unit as _omm_unit

        frame = _np.random.randint(self._dcd.num_frames)

        state = context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(_omm_unit.nanometer)
        box_old = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            _omm_unit.nanometer
        )
        e_old = state.getPotentialEnergy().value_in_unit(_omm_unit.kilojoule_per_mole)

        coords, box_new = self._dcd.read(frame)

        new_positions = _np.empty_like(positions)
        new_positions[self._reservoir_idxs] = coords
        for parent, dummies in self._dummy_groups:
            new_positions[dummies] = new_positions[parent] + _min_image(
                positions[dummies] - positions[parent], box_old
            )
        new_positions[self._ligand_idxs] = rebuild_ligand(
            positions[self._ligand_idxs],
            positions[self._anchor_idxs],
            box_old,
            new_positions[self._anchor_idxs],
            box_new,
        )

        context.setPeriodicBoxVectors(*(box_new * _omm_unit.nanometer))
        context.setPositions(new_positions * _omm_unit.nanometer)
        context.setVelocitiesToTemperature(self._temperature)

        e_new = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(_omm_unit.kilojoule_per_mole)
        )

        return frame, e_new - e_old
