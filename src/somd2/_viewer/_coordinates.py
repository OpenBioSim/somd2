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

"""
The coordinates saved by a run, for the viewer.
"""

__all__ = ["Mismatch", "SavedCoordinates", "read_coordinates"]

import numpy as _np


class Mismatch(Exception):
    """
    The saved coordinates don't match the topology.
    """


class SavedCoordinates:
    """
    The coordinates of the whole system in nm, with the box vectors in nm and
    the simulation time in ps if known, and where each molecule's particles
    are among them.
    """

    def __init__(
        self, positions, box=None, time_ps=None, virtual_sites=None, atoms_only=False
    ):
        self.positions = positions
        self.box = box
        self.time_ps = time_ps
        # Rows of [first atom index, number of virtual sites], or None if
        # unknown. Not needed if the coordinates are of the atoms alone.
        self.virtual_sites = virtual_sites
        self.atoms_only = atoms_only

    def molecule(self, first_atom, num_atoms, num_system_atoms):
        """
        The coordinates of a molecule's atoms, given the index of its first
        atom and its number of atoms in a topology of num_system_atoms atoms.
        """
        if self.atoms_only:
            start = first_atom
        elif self.virtual_sites is not None:
            # A molecule's virtual sites follow its atoms.
            start = first_atom + sum(
                int(n) for atom, n in self.virtual_sites if atom < first_atom
            )
        elif first_atom == 0 or len(self.positions) == num_system_atoms:
            # Without the layout, the atom index is only trusted where virtual
            # sites can't have moved it.
            start = first_atom
        else:
            raise Mismatch()
        if len(self.positions) < start + num_atoms:
            raise Mismatch()
        return self.positions[start : start + num_atoms]


def read_coordinates(path):
    """
    Read the coordinates of the λ = 0 window from a checkpoint, the replica
    exchange state, or a legacy stream file.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".npz":
        with _np.load(path) as checkpoint:
            time_ps = checkpoint.get("time_ps")
            return SavedCoordinates(
                checkpoint["positions"],
                box=checkpoint.get("box"),
                time_ps=float(time_ps[0]) if time_ps is not None else None,
                virtual_sites=checkpoint.get("virtual_sites"),
            )

    if path.suffix == ".s3":
        import sire as _sr
        from sire.io import get_coords_array

        from ._data import sire_lock

        with sire_lock:
            # Perturbable molecules only have coordinates for each end state
            # until linked to one.
            system = _sr.morph.link_to_reference(_sr.stream.load(str(path)))
            positions = get_coords_array(system, units=_sr.units.nanometer)
        return SavedCoordinates(positions, atoms_only=True)

    return _read_repex(path)


def _read_repex(path):
    import openmm.unit as _unit

    from ._data import _RepexUnpickler

    with open(path, "rb") as f:
        state = _RepexUnpickler(f).load()

    # The saved states move with the mixing, so the first is always the
    # configuration in the λ = 0 window. Older states are stored the other
    # way round, as the runner also allows for.
    states = getattr(state, "_openmm_states", None) or []
    if not states:
        return None
    window = 0
    if not hasattr(state, "_num_slots"):
        window = int(_np.argsort(state._states)[0])
    saved = states[window]
    if saved is None:
        return None

    # Older states are OpenMM State objects rather than dicts.
    if isinstance(saved, dict):
        positions, box = saved["positions"], saved.get("box")
    else:
        positions = saved.getPositions(asNumpy=True)
        box = saved.getPeriodicBoxVectors(asNumpy=True)

    def nanometres(value):
        return _np.asarray(value.value_in_unit(_unit.nanometer))

    return SavedCoordinates(
        nanometres(positions),
        box=nanometres(box) if box is not None else None,
        time_ps=_picoseconds(getattr(state, "_time", None)),
        virtual_sites=getattr(state, "_virtual_sites", None),
    )


def _picoseconds(time):
    """
    A simulation time in ps, from a Sire unit, or None if it isn't known.
    """
    try:
        return float(time.to("ps"))
    except Exception:
        return None
