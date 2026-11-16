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

__all__ = ["auto_selection", "selection_geometry", "selection_sites", "ghost_waters"]

# The volume of a heavy atom and its hydrogens at protein density, in waters.
_HEAVY_ATOM_WATERS = 0.58

# The margin of ghost waters for fluctuations, as a fraction of the waters that
# fill the empty sphere, and its minimum.
_GHOST_WATER_MARGIN = 0.1
_MIN_GHOST_WATER_MARGIN = 10

# Selected atoms closer than this, in Angstrom, belong to the same site.
_SITE_DISTANCE = 4.0


def perturbed_atoms(mol):
    """
    Return the indices of the atoms of a perturbable molecule that change
    between the end states: atoms whose element changes, including ghost
    atoms, and heavy atoms whose Lennard-Jones parameters change. Hydrogens
    whose Lennard-Jones parameters change are left out, since this is often
    only a change in atom type caused by changes elsewhere in the molecule. If
    no atoms change, then all atoms are returned.

    Parameters
    ----------

    mol: sire.mol.Molecule
        The perturbable molecule.

    Returns
    -------

    indices: list of int
        The indices of the perturbed atoms within the molecule.
    """

    def changes(atom):
        element = atom.property("element0").symbol()
        if element != atom.property("element1").symbol():
            return True
        return element != "H" and atom.property("LJ0") != atom.property("LJ1")

    indices = [i for i, atom in enumerate(mol.atoms()) if changes(atom)]
    return indices or list(range(mol.num_atoms()))


def vanishing_atoms(mol):
    """
    Return the indices of the atoms of a perturbable molecule that vanish in
    either end state, i.e. ghost atoms, or atoms with no charge and no
    Lennard-Jones interactions, as for a decoupled ligand.

    Parameters
    ----------

    mol: sire.mol.Molecule
        The perturbable molecule.

    Returns
    -------

    indices: list of int
        The indices of the vanishing atoms within the molecule.
    """

    def decoupled(atom, end):
        return (
            atom.property(f"charge{end}").value() == 0
            and atom.property(f"LJ{end}").epsilon().value() == 0
        )

    return [
        i
        for i, atom in enumerate(mol.atoms())
        if "Xx"
        in (atom.property("element0").symbol(), atom.property("element1").symbol())
        or decoupled(atom, 0)
        or decoupled(atom, 1)
    ]


def auto_selection(system):
    """
    Return a selection string for the perturbed atoms of the perturbable
    molecules in the system. Perturbable waters and single-atom ions, e.g.
    co-alchemical ions, are left out unless they are the only perturbable
    molecules. Molecules are selected by index, rather than number, so that the
    selection also works for the system reloaded from file.

    Parameters
    ----------

    system: sire.system.System
        The system containing the perturbable molecules.

    Returns
    -------

    selection: str
        The selection string.
    """
    perturbable = system.molecules("property is_perturbable")
    try:
        solutes = system.molecules("property is_perturbable and not water")
    except KeyError:
        solutes = []
    solutes = [mol for mol in solutes if mol.num_atoms() > 1] or perturbable

    molecules = system.molecules()
    parts = []
    for mol in solutes:
        indices = ",".join(str(i) for i in perturbed_atoms(mol))
        parts.append(f"(molidx {molecules.find(mol)} and atomidx {indices})")
    return " or ".join(parts)


def _space(system):
    """
    Return the periodic space of the system, or None if it isn't periodic.
    """
    try:
        space = system.property("space")
    except Exception:
        return None
    return space if space.is_periodic() else None


def _box_matrix(space):
    """
    Return the box vectors of a periodic space as the columns of a matrix.
    """
    import numpy as _np

    if hasattr(space, "vector0"):
        vectors = [space.vector0(), space.vector1(), space.vector2()]
        return _np.array(
            [[v.x().value(), v.y().value(), v.z().value()] for v in vectors]
        ).T

    dimensions = space.dimensions()
    return _np.diag(
        [dimensions.x().value(), dimensions.y().value(), dimensions.z().value()]
    )


def _minimum_image(delta, space):
    """
    Return the minimum image of each displacement vector in a periodic space.
    """
    import itertools as _itertools

    import numpy as _np

    if space is None:
        return delta

    box = _box_matrix(space)
    fractional = _np.linalg.solve(box, delta.T)
    fractional -= _np.round(fractional)

    # Rounding the fractional coordinates isn't the minimum image for every
    # triclinic box, so take the nearest of the neighbouring images.
    shifts = _np.array(list(_itertools.product((-1, 0, 1), repeat=3))).T
    images = box @ (fractional[:, None, :] + shifts[:, :, None]).reshape(3, -1)
    images = images.reshape(3, shifts.shape[1], -1)
    nearest = _np.linalg.norm(images, axis=0).argmin(axis=0)
    return images[:, nearest, _np.arange(delta.shape[0])].T


def _positions(system, selection, end):
    """
    Return the positions of the selected atoms in an end state, in the order
    of the atoms in the system. Molecules without end-state coordinates use
    their current coordinates.
    """
    import numpy as _np
    from sire.io import get_coords_array as _get_coords_array

    atoms = system[selection].atoms()
    indices = _np.array(system.atoms().find(atoms))
    xyz = _np.empty((len(indices), 3))

    name = f"coordinates{end}"
    for mol in atoms.molecules():
        offset = system.atoms().find(mol.atoms()[0])
        mask = (indices >= offset) & (indices < offset + mol.num_atoms())
        prop = name if mol.has_property(name) else "coordinates"
        coords = _np.asarray(_get_coords_array(mol, map={"coordinates": prop}))
        xyz[mask] = coords[indices[mask] - offset]

    return xyz


def selection_geometry(system, selection):
    """
    Return the centre of geometry of the selected atoms in the reference state,
    and their largest distance from their own centre of geometry in either end
    state, in Angstrom. The GCMC sampler re-centres the sphere on the current
    positions, so each end state is measured from its own centre. Positions are
    imaged next to each other first, so that atoms on either side of a periodic
    boundary are handled correctly.

    Parameters
    ----------

    system: sire.system.System
        The system.

    selection: str
        A selection string for the atoms.

    Returns
    -------

    centre: numpy.ndarray
        The centre of geometry in the reference state, in Angstrom.

    extent: float
        The extent of the selected atoms in Angstrom.
    """
    import numpy as _np

    space = _space(system)

    centre = None
    extent = 0.0
    for end in (0, 1):
        xyz = _positions(system, selection, end)
        xyz = xyz[0] + _minimum_image(xyz - xyz[0], space)
        cog = xyz.mean(axis=0)
        if centre is None:
            centre = cog
        extent = max(extent, float(_np.linalg.norm(xyz - cog, axis=1).max()))

    return centre, extent


def selection_sites(system, selection):
    """
    Return the centres of geometry of the separate sites of the selected atoms
    in the reference state, where atoms closer than 4 Angstrom, directly or
    through other selected atoms, belong to the same site.

    Parameters
    ----------

    system: sire.system.System
        The system.

    selection: str
        A selection string for the atoms.

    Returns
    -------

    centres: list of numpy.ndarray
        The centre of geometry of each site, in Angstrom.
    """
    import numpy as _np

    xyz = _positions(system, selection, 0)
    xyz = xyz[0] + _minimum_image(xyz - xyz[0], _space(system))
    close = _np.linalg.norm(xyz[:, None] - xyz[None], axis=2) < _SITE_DISTANCE

    site = -_np.ones(len(xyz), dtype=int)
    centres = []
    for start in range(len(xyz)):
        if site[start] >= 0:
            continue
        site[start] = len(centres)
        stack = [start]
        while stack:
            i = stack.pop()
            for j in _np.flatnonzero(close[i] & (site < 0)):
                site[j] = site[start]
                stack.append(j)
        centres.append(xyz[site == site[start]].mean(axis=0))

    return centres


def ghost_waters(system, centre, radius, standard_volume):
    """
    Return the number of ghost waters needed to fill the free volume of a GCMC
    sphere: the waters that would fill the empty sphere, less
    the waters already inside and the volume of the heavy atoms of the
    environment. Atoms of perturbable molecules that vanish in either end
    state don't count towards the occupied volume. A margin is added for
    fluctuations.

    Parameters
    ----------

    system: sire.system.System
        The system.

    centre: numpy.ndarray
        The centre of the sphere in Angstrom.

    radius: float
        The radius of the sphere in Angstrom.

    standard_volume: float
        The volume of a water molecule in bulk water, in cubic Angstrom.

    Returns
    -------

    num_waters: int
        The number of ghost waters.
    """
    import math as _math

    import numpy as _np
    from sire.io import get_coords_array as _get_coords_array

    space = _space(system)
    xyz = _np.asarray(_get_coords_array(system))
    distances = _np.linalg.norm(_minimum_image(xyz - centre, space), axis=1)
    inside = distances < radius

    def indices(selection):
        try:
            return _np.array(system.atoms().find(system[selection].atoms()), dtype=int)
        except KeyError:
            return _np.array([], dtype=int)

    vanishing = []
    for mol in system.molecules("property is_perturbable"):
        offset = system.atoms().find(mol.atoms()[0])
        vanishing.extend(offset + i for i in vanishing_atoms(mol))

    heavy = _np.setdiff1d(indices("(not water) and (not element H)"), vanishing)
    occupied = int(inside[heavy].sum())
    waters = int(inside[indices("water and element O")].sum())

    empty = 4.0 / 3.0 * _math.pi * radius**3 / standard_volume
    free = max(_math.ceil(empty - _HEAVY_ATOM_WATERS * occupied - waters), 0)
    margin = max(_math.ceil(_GHOST_WATER_MARGIN * empty), _MIN_GHOST_WATER_MARGIN)
    return free + margin
