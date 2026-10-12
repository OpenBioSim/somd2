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
A Blender script that sets up a binding site or mutation, as shown by the
viewer, for rendering with Molecular Nodes.
"""

__all__ = ["blender_script"]

# Backbone atoms, which are left out of residues drawn over the cartoon.
_BACKBONE = {"N", "C", "O", "OXT"}

# Covalent radii in Å, for finding the bonds of atoms that residue templates
# don't cover, e.g. of a ligand.
_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def blender_script(site, name, state=0, show_waters=True, view=None, aspect=None):
    """
    A Blender script for the binding site or mutation.

    Parameters
    ----------

    site: dict
        The binding site, as given by binding_site.

    name: str
        The name of the run.

    state: int
        The end state of a mutation to show.

    show_waters: bool
        Whether to show the waters.

    view: dict
        The viewer's view: the "centre" it looks at and the camera's
        "distance" from it, in Å, the "rotation" of the scene as a quaternion
        (x, y, z, w), and the vertical field of view, "fov", in degrees. The
        view is chosen from the structure if None.

    aspect: float
        The width of the view divided by its height, which sets that of the
        render.

    Returns
    -------

    script: str
        The script.
    """
    import numpy as _np

    if site["kind"] == "mutation":
        shown = site["states"][state]
        detail = shown["nearby"]
        focus = shown["mutated"]
        ligand = shown["ligand"]
    else:
        shown = site
        detail = site["pocket"]
        focus = site["ligand"]
        ligand = site["ligand"]

    lines = [
        line
        for line in shown["pdb"].splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    serials = set(shown["waters"])
    if not show_waters:
        lines = [line for line in lines if int(line[6:11]) not in serials]
    records = {int(line[6:11]): line for line in lines}

    def position(serial):
        record = records[serial]
        return _np.array([float(record[x : x + 8]) for x in (30, 38, 46)])

    # Residues drawn over the cartoon only from Cα outwards, as in the viewer.
    def side_chain(serial):
        record = records[serial]
        atom, residue = record[12:16].strip(), record[17:20].strip()
        if residue == "GLY":
            return False
        return atom not in _BACKBONE or (atom == "N" and residue == "PRO")

    detail = [s for s in detail if s in records and side_chain(s)]
    waters = [s for s in shown["waters"] if s in records]
    bonded = set(focus) | set(ligand) | set(waters)

    pdb = "\n".join(lines + _conect(records, bonded) + ["END"]) + "\n"

    # A mutated residue's name, e.g. MUT, isn't recognised as protein, which
    # would break the cartoon.
    cartoon = "protein"
    if site["kind"] == "mutation":
        cartoon = f"protein or {_bynum(focus)}"
    styles = [("cartoon", cartoon, "(0.8, 0.8, 0.8, 1.0)")]
    if detail:
        styles.append(("sticks", _bynum(detail), "None"))
    if site["kind"] == "mutation":
        styles.append(("ball_and_stick", _bynum(focus), "None"))
        if ligand:
            styles.append(("ball_and_stick", _bynum(ligand), "None"))
    else:
        styles.append(("ball_and_stick", _bynum(ligand), "None"))
    if waters:
        styles.append(("sticks", _bynum(waters), "None"))

    if view is None:
        view = _default_view(
            _np.array([position(s) for s in focus]),
            _np.array([position(s) for s in records if records[s].startswith("ATOM")]),
        )

    height = 1080
    width = height if not aspect else 2 * round(height * aspect / 2)

    # Shown with the waters, as in the viewer.
    region = shown.get("gcmc") if show_waters else None

    what = "The mutation" if site["kind"] == "mutation" else "The binding site"
    return _SCRIPT.format(
        title=f"{what} of {name}",
        pdb=pdb,
        styles="\n".join(
            f'mol.add_style("{style}", selection="{selection}", color={color})'
            for style, selection, color in styles
        ),
        centre=[round(float(x), 3) for x in view["centre"]],
        rotation=[round(float(x), 6) for x in view["rotation"]],
        distance=round(float(view["distance"]), 3),
        fov=round(float(view["fov"]), 3),
        width=width,
        height=height,
        gcmc_centre=(
            [round(float(x), 3) for x in region["centre"]] if region else None
        ),
        gcmc_radius=round(float(region["radius"]), 3) if region else None,
    )


def _bynum(serials):
    """
    An MDAnalysis selection of atoms by their serial numbers, as ranges.
    """
    serials = sorted(serials)
    ranges = []
    start = previous = serials[0]
    for serial in serials[1:]:
        if serial != previous + 1:
            ranges.append((start, previous))
            start = serial
        previous = serial
    ranges.append((start, previous))
    return "bynum " + " ".join(f"{a}:{b}" if a != b else f"{a}" for a, b in ranges)


def _conect(records, serials):
    """
    CONECT records for the bonds, found by distance, between the given atoms
    within each residue, which templates may not cover.
    """
    import numpy as _np

    residues = {}
    for serial in serials:
        residues.setdefault(records[serial][17:27], []).append(serial)

    bonds = {}
    for members in residues.values():
        xyz = _np.array(
            [
                [
                    float(records[s][30:38]),
                    float(records[s][38:46]),
                    float(records[s][46:54]),
                ]
                for s in members
            ]
        )
        radii = _np.array([_RADII.get(records[s][76:78].strip(), 0.8) for s in members])
        distances = _np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        limit = 1.25 * (radii[:, None] + radii[None, :])
        for i, j in zip(*_np.where(_np.triu(distances < limit, k=1))):
            bonds.setdefault(members[i], []).append(members[j])
            bonds.setdefault(members[j], []).append(members[i])

    lines = []
    for serial, partners in sorted(bonds.items()):
        # A record holds at most four bonded atoms.
        partners = sorted(partners)
        for i in range(0, len(partners), 4):
            lines.append(
                f"CONECT{serial:>5}" + "".join(f"{p:>5}" for p in partners[i : i + 4])
            )
    return lines


def _default_view(focus, protein):
    """
    A view of the focus atoms from outside the protein, along the line from
    its centre through them, fitting them in a 20° field of view.
    """
    import numpy as _np

    centre = focus.mean(axis=0)
    direction = centre - protein.mean(axis=0)
    direction /= _np.linalg.norm(direction)
    radius = _np.linalg.norm(focus - centre, axis=1).max() + 6.0
    distance = radius / _np.tan(_np.radians(10.0))

    # The rotation that turns the direction to the camera onto +z.
    z = _np.array([0.0, 0.0, 1.0])
    axis = _np.cross(direction, z)
    sine, cosine = _np.linalg.norm(axis), float(direction @ z)
    if sine < 1e-8:
        rotation = [0.0, 0.0, 0.0, 1.0] if cosine > 0 else [1.0, 0.0, 0.0, 0.0]
    else:
        half = _np.arctan2(sine, cosine) / 2
        axis /= sine
        rotation = list(axis * _np.sin(half)) + [_np.cos(half)]
    return {"centre": centre, "rotation": rotation, "distance": distance, "fov": 20.0}


_SCRIPT = '''"""
{title}, as shown by the SOMD2 viewer, for rendering
with Molecular Nodes (https://github.com/BradyAJohnston/MolecularNodes).

Run it in Blender's Scripting tab with the Molecular Nodes extension
installed, then render with F12. It can be run again, e.g. after changing the
settings below, and replaces what it set up before.
"""

import tempfile
from math import radians
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion, Vector

# Blender installs extensions under its own namespace.
try:
    import molecularnodes as mn
except ModuleNotFoundError:
    import bl_ext.blender_org.molecularnodes as mn

# Settings that can be changed.
RESOLUTION = ({width}, {height})
TRANSPARENT = False
# How the GCMC region is drawn: "wireframe", "shell" or None to leave it out.
GCMC_STYLE = "wireframe"

# The viewer's view: the point looked at and the camera's distance from it,
# in Å, the scene's rotation as a quaternion (x, y, z, w), and the vertical
# field of view in degrees.
CENTRE = {centre}
ROTATION = {rotation}
DISTANCE = {distance}
FOV = {fov}

# The centre and radius of the GCMC region in Å, if there is one.
GCMC_CENTRE = {gcmc_centre}
GCMC_RADIUS = {gcmc_radius}

PDB = """{pdb}"""

# Written to a temporary file, since Molecular Nodes loads from files.
path = Path(tempfile.mkdtemp()) / "structure.pdb"
path.write_text(PDB)

canvas = mn.Canvas(resolution=RESOLUTION, transparent=TRANSPARENT)
canvas.clear()

mol = mn.Molecule.load(str(path))
{styles}

# The camera is placed as in the viewer, which rotates the scene rather than
# the camera, so the camera turns the opposite way.
x, y, z, w = ROTATION
turn = Quaternion((w, x, y, z)).conjugated()
camera = bpy.context.scene.camera
camera.location = (Vector(CENTRE) + turn @ Vector((0.0, 0.0, DISTANCE))) * mol.world_scale
camera.rotation_mode = "QUATERNION"
camera.rotation_quaternion = turn
camera.data.sensor_fit = "VERTICAL"
camera.data.angle = radians(FOV)
camera.data.clip_start = DISTANCE * mol.world_scale / 100
camera.data.clip_end = DISTANCE * mol.world_scale * 100

if GCMC_RADIUS is not None and GCMC_STYLE is not None:
    # Built from data, since operators need a context that scripts lack.
    mesh = bpy.data.meshes.new("GCMC region")
    shape = bmesh.new()
    bmesh.ops.create_uvsphere(
        shape, u_segments=32, v_segments=16, radius=GCMC_RADIUS * mol.world_scale
    )
    shape.to_mesh(mesh)
    shape.free()
    sphere = bpy.data.objects.new("GCMC region", mesh)
    sphere.location = Vector(GCMC_CENTRE) * mol.world_scale
    bpy.context.scene.collection.objects.link(sphere)
    material = bpy.data.materials.new("GCMC region")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.15, 0.39, 0.79, 1.0)
    if GCMC_STYLE == "wireframe":
        wireframe = sphere.modifiers.new("Wireframe", "WIREFRAME")
        wireframe.thickness = 0.08 * mol.world_scale
        shader.inputs["Alpha"].default_value = 0.6
    else:
        for polygon in sphere.data.polygons:
            polygon.use_smooth = True
        shader.inputs["Alpha"].default_value = 0.15
    # Transparency is set differently before Blender 4.2.
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    else:
        material.blend_method = "BLEND"
    sphere.data.materials.append(material)
'''
