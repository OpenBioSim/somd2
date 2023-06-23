import sire as sr
import os
import re
import argparse
import csv

parser = argparse.ArgumentParser(
    prog="Add Anchors",
    description="A scipt to generate positional restraints for use in SOMD",
)

parser.add_argument(
    "-i",
    "--input",
    help="Input files - should contain sire compatible coordinate and topology files.",
    nargs=2,
    required=True,
)

parser.add_argument(
    "-r",
    "--restrained",
    help="Optional file - take in a .csv file that contains a list of restrained\
        atoms and use those. Bypasses sire operations used to find atoms. Assumes that the file is a single line containing only the list of atom numbers",
    type=str,
)

parser.add_argument(
    "-d",
    "--distances",
    help="The distances between which to restrain atoms. Default is between 12.0 and 15.0 angstrom.",
    nargs=2,
    type=float,
    default=[12.0, 15.0],
)

parser.add_argument(
    "-o",
    "--outname",
    help="Name of output file. If not defined the output file will be {name of first input file}_restrained.{filetype}",
    type=str,
)

args = parser.parse_args()

mols = sr.load(args.input[0], args.input[1])

if not args.restrained:
    if args.distances[0] >= args.distances[1]:
        raise ValueError(
            "Restraint distances are in the wrong order, please order -d small big"
        )
    # Use sire to find all heavy atoms between 12 and 15 angstrom of the ligand
    sr.search.set_token("lnd", "resname LIG")
    ligand = mols["lnd"]
    residues = mols[
        "((atoms within %s of lnd) and (not atoms within %s of lnd)) and protein"
        % (args.distances[1], args.distances[0])
    ]
    heavy = residues["not atomname /H*/"]
    an = heavy.numbers()
    restrained_atoms = []
    for at in an:
        restrained_atoms.append(int(re.findall(r"\d+", str(at))[0]))

else:
    with open(args.restrained) as f:
        reader = csv.reader(f)
        data = list(reader)
    restrained_atoms = [int(x) for x in data[0]]

print("Atomnums for restrained atoms:")
print(restrained_atoms)
# restrained_atoms = [18612, 18613, 18614, 18615, 18616, 18617, 18618]

n_existing_atoms = mols.num_atoms()
n_existing_residues = mols.num_residues()

newmol = sr.mol.Molecule("dummies")

editor = newmol.edit()

# Create a residue
editor = (
    editor.add(sr.mol.ResName("Re"))
    .renumber(sr.mol.ResNum(n_existing_residues + 1))
    .molecule()
)

for i in range(0, len(restrained_atoms)):
    editor = (
        editor.add(sr.mol.AtomName("Re"))
        .renumber(sr.mol.AtomNum(n_existing_atoms + i + 1))
        .reparent(sr.mol.ResIdx(0))
        .molecule()
    )

mol = editor.commit()

cursor = mol.cursor()["atomname Re"]

# need to set the properties to the correct type...
cursor[0]["charge"] = 1 * sr.units.mod_electron
cursor[0]["mass"] = 1 * sr.units.g_per_mol

for i in range(0, len(cursor)):
    atom = cursor.atom(i)
    restrained_at = mols["atomnum %i" % restrained_atoms[i]]
    atom["coordinates"] = restrained_at.property("coordinates")
    atom["charge"] = 0 * sr.units.mod_electron
    atom["element"] = sr.mol.Element(0)
    atom["mass"] = 0 * sr.units.g_per_mol
    atom["atomtype"] = "DM"
    atom["LJ"] = sr.mm.LJParameter(1 * sr.units.angstrom, 0 * sr.units.kcal_per_mol)

mol = cursor.molecule().commit()

mols.add(mol)

if args.outname:
    f = sr.save(mols, args.outname, format=["PDB"])
    f = sr.save(mols, args.outname, format=["PRM7", "RST7"])

else:
    origname = args.input[0].split(".")[0]
    f = sr.save(mols, origname + "_restrained", format=["PDB"])
    f = sr.save(mols, origname + "_restrained", format=["PRM7", "RST7"])

# load to check
mols = sr.load(f)

# list of tuples of atoms to save, will be used to specify restraints
# in somd-freenrg
paired_atoms = {}

last = mols[-1]
for i in range(0, len(last.atoms())):
    atom = last.atom(i)
    restrained_at = mols["atomnum %i" % restrained_atoms[i]]
    paired_atoms[atom.number().value()] = restrained_at.number().value()
    # print(atom, atom.property("charge"), atom.property("LJ"), atom.residue())

ofile = open("restraint.cfg", "w")
ofile.write("use restraints = True\n")
ofile.write("restrained atoms = %s\n" % paired_atoms)
ofile.close()
print("Atom pairs {dummy atom: original atom}:")
print(paired_atoms)
