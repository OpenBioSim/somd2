import sys
import os
import subprocess

# Get the name of the script.
script = os.path.abspath(sys.argv[0])

# we want to import the 'get_requirements' package from this directory
sys.path.insert(0, os.path.dirname(script))

# go up one directories to get the source directory
# (this script is in BioSimSpace/actions/)
srcdir = os.path.dirname(os.path.dirname(script))

condadir = os.path.join(srcdir, "recipes", "somd2")

print(f"conda recipe in {condadir}")

# Store the name of the recipe and template YAML files.
recipe = os.path.join(condadir, "meta.yaml")
template = os.path.join(condadir, "template.yaml")

gitdir = os.path.join(srcdir, ".git")


def run_cmd(cmd):
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    return str(p.stdout.read().decode("utf-8")).lstrip().rstrip()


# Get the remote.
remote = run_cmd(
    f"git --git-dir={gitdir} --work-tree={srcdir} config --get remote.origin.url"
)
print(remote)

# Get the branch.
branch = run_cmd(
    f"git --git-dir={gitdir} --work-tree={srcdir} rev-parse --abbrev-ref HEAD"
)
print(branch)

lines = open(template, "r").readlines()

with open(recipe, "w") as FILE:
    for line in lines:
        line = line.replace("SOMD2_REMOTE", remote)
        line = line.replace("SOMD2_BRANCH", branch)

        FILE.write(line)
