from inputOutput.pdb import *
import numpy as np



cell = Cell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)

# Create some dummy atoms
atoms = [
    Atom(idx=1, name="N",  resName="MET", chain="A", resIdx=1, x=1.0, y=2.0, z=3.0, el="N"),
    Atom(idx=2, name="CA", resName="MET", chain="A", resIdx=1, x=2.5, y=3.0, z=4.0, el="C"),
    Atom(idx=3, name="O",  resName="MET", chain="A", resIdx=1, x=4.0, y=4.5, z=5.0, el="O"),
]

# Write to file
out_file = "test_cell_based.pdb"
writePdb(out_file, atoms, cell, spg="P 21 21 21")

# Print confirmation + file preview
print(f"\nSuccessfully wrote '{out_file}'\nPreview:")
with open(out_file, "r") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i > 10:
            break