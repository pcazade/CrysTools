from inputOutput.poscar import *

import numpy as np
import math

# --- minimal args container ---
class Args:
    sd = True          # write Selective dynamics
    potcar = False     # don't write POTCAR
    kpoints = False    # don't write KPOINTS
    incar = False      # don't write INCAR



# ---- create a simple cubic cell ----
hmat = np.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 5.0],
])
cell = Cell(hmat=hmat)

# ---- create two atoms in Direct coordinates ----
a1 = Atom()
a1.el = "Si"
a1.x, a1.y, a1.z = 0.0, 0.0, 0.0

a2 = Atom()
a2.el = "Si"
a2.x, a2.y, a2.z = 0.5, 0.5, 0.5

atoms = [a1, a2]

# ---- args ----
args = Args()

# ---- call the function ----
writePoscar("POSCAR_test", atoms, cell, args)

print("writePoscar() ran successfully.")
print("Check file: POSCAR_test")
