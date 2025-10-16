from numpy.linalg import det
from core.atom import Atom
from core.cell import Cell
from crystools import typeatom, cart2frac
import numpy as np


hmat = np.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 7.0]
])
cell = Cell(hmat=hmat)

# --- Define a few atoms (Cartesian coordinates) ---
atoms = [
    Atom(x=2.5, y=2.5, z=3.5),   # should map to (0.5, 0.5, 0.5)
    Atom(x=0.0, y=0.0, z=0.0),   # should map to (0.0, 0.0, 0.0)
    Atom(x=5.0, y=5.0, z=7.0)    # should map to (1.0, 1.0, 1.0)
]

print("Before conversion (Cartesian):")
for a in atoms:
    print(a)

# --- Run the conversion ---
cell.cart2frac(atoms)

print("\nAfter conversion (Fractional):")
for a in atoms:
    print(a)

# --- Optional: print matrix info ---
print("\nCell.hmat:\n", cell.hmat)
print("\nInverse transpose used:\n", np.transpose(np.linalg.inv(cell.hmat)))