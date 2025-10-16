from numpy.linalg import det
from core.atom import Atom
from core.cell import Cell
from crystools import typeatom, cart2frac
import numpy as np


cell = Cell()
cell.hmat = np.array([[10.0, 0.0, 0.0],
                      [0.0, 10.0, 0.0],
                      [0.0, 0.0, 10.0]], dtype=float)

# === Define some atoms, some of which lie outside the box ===
atoms = [
    Atom(x=12.3,  y=5.1,  z=3.7),   # outside along x
    Atom(x=-2.4,  y=1.0,  z=9.9),   # negative x
    Atom(x=3.0, y=11.2, z=-0.5),    # y too big, z negative
    Atom(x=5.0,  y=5.0,  z=5.0)     # already inside
]

print("Before applying PBC:")
for at in atoms:
    print(at)

# === Apply PBC in Cartesian mode ===
cell.pbc(atoms, isScaled=False)

print("\nAfter applying PBC:")
for at in atoms:
    print(at)