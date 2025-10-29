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

print(cell.gmat[2,:])

# --- Call wz() to compute reciprocal lattice and volume ---
ra, rb, rc, vol = cell.wz()
#
# # --- Print results ---
# print("Real-space lattice matrix (hmat):")
# print(cell.hmat)
#
# print("\nReciprocal lattice vectors (Å⁻¹):")
# print("a* =", ra)
# print("b* =", rb)
# print("c* =", rc)
#
# print(f"\nUnit cell volume: {vol:.4f}")