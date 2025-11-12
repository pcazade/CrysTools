from inputOutput.pdb import *
import numpy as np


atoms, cell, spg = readPdb("../data/samplePdbFile.pdb")

# --- check what we got ---
print("\nPDB successfully read.")
print(f"Total atoms parsed: {len(atoms)}")
print(f"Space group: {spg.strip()}")
print("Cell parameters:")
print(f"  a={cell.a:.2f}, b={cell.b:.2f}, c={cell.c:.2f}")
print(f"  alpha={cell.alpha:.2f}, beta={cell.beta:.2f}, gamma={cell.gamma:.2f}")
print("hmat matrix:\n", np.array_str(cell.hmat, precision=2, suppress_small=True))

print("\nFirst atom details:")
if atoms:
    a = atoms[0]
    print(f"  idx={a.idx}, name={a.name.strip()}, resName={a.resName.strip()}, "
          f"chain={a.chain}, x={a.x:.3f}, y={a.y:.3f}, z={a.z:.3f}")
else:
    print("  No atoms found!")