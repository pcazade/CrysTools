from inputOutput.gro import readGro
from inputOutput.itp import *

gro_file = "../data/fwn.gro"   # put your .gro filename here
chain_id = "A"

atoms, cell = readGro(gro_file, chain=chain_id)

print("=== READ GRO TEST ===")
print(f"Number of atoms: {len(atoms)}")
print()

# Print first 5 atoms
print("First 5 atoms:")
for a in atoms[:5]:
    print(f"Atom {a.idx:4d} | {a.resName:<4s} {a.name:<4s} | "
          f"x={a.x:8.3f} y={a.y:8.3f} z={a.z:8.3f}")

print("\n=== CELL PARAMETERS ===")
print(f"a     = {cell.a:.4f} Å")
print(f"b     = {cell.b:.4f} Å")
print(f"c     = {cell.c:.4f} Å")
print(f"alpha = {cell.alpha:.3f}°")
print(f"beta  = {cell.beta:.3f}°")
print(f"gamma = {cell.gamma:.3f}°")

print("\n=== HMAT ===")
print(cell.hmat)

print("\n=== GMAT (inverse) ===")
print(cell.gmat)