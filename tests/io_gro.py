import numpy as np

from inputOutput.gro import readGro, writeGro

# gro_file = "../data/fwn.gro"   # put your .gro filename here
# chain_id = "A"
#
# atoms, cell = readGro(gro_file, chain=chain_id)
#
# print("=== READ GRO TEST ===")
# print(f"Number of atoms: {len(atoms)}")
# print()
#
# # Print first 5 atoms
# print("First 5 atoms:")
# for a in atoms[:5]:
#     print(f"Atom {a.idx:4d} | {a.resName:<4s} {a.name:<4s} | "
#           f"x={a.x:8.3f} y={a.y:8.3f} z={a.z:8.3f}")
#
# print("\n=== CELL PARAMETERS ===")
# print(f"a     = {cell.a:.4f} Å")
# print(f"b     = {cell.b:.4f} Å")
# print(f"c     = {cell.c:.4f} Å")
# print(f"alpha = {cell.alpha:.3f}°")
# print(f"beta  = {cell.beta:.3f}°")
# print(f"gamma = {cell.gamma:.3f}°")
#
# print("\n=== HMAT ===")
# print(cell.hmat)
#
# print("\n=== GMAT (inverse) ===")
# print(cell.gmat)



# -------------------------------------------------------------------------------



def compare_atoms(a1, a2):
    """Quick helper to compare two Atom objects"""
    attrs = ["resIdx", "resName", "name", "idx", "x", "y", "z"]
    for at in attrs:
        v1 = getattr(a1, at)
        v2 = getattr(a2, at)
        if isinstance(v1, str):
            if v1 != v2:
                print(f"Mismatch in {at}: {v1} != {v2}")
                return False
        else:
            if abs(v1 - v2) > 1e-6:
                print(f"Mismatch in {at}: {v1} != {v2}")
                return False
    return True



input_gro = "../data/fwn.gro"
output_gro = "written.gro"

print("=== Reading original GRO ===")
atoms1, cell1 = readGro(input_gro, chain="A")
print(f"Read {len(atoms1)} atoms.")

print("\n=== Writing GRO back ===")
writeGro(output_gro, atoms1, cell1, is_scaled=False)
print(f"Wrote {output_gro}")

print("\n=== Reading written GRO ===")
atoms2, cell2 = readGro(output_gro, chain="A")
print(f"Read {len(atoms2)} atoms.")

print("\n=== Checking atom count ===")
if len(atoms1) != len(atoms2):
    print("ERROR: Atom count mismatch!")
else:
    print("Atom count matches")

print("\n=== Checking coordinates for first 5 atoms ===")
for i in range(min(5, len(atoms1))):
    ok = compare_atoms(atoms1[i], atoms2[i])
    if not ok:
        print(f"Atom {i+1} differs!")
    else:
        print(f"Atom {i+1} OK")

print("\n=== Checking HMAT ===")
print("Original HMAT:")
print(cell1.hmat)
print("\nWritten HMAT:")
print(cell2.hmat)
if np.allclose(cell1.hmat, cell2.hmat, atol=1e-6):
    print("HMAT matches")
else:
    print("HMAT mismatch!")

print("\n==== TEST COMPLETE ====")