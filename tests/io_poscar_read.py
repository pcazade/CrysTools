from inputOutput.poscar import *

atoms, cell, isDirect, isSelective = readPoscar("../data/samplePoscar.poscar")

print("=== POSCAR READ TEST ===")
print(f"isDirect     : {isDirect}")
print(f"isSelective  : {isSelective}")
print()

print("Cell (hmat):")
print(cell.hmat)
print()

print("Atoms:")
for a in atoms:
    print(
        f"idx={a.idx:2d}, el={a.el}, "
        f"x={a.x:.3f}, y={a.y:.3f}, z={a.z:.3f}",
        end=""
    )
    if hasattr(a, "selective_flags"):
        print(f", selective={getattr(a, 'selective_flags', None)}")
    else:
        print()
