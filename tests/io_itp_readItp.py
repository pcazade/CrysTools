from inputOutput.itp import *

# Option 1: using itpReader()
# tops = []
# ITP.itpReader("../data/sampleItpFile.itp", tops)
# print(f"Parsed {len(tops)} molecule(s).")
# print("Molecule name:", tops[0].molName)
# print("Number of atoms:", len(tops[0].atoms))
# print("First atom:", tops[0].atoms[0])

# Option 2: using the convenience wrapper
tops2 = readItp("../data/sampleItpFile.itp")
# print(f"\nParsed {len(tops2)} molecule(s) using parse().")
# print("Molecule Name:", tops2[0].molName)
# print("nrexcl:", tops2[0].nrexcl)
# print("Atoms Idx:", [a.idx for a in tops2[0].atoms])
# print("Mass:", [a.m for a in tops2[0].atoms])
# print("Bonds:", [b for b in tops2[0].bonds])
print("Atoms:", [a for a in tops2[0].atoms])
print("Bonds:", [b for b in tops2[0].bonds])
print("Pairs:", [p for p in tops2[0].pairs])
print("Angles:", [a for a in tops2[0].angles])
print("Angles:", [a for a in tops2[0].angles])
print("Dihedrals:", [d for d in tops2[0].dihedrals])
print("Impropers:", [i for i in tops2[0].impropers])
print("cmap:", [c for c in tops2[0].cmap])
