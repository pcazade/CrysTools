import math
import numpy as np
from core import Atom, Cell


def readPoscar(fName):
    """
        Read a VASP POSCAR/CONTCAR file.

        Returns:
            atoms       : list[Atom]  (positions stored exactly as in file)
            cell        : Cell        (built from lattice hmat in Angstrom)
            isDirect    : bool        (True if "Direct", False if "Cartesian")
            isSelective : bool        (True if "Selective dynamics" is present)
        """

    isSelective = False
    isDirect = False

    with open(fName, "r", encoding="utf-8") as f:
        title = f.readline().strip()  # POSCAR line 1 (unused)
        coeff = float(f.readline().split()[0])  # POSCAR line 2

        # --- lattice vectors (3 lines) -> hmat (3x3) ---
        hmat = np.zeros((3, 3), dtype=float)
        for i in range(3):
            parts = f.readline().split()
            if len(parts) < 3:
                raise ValueError("Bad POSCAR: lattice vector line must have 3 numbers.")
            hmat[i, :] = [float(parts[0]), float(parts[1]), float(parts[2])]  # store
        hmat *= coeff  # apply scaling factor (common case: coeff > 0)

        # build Cell object from hmat
        cell = Cell(hmat=hmat)  # __post_init__ will fill a,b,c,alpha,beta,gamma,gmat

        # --- element symbols line ---
        lel = f.readline().split()  # e.g., ["Si", "O"]

        # --- element counts line ---
        parts = f.readline().split()  # e.g., ["2", "4"]
        nel = [int(x) for x in parts]

        # --- optional "Selective dynamics" + coordinate mode line ---
        line = f.readline().strip()
        if line.lower().startswith("s") or "selective" in line.lower():
            isSelective = True
            line = f.readline().strip()  # now should be Direct/Cartesian

        if line.lower().startswith("d"):
            isDirect = True
        elif line.lower().startswith("c"):
            isDirect = False
        else:
            raise ValueError(f'Bad POSCAR: expected "Direct" or "Cartesian", got: {line!r}')

        # --- read atoms ---
        atoms = []
        idx = 0

        for i, el in enumerate(lel):
            for _ in range(nel[i]):
                parts = f.readline().split()
                if len(parts) < 3:
                    raise ValueError("Bad POSCAR: coordinate line must have at least 3 numbers.")

                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])

                # If selective dynamics exists, POSCAR usually has 3 extra flags after xyz.
                # We keep it simple: parse them if present, otherwise ignore.
                # Example: ["0.1","0.2","0.3","T","T","F"]
                selective_flags = None
                if isSelective and len(parts) >= 6:
                    selective_flags = parts[3:6]  # ["T","T","F"] (optional use)

                idx += 1
                a = Atom()
                a.name = el
                a.el = el
                a.idx = idx
                a.x, a.y, a.z = x, y, z

                atoms.append(a)

    return atoms, cell, isDirect, isSelective





# def writePoscar(fName, atoms:list[Atom], cell:Cell, args):
#     if (args.potcar):
#         fp = open("POTCAR", 'w')
#     fo = open(fName, 'w')
#     fo.write("Written by cp2k2pdb.py by P.-A. Cazade\n")
#     fo.write("   1.00000000000000\n")
#     fo.write(" %22.16f%22.16f%22.16f\n" % (a.x, a.y, a.z))
#     fo.write(" %22.16f%22.16f%22.16f\n" % (b.x, b.y, b.z))
#     fo.write(" %22.16f%22.16f%22.16f\n" % (c.x, c.y, c.z))
#     lel = []
#     nel = []
#     for at in atoms:
#         at.inferAtom()
#         if (at.el.strip() not in lel):
#             lel.append(at.el.strip())
#             nel.append(1)
#         else:
#             i = lel.index(at.el.strip())
#             nel[i] += 1
#     for el in lel:
#         fo.write("%5s" % (el))
#     fo.write("\n")
#     for n in nel:
#         fo.write("%6d" % (n))
#     fo.write("\n")
#     if (args.sd):
#         fo.write("Selective dynamics\n")
#     fo.write("Direct\n")
#     for el in lel:
#         if (args.potcar):
#             fpot = args.potcar_source[0].strip() + '/' + el.strip() + '/POTCAR'
#             ft = open(fpot, 'r')
#             for line in ft:
#                 fp.write(line)
#             ft.close()
#         for at in atoms:
#             if (at.el.strip() == el):
#                 at.x -= math.floor(at.x)
#                 at.y -= math.floor(at.y)
#                 at.z -= math.floor(at.z)
#                 if (args.sd):
#                     fo.write("%20.16f%20.16f%20.16f T T T\n" % (at.x, at.y, at.z))
#                 else:
#                     fo.write("%20.16f%20.16f%20.16f\n" % (at.x, at.y, at.z))
#     fo.close()
#     if (args.potcar):
#         fp.close()
#     if (args.kpoints):
#         fk = open("KPOINTS", 'w')
#         ra, rb, rc, vol = cell.wz()
#         is1 = round(ra.norm / args.kgrid[0])
#         is2 = round(rb.norm / args.kgrid[0])
#         is3 = round(rc.norm / args.kgrid[0])
#         fk.write("A\n")
#         fk.write("0\n")
#         fk.write("G\n")
#         fk.write("%d %d %d\n" % (is1, is2, is3))
#         fk.write("0 0 0\n")
#         fk.close()
#     if (args.incar):
#         fc = open("INCAR", 'w')
#         fc.write("Relax\n")
#         fc.write("\n")
#         fc.write("ISTART = 0\n")
#         fc.write("ICHARG = 2\n")
#         fc.write("\n")
#         fc.write("PREC = Accurate\n")
#         fc.write("EDIFF = 0.000001\n")
#         fc.write("EDIFFG = 0.001\n")
#         fc.write("\n")
#         fc.write("IBRION = 2\n")
#         fc.write("NSW = 199\n")
#         fc.write("\n")
#         fc.write("ISMEAR = 0\n")
#         fc.write("SIGMA = 0.05\n")
#         fc.write("\n")
#         if ("IONS" in args.cp2k_opt[0]):
#             fc.write("ISIF = 2\n")
#         elif ("CELL" in args.cp2k_opt[0]):
#             if (args.cp2k_opt_angles):
#                 fc.write("ISIF = 8\n")
#             else:
#                 fc.write("ISIF = 3\n")
#         fc.write("\n")
#         fc.write("ENCUT = 800\n")
#         fc.write("\n")
#         fc.write("NPAR = 8\n")
#         fc.write("\n")
#         fc.write("LCHARG = .FALSE.\n")
#         fc.write("LWAVE = .FALSE.\n")
#         fc.write("\n")
#         if (args.d3):
#             fc.write("#DFT-D3\n")
#             fc.write("IVDW = 11\n")
#             fc.write("VDW_RADIUS = 50.2\n")
#             fc.write("VDW_CNRADIUS = 20.0\n")
#             fc.write("VDW_S6 = 1.0\n")
#             fc.write("VDW_SR = 1.217\n")
#             fc.write("VDW_S8 = 0.722\n")
#             fc.write("\n")
#         fc.close()
#     return