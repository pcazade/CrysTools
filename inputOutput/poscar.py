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





def writePoscar(fname, atoms:list[Atom], cell:Cell, args):
    """
        Write a VASP POSCAR file (Direct coordinates).

        Also optionally writes:
          - POTCAR   if args.potcar is True (concatenates per-element POTCARs)
          - KPOINTS  if args.kpoints is True
          - INCAR    if args.incar is True

        Notes:
          - Writes "Direct" coordinates (fractional). This function does not convert;
            it assumes Atom.x/y/z are already fractional.
          - If args.sd is True, writes 'Selective dynamics' and 'T T T' for each atom.
        """

    # ---- element list (preserve first-seen order) + counts ----
    lel = []
    nel = []
    for at in atoms:
        at.inferAtom()  # keep your behavior
        el = at.el.strip()
        if el not in lel:
            lel.append(el)
            nel.append(1)
        else:
            nel[lel.index(el)] += 1

    # ---- write POSCAR ----
    with open(fname, "w", encoding="utf-8") as fo:
        fo.write("Written by cp2k2pdb.py by P.-A. Cazade\n")
        fo.write("   1.00000000000000\n")

        # lattice vectors from Cell
        # assuming hmat rows are the a, b, c vectors
        for i in range(3):
            vx, vy, vz = cell.hmat[i, 0], cell.hmat[i, 1], cell.hmat[i, 2]
            fo.write(f" {vx:22.16f}{vy:22.16f}{vz:22.16f}\n")

        # element symbols line
        fo.write("".join(f"{el:5s}" for el in lel) + "\n")

        # element counts line
        fo.write("".join(f"{n:6d}" for n in nel) + "\n")

        # selective dynamics + coordinate mode
        if getattr(args, "sd", False):
            fo.write("Selective dynamics\n")
        fo.write("Direct\n")

        # group atoms by element in lel order (like your original)
        for el in lel:
            for at in atoms:
                if at.el.strip() != el:
                    continue

                # wrap fractional coords into [0,1)
                x = at.x - math.floor(at.x)
                y = at.y - math.floor(at.y)
                z = at.z - math.floor(at.z)

                if getattr(args, "sd", False):
                    fo.write(f"{x:20.16f}{y:20.16f}{z:20.16f} T T T\n")
                else:
                    fo.write(f"{x:20.16f}{y:20.16f}{z:20.16f}\n")

    # ---- optionally write POTCAR ----
    if getattr(args, "potcar", False):
        potcar_root = args.potcar_source[0].strip()
        with open("POTCAR", "w", encoding="utf-8") as fp:
            for el in lel:
                fpot = f"{potcar_root}/{el}/POTCAR"
                with open(fpot, "r", encoding="utf-8") as ft:
                    fp.writelines(ft.readlines())

    # ---- optionally write KPOINTS ----
    if getattr(args, "kpoints", False):
        # your original logic uses cell.wz() and ra.norm, rb.norm, rc.norm
        ra, rb, rc, vol = cell.wz()

        # NOTE: original code used args.kgrid[0] for all three, keep same behavior
        step = args.kgrid[0]
        is1 = round(ra.norm / step)
        is2 = round(rb.norm / step)
        is3 = round(rc.norm / step)

        with open("KPOINTS", "w", encoding="utf-8") as fk:
            fk.write("A\n")
            fk.write("0\n")
            fk.write("G\n")
            fk.write(f"{is1:d} {is2:d} {is3:d}\n")
            fk.write("0 0 0\n")

    # ---- optionally write INCAR ----
    if getattr(args, "incar", False):
        with open("INCAR", "w", encoding="utf-8") as fc:
            fc.write("Relax\n\n")
            fc.write("ISTART = 0\n")
            fc.write("ICHARG = 2\n\n")
            fc.write("PREC = Accurate\n")
            fc.write("EDIFF = 0.000001\n")
            fc.write("EDIFFG = 0.001\n\n")
            fc.write("IBRION = 2\n")
            fc.write("NSW = 199\n\n")
            fc.write("ISMEAR = 0\n")
            fc.write("SIGMA = 0.05\n\n")

            # ISIF logic copied from your code
            if "IONS" in args.cp2k_opt[0]:
                fc.write("ISIF = 2\n")
            elif "CELL" in args.cp2k_opt[0]:
                if getattr(args, "cp2k_opt_angles", False):
                    fc.write("ISIF = 8\n")
                else:
                    fc.write("ISIF = 3\n")
            fc.write("\n")

            fc.write("ENCUT = 800\n\n")
            fc.write("NPAR = 8\n\n")
            fc.write("LCHARG = .FALSE.\n")
            fc.write("LWAVE = .FALSE.\n\n")

            if getattr(args, "d3", False):
                fc.write("#DFT-D3\n")
                fc.write("IVDW = 11\n")
                fc.write("VDW_RADIUS = 50.2\n")
                fc.write("VDW_CNRADIUS = 20.0\n")
                fc.write("VDW_S6 = 1.0\n")
                fc.write("VDW_SR = 1.217\n")
                fc.write("VDW_S8 = 0.722\n\n")

    return
