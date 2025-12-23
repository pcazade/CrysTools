import math

import numpy as np

from core import Atom, Cell


def readPoscar(fName):
    isSelective = False
    isScaled = False
    fo = open(fName, 'r')
    title = fo.readline()
    coeff = float(fo.readline())
    # --- read 3 lattice vectors into a 3x3 matrix ---
    hmat = np.zeros((3, 3), dtype=float)
    for i in range(3):
        words = fo.readline().split()
        hmat[i, 0] = float(words[0]) * coeff
        hmat[i, 1] = float(words[1]) * coeff
        hmat[i, 2] = float(words[2]) * coeff
    # a = lattice_vect()
    # b = lattice_vect()
    # c = lattice_vect()
    # words = fo.readline().split()

    # build Cell object from hmat
    cell = Cell(hmat=hmat)  # __post_init__ will fill a,b,c,alpha,beta,gamma,gmat





    # a.x = float(words[0]) * coeff
    # a.y = float(words[1]) * coeff
    # a.z = float(words[2]) * coeff
    # words = fo.readline().split()
    # b.x = float(words[0]) * coeff
    # b.y = float(words[1]) * coeff
    # b.z = float(words[2]) * coeff
    # words = fo.readline().split()
    # c.x = float(words[0]) * coeff
    # c.y = float(words[1]) * coeff
    # c.z = float(words[2]) * coeff
    words = fo.readline().split()
    lel = []
    for w in words:
        lel.append(w)

    words = fo.readline().split()
    nel = []
    for w in words:
        nel.append(int(w))

    line = fo.readline()
    if (("selective" in line.lower()) or (line[0].lower() == 's')):
        isSelective = True
        line = fo.readline()

    if ("direct" in line.lower()):
        isScaled = True

    atoms = []
    for i in range(len(nel)):
        for j in range(nel[i]):
            line = fo.readline()
            words = line.split()
            atoms.append(Atom())
            idx = len(atoms) - 1
            atoms[idx].name = lel[i]
            atoms[idx].el = lel[i]
            atoms[idx].idx = idx + 1
            atoms[idx].x = float(words[0])
            atoms[idx].y = float(words[1])
            atoms[idx].z = float(words[2])

    # you no longer need a.norm, b.norm, c.norm here:
    # - lengths are in cell.a, cell.b, cell.c
    # - full vectors are in cell.hmat[0], cell.hmat[1], cell.hmat[2]

    # a.norm = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) #use
    # b.norm = math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z)
    # c.norm = math.sqrt(c.x * c.x + c.y * c.y + c.z * c.z)

    fo.close()
    return (atoms, cell, isScaled)



def writePoscar(fName, atoms:list[Atom], cell:Cell, args):
    if (args.potcar):
        fp = open("POTCAR", 'w')
    fo = open(fName, 'w')
    fo.write("Written by cp2k2pdb.py by P.-A. Cazade\n")
    fo.write("   1.00000000000000\n")
    fo.write(" %22.16f%22.16f%22.16f\n" % (a.x, a.y, a.z))
    fo.write(" %22.16f%22.16f%22.16f\n" % (b.x, b.y, b.z))
    fo.write(" %22.16f%22.16f%22.16f\n" % (c.x, c.y, c.z))
    lel = []
    nel = []
    for at in atoms:
        at.inferAtom()
        if (at.el.strip() not in lel):
            lel.append(at.el.strip())
            nel.append(1)
        else:
            i = lel.index(at.el.strip())
            nel[i] += 1
    for el in lel:
        fo.write("%5s" % (el))
    fo.write("\n")
    for n in nel:
        fo.write("%6d" % (n))
    fo.write("\n")
    if (args.sd):
        fo.write("Selective dynamics\n")
    fo.write("Direct\n")
    for el in lel:
        if (args.potcar):
            fpot = args.potcar_source[0].strip() + '/' + el.strip() + '/POTCAR'
            ft = open(fpot, 'r')
            for line in ft:
                fp.write(line)
            ft.close()
        for at in atoms:
            if (at.el.strip() == el):
                at.x -= math.floor(at.x)
                at.y -= math.floor(at.y)
                at.z -= math.floor(at.z)
                if (args.sd):
                    fo.write("%20.16f%20.16f%20.16f T T T\n" % (at.x, at.y, at.z))
                else:
                    fo.write("%20.16f%20.16f%20.16f\n" % (at.x, at.y, at.z))
    fo.close()
    if (args.potcar):
        fp.close()
    if (args.kpoints):
        fk = open("KPOINTS", 'w')
        ra, rb, rc, vol = cell.wz()
        is1 = round(ra.norm / args.kgrid[0])
        is2 = round(rb.norm / args.kgrid[0])
        is3 = round(rc.norm / args.kgrid[0])
        fk.write("A\n")
        fk.write("0\n")
        fk.write("G\n")
        fk.write("%d %d %d\n" % (is1, is2, is3))
        fk.write("0 0 0\n")
        fk.close()
    if (args.incar):
        fc = open("INCAR", 'w')
        fc.write("Relax\n")
        fc.write("\n")
        fc.write("ISTART = 0\n")
        fc.write("ICHARG = 2\n")
        fc.write("\n")
        fc.write("PREC = Accurate\n")
        fc.write("EDIFF = 0.000001\n")
        fc.write("EDIFFG = 0.001\n")
        fc.write("\n")
        fc.write("IBRION = 2\n")
        fc.write("NSW = 199\n")
        fc.write("\n")
        fc.write("ISMEAR = 0\n")
        fc.write("SIGMA = 0.05\n")
        fc.write("\n")
        if ("IONS" in args.cp2k_opt[0]):
            fc.write("ISIF = 2\n")
        elif ("CELL" in args.cp2k_opt[0]):
            if (args.cp2k_opt_angles):
                fc.write("ISIF = 8\n")
            else:
                fc.write("ISIF = 3\n")
        fc.write("\n")
        fc.write("ENCUT = 800\n")
        fc.write("\n")
        fc.write("NPAR = 8\n")
        fc.write("\n")
        fc.write("LCHARG = .FALSE.\n")
        fc.write("LWAVE = .FALSE.\n")
        fc.write("\n")
        if (args.d3):
            fc.write("#DFT-D3\n")
            fc.write("IVDW = 11\n")
            fc.write("VDW_RADIUS = 50.2\n")
            fc.write("VDW_CNRADIUS = 20.0\n")
            fc.write("VDW_S6 = 1.0\n")
            fc.write("VDW_SR = 1.217\n")
            fc.write("VDW_S8 = 0.722\n")
            fc.write("\n")
        fc.close()
    return