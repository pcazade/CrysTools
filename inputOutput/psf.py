import math

from core import Topology, Atom


def readPsf(fname):
    fi = open(fname, 'r')
    psf = Topology()
    psf.molName = ''
    psf.atoms = []
    psf.bonds = []
    psf.pairs = []
    psf.angles = []
    psf.dihedrals = []
    psf.impropers = []
    psf.cmap = []
    isAtom = False
    isBond = False
    isTheta = False
    isPhi = False
    isImphi = False
    isCrt = False
    isExt = False
    isFirst = True
    for line in fi:
        if (isFirst):
            isFirst = False
            if ('EXT' in line):
                isExt = True
        words = line.split()
        if (len(words) == 0):
            continue
        if ('!NATOM' in line):
            psf.nAtoms = int(words[0])
            # psf.atoms=[]
            isAtom = True
            i = 0
            continue
        if ('!NBOND' in line):
            psf.nBonds = int(words[0])
            # psf.bonds=[]
            isBond = True
            i = 0
            continue
        if ('!NTHETA' in line):
            psf.nAngles = int(words[0])
            # psf.theta=[]
            isTheta = True
            i = 0
            continue
        if ('!NPHI' in line):
            psf.nDihedrals = int(words[0])
            # psf.phi=[]
            isPhi = True
            i = 0
            continue
        if ('!NIMPHI' in line):
            psf.nImpropers = int(words[0])
            # psf.imphi=[]
            isImphi = True
            i = 0
            continue
        if ('!NCRTERM' in line):
            psf.nCmap = int(words[0])
            # psf.crt=[]
            isCrt = True
            i = 0
            continue
        if (isAtom and i < psf.nAtoms):
            psf.atoms.append(Atom())
            psf.atoms[i].idx = int(words[0])
            psf.atoms[i].segName = words[1]
            psf.atoms[i].resIdx = int(words[2])
            psf.atoms[i].resName = words[3]
            psf.atoms[i].name = words[4]
            psf.atoms[i].aType = words[5]
            psf.atoms[i].q = float(words[6])
            psf.atoms[i].m = float(words[7])
            psf.atoms[i].inferAtom()
            i += 1
        if (isBond and i < psf.nBonds):
            if (isExt):
                for j in range(int(len(line) / 20)):
                    st = line[j * 20:(j + 1) * 20]
                    psf.bonds.append([])
                    psf.bonds[i].append(int(st[0:10]))
                    psf.bonds[i].append(int(st[10:20]))
                    i += 1
            else:
                for j in range(int(len(line) / 16)):
                    st = line[j * 16:(j + 1) * 16]
                    psf.bonds.append([])
                    psf.bonds[i].append(int(st[0:8]))
                    psf.bonds[i].append(int(st[8:16]))
                    i += 1
        if (isTheta and i < psf.nAngles):
            if (isExt):
                for j in range(int(len(line) / 30)):
                    st = line[j * 30:(j + 1) * 30]
                    psf.angles.append([])
                    psf.angles[i].append(int(st[0:10]))
                    psf.angles[i].append(int(st[10:20]))
                    psf.angles[i].append(int(st[20:30]))
                    i += 1
            else:
                for j in range(int(len(line) / 24)):
                    st = line[j * 24:(j + 1) * 24]
                    psf.angles.append([])
                    psf.angles[i].append(int(st[0:8]))
                    psf.angles[i].append(int(st[8:16]))
                    psf.angles[i].append(int(st[16:24]))
                    i += 1
        if (isPhi and i < psf.nDihedrals):
            if (isExt):
                for j in range(int(len(line) / 40)):
                    st = line[j * 40:(j + 1) * 40]
                    psf.dihedrals.append([])
                    psf.dihedrals[i].append(int(st[0:10]))
                    psf.dihedrals[i].append(int(st[10:20]))
                    psf.dihedrals[i].append(int(st[20:30]))
                    psf.dihedrals[i].append(int(st[30:40]))
                    i += 1
            else:
                for j in range(int(len(line) / 32)):
                    st = line[j * 32:(j + 1) * 32]
                    psf.dihedrals.append([])
                    psf.dihedrals[i].append(int(st[0:8]))
                    psf.dihedrals[i].append(int(st[8:16]))
                    psf.dihedrals[i].append(int(st[16:24]))
                    psf.dihedrals[i].append(int(st[24:32]))
                    i += 1
        if (isImphi and i < psf.nImpropers):
            if (isExt):
                for j in range(int(len(line) / 40)):
                    st = line[j * 40:(j + 1) * 40]
                    psf.impropers.append([])
                    psf.impropers[i].append(int(st[0:10]))
                    psf.impropers[i].append(int(st[10:20]))
                    psf.impropers[i].append(int(st[20:30]))
                    psf.impropers[i].append(int(st[30:40]))
                    i += 1
            else:
                for j in range(int(len(line) / 32)):
                    st = line[j * 32:(j + 1) * 32]
                    psf.impropers.append([])
                    psf.impropers[i].append(int(st[0:8]))
                    psf.impropers[i].append(int(st[8:16]))
                    psf.impropers[i].append(int(st[16:24]))
                    psf.impropers[i].append(int(st[24:32]))
                    i += 1
        if (isCrt and i < psf.nCmap):
            if (isExt):
                psf.cmap.append([])
                psf.cmap[i].append(int(line[0:10]))
                psf.cmap[i].append(int(line[10:20]))
                psf.cmap[i].append(int(line[20:30]))
                psf.cmap[i].append(int(line[30:40]))
                # psf.cmap[i].append(int(line[40:50]))
                # psf.cmap[i].append(int(line[50:60]))
                # psf.cmap[i].append(int(line[60:70]))
                psf.cmap[i].append(int(line[70:80]))
                i += 1
            else:
                psf.cmap.append([])
                psf.cmap[i].append(int(line[0:8]))
                psf.cmap[i].append(int(line[8:16]))
                psf.cmap[i].append(int(line[16:24]))
                psf.cmap[i].append(int(line[24:32]))
                # psf.cmap[i].append(int(line[32:40]))
                # psf.cmap[i].append(int(line[40:48]))
                # psf.cmap[i].append(int(line[48:56]))
                psf.cmap[i].append(int(line[56:64]))
                i += 1
        if (isAtom and i >= psf.nAtoms):
            isAtom = False
        if (isBond and i >= psf.nBonds):
            isBond = False
        if (isTheta and i >= psf.nAngles):
            isTheta = False
        if (isPhi and i >= psf.nDihedrals):
            isPhi = False
        if (isImphi and i >= psf.nImpropers):
            isImphi = False
        if (isCrt and i >= psf.nCmap):
            isCrt = False
    fi.close()
    return psf



def writePsf(outName, psf, args):
    fo = open(outName, 'w')
    if (args.cppsf):
        fo.write("PSF EXT\n")
    else:
        fo.write("PSF CMAP XPLOR EXT\n")
    fo.write("%10d !NTITLE\n" % (1))
    fo.write(" SYSTEM\n")
    fo.write("\n")
    fo.write("%10d !NATOM\n" % (psf.nAtoms))
    for at in psf.atoms:
        if (at.resName.strip() == 'SOL'):
            at.resName = 'TIP3'
            if (at.name.strip() == 'OW'):
                at.name = 'OH2'
            elif (at.name.strip() == 'HW1'):
                at.name = 'H1'
            elif (at.name.strip() == 'HW2'):
                at.name = 'H2'
        if (at.resName.strip() == 'NA'):
            at.resName = 'SOD'
            at.name = 'SOD'
        if (args.cppsf):
            fo.write(
                "%10d %-7s  %-8i %-7s  %-6s  %-6s%10.6f      %8.3f           %1d\n" % (at.idx, at.segName, at.resIdx,
                                                                                       at.resName, at.name, at.aType,
                                                                                       at.q, at.m, 0))
        else:
            # fo.write("%8d %-4s %-4d %-4s %-4s %-5s %9.6f %13.4f %11d\n" % (at.idx,at.segName,at.resIdx,at.resName,at.name,at.aType,at.q,at.m,0))
            fo.write("%10d %-8s %-8i %-8s %-8s %-6s %10.6f %13.4f %11d\n" % (at.idx, at.segName, at.resIdx, at.resName,
                                                                             at.name, at.aType, at.q, at.m, 0))

    fo.write("\n")
    fo.write("%10d !NBOND: bonds\n" % (psf.nBonds))
    i = 0
    for bd in psf.bonds:
        fo.write("%10d%10d" % (bd[0], bd[1]))
        i += 1
        if (i % 4 == 0 or i == psf.nBonds):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NTHETA: angles\n" % (psf.nAngles))
    i = 0
    for th in psf.angles:
        fo.write("%10d%10d%10d" % (th[0], th[1], th[2]))
        i += 1
        if (i % 3 == 0 or i == psf.nAngles):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NPHI: dihedrals\n" % (psf.nDihedrals))
    i = 0
    for ph in psf.dihedrals:
        fo.write("%10d%10d%10d%10d" % (ph[0], ph[1], ph[2], ph[3]))
        i += 1
        if (i % 2 == 0 or i == psf.nDihedrals):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NIMPHI: impropers\n" % (psf.nImpropers))
    i = 0
    for ph in psf.impropers:
        fo.write("%10d%10d%10d%10d" % (ph[0], ph[1], ph[2], ph[3]))
        i += 1
        if (i % 2 == 0 or i == psf.nImpropers):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NDON: donors\n" % (0))
    fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NACC: acceptors\n" % (0))
    fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NCRTERM: cross-terms\n" % (psf.nCmap))
    i = 0
    for ct in psf.cmap:
        fo.write("%10d%10d%10d%10d%10d%10d%10d%10d\n" % (ct[0], ct[1], ct[2], ct[3], ct[1], ct[2], ct[3], ct[4]))
    fo.write("\n")
    fo.close()
    return


