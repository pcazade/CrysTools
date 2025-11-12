# === Refactor for crystools.py: Read Write TOP file - OOP methods ===
from __future__ import annotations
import re
from dataclasses import dataclass, field  # only needed here if you keep local Atom/Topol; otherwise remove
from typing import List, Optional, Tuple
from core.topology import Topology
from core.atom import Atom
from core.topology import Topology
from itp import readItp
import math
import numpy as np

def readTop(fname):
    itp = []
    isMolecules = False
    isMolType = False
    isAtom = False
    isPair = False
    isBond = False
    isAngl = False
    isDihe = False
    isImpr = False
    isCmap = False
    fi = open(fname, 'r')
    for line in fi:
        # if(len(itp)>2):
        #    print(line)
        #    print(itp[2].molName,len(itp[2].atoms),len(itp[2].bonds),len(itp[2].pairs))
        words = line.split()
        if (len(words) == 0):
            continue
        if ('#include' in words[0]):
            if ('forcefield.itp' in words[1]):
                ff = str(words[1])
                continue
            fname = words[1].strip('"')
            # readItp(fname, itp)
            readItp(fname)
        if (';' in words[0] or '#' in words[0]):
            continue
        if ('[ molecules ]' in line):
            isMolecules = True
            top = Topology()
            print("Okay")
            top.molName = 'System'
            top.atoms = []
            top.bonds = []
            top.pairs = []
            top.angles = []
            top.dihedrals = []
            top.impropers = []
            top.cmap = []
            continue
        if ('[ moleculetype ]' in line):
            itp.append(Topology())
            itp[-1].molName = ''
            itp[-1].atoms = []
            itp[-1].bonds = []
            itp[-1].pairs = []
            itp[-1].angles = []
            itp[-1].dihedrals = []
            itp[-1].impropers = []
            itp[-1].cmap = []
            isMolecules = False
            isMolType = True
            isAtom = False
            isPair = False
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if ('[ atoms ]' in line):
            # itp[-1].atoms=[]
            isMolecules = False
            isMolType = False
            isAtom = True
            isPair = False
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if ('[ pairs ]' in line):
            # itp[-1].pairs=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = True
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if ('[ bonds ]' in line):
            # itp[-1].bonds=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = True
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if ('[ angles ]' in line):
            # itp[-1].angles=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = False
            isAngl = True
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if ('[ dihedrals ]' in line and not isDihe):
            # itp[-1].dihedrals=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = False
            isAngl = False
            isDihe = True
            isImpr = False
            isCmap = False
            continue
        if ('[ dihedrals ]' in line and isDihe):
            # itp[-1].impropers=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = True
            isCmap = False
            continue
        if ('[ cmap ]' in line):
            # itp[-1].cmap=[]
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = True
            continue
        if ('[' in line and (isAtom or isPair or isBond or isAngl or isDihe or isImpr or isCmap)):
            isMolecules = False
            isMolType = False
            isAtom = False
            isPair = False
            isBond = False
            isAngl = False
            isDihe = False
            isImpr = False
            isCmap = False
            continue
        if (isMolType):
            itp[-1].molName = str(words[0])
            continue
        if (isAtom):
            idx = int(words[0]) - 1
            nr = int(words[2])
            itp[-1].atoms.append(Atom())
            itp[-1].atoms[idx].idx = idx + 1
            itp[-1].atoms[idx].aType = words[1].strip()
            itp[-1].atoms[idx].resIdx = nr
            itp[-1].atoms[idx].resName = words[3].strip()
            itp[-1].atoms[idx].name = words[4].strip()
            itp[-1].atoms[idx].q = float(words[6])
            if (len(words) > 7):
                itp[-1].atoms[idx].m = float(words[7])
            continue
        if (isPair):
            itp[-1].pairs.append([int(words[0]), int(words[1]), int(words[2])])
            continue
        if (isBond):
            itp[-1].bonds.append([int(words[0]), int(words[1]), int(words[2])])
            continue
        if (isAngl):
            itp[-1].angles.append([int(words[0]), int(words[1]), int(words[2]), int(words[3])])
            continue
        if (isDihe):
            itp[-1].dihedrals.append([int(words[0]), int(words[1]), int(words[2]), int(words[3]), int(words[4])])
            continue
        if (isImpr):
            itp[-1].impropers.append([int(words[0]), int(words[1]), int(words[2]), int(words[3]), int(words[4])])
            continue
        if (isCmap):
            itp[-1].cmap.append(
                [int(words[0]), int(words[1]), int(words[2]), int(words[3]), int(words[4]), int(words[5])])
            continue
        if (isMolecules):
            nMol = int(words[1])
            molName = str(words[0])
            # print(isMolecules,nMol,molName)
            idx = 0
            for i in range(len(itp)):
                if (itp[i].molName.strip() == molName.strip()):
                    idx = i
                    break
            for i in range(nMol):
                if (molName.strip() == 'SOL'):
                    segName = 'W'
                elif (molName.strip() == 'NA'):
                    segName = 'S'
                elif (molName.strip() == 'CL'):
                    segName = 'C'
                elif (len(itp[idx].atoms) < 50):
                    segName = 'O'
                else:
                    segName = 'P' + str(i + 1)

                iShift = len(top.atoms)
                rShift = 0
                if (iShift > 0):
                    rShift = top.atoms[iShift - 1].resIdx
                for at in itp[idx].atoms:
                    top.atoms.append(Atom())
                    Atom.copyFromAtom(top.atoms[-1], at)
                    Atom.inferAtom(top.atoms[-1])
                    top.atoms[-1].idx += iShift
                    top.atoms[-1].resIdx += rShift
                    top.atoms[-1].segName = segName
                for pair in itp[idx].pairs:
                    top.pairs.append([pair[0], pair[1], pair[2]])
                for bond in itp[idx].bonds:
                    top.bonds.append([bond[0], bond[1], bond[2]])
                for ang in itp[idx].angles:
                    top.angles.append([ang[0], ang[1], ang[2], ang[3]])
                for dih in itp[idx].dihedrals:
                    top.dihedrals.append([dih[0], dih[1], dih[2], dih[3], dih[4]])
                for imp in itp[idx].impropers:
                    top.impropers.append([imp[0], imp[1], imp[2], imp[3], imp[4]])
                for cm in itp[idx].cmap:
                    top.cmap.append([cm[0], cm[1], cm[2], cm[3], cm[4], cm[5]])
            # if(len(itp)>2):
            #    print(line)
            #    print(itp[2].molName,len(itp[2].atoms),len(itp[2].bonds),len(itp[2].pairs))
            #    if(len(itp[2].pairs)>0):
            #        print(itp[2].pairs[0])
            #    print(idx)
            continue
    top.ff = ff
    top.nAtoms = len(top.atoms)
    top.nBonds = len(top.bonds)
    top.nAngles = len(top.angles)
    top.nDihedrals = len(top.dihedrals)
    top.nImpropers = len(top.impropers)
    top.nCmap = len(top.cmap)
    fi.close()
    return (top)


def writeTop(outName, top):
    fo = open(outName, 'w')
    fo.write('#include "%s"\n' % (top.ff))
    fo.write('\n')
    fo.write('[ moleculetype ]\n')
    fo.write('%s\t3\n' % (top.molName))
    fo.write('\n')
    fo.write('[ atoms ]\n')
    qt = 0.0
    for at in top.atoms:
        qt += at.q
        fo.write(
            "%6d %10s %6d %6s %6s %6d %10.3f %10.3f  ; qtot %f\n" % (at.idx, at.aType, at.resIdx, at.resName, at.name,
                                                                     at.idx, at.q, at.m, qt))
    fo.write('\n')

    if (len(top.atoms) > 0):
        fo.write('[ bonds ]\n')
        for bd in top.bonds:
            fo.write("%5d %5d %5d\n" % (bd[0], bd[1], bd[2]))
        fo.write('\n')

    if (len(top.pairs) > 0):
        fo.write('[ pairs ]\n')
        for pr in top.pairs:
            fo.write("%5d %5d %5d\n" % (pr[0], pr[1], pr[2]))
        fo.write('\n')

    if (len(top.angles) > 0):
        fo.write('[ angles ]\n')
        for ag in top.angles:
            fo.write("%5d %5d %5d %5d\n" % (ag[0], ag[1], ag[2], ag[3]))
        fo.write('\n')

    if (len(top.dihedrals) > 0):
        fo.write('[ dihedrals ]\n')
        for dl in top.dihedrals:
            fo.write("%5d %5d %5d %5d %5d\n" % (dl[0], dl[1], dl[2], dl[3], dl[4]))
        fo.write('\n')

    if (len(top.impropers) > 0):
        fo.write('[ dihedrals ]\n')
        for dl in top.impropers:
            fo.write("%5d %5d %5d %5d %5d\n" % (dl[0], dl[1], dl[2], dl[3], dl[4]))
        fo.write('\n')

    if (len(top.cmap) > 0):
        fo.write('[ cmap ]\n')
        for cp in top.cmap:
            fo.write("%5d %5d %5d %5d %5d %5d\n" % (cp[0], cp[1], cp[2], cp[3], cp[4], cp[5]))
        fo.write('\n')

    words = outName.split('.')
    posrename = '"posre_' + words[0].strip() + '.itp"'
    fo.write("; Include Position restraint file\n")
    fo.write("#include %s\n" % (posrename))
    fo.write('[ system ]\n')
    fo.write('System\n')
    fo.write('\n')
    fo.write('[ molecules ]\n')
    fo.write('%s   1\n' % (top.molName))

    fo.close()
    return