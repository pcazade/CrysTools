# === Refactor for crystools.py: Read Write PDB file - OOP methods ===
from __future__ import annotations
import re
from dataclasses import dataclass, field  # only needed here if you keep local Atom/Topol; otherwise remove
from typing import List, Optional, Tuple
from core.topology import Topology
from core.atom import Atom
from core.cell import Cell
import math
import numpy as np


def readPdb(fName:str)-> Tuple[List[Atom], Cell, str]:
    chainList:List[str]=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    # fi=open(fName,'r')
    isCryst:bool=False
    isTer:bool=False
    atoms:List[Atom]=[]
    i:int=0
    nter:int=1
    spg:str='P1'
    cell:Cell|None = None

    with open(fName, "r") as fi:
        for line in fi:
            lineHead=line[0:6]
            if lineHead == "HEADER" or line.startswith("TITLE") or lineHead == "REMARK" or lineHead == "ANISOU" or lineHead == "COMPND" or lineHead == "CONECT":
                continue
            # if(lineHead=="REMARK"): #PDB files contain comments and metadata prefixed with REMARK.
            #     continue
            # elif(lineHead=="ANISOU"): #anisotropic displacement parameters (vibration tensors) for atoms. extra crystallographic refinement details, not required for basic coordinate parsing.
            #     continue
            # elif(lineHead=="COMPND"): #compound name and molecular information. part of the PDB “header” metadata.
            #     continue
            # elif(lineHead=="CONECT"): #Lists bonds between atoms. not needed when you’re reading coordinates for crystallographic analysis.
            #     continue
            elif(lineHead=="CRYST1"): #
                words=line.split()
                # a,b,c=crystobox(words)
                cell = Cell(
                    a=float(words[1]),
                    b=float(words[2]),
                    c=float(words[3]),
                    alpha=float(words[4]),
                    beta=float(words[5]),
                    gamma=float(words[6]),
                )
                spg=line[55:66]
                isCryst=True
                continue
            elif line.startswith("TER"):
                nr=0
                if not isTer:
                    nter+=1
                isTer=True
                continue
            elif line.startswith("END"):
                break
            isTer=False
            atoms.append(Atom())
            atoms[i].header=line[0:6]
            atoms[i].idx=i+1
            atoms[i].name=line[12:16]
            atoms[i].loc=line[16]
            if(len(line[17:21].strip())>0):
                atoms[i].resName=line[17:21]
            #atoms[i].chain=line[21]
            atoms[i].chain=chainList[(nter-1) % len(chainList)]
            if(i==0):
                oldr=line[22:26]
                nr=1
            elif(line[22:26]!=oldr):
                nr=nr+1
            oldr=line[22:26]
            atoms[i].resIdx=nr
            atoms[i].inser=line[26]
            atoms[i].x=float(line[30:38])
            atoms[i].y=float(line[38:46])
            atoms[i].z=float(line[46:54])
            atoms[i].occ=float(line[54:60])
            atoms[i].beta=float(line[60:66])
            atoms[i].segName=line[72:76]
            atoms[i].el=line[76:78]
            i=i+1
    # If no CRYST1 was found, return a default/empty Cell once.
    if not isCryst:
        cell = Cell()  # hmat/gmat zeros by default
    return atoms,cell,spg



def writePdb(fName: str, atoms: List[Atom], cell: Cell, spg: str = "P1")-> None:
    # fo = open(fName, 'w')

    with open(fName, "w") as fo:
        a_len = b_len = c_len = 0.0
        al = be = ga = 90.0
        if hasattr(cell, "hmat") and isinstance(cell.hmat, np.ndarray) and cell.hmat.shape == (3, 3) and np.any(
                cell.hmat):
            av, bv, cv = cell.hmat[:, 0], cell.hmat[:, 1], cell.hmat[:, 2]

            def _norm(v):
                return float(np.linalg.norm(v))

            def _cos(u, v):
                nu, nv = _norm(u), _norm(v)
                if nu == 0.0 or nv == 0.0: return 0.0
                x = float(np.dot(u, v) / (nu * nv))
                return max(-1.0, min(1.0, x))

            a_len, b_len, c_len = _norm(av), _norm(bv), _norm(cv)
            al = math.degrees(math.acos(_cos(bv, cv)))  # α = <(b,c)
            be = math.degrees(math.acos(_cos(av, cv)))  # β = <(a,c)
            ga = math.degrees(math.acos(_cos(av, bv)))  # γ = <(a,b)



        # if (a.norm > 0. and b.norm > 0. and c.norm > 0.):
        #     al = math.acos((b.x * c.x + b.y * c.y + b.z * c.z) / (b.norm * c.norm)) * 180. / math.pi
        #     be = math.acos((a.x * c.x + a.y * c.y + a.z * c.z) / (a.norm * c.norm)) * 180. / math.pi
        #     ga = math.acos((a.x * b.x + a.y * b.y + a.z * b.z) / (a.norm * b.norm)) * 180. / math.pi
        # else:
        #     al = 90.
        #     be = 90.
        #     ga = 90.


        fo.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d\n" % (a_len, b_len, c_len, al, be, ga, spg, 1))

        # Handle empty atom list gracefully
        if not atoms:
            fo.write("END\n")
            return

        oldSegName = atoms[0].segName.strip()
        oldChain = atoms[0].chain.strip()
        oldResIdx = atoms[0].resIdx
        nr = 1
        na = 1
        for at in atoms:
            # typeatom(at)
            at.inferAtom()
            if (at.resIdx != oldResIdx):
                nr = nr + 1
                oldResIdx = at.resIdx
            if (nr > 9999):
                nr = 1
            at.resIdx = nr
            at.idx = na
            if ((oldChain.strip() != at.chain.strip()) or (oldSegName.strip() != at.segName.strip())):
                fo.write("TER\n")
            fo.write("%6s%5d %4s %4s%c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%-2s\n" % (at.header, at.idx, at.name,
                                                                                           at.resName, at.chain, at.resIdx,
                                                                                           at.x, at.y, at.z, at.occ,
                                                                                           at.beta, at.segName, at.el))
            na += 1
            if (na > 99999):
                na = 1
            oldSegName = at.segName.strip()
            oldChain = at.chain.strip()
        fo.write("END\n")
        # fo.close()
    return