# === Refactor for crystools.py: Read Write ITP file - OOP methods ===
from __future__ import annotations
import re
from dataclasses import dataclass, field  # only needed here if you keep local Atom/Topol; otherwise remove
from typing import List, Optional
from core.topology import Topology
from core.atom import Atom


def readPdb(fName):
    chainList=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    fi=open(fName,'r')
    isCryst=False
    isTer=False
    atoms=[]
    i=0
    nter=1
    spg='P1'
    for line in fi:
        if(line[0:6]=="REMARK"):
            continue
        elif(line[0:6]=="ANISOU"):
            continue
        elif(line[0:6]=="COMPND"):
            continue
        elif(line[0:6]=="CONECT"):
            continue
        elif(line[0:6]=="CRYST1"):
            words=line.split()
            a,b,c=crystobox(words)
            spg=line[55:66]
            isCryst=True
            continue
        elif(line[0:3]=="TER"):
            nr=0
            if(not isTer):
                nter+=1
            isTer=True
            continue
        elif(line[0:3]=="END"):
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
    if(not isCryst):
        a=lattice_vect()
        b=lattice_vect()
        c=lattice_vect()
    fi.close()
    return atoms,a,b,c,spg