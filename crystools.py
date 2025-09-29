#!/usr/bin/python

#################################################################################
# This program is free software: you can redistribute it and/or modify it under #
# the terms of the GNU General Public License as published by the Free Software #
# Foundation, either version 3 of the License, or (at your option) any later    #
# version.                                                                      #
#                                                                               #
# This program is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                          #
# See the GNU General Public License for more details.                          #
#                                                                               #
# You should have received a copy of the GNU General Public License along with  #
# this program.                                                                 #
# If not, see <https://www.gnu.org/licenses/>.                                  #
#################################################################################

#################################################################################
# CrysTools.py written by P.-A. Cazade                                          #
# Copyright (C) 2019 - 2025  P.-A. Cazade                                       #
#################################################################################

import os
import sys
import math
import random
import copy
import numpy as np
import numpy.linalg as la
import ase.io
import spglib as sg
import argparse as ap
import numba as nb
import scipy as sp

class Atom(object):
    header="ATOM  "
    idx=1
    name=" H  "
    loc=' '
    resName="DUM"
    chain='A'
    resIdx=1
    aType="DUM"
    inser=' '
    x=0.0
    y=0.0
    z=0.0
    vx=0.0
    vy=0.0
    vz=0.0
    occ=1.0
    beta=0.0
    segName="P1  "
    element=" H"
    chg="1 "
    q=0.0
    m=0.0
    nAt=0
    el=' '
    sig=2.0

class Cell:
    def __init__(self):
        self.a=0.0
        self.b=0.0
        self.c=0.0
        self.al=0.0
        self.be=0.0
        self.ga=0.0
        self.hmat=np.zeros((3,3))
        self.gmat=np.zeros((3,3))

    def lp2box(self,a,b,c,al,be,ga):
        self.a=a
        self.b=b
        self.c=c
        self.al=al
        self.be=be
        self.ga=ga
        degtorad=math.pi/180.0
        if (al==90.0):
            cosa=0.0
        else:
            alp=al*degtorad
            cosa=math.cos(alp)
        if(be==90.0):
            cosb=0.0
        else:
            bet=be*degtorad
            cosb=math.cos(bet)
        if (ga==90.0):
            sing=1.0
            cosg=0.0
        else:
            gam=ga*degtorad
            sing=math.sin(gam)
            cosg=math.cos(gam)
        self.hmat[0,1]=0.0
        self.hmat[0,2]=0.0
        self.hmat[1,2]=0.0
        self.hmat[0,0]=a
        self.hmat[1,0]=b*cosg
        self.hmat[1,1]=b*sing
        self.hmat[2,0]=c*cosb
        self.hmat[2,1]=c*(cosa-cosg*cosb)/sing
        t=cc.y/c
        self.hmat[2,2]=c*math.sqrt(1.0-(cosb*cosb)-(t*t))
        self.gmat=la.inv(self.hmat)

    def mat2box(self,hmat):
        rad2deg=180.0/math.pi
        for i in range(3):
            for j in range(3):
                self.hmat[i,j]=hmat[i,j]
        self.a=la.norm(hmat[0])
        self.b=la.norm(hmat[1])
        self.c=la.norm(hmat[2])
        al=rad2deg*math.acos(np.sum(hmat[1]*hmat[2])/(self.b*self.c))
        be=rad2deg*math.acos(np.sum(hmat[0]*hmat[2])/(self.a*self.c))
        ga=rad2deg*math.acos(np.sum(hmat[0]*hmat[1])/(self.a*self.b))

class Topol(object):
    molName=''
    atoms=[]
    bonds=[]
    pairs=[]
    angles=[]
    dihedrals=[]
    impropers=[]
    cmap=[]

class Psf(object):
    atoms=None
    bonds=None
    theta=None
    phi=None
    imphi=None
    crt=None
    nAtom=0
    nBond=0
    nTheta=0
    nPhi=0
    nImphi=0
    nCrt=0

class lattice_vect(object):
    norm=0.0
    x=0.0
    y=0.0
    z=0.0

class Zmat(object):
    header="ATOM  "
    idx=1
    name=" H  "
    loc=' '
    resName="DUM"
    chain='A'
    resIdx=1
    inser=' '
    a=-1
    b=-1
    c=-1
    rl='NA'
    al='NA'
    dl='NA'
    dist=1.0
    ang=90.0
    tor=180.0
    activeTor=False
    occ=1.0
    beta=0.0
    segName="P1  "
    element=" H"
    chg="1 "
    q=0.0
    m=0.0
    nAt=0
    el=' '
    sig=2.0

class Residue(object):
    nAt=0
    fAt=0
    lAt=0
    resName="DUM"
    segName="P1  "

class Site(object):
    at1=0
    at2=0
    at3=0
    resIdx=0
    resName="DUM"
    segName="P1  "
    isGlyc=False

class Link(object):
    at1="C1"
    at2="OG1"
    rName1="AGAL"
    rName2="THR"
    idx1=1
    idx2=1

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def isint(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def crystobox(words):
    degtorad=math.pi/180.0
    a=float(words[1])
    b=float(words[2])
    c=float(words[3])
    al=float(words[4])
    be=float(words[5])
    ga=float(words[6])
    aa=lattice_vect()
    bb=lattice_vect()
    cc=lattice_vect()
    if (al==90.0):
        cosa=0.0
    else:
        alp=al*degtorad
        cosa=math.cos(alp)
    if(be==90.0):
        cosb=0.0
    else:
        bet=be*degtorad
        cosb=math.cos(bet)
    if (ga==90.0):
        sing=1.0
        cosg=0.0
    else:
        gam=ga*degtorad
        sing=math.sin(gam)
        cosg=math.cos(gam)
    aa.norm=a
    bb.norm=b
    cc.norm=c
    aa.y=0.0
    aa.z=0.0
    bb.z=0.0
    aa.x=a
    bb.x=b*cosg
    bb.y=b*sing
    cc.x=c*cosb
    cc.y=c*(cosa-cosg*cosb)/sing
    t=cc.y/c
    cc.z=c*math.sqrt(1.0-(cosb*cosb)-(t*t))
    return(aa,bb,cc)

def boxtocrys(a,b,c):
    a.norm=math.sqrt((a.x*a.x)+(a.y*a.y)+(a.z*a.z))
    b.norm=math.sqrt((b.x*b.x)+(b.y*b.y)+(b.z*b.z))
    c.norm=math.sqrt((c.x*c.x)+(c.y*c.y)+(c.z*c.z))
    if(a.norm>0. and b.norm>0. and c.norm>0.):
        al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
        be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
        ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    else:
        al=90.
        be=90.
        ga=90.
    return(a,norm,b.norm,c.norm,al,be,ga)

def cpzmat(zmat,atom):
    atom.idx=zmat.idx
    atom.name=zmat.name
    atom.loc=zmat.loc
    atom.resName=zmat.resName
    atom.chain=zmat.chain
    atom.resIdx=zmat.resIdx
    atom.inser=zmat.inser
    atom.occ=zmat.occ
    atom.beta=zmat.beta
    atom.segName=zmat.segName
    atom.element=zmat.element
    atom.chg=zmat.chg
    atom.nAt=zmat.nAt
    atom.el=zmat.el
    atom.sig=zmat.sig
    atom.q=zmat.q
    atom.m=zmat.m
    return

def cpatom(atom,zmat):
    zmat.x=atom.x
    zmat.y=atom.y
    zmat.z=atom.z
    zmat.idx=atom.idx
    zmat.name=atom.name
    zmat.loc=atom.loc
    zmat.resName=atom.resName
    zmat.chain=atom.chain
    zmat.resIdx=atom.resIdx
    zmat.inser=atom.inser
    zmat.occ=atom.occ
    zmat.beta=atom.beta
    zmat.segName=atom.segName
    zmat.element=atom.element
    zmat.chg=atom.chg
    zmat.nAt=atom.nAt
    zmat.el=atom.el
    zmat.sig=atom.sig
    zmat.q=atom.q
    zmat.m=atom.m
    return

def cpAtom(at1,at2):
    at1.header=at2.header
    at1.idx=at2.idx
    at1.name=at2.name
    at1.loc=at2.loc
    at1.resName=at2.resName
    at1.aType=at2.aType
    at1.chain=at2.chain
    at1.resIdx=at2.resIdx
    at1.inser=at2.inser
    at1.x=at2.x
    at1.y=at2.y
    at1.z=at2.z
    at1.vx=at2.vx
    at1.vy=at2.vy
    at1.vz=at2.vz
    at1.occ=at2.occ
    at1.beta=at2.beta
    at1.segName=at2.segName
    at1.element=at2.element
    at1.el=at2.el
    at1.chg=at2.chg
    at1.q=at2.q
    at1.m=at2.m
    at1.nAt=at1.nAt
    at1.sig=at2.sig
    return

def cpPsf(psfo,psfi,atShift,resShift):
    k=len(psfo.atoms)
    for j in range(psfi.nAtom):
        psfo.atoms.append(Atom())
        cpAtom(psfo.atoms[k],psfi.atoms[j])
        psfo.atoms[k].idx+=atShift
        psfo.atoms[k].resIdx+=resShift
        k=k+1
    k=len(psfo.bonds)
    for j in range(psfi.nBond):
        psfo.bonds.append([])
        psfo.bonds[k].append(psfi.bonds[j][0]+atShift)
        psfo.bonds[k].append(psfi.bonds[j][1]+atShift)
        k=k+1
    k=len(psfo.theta)
    for j in range(psfi.nTheta):
        psfo.theta.append([])
        psfo.theta[k].append(psfi.theta[j][0]+atShift)
        psfo.theta[k].append(psfi.theta[j][1]+atShift)
        psfo.theta[k].append(psfi.theta[j][2]+atShift)
        k=k+1
    k=len(psfo.phi)
    for j in range(psfi.nPhi):
        psfo.phi.append([])
        psfo.phi[k].append(psfi.phi[j][0]+atShift)
        psfo.phi[k].append(psfi.phi[j][1]+atShift)
        psfo.phi[k].append(psfi.phi[j][2]+atShift)
        psfo.phi[k].append(psfi.phi[j][3]+atShift)
        k=k+1
    k=len(psfo.imphi)
    for j in range(psfi.nImphi):
        psfo.imphi.append([])
        psfo.imphi[k].append(psfi.imphi[j][0]+atShift)
        psfo.imphi[k].append(psfi.imphi[j][1]+atShift)
        psfo.imphi[k].append(psfi.imphi[j][2]+atShift)
        psfo.imphi[k].append(psfi.imphi[j][3]+atShift)
        k=k+1
    k=len(psfo.crt)
    for j in range(psfi.nCrt):
        psfo.crt.append([])
        psfo.crt[k].append(psfi.crt[j][0]+atShift)
        psfo.crt[k].append(psfi.crt[j][1]+atShift)
        psfo.crt[k].append(psfi.crt[j][2]+atShift)
        psfo.crt[k].append(psfi.crt[j][3]+atShift)
        psfo.crt[k].append(psfi.crt[j][4]+atShift)
        psfo.crt[k].append(psfi.crt[j][5]+atShift)
        psfo.crt[k].append(psfi.crt[j][6]+atShift)
        psfo.crt[k].append(psfi.crt[j][7]+atShift)
        k=k+1
    return

def sigatom(atom):
    
    sig=[['H',1.2],
         ['C',1.7],
         ['N',1.55],
         ['O',1.52],
         ['F',1.47],
         ['P',1.8],
         ['S',1.8],
         ['Cl',1.75],
         ['Cu',1.4],
         ['Li',1.82],
         ['Na',2.27],
         ['K',2.75],
         ['I',1.98]]

    if(atom.name.strip()=='CAL' or atom.name.strip()=='Ca'):
        atom.el='Ca'
    elif(atom.name.strip()=='CLA' or atom.name.strip()[0:3]=='CLG' or atom.name.strip()[0:3]=='Cl'):
        atom.el='Cl'
    elif(atom.name.strip()=='SOD' or atom.name.strip()=='Na'):
        atom.el='Na'
    elif(atom.name.strip()=='MGA' or atom.name.strip()=='MG' or atom.name.strip()=='Mg'):
        atom.el='Mg'
    elif(atom.name.strip()=='ZN' or atom.name.strip()=='Zn'):
        atom.el='Zn'
    elif(atom.name.strip()=='POT' or atom.name.strip()=='K'):
        atom.el='K'
    elif(atom.name.strip()=='RUB'):
        atom.el='Rb'
    elif(atom.name.strip()=='FE' or atom.name.strip()=='Fe'):
        atom.el='Fe'
    elif(atom.name.strip()=='CES' or atom.name.strip()=='Ce'):
        atom.el='Ce'
    elif(atom.name.strip()=='CAD' or atom.name.strip()=='Cd'):
        atom.el='Cd'
    elif(atom.name.strip()=='ALG1' or atom.name.strip()=='Al'):
        atom.el='Al'
    elif(atom.name.strip()[0:2]=='BR' or atom.name.strip()=='Br'):
        atom.el='Br'
    elif(atom.name.strip()[0:2]=='AU' or atom.name.strip()=='Au'):
        atom.el='Au'
    elif(atom.name.strip()=='BAR'):
        atom.el='Ba'
    elif(atom.name.strip()=='LIT'):
        atom.el='Li'
    elif(len(atom.name.strip())>1 and atom.name.strip()[1].islower()):
        atom.el=atom.name.strip()[0:2]
    else:
        atom.el=' '+atom.name.strip()[0]
    
    for i in range(len(sig)):
        if(atom.el==sig[i][0]):
            atom.sig=sig[i][1]
            break
    return

def readTop(fname):
    itp=[]
    isMolecules=False
    isMolType=False
    isAtom=False
    isPair=False
    isBond=False
    isAngl=False
    isDihe=False
    isImpr=False
    isCmap=False
    fi=open(fname,'r')
    for line in fi:
        #if(len(itp)>2):
        #    print(line)
        #    print(itp[2].molName,len(itp[2].atoms),len(itp[2].bonds),len(itp[2].pairs))
        words=line.split()
        if(len(words)==0):
            continue
        if('#include' in words[0]):
            if('forcefield.itp' in words[1]):
                continue
            fname=words[1].strip('"')
            readItp(fname,itp)
        if(';' in words[0] or '#' in words[0]):
            continue
        if('[ molecules ]' in line):
            isMolecules=True
            top=Topol()
            top.molName=''
            top.atoms=[]
            top.bonds=[]
            top.pairs=[]
            top.angles=[]
            top.dihedrals=[]
            top.impropers=[]
            top.cmap=[]
            continue
        if('[ moleculetype ]' in line):
            itp.append(Topol())
            isMolecules=False
            isMolType=True
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ atoms ]' in line):
            itp[-1].atoms=[]
            isMolecules=False
            isMolType=False
            isAtom=True
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ pairs ]' in line):
            itp[-1].pairs=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=True
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ bonds ]' in line):
            itp[-1].bonds=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=True
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ angles ]' in line):
            itp[-1].angles=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=True
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and not isDihe):
            itp[-1].dihedrals=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=True
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and isDihe):
            itp[-1].impropers=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=True
            isCmap=False
            continue
        if('[ cmap ]' in line):
            itp[-1].cmap=[]
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=True
            continue
        if('[' in line and (isAtom or isPair or isBond or isAngl or isDihe or isImpr or isCmap) ):
            isMolecules=False
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            break
        if(isMolType):
            itp[-1].molName=str(words[0])
            continue
        if(isAtom):
            idx=int(words[0])-1
            nr=int(words[2])
            itp[-1].atoms.append(Atom())
            itp[-1].atoms[idx].idx=idx+1
            itp[-1].atoms[idx].aType=words[1].strip()
            itp[-1].atoms[idx].resIdx=nr
            itp[-1].atoms[idx].resName=words[3].strip()
            itp[-1].atoms[idx].name=words[4].strip()
            itp[-1].atoms[idx].q=float(words[6])
            if(len(words)>7):
                itp[-1].atoms[idx].m=float(words[7])
            continue
        if(isPair):
            itp[-1].pairs.append([int(words[0]),int(words[1]),int(words[2])])
            continue
        if(isBond):
            itp[-1].bonds.append([int(words[0]),int(words[1]),int(words[2])])
            continue
        if(isAngl):
            itp[-1].angles.append([int(words[0]),int(words[1]),int(words[2]),int(words[3])])
            continue
        if(isDihe):
            itp[-1].dihedrals.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4])])
            continue
        if(isImpr):
            itp[-1].impropers.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4])])
            continue
        if(isCmap):
            itp[-1].cmap.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4]),int(words[5])])
            continue
        if(isMolecules):
            nMol=int(words[1])
            molName=str(words[0])
            #print(isMolecules,nMol,molName)
            idx=0
            for i in range(len(itp)):
                if(itp[i].molName.strip()==molName.strip()):
                    idx=i
                    break
            for i in range(nMol):
                iShift=len(top.atoms)
                rShift=0
                if(iShift>0):
                    rShift=top.atoms[iShift-1].resIdx
                for at in itp[idx].atoms:
                    top.atoms.append(Atom())
                    cpAtom(top.atoms[-1],at)
                    top.atoms[-1].idx+=iShift
                    top.atoms[-1].resIdx+=rShift
                for pair in itp[idx].pairs:
                    top.pairs.append([pair[0],pair[1],pair[2]])
                for bond in itp[idx].bonds:
                    top.bonds.append([bond[0],bond[1],bond[2]])
                for ang in itp[idx].angles:
                    top.angles.append([ang[0],ang[1],ang[2],ang[3]])
                for dih in itp[idx].dihedrals:
                    top.dihedrals.append([dih[0],dih[1],dih[2],dih[3],dih[4]])
                for imp in itp[idx].impropers:
                    top.impropers.append([imp[0],imp[1],imp[2],imp[3],imp[4]])
                for cm in itp[idx].cmap:
                    top.cmap.append([cm[0],cm[1],cm[2],cm[3],cm[4],cm[5]])
            #if(len(itp)>2):
            #    print(line)
            #    print(itp[2].molName,len(itp[2].atoms),len(itp[2].bonds),len(itp[2].pairs))
            #    if(len(itp[2].pairs)>0):
            #        print(itp[2].pairs[0])
            #    print(idx)
            continue
    fi.close()
    return(top)

def readItp(fname,itp):
    #itp.append(Topol())
    isMolType=False
    isAtom=False
    isPair=False
    isBond=False
    isAngl=False
    isDihe=False
    isImpr=False
    isCmap=False
    fi=open(fname,'r')
    for line in fi:
        words=line.split()
        if(len(words)==0):
            continue
        if(';' in words[0] or '#' in words[0]):
            continue
        if('[ moleculetype ]' in line):
            itp.append(Topol())
            isMolType=True
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ atoms ]' in line):
            itp[-1].atoms=[]
            isMolType=False
            isAtom=True
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ pairs ]' in line):
            itp[-1].pairs=[]
            isMolType=False
            isAtom=False
            isPair=True
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ bonds ]' in line):
            itp[-1].bonds=[]
            isMolType=False
            isAtom=False
            isPair=False
            isBond=True
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ angles ]' in line):
            itp[-1].angles=[]
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=True
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and not isDihe):
            itp[-1].dihedrals=[]
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=True
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and isDihe):
            itp[-1].impropers=[]
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=True
            isCmap=False
            continue
        if('[ cmap ]' in line):
            itp[-1].cmap=[]
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=True
            continue
        if('[' in line and (isAtom or isPair or isBond or isAngl or isDihe or isImpr or isCmap) ):
            isMolType=False
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            break
        if(isMolType):
            itp[-1].molName=str(words[0])
            continue
        if(isAtom):
            idx=int(words[0])-1
            nr=int(words[2])
            itp[-1].atoms.append(Atom())
            itp[-1].atoms[idx].idx=idx+1
            itp[-1].atoms[idx].aType=words[1].strip()
            itp[-1].atoms[idx].resIdx=nr
            itp[-1].atoms[idx].resName=words[3].strip()
            itp[-1].atoms[idx].name=words[4].strip()
            itp[-1].atoms[idx].q=float(words[6])
            if(len(words)>7):
                itp[-1].atoms[idx].m=float(words[7])
            continue
        if(isPair):
            itp[-1].pairs.append([int(words[0]),int(words[1]),int(words[2])])
            continue
        if(isBond):
            itp[-1].bonds.append([int(words[0]),int(words[1]),int(words[2])])
            continue
        if(isAngl):
            itp[-1].angles.append([int(words[0]),int(words[1]),int(words[2]),int(words[3])])
            continue
        if(isDihe):
            itp[-1].dihedrals.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4])])
            continue
        if(isImpr):
            itp[-1].impropers.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4])])
            continue
        if(isCmap):
            itp[-1].cmap.append([int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4]),int(words[5])])
            continue
    fi.close()
    return

def readItpAppend(fname,top):
    iShift=len(top.atoms)
    rShift=top.atoms[iShift-1].resIdx
    isAtom=False
    isPair=False
    isBond=False
    isAngl=False
    isDihe=False
    isImpr=False
    isCmap=False
    fi=open(fname,'r')
    for line in fi:
        words=line.split()
        if(len(words)==0):
            continue
        if(';' in words[0] or '#' in words[0]):
            continue
        if('[ atoms ]' in line):
            isAtom=True
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ pairs ]' in line):
            isAtom=False
            isPair=True
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ bonds ]' in line):
            isAtom=False
            isPair=False
            isBond=True
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ angles ]' in line):
            isAtom=False
            isPair=False
            isBond=False
            isAngl=True
            isDihe=False
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and not isDihe):
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=True
            isImpr=False
            isCmap=False
            continue
        if('[ dihedrals ]' in line and isDihe):
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=True
            isCmap=False
            continue
        if('[ cmap ]' in line):
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=True
            continue
        if('[' in line and (isAtom or isPair or isBond or isAngl or isDihe or isImpr or isCmap) ):
            isAtom=False
            isPair=False
            isBond=False
            isAngl=False
            isDihe=False
            isImpr=False
            isCmap=False
            break
        if(isAtom):
            idx=int(words[0])-1+iShift
            nr=int(words[2])+rShift
            top.atoms.append(Atom())
            top.atoms[idx].idx=idx+1
            top.atoms[idx].aType=words[1].strip()
            top.atoms[idx].resIdx=nr
            top.atoms[idx].resName=words[3].strip()
            top.atoms[idx].name=words[4].strip()
            top.atoms[idx].q=float(words[6])
            top.atoms[idx].m=float(words[7])
            continue
        if(isPair):
            top.pairs.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])])
            continue
        if(isBond):
            top.bonds.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])])
            continue
        if(isAngl):
            top.angles.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])+iShift,int(words[3])])
            continue
        if(isDihe):
            top.dihedrals.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])+iShift,int(words[3])+iShift,int(words[4])])
            continue
        if(isImpr):
            top.impropers.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])+iShift,int(words[3])+iShift,int(words[4])])
            continue
        if(isCmap):
            top.cmap.append([int(words[0])+iShift,int(words[1])+iShift,int(words[2])+iShift,int(words[3])+iShift,int(words[4])+iShift,int(words[5])])
            continue
    fi.close()
    return

def writeTop(fname,top):
    fo=open(fname,'w')
    fo.write('[ atoms ]\n')
    qt=0.0
    for at in top.atoms:
        qt+=at.q
        fo.write("%6d %10s %6d %6s %6s %6d %10.3f %10.3f  ; qtot %f\n" % (at.idx,at.aType,at.resIdx,at.resName,at.name,at.idx,at.q,at.m,qt))
    fo.write('\n')

    if(len(top.atoms)>0):
        fo.write('[ bonds ]\n')
        for bd in top.bonds:
            fo.write("%5d %5d %5d\n" % (bd[0],bd[1],bd[2]))
        fo.write('\n')

    if(len(top.pairs)>0):
        fo.write('[ pairs ]\n')
        for pr in top.pairs:
            fo.write("%5d %5d %5d\n" % (pr[0],pr[1],pr[2]))
        fo.write('\n')

    if(len(top.angles)>0):
        fo.write('[ angles ]\n')
        for ag in top.angles:
            fo.write("%5d %5d %5d %5d\n" % (ag[0],ag[1],ag[2],ag[3]))
        fo.write('\n')

    if(len(top.dihedrals)>0):
        fo.write('[ dihedrals ]\n')
        for dl in top.dihedrals:
            fo.write("%5d %5d %5d %5d %5d\n" % (dl[0],dl[1],dl[2],dl[3],dl[4]))
        fo.write('\n')

    if(len(top.impropers)>0):
        fo.write('[ dihedrals ]\n')
        for dl in top.impropers:
            fo.write("%5d %5d %5d %5d %5d\n" % (dl[0],dl[1],dl[2],dl[3],dl[4]))
        fo.write('\n')

    if(len(top.cmap)>0):
        fo.write('[ cmap ]\n')
        for cp in top.cmap:
            fo.write("%5d %5d %5d %5d %5d %5d\n" % (cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]))
        fo.write('\n')
    
    words=fname.split('.')
    posrename='"posre_'+words[0].strip()+'.itp"'
    fo.write("; Include Position restraint file\n")
    fo.write("#include %s\n" % (posrename))

    fo.close()
    return

def readPsf(fname):
    fi=open(fname,'r')
    psf=Psf()
    isAtom=False
    isBond=False
    isTheta=False
    isPhi=False
    isImphi=False
    isCrt=False
    isExt=False
    isFirst=True
    for line in fi:
        if(isFirst):
            isFirst=False
            if('EXT' in line):
                isExt=True
        words=line.split()
        if(len(words)==0):
            continue
        if('!NATOM' in line):
            psf.nAtom=int(words[0])
            psf.atoms=[]
            isAtom=True
            i=0
            continue
        if('!NBOND' in line):
            psf.nBond=int(words[0])
            psf.bonds=[]
            isBond=True
            i=0
            continue
        if('!NTHETA' in line):
            psf.nTheta=int(words[0])
            psf.theta=[]
            isTheta=True
            i=0
            continue
        if('!NPHI' in line):
            psf.nPhi=int(words[0])
            psf.phi=[]
            isPhi=True
            i=0
            continue
        if('!NIMPHI' in line):
            psf.nImphi=int(words[0])
            psf.imphi=[]
            isImphi=True
            i=0
            continue
        if('!NCRTERM' in line):
            psf.nCrt=int(words[0])
            psf.crt=[]
            isCrt=True
            i=0
            continue
        if(isAtom and i<psf.nAtom):
            psf.atoms.append(Atom())
            psf.atoms[i].idx=int(words[0])
            psf.atoms[i].segName=words[1]
            psf.atoms[i].resIdx=int(words[2])
            psf.atoms[i].resName=words[3]
            psf.atoms[i].name=words[4]
            psf.atoms[i].atType=words[5]
            psf.atoms[i].q=float(words[6])
            psf.atoms[i].m=float(words[7])
            i+=1
        if(isBond and i<psf.nBond):
            if(isExt):
                for j in range(int(len(line)/20)):
                    st=line[j*20:(j+1)*20]
                    psf.bonds.append([])
                    psf.bonds[i].append(int(st[0:10]))
                    psf.bonds[i].append(int(st[10:20]))
                    i+=1
            else:
                for j in range(int(len(line)/16)):
                    st=line[j*16:(j+1)*16]
                    psf.bonds.append([])
                    psf.bonds[i].append(int(st[0:8]))
                    psf.bonds[i].append(int(st[8:16]))
                    i+=1
        if(isTheta and i<psf.nTheta):
            if(isExt):
                for j in range(int(len(line)/30)):
                    st=line[j*30:(j+1)*30]
                    psf.theta.append([])
                    psf.theta[i].append(int(st[0:10]))
                    psf.theta[i].append(int(st[10:20]))
                    psf.theta[i].append(int(st[20:30]))
                    i+=1
            else:
                for j in range(int(len(line)/24)):
                    st=line[j*24:(j+1)*24]
                    psf.theta.append([])
                    psf.theta[i].append(int(st[0:8]))
                    psf.theta[i].append(int(st[8:16]))
                    psf.theta[i].append(int(st[16:24]))
                    i+=1
        if(isPhi and i<psf.nPhi):
            if(isExt):
                for j in range(int(len(line)/40)):
                    st=line[j*40:(j+1)*40]
                    psf.phi.append([])
                    psf.phi[i].append(int(st[0:10]))
                    psf.phi[i].append(int(st[10:20]))
                    psf.phi[i].append(int(st[20:30]))
                    psf.phi[i].append(int(st[30:40]))
                    i+=1
            else:
                for j in range(int(len(line)/32)):
                    st=line[j*32:(j+1)*32]
                    psf.phi.append([])
                    psf.phi[i].append(int(st[0:8]))
                    psf.phi[i].append(int(st[8:16]))
                    psf.phi[i].append(int(st[16:24]))
                    psf.phi[i].append(int(st[24:32]))
                    i+=1
        if(isImphi and i<psf.nImphi):
            if(isExt):
                for j in range(int(len(line)/40)):
                    st=line[j*40:(j+1)*40]
                    psf.imphi.append([])
                    psf.imphi[i].append(int(st[0:10]))
                    psf.imphi[i].append(int(st[10:20]))
                    psf.imphi[i].append(int(st[20:30]))
                    psf.imphi[i].append(int(st[30:40]))
                    i+=1
            else:
                for j in range(int(len(line)/32)):
                    st=line[j*32:(j+1)*32]
                    psf.imphi.append([])
                    psf.imphi[i].append(int(st[0:8]))
                    psf.imphi[i].append(int(st[8:16]))
                    psf.imphi[i].append(int(st[16:24]))
                    psf.imphi[i].append(int(st[24:32]))
                    i+=1
        if(isCrt and i<psf.nCrt):
            if(isExt):
                psf.crt.append([])
                psf.crt[i].append(int(line[0:10]))
                psf.crt[i].append(int(line[10:20]))
                psf.crt[i].append(int(line[20:30]))
                psf.crt[i].append(int(line[30:40]))
                psf.crt[i].append(int(line[40:50]))
                psf.crt[i].append(int(line[50:60]))
                psf.crt[i].append(int(line[60:70]))
                psf.crt[i].append(int(line[70:80]))
                i+=1
            else:
                psf.crt.append([])
                psf.crt[i].append(int(line[0:8]))
                psf.crt[i].append(int(line[8:16]))
                psf.crt[i].append(int(line[16:24]))
                psf.crt[i].append(int(line[24:32]))
                psf.crt[i].append(int(line[32:40]))
                psf.crt[i].append(int(line[40:48]))
                psf.crt[i].append(int(line[48:56]))
                psf.crt[i].append(int(line[56:64]))
                i+=1
        if(isAtom and i>=psf.nAtom):
            isAtom=False
        if(isBond and i>=psf.nBond):
            isBond=False
        if(isTheta and i>=psf.nTheta):
            isTheta=False
        if(isPhi and i>=psf.nPhi):
            isPhi=False
        if(isImphi and i>=psf.nImphi):
            isImphi=False
        if(isCrt and i>=psf.nCrt):
            isCrt=False
    fi.close()
    return psf

def writePsf(fname,psf):
    fo=open(fname,'w')
    fo.write("PSF CMAP XPLOR EXT\n")
    fo.write("%10d !NTITLE\n" % (1))
    fo.write(" BA3 CRYSTAL\n")
    fo.write("\n")
    fo.write("%10d !NATOM\n" % (psf.nAtom))
    for at in psf.atoms:
        #fo.write("%8d %-4s %-4d %-4s %-4s %-5s %9.6f %13.4f %11d\n" % (at.idx,at.segName,at.resIdx,at.resName,at.name,at.atType,at.q,at.m,0))
        fo.write("%10d %-8s %-8i %-8s %-8s %-6s %10.6f %13.4f %11d\n" % (at.idx,at.segName,at.resIdx,at.resName,at.name,at.atType,at.q,at.m,0))
    fo.write("\n")
    fo.write("%10d !NBOND: bonds\n" % (psf.nBond))
    i=0
    for bd in psf.bonds:
        fo.write("%10d%10d" % (bd[0],bd[1]))
        i+=1
        if(i%4 == 0 or i == psf.nBond):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NTHETA: angles\n" % (psf.nTheta))
    i=0
    for th in psf.theta:
        fo.write("%10d%10d%10d" % (th[0],th[1],th[2]))
        i+=1
        if(i%3 == 0 or i == psf.nTheta):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NPHI: dihedrals\n" % (psf.nPhi))
    i=0
    for ph in psf.phi:
        fo.write("%10d%10d%10d%10d" % (ph[0],ph[1],ph[2],ph[3]))
        i+=1
        if(i%2 == 0 or i == psf.nPhi):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NIMPHI: impropers\n" % (psf.nImphi))
    i=0
    for ph in psf.imphi:
        fo.write("%10d%10d%10d%10d" % (ph[0],ph[1],ph[2],ph[3]))
        i+=1
        if(i%2 == 0 or i == psf.nImphi):
            fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NDON: donors\n" % (0))
    fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NACC: acceptors\n" % (0))
    fo.write("\n")
    fo.write("\n")
    fo.write("%10d !NCRTERM: cross-terms\n" % (psf.nCrt))
    i=0
    for ct in psf.crt:
        fo.write("%10d%10d%10d%10d%10d%10d%10d%10d\n"% (ct[0],ct[1],ct[2],ct[3],ct[4],ct[5],ct[6],ct[7]))
    fo.write("\n")
    fo.close()
    return

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

def writePdb(fName,atoms,a,b,c):
    fo=open(fName,'w')
    if(a.norm>0. and b.norm>0. and c.norm>0.):
        al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
        be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
        ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    else:
        al=90.
        be=90.
        ga=90.
    fo.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d\n" % (a.norm,b.norm,c.norm,al,be,ga,'P1',1))
    oldChain=atoms[0].chain
    oldResIdx=atoms[0].resIdx
    nr=1
    na=1
    for at in atoms:
        typeatom(at)
        if(at.resIdx!=oldResIdx):
            nr=nr+1
            oldResIdx=at.resIdx
        if(nr>9999):
            nr=1
        at.resIdx=nr
        at.idx=na
        fo.write("%6s%5d %4s %4s%c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%-2s\n" % (at.header,at.idx,at.name,at.resName,at.chain,at.resIdx,at.x,at.y,at.z,at.occ,at.beta,at.segName,at.el))
        na+=1
        if(na>99999):
            na=1
    fo.write("END\n")
    fo.close()
    return

def readGro(fName,chain):
    fi=open(fName,'r')
    atoms=[]
    title=fi.readline()
    nAtom=int(fi.readline())
    for i in range(nAtom):
        line=fi.readline()
        atoms.append(Atom())
        if(i==0):
            oldr=int(line[0:5])
            nr=oldr #1
        elif(int(line[0:5])!=oldr):
            nr=nr+1
        oldr=int(line[0:5])
        atoms[i].resIdx=nr
        atoms[i].resName=line[5:10].strip()
        atoms[i].name=line[10:15].strip()
        atoms[i].idx=i+1
        atoms[i].x=float(line[20:28])*10.
        atoms[i].y=float(line[28:36])*10.
        atoms[i].z=float(line[36:44])*10.
        if(len(line)>=68):
            atoms[i].vx=float(line[44:52])
            atoms[i].vy=float(line[52:60])
            atoms[i].vz=float(line[60:68])
        atoms[i].chain=chain
    words=fi.readline().split()
    a=lattice_vect()
    b=lattice_vect()
    c=lattice_vect()
    a.x=float(words[0])*10.
    b.y=float(words[1])*10.
    c.z=float(words[2])*10.
    if(len(words)==9):
        a.y=float(words[3])*10.
        a.z=float(words[4])*10.
        b.x=float(words[5])*10.
        b.z=float(words[6])*10.
        c.x=float(words[7])*10.
        c.y=float(words[8])*10.
    a.norm=math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)
    b.norm=math.sqrt(b.x*b.x+b.y*b.y+b.z*b.z)
    c.norm=math.sqrt(c.x*c.x+c.y*c.y+c.z*c.z)
    fi.close()
    return atoms,a,b,c

def writeGro(fName,atoms,a,b,c,isScaled):
    h=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
    th=np.transpose(h)
    fo=open(fName,'w')
    fo.write('File written by cryst.py from P.-A. Cazade\n')
    fo.write('%d\n' % (len(atoms)))
    for at in atoms:
        typeatom(at)
        resIdx=at.resIdx%100000
        idx=at.idx%100000
        if(isScaled):
            uvw=np.array([at.x,at.y,at.z])
            xyz=np.matmul(th,uvw)
        else:
            xyz=np.array([at.x,at.y,at.z])
        fo.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n' % (resIdx,at.resName,at.name,idx,xyz[0]*0.1,xyz[1]*0.1,xyz[2]*0.1))
    if(max(abs(a.y),abs(a.z),abs(b.x),abs(b.z),abs(c.x),abs(c.y))>1.e-6):
        fo.write('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n' % (a.x*0.1,b.y*0.1,c.z*0.1,a.y*0.1,a.z*0.1,b.x*0.1,b.z*0.1,c.x*0.1,c.y*0.1))
    else:
        fo.write('%10.5f%10.5f%10.5f\n' % (a.x*0.1,b.y*0.1,c.z*0.1))
    fo.close()
    return

def readGzmat(fName,atoms):
    
    fi=open(fName,"r")
    
    zmat=[]
    for i in range(5):
        fi.readline()
    i=0
    first=True
    second=True
    third=True
    for line in fi:
        words=line.split()
        if(len(words)==0):
            break
        elif(len(words)==1 and first):
            first=False
            zmat.append(Zmat())
            cpatom(atoms[i],zmat[i])
        elif(len(words)==3 and second):
            second=False
            zmat.append(Zmat())
            cpatom(atoms[i],zmat[i])
            zmat[i].a=int(words[1])-1
            zmat[i].rl=words[2].strip()
        elif(len(words)==5 and third):
            third=False
            zmat.append(Zmat())
            cpatom(atoms[i],zmat[i])
            zmat[i].a=int(words[1])-1
            zmat[i].rl=words[2].strip()
            zmat[i].b=int(words[3])-1
            zmat[i].al=words[4].strip()
        elif(len(words)==7):
            zmat.append(Zmat())
            cpatom(atoms[i],zmat[i])
            zmat[i].a=int(words[1])-1
            zmat[i].rl=words[2].strip()
            zmat[i].b=int(words[3])-1
            zmat[i].al=words[4].strip()
            zmat[i].c=int(words[5])-1
            zmat[i].dl=words[6].strip()
        elif(len(words)==8):
            zmat.append(Zmat())
            cpatom(atoms[i],zmat[i])
            zmat[i].a=int(words[1])-1
            zmat[i].rl=words[2].strip()
            zmat[i].b=int(words[3])-1
            zmat[i].al=words[4].strip()
            zmat[i].c=int(words[5])-1
            zmat[i].dl=words[6].strip()
            if(int(words[7])==1):
                zmat[i].activeTor=True
        elif(len(words)<7 and (not first) and (not second) and (not third)):
            break
        i=i+1
    
    for line in fi:
        words=line.split()
        if(len(words)==0):
            continue
        for i in range(len(zmat)):
            if(words[0].strip()==zmat[i].rl or words[0].strip()==(zmat[i].rl+'=')):
                zmat[i].dist=float(words[1])
                break
            elif(words[0].strip()==zmat[i].al or words[0].strip()==(zmat[i].al+'=')):
                zmat[i].ang=float(words[1])
                break
            elif(words[0].strip()==zmat[i].dl or words[0].strip()==(zmat[i].dl+'=')):
                zmat[i].tor=float(words[1])
                break
    
    return zmat

def rotate(cart,at1,at2,at3):
    
    ux=at2.x-at1.x
    uy=at2.y-at1.y
    uz=at2.z-at1.z
    
    ru=math.sqrt(ux*ux+uy*uy+uz*uz)
    
    ux=ux/ru
    uy=uy/ru
    uz=uz/ru
    
    wx=at3.x-at2.x
    wy=at3.y-at2.y
    wz=at3.z-at2.z
    
    rw=math.sqrt(wx*wx+wy*wy+wz*wz)
    
    wx=wx/rw
    wy=wy/rw
    wz=wz/rw
    
    vx=uy*wz-uz*wy
    vy=uz*wx-ux*wz
    vz=ux*wy-uy*wx
    
    rv=math.sqrt(vx*vx+vy*vy+vz*vz)
    
    vx=-vx/rv
    vy=-vy/rv
    vz=-vz/rv
    
    wx=uy*vz-uz*vy
    wy=uz*vx-ux*vz
    wz=ux*vy-uy*vx
    
    rw=math.sqrt(wx*wx+wy*wy+wz*wz)
    
    wx=wx/rw
    wy=wy/rw
    wz=wz/rw
    
    for i in range(len(cart)):
        
        xt=ux*cart[i].x+vx*cart[i].y+wx*cart[i].z
        yt=uy*cart[i].x+vy*cart[i].y+wy*cart[i].z
        zt=uz*cart[i].x+vz*cart[i].y+wz*cart[i].z
        
        cart[i].x=xt+at1.x
        cart[i].y=yt+at1.y
        cart[i].z=zt+at1.z
    
    return

def zmat2cart(zmat):
    cart=[]
    for i in range(len(zmat)):
        cart.append(Atom())
        if(zmat[i].a<0):
            cart[i].x=0.0
            cart[i].y=0.0
            cart[i].z=0.0
            cpzmat(zmat[i],cart[i])
            continue
        else:
            dist=zmat[i].dist
            ax=cart[zmat[i].a].x
            ay=cart[zmat[i].a].y
            az=cart[zmat[i].a].z
        
        if(zmat[i].b<0):
            cart[i].x=dist
            cart[i].y=0.0
            cart[i].z=0.0
            cpzmat(zmat[i],cart[i])
            continue
        else:
            ang=math.radians(zmat[i].ang)
            bx=cart[zmat[i].b].x
            by=cart[zmat[i].b].y
            bz=cart[zmat[i].b].z
        
        if(zmat[i].c<0):
            tor=math.radians(90.0)
            cx=0.0
            cy=1.0
            cz=0.0
        else:
            tor=math.radians(zmat[i].tor)
            cx=cart[zmat[i].c].x
            cy=cart[zmat[i].c].y
            cz=cart[zmat[i].c].z
            
        #print i,math.degrees(ang),math.degrees(tor)
        
        ux=ax-bx
        uy=ay-by
        uz=az-bz
        
        vx=ax-cx
        vy=ay-cy
        vz=az-cz
        
        nx=uy*vz-uz*vy
        ny=uz*vx-ux*vz
        nz=ux*vy-uy*vx
        
        nnx=uy*nz-uz*ny
        nny=uz*nx-ux*nz
        nnz=ux*ny-uy*nx
        
        rn=math.sqrt(nx*nx+ny*ny+nz*nz)
        
        nx=nx/rn
        ny=ny/rn
        nz=nz/rn
        
        rnn=math.sqrt(nnx*nnx+nny*nny+nnz*nnz)
        
        nnx=nnx/rnn
        nny=nny/rnn
        nnz=nnz/rnn
        
        nx=nx*(-math.sin(tor))
        ny=ny*(-math.sin(tor))
        nz=nz*(-math.sin(tor))
        
        nnx=nnx*math.cos(tor)
        nny=nny*math.cos(tor)
        nnz=nnz*math.cos(tor)
        
        wx=nx+nnx
        wy=ny+nny
        wz=nz+nnz
        
        rw=math.sqrt(wx*wx+wy*wy+wz*wz)
        
        wx=wx/rw
        wy=wy/rw
        wz=wz/rw
        
        wx=wx*dist*math.sin(ang)
        wy=wy*dist*math.sin(ang)
        wz=wz*dist*math.sin(ang)
        
        ru=math.sqrt(ux*ux+uy*uy+uz*uz)
        
        ux=ux/ru
        uy=uy/ru
        uz=uz/ru
        
        ux=ux*dist*math.cos(ang)
        uy=uy*dist*math.cos(ang)
        uz=uz*dist*math.cos(ang)
        
        cart[i].x=ax+wx-ux
        cart[i].y=ay+wy-uy
        cart[i].z=az+wz-uz
        
        cpzmat(zmat[i],cart[i])
    
    return cart

def polymerize(zmat,atoms,idx1,idx2,idx3,tor1,tor2,tor3):
    cart=[]
    zmat[0].dist=1.43
    zmat[0].ang=111.3
    zmat[0].tor=tor1
    zmat[1].ang=110.09
    zmat[1].tor=tor2
    zmat[2].tor=tor3
    for i in range(len(zmat)):
        cart.append(Atom())
        if(zmat[i].a<0):
            dist=zmat[i].dist
            ax=atoms[idx3].x
            ay=atoms[idx3].y
            az=atoms[idx3].z
            l=idx2
            m=idx1
        else:
            dist=zmat[i].dist
            ax=cart[zmat[i].a].x
            ay=cart[zmat[i].a].y
            az=cart[zmat[i].a].z
            l=idx3
            m=idx2
        
        if(zmat[i].b<0):
            ang=math.radians(zmat[i].ang)
            bx=atoms[l].x
            by=atoms[l].y
            bz=atoms[l].z
        else:
            ang=math.radians(zmat[i].ang)
            bx=cart[zmat[i].b].x
            by=cart[zmat[i].b].y
            bz=cart[zmat[i].b].z
            m=idx3
        
        if(zmat[i].c<0):
            tor=math.radians(zmat[i].tor)
            cx=atoms[m].x
            cy=atoms[m].y
            cz=atoms[m].z
        else:
            tor=math.radians(zmat[i].tor)
            cx=cart[zmat[i].c].x
            cy=cart[zmat[i].c].y
            cz=cart[zmat[i].c].z
            
        #print i,math.degrees(ang),math.degrees(tor)
        
        ux=ax-bx
        uy=ay-by
        uz=az-bz
        
        vx=ax-cx
        vy=ay-cy
        vz=az-cz
        
        nx=uy*vz-uz*vy
        ny=uz*vx-ux*vz
        nz=ux*vy-uy*vx
        
        nnx=uy*nz-uz*ny
        nny=uz*nx-ux*nz
        nnz=ux*ny-uy*nx
        
        rn=math.sqrt(nx*nx+ny*ny+nz*nz)
        
        nx=nx/rn
        ny=ny/rn
        nz=nz/rn
        
        rnn=math.sqrt(nnx*nnx+nny*nny+nnz*nnz)
        
        nnx=nnx/rnn
        nny=nny/rnn
        nnz=nnz/rnn
        
        nx=nx*(-math.sin(tor))
        ny=ny*(-math.sin(tor))
        nz=nz*(-math.sin(tor))
        
        nnx=nnx*math.cos(tor)
        nny=nny*math.cos(tor)
        nnz=nnz*math.cos(tor)
        
        wx=nx+nnx
        wy=ny+nny
        wz=nz+nnz
        
        rw=math.sqrt(wx*wx+wy*wy+wz*wz)
        
        wx=wx/rw
        wy=wy/rw
        wz=wz/rw
        
        wx=wx*dist*math.sin(ang)
        wy=wy*dist*math.sin(ang)
        wz=wz*dist*math.sin(ang)
        
        ru=math.sqrt(ux*ux+uy*uy+uz*uz)
        
        ux=ux/ru
        uy=uy/ru
        uz=uz/ru
        
        ux=ux*dist*math.cos(ang)
        uy=uy*dist*math.cos(ang)
        uz=uz*dist*math.cos(ang)
        
        cart[i].x=ax+wx-ux
        cart[i].y=ay+wy-uy
        cart[i].z=az+wz-uz
        
        cpzmat(zmat[i],cart[i])
    
    return cart

def adjustDist(cart,at,d):
    xt=cart[3].x-at.x
    yt=cart[3].y-at.y
    zt=cart[3].z-at.z
    
    s=d/math.sqrt(xt*xt+yt*yt+zt*zt)
    
    dx=xt*(s-1)
    dy=yt*(s-1)
    dz=zt*(s-1)
    
    for i in range(len(cart)):
        cart[i].x=cart[i].x+dx
        cart[i].y=cart[i].y+dy
        cart[i].z=cart[i].z+dz
    
    return

def countRes(atoms):
    residues=[]
    oldr=-1
    k=-1
    for i in range(len(atoms)):
        if(atoms[i].resIdx==oldr):
            residues[k].lAt=i
            residues[k].nAt=nAt
        else:
            oldr=atoms[i].resIdx
            nAt=atoms[i].nAt
            k=k+1
            residues.append(Residue())
            residues[k].fAt=i
            residues[k].lAt=i
            residues[k].nAt=nAt
            residues[k].resName=atoms[i].resName
            residues[k].segName=atoms[i].segName
    
    return residues

def pbc(atoms,a,b,c,isScaled):
    ra,rb,rc,vol=wz(a,b,c)
    if(isScaled):
        for at in atoms:
            at.x-=math.floor(at.x)
            at.y-=math.floor(at.y)
            at.z-=math.floor(at.z)
    else:
        for at in atoms:
            xt=at.x*ra.x+at.y*ra.y+at.z*ra.z
            yt=at.x*rb.x+at.y*rb.y+at.z*rb.z
            zt=at.x*rc.x+at.y*rc.y+at.z*rc.z
            xt-=math.floor(xt)
            yt-=math.floor(yt)
            zt-=math.floor(zt)
            at.x=xt*a.x+yt*b.x+zt*c.x
            at.y=xt*a.y+yt*b.y+zt*c.y
            at.z=xt*a.z+yt*b.z+zt*c.z
    return

def getCutoffsCP2K(fname):
    fi=open(fname,'r')
    for line in fi:
        words=line.split()
        if("QS| Density cutoff" in line):
            cut=2.0*float(words[4])
        if("QS| Relative density cutoff" in line):
            rcut=2.0*float(words[5])
            break
    return cut,rcut

def getEnergyCP2K(fname):
    en=0.0
    fi=open(fname,'r')
    for line in fi:
        words=line.split()
        if("ENERGY| Total FORCE_EVAL" in line):
            en=float(words[8])
            break
    return en

def getMgridCP2K(fname):
    fi=open(fname,'r')
    grid=[]
    for line in fi:
        words=line.split()
        if("count for grid" in line):
            grid.append(int(words[4]))
        if("total gridlevel count" in line):
            break
    return grid

def getDipoleCP2K(fname):
    fi=open(fname,'r')
    fi.seek(0,2)
    eof = fi.tell()
    fi.seek(0,0)
    isLast=False
    while (fi.tell() < eof):
        line=fi.readline()
        words=line.split()
        if('CELL| Vector a' in line):
            a=lattice_vect()
            a.x=float(words[4])
            a.y=float(words[5])
            a.z=float(words[6])
            continue
        if('CELL| Vector b' in line):
            b=lattice_vect()
            b.x=float(words[4])
            b.y=float(words[5])
            b.z=float(words[6])
            continue
        if('CELL| Vector c' in line):
            c=lattice_vect()
            c.x=float(words[4])
            c.y=float(words[5])
            c.z=float(words[6])
            continue
        if('GEOMETRY OPTIMIZATION COMPLETED' in line or 'satisfied .... run CONVERGED' in line):
            isLast=True
            continue
        if('Dipole moment [Debye]' in line and isLast):
            line=fi.readline()
            words=line.split()
            mu=np.array([float(words[1]),float(words[3]),float(words[5])])
        if('MM DIPOLE [BERRY PHASE](Debye)|' in line and isLast):
            mu=np.array([float(words[4]),float(words[5]),float(words[6])])
        if('MM_DIPOLE| Moment [Debye]' in line and isLast):
            mu=np.array([float(words[3]),float(words[4]),float(words[5])])
    return mu,a,b,c

def getStressTensorCP2K(fname):
    if os.path.isfile(fname):
        fi=open(fname,'r')
        fi.seek(0,2)
        eof = fi.tell()
        fi.seek(0,0)
        isLast=False
        stress=np.zeros((3,3))
        while (fi.tell() < eof):
            line=fi.readline()
            words=line.split()
            if('CELL| Vector a' in line):
                a=lattice_vect()
                a.x=float(words[4])
                a.y=float(words[5])
                a.z=float(words[6])
                continue
            if('CELL| Vector b' in line):
                b=lattice_vect()
                b.x=float(words[4])
                b.y=float(words[5])
                b.z=float(words[6])
                continue
            if('CELL| Vector c' in line):
                c=lattice_vect()
                c.x=float(words[4])
                c.y=float(words[5])
                c.z=float(words[6])
                continue
            if('STRESS TENSOR [GPa]' in line):
                isLast=True
                continue
            if('STRESS| Analytical stress tensor [GPa]' in line):
                isLast=True
                continue
            if('X                       Y                       Z' in line):
                continue
            if('X               Y               Z' in line):
                continue
            if('x                   y                   z' in line):
                continue
            if(('  X ' in line or 'STRESS|      x ' in line) and isLast):
                stress[0,0]=float(words[2])
                stress[0,1]=float(words[3])
                stress[0,2]=float(words[4])
                continue
            if(('  Y ' in line or 'STRESS|      y ' in line) and isLast):
                stress[1,0]=float(words[2])
                stress[1,1]=float(words[3])
                stress[1,2]=float(words[4])
                continue
            if(('  Z ' in line or 'STRESS|      z ' in line) and isLast):
                stress[2,0]=float(words[2])
                stress[2,1]=float(words[3])
                stress[2,2]=float(words[4])
                isLast=False
                continue
    else:
        stress=np.zeros((3,3))
        a=lattice_vect()
        b=lattice_vect()
        c=lattice_vect()
    return(stress,a,b,c)

def getBoxCP2K(fname):
    fi=open(fname,'r')
    isA=False
    isB=False
    isC=False
    isCell=False
    isABC=False
    isALBEGA=False
    cell_par=[]
    for line in fi:
        words=line.split()
        if("&CELL" in line):
            cell_par.append('cell')
            isCell=True
            continue
        if("&END CELL" in line):
            isCell=False
            continue
        if(words[0]=='ABC' and isCell):
            cell_par.append(words[1])
            cell_par.append(words[2])
            cell_par.append(words[3])
            isABC=True
        if(words[0]=='ALPHA_BETA_GAMMA' and isCell):
            cell_par.append(words[1])
            cell_par.append(words[2])
            cell_par.append(words[3])
            isALBEGA=True
        if(words[0]=='A' and isCell):
            a=lattice_vect()
            a.x=float(words[1])
            a.y=float(words[2])
            a.z=float(words[3])
            isA=True
            continue
        if(words[0]=='B' and isCell):
            b=lattice_vect()
            b.x=float(words[1])
            b.y=float(words[2])
            b.z=float(words[3])
            isB=True
            continue
        if(words[0]=='C' and isCell):
            c=lattice_vect()
            c.x=float(words[1])
            c.y=float(words[2])
            c.z=float(words[3])
            isC=True
            continue
        if(isA and isB and isC):
            break
        if(isABC and isALBEGA):
            a,b,c=crystobox(cell_par)
            break
    return a,b,c

def getCoordCP2K(fname):
    fi=open(fname,'r')
    atoms=[]
    isCoord=False
    isScaled=False
    for line in fi:
        words=line.split()
        if("&COORD" in line):
            isCoord=True
            continue
        if("UNIT" in line):
            continue
        if("SCALED" in line):
            words=line.split()
            if(len(words)<=1):
                isScaled=True
            elif(words[1].strip()=='T'):
                isScaled=True
            else:
                isScaled=False
            continue
        if("&END COORD" in line):
            isCoord=False
            break
        if(isCoord):
            atoms.append(Atom())
            atoms[len(atoms)-1].name=words[0]
            atoms[len(atoms)-1].x=float(words[1])
            atoms[len(atoms)-1].y=float(words[2])
            atoms[len(atoms)-1].z=float(words[3])
            atoms[len(atoms)-1].el=words[0]
    return atoms,isScaled

def typeatom(atom):
    if(atom.name.strip()=='CAL' or atom.name.strip()=='Ca'):
        atom.el='Ca'
    elif(atom.name.strip()=='CLA' or atom.name.strip()[0:3]=='CLG' or atom.name.strip()[0:3]=='Cl'):
        atom.el='Cl'
    elif(atom.name.strip()=='SOD' or atom.name.strip()=='Na'):
        atom.el='Na'
    elif(atom.name.strip()=='MGA' or atom.name.strip()=='MG' or atom.name.strip()=='Mg'):
        atom.el='Mg'
    elif(atom.name.strip()=='ZN' or atom.name.strip()=='Zn'):
        atom.el='Zn'
    elif(atom.name.strip()=='POT' or atom.name.strip()=='K'):
        atom.el='K'
    elif(atom.name.strip()=='RUB'):
        atom.el='Rb'
    elif(atom.name.strip()=='FE' or atom.name.strip()=='Fe'):
        atom.el='Fe'
    elif(atom.name.strip()=='CES' or atom.name.strip()=='Ce'):
        atom.el='Ce'
    elif(atom.name.strip()=='CAD' or atom.name.strip()=='Cd'):
        atom.el='Cd'
    elif(atom.name.strip()=='ALG1' or atom.name.strip()=='Al'):
        atom.el='Al'
    elif(atom.name.strip()[0:2]=='BR' or atom.name.strip()=='Br'):
        atom.el='Br'
    elif(atom.name.strip()[0:2]=='AU' or atom.name.strip()=='Au'):
        atom.el='Au'
    elif(atom.name.strip()=='BAR'):
        atom.el='Ba'
    elif(atom.name.strip()=='LIT'):
        atom.el='Li'
    elif(len(atom.name.strip())>1 and atom.name.strip()[1].islower()):
        atom.el=atom.name.strip()[0:2]
    else:
        atom.el=' '+atom.name.strip()[0]
    return

def cart2frac(atoms,a,b,c):
    r=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
    ir=la.inv(r)
    tir=np.transpose(ir)
    for at in atoms:
        xyz=np.array([at.x,at.y,at.z])
        uvw=np.matmul(tir,xyz)
        at.x=uvw[0]
        at.y=uvw[1]
        at.z=uvw[2]
    return

def frac2cart(atoms,a,b,c):
    r=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
    tr=np.transpose(r)
    for at in atoms:
        uvw=np.array([at.x,at.y,at.z])
        xyz=np.matmul(tr,uvw)
        at.x=xyz[0]
        at.y=xyz[1]
        at.z=xyz[2]
    return

def wz(a,b,c):
    ra=lattice_vect()
    rb=lattice_vect()
    rc=lattice_vect()
    a.norm=math.sqrt((a.x*a.x)+(a.y*a.y)+(a.z*a.z))
    b.norm=math.sqrt((b.x*b.x)+(b.y*b.y)+(b.z*b.z))
    c.norm=math.sqrt((c.x*c.x)+(c.y*c.y)+(c.z*c.z))
    ra.x=b.y*c.z-c.y*b.z
    ra.y=b.z*c.x-c.z*b.x
    ra.z=b.x*c.y-c.x*b.y
    rb.x=c.y*a.z-a.y*c.z
    rb.y=c.z*a.x-a.z*c.x
    rb.z=c.x*a.y-a.x*c.y
    rc.x=a.y*b.z-b.y*a.z
    rc.y=a.z*b.x-b.z*a.x
    rc.z=a.x*b.y-b.x*a.y
    det=(a.x*ra.x)+(a.y*ra.y)+(a.z*ra.z)
    if(det>sys.float_info.min):
        ra.x/=det
        ra.y/=det
        ra.z/=det
        rb.x/=det
        rb.y/=det
        rb.z/=det
        rc.x/=det
        rc.y/=det
        rc.z/=det
    ra.norm=math.sqrt((ra.x*ra.x)+(ra.y*ra.y)+(ra.z*ra.z))
    rb.norm=math.sqrt((rb.x*rb.x)+(rb.y*rb.y)+(rb.z*rb.z))
    rc.norm=math.sqrt((rc.x*rc.x)+(rc.y*rc.y)+(rc.z*rc.z))
    vol=math.fabs(det)
    return(ra,rb,rc,vol)

def writePoscar(fName,atoms,a,b,c,args):
    if(args.potcar):
        fp=open("POTCAR",'w')
    fo=open(fName,'w')
    fo.write("Written by cp2k2pdb.py by P.-A. Cazade\n")
    fo.write("   1.00000000000000\n")
    fo.write(" %22.16f%22.16f%22.16f\n" % (a.x,a.y,a.z))
    fo.write(" %22.16f%22.16f%22.16f\n" % (b.x,b.y,b.z))
    fo.write(" %22.16f%22.16f%22.16f\n" % (c.x,c.y,c.z))
    lel=[]
    nel=[]
    for at in atoms:
        typeatom(at)
        if(at.el.strip() not in lel):
            lel.append(at.el.strip())
            nel.append(1)
        else:
            i=lel.index(at.el.strip())
            nel[i]+=1
    for el in lel:
        fo.write("%5s" % (el))
    fo.write("\n")
    for n in nel:
        fo.write("%6d" % (n))
    fo.write("\n")
    if(args.sd):
        fo.write("Selective dynamics\n")
    fo.write("Direct\n")
    for el in lel:
        if(args.potcar):
            fpot=args.potcar_source[0].strip()+'/'+el.strip()+'/POTCAR'
            ft=open(fpot,'r')
            for line in ft:
                fp.write(line)
            ft.close()
        for at in atoms:
            if(at.el.strip()==el):
                at.x-=math.floor(at.x)
                at.y-=math.floor(at.y)
                at.z-=math.floor(at.z)
                if(args.sd):
                    fo.write("%20.16f%20.16f%20.16f T T T\n" % (at.x,at.y,at.z))
                else:
                    fo.write("%20.16f%20.16f%20.16f\n" % (at.x,at.y,at.z))
    fo.close()
    if(args.potcar):
        fp.close()
    if(args.kpoints):
        fk=open("KPOINTS",'w')
        ra,rb,rc,vol=wz(a,b,c)
        is1=round(ra.norm/args.kgrid[0])
        is2=round(rb.norm/args.kgrid[0])
        is3=round(rc.norm/args.kgrid[0])
        fk.write("A\n")
        fk.write("0\n")
        fk.write("G\n")
        fk.write("%d %d %d\n" % (is1,is2,is3))
        fk.write("0 0 0\n")
        fk.close()
    if(args.incar):
        fc=open("INCAR",'w')
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
        if("IONS" in args.cp2k_opt[0]):
            fc.write("ISIF = 2\n")
        elif("CELL" in args.cp2k_opt[0]):
            if(args.cp2k_opt_angles):
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
        if(args.d3):
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

def readPoscar(fName):
    isSelective=False
    isScaled=False
    fo=open(fName,'r')
    title=fo.readline()
    coeff=float(fo.readline())
    a=lattice_vect()
    b=lattice_vect()
    c=lattice_vect()
    words=fo.readline().split()
    a.x=float(words[0])*coeff
    a.y=float(words[1])*coeff
    a.z=float(words[2])*coeff
    words=fo.readline().split()
    b.x=float(words[0])*coeff
    b.y=float(words[1])*coeff
    b.z=float(words[2])*coeff
    words=fo.readline().split()
    c.x=float(words[0])*coeff
    c.y=float(words[1])*coeff
    c.z=float(words[2])*coeff
    words=fo.readline().split()
    lel=[]
    for w in words:
        lel.append(w)
    words=fo.readline().split()
    nel=[]
    for w in words:
        nel.append(int(w))
    line=fo.readline()
    if(("selective" in line.lower()) or (line[0].lower()=='s')):
        isSelective=True
        line=fo.readline()
    if("direct" in line.lower()):
        isScaled=True
    atoms=[]
    for i in range(len(nel)):
        for j in range(nel[i]):
            line=fo.readline()
            words=line.split()
            atoms.append(Atom())
            idx=len(atoms)-1
            atoms[idx].name=lel[i]
            atoms[idx].el=lel[i]
            atoms[idx].idx=idx+1
            atoms[idx].x=float(words[0])
            atoms[idx].y=float(words[1])
            atoms[idx].z=float(words[2])
    a.norm=math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)
    b.norm=math.sqrt(b.x*b.x+b.y*b.y+b.z*b.z)
    c.norm=math.sqrt(c.x*c.x+c.y*c.y+c.z*c.z)
    fo.close()
    return(atoms,a,b,c,isScaled)

def readGen(fName):
    fo=open(fName,'r')
    words=fo.readline().split()
    nAtom=int(words[0])
    sysType=words[1]
    words=fo.readline().split()
    lel=[]
    for w in words:
        lel.append(w)
    atoms=[]
    for i in range(nAtom):
        line=fo.readline()
        words=line.split()
        idx=int(words[0])
        iel=int(words[1])-1
        atoms.append(Atom())
        atoms[i].name=lel[iel]
        atoms[i].el=lel[iel]
        atoms[i].idx=idx
        atoms[i].x=float(words[2])
        atoms[i].y=float(words[3])
        atoms[i].z=float(words[4])
    a=lattice_vect()
    b=lattice_vect()
    c=lattice_vect()
    words=fo.readline().split()
    shift=[float(words[0]),float(words[1]),float(words[2])]
    words=fo.readline().split()
    a.x=float(words[0])
    a.y=float(words[1])
    a.z=float(words[2])
    words=fo.readline().split()
    b.x=float(words[0])
    b.y=float(words[1])
    b.z=float(words[2])
    words=fo.readline().split()
    c.x=float(words[0])
    c.y=float(words[1])
    c.z=float(words[2])
    a.norm=math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)
    b.norm=math.sqrt(b.x*b.x+b.y*b.y+b.z*b.z)
    c.norm=math.sqrt(c.x*c.x+c.y*c.y+c.z*c.z)
    fo.close()
    return(atoms,a,b,c,sysType)

def writeGen(fName,atoms,a,b,c,sysType):
    fo=open(fName,'w')
    fo.write("%d %s\n" % (len(atoms),sysType))
    lel=[]
    nel=[]
    for at in atoms:
        typeatom(at)
        if(at.el.strip() not in lel):
            lel.append(at.el.strip())
    for el in lel:
        fo.write(" %s" % (el))
    fo.write("\n")
    idx=0
    for i in range(len(lel)):
        el=lel[i]
        for at in atoms:
            if(at.el.strip()==el):
                idx+=1
                if('F' in sysType):
                    at.x-=math.floor(at.x)
                    at.y-=math.floor(at.y)
                    at.z-=math.floor(at.z)
                fo.write("%5d%3d%20.10E%20.10E%20.10E\n" % (idx,i+1,at.x,at.y,at.z))
    fo.write(" %22.10E%22.10E%22.10E\n" % (0.0,0.0,0.0))
    fo.write(" %22.10E%22.10E%22.10E\n" % (a.x,a.y,a.z))
    fo.write(" %22.10E%22.10E%22.10E\n" % (b.x,b.y,b.z))
    fo.write(" %22.10E%22.10E%22.10E\n" % (c.x,c.y,c.z))
    fo.close()
    return

def readXyz(fName):
    fi=open(fName,'r')
    words=fi.readline().split()
    natom=int(words[0])
    words=fi.readline().split()
    if(isfloat(words[0]) and len(words)>=6):
        words.insert(0,'CRYST')
        a,b,c=crystobox(words)
    else:
        a=lattice_vect()
        b=lattice_vect()
        c=lattice_vect()
    atoms=[]
    for i in range(natom):
        words=fi.readline().split()
        atoms.append(Atom())
        atoms[i].name=words[0]
        atoms[i].x=float(words[1])
        atoms[i].y=float(words[2])
        atoms[i].z=float(words[3])
    fi.close()
    return(atoms,a,b,c)

def writeXyz(fName,atoms,a,b,c):
    fo=open(fName,'w')
    fo.write("%d\n" % (len(atoms)))
    if(a.norm>0.0 and b.norm>0.0 and c.norm>0.0):
        al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
        be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
        ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
        fo.write("%lf %lf %lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,al,be,ga))
    else:
        fo.write("%s\n" % ("System"))
    for at in atoms:
        typeatom(at)
        fo.write("%s %20.10E %20.10E %20.10E\n" % (at.el,at.x,at.y,at.z))
    fo.close()
    return

def readCif(fName):
    data=ase.io.read(fName)
    natom=len(data.positions)
    a=lattice_vect()
    b=lattice_vect()
    c=lattice_vect()
    a.x=data.cell[0][0]
    a.y=data.cell[0][1]
    a.z=data.cell[0][2]
    b.x=data.cell[1][0]
    b.y=data.cell[1][1]
    b.z=data.cell[1][2]
    c.x=data.cell[2][0]
    c.y=data.cell[2][1]
    c.z=data.cell[2][2]
    a.norm=math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)
    b.norm=math.sqrt(b.x*b.x+b.y*b.y+b.z*b.z)
    c.norm=math.sqrt(c.x*c.x+c.y*c.y+c.z*c.z)
    atoms=[]
    k=0
    for i in range(0,natom):
        atoms.append(Atom())
        atoms[k].name=data.symbols[i]
        atoms[k].el=data.symbols[i]
        atoms[k].idx=k+1
        atoms[k].x=data.positions[i][0]
        atoms[k].y=data.positions[i][1]
        atoms[k].z=data.positions[i][2]
        k=k+1
    return(atoms,a,b,c)

def readCry(fName):
    data=ase.io.read(fName,format='crystal')
    natom=len(data.positions)
    a=lattice_vect()
    b=lattice_vect()
    c=lattice_vect()
    a.x=data.cell[0][0]
    a.y=data.cell[0][1]
    a.z=data.cell[0][2]
    b.x=data.cell[1][0]
    b.y=data.cell[1][1]
    b.z=data.cell[1][2]
    c.x=data.cell[2][0]
    c.y=data.cell[2][1]
    c.z=data.cell[2][2]
    a.norm=math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)
    b.norm=math.sqrt(b.x*b.x+b.y*b.y+b.z*b.z)
    c.norm=math.sqrt(c.x*c.x+c.y*c.y+c.z*c.z)
    atoms=[]
    k=0
    for i in range(natom):
        atoms.append(Atom())
        atoms[k].name=data.symbols[i]
        atoms[k].el=data.symbols[i]
        atoms[k].idx=k+1
        atoms[k].x=data.positions[i][0]
        atoms[k].y=data.positions[i][1]
        atoms[k].z=data.positions[i][2]
        k=k+1
    return(atoms,a,b,c)

def writeFdf(fName,atoms,a,b,c,sysType,isScaled):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
        'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
        'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
        'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn']
    wz(a,b,c)
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    NL=""
    w=fName.split('.')
    for i in range(len(w)-1):
        if(i==0):
            NL=w[i]
        else:
            NL=NL+'_'+w[i]
    fn=NL+".fdf"
    fo=open(fn,'w')
    fo.write("SystemName %s\n" % (NL))
    fo.write("SystemLabel %s\n" % (NL))
    fo.write("\n")
    lel=[]
    nel=[]
    for at in atoms:
        typeatom(at)
        if(at.el.strip() not in lel):
            lel.append(at.el.strip())
    fo.write("NumberOfAtoms %d\n" % (len(atoms)))
    fo.write("NumberOfSpecies %d\n" % (len(lel)))
    fo.write("\n")
    fo.write("%block ChemicalSpeciesLabel\n")
    i=0
    for el in lel:
        i=i+1
        atn=ks.index(el.strip())+1
        fo.write("%d %d %s\n" % (i,atn,el))
    fo.write("%endblock ChemicalSpeciesLabel\n")
    fo.write("\n")
    fo.write("PAO.BasisSize    DZP\n")
    fo.write("PAO.EnergyShift  0.01 Ry\n")
    fo.write("XC.functional    GGA\n")
    fo.write("XC.authors       PBE\n")
    fo.write("\n")
    fo.write("%block kgrid.MonkhorstPack\n")
    kx=math.ceil(24./a.norm)
    ky=math.ceil(24./b.norm)
    kz=math.ceil(24./c.norm)
    fo.write("%d %d %d %f\n" % (kx,0,0,0.0))
    fo.write("%d %d %d %f\n" % (0,ky,0,0.0))
    fo.write("%d %d %d %f\n" % (0,0,kz,0.0))
    fo.write("%endblock kgrid.MonkhorstPack\n")
    fo.write("\n")
    fo.write("LatticeConstant 1.0 Ang\n")
    fo.write("%block LatticeVectors\n")
    fo.write(" %22.10E%22.10E%22.10E\n" % (a.x,a.y,a.z))
    fo.write(" %22.10E%22.10E%22.10E\n" % (b.x,b.y,b.z))
    fo.write(" %22.10E%22.10E%22.10E\n" % (c.x,c.y,c.z))
    fo.write("%endblock LatticeVectors\n")
    fo.write("\n")
    if('F' in sysType or isScaled):
        fo.write("AtomicCoordinatesFormat  Fractional\n")
    else:
        fo.write("AtomicCoordinatesFormat  Ang\n")
    fo.write("%block AtomicCoordinatesAndAtomicSpecies\n")
    for at in atoms:
        iel=lel.index(at.el.strip())+1
        if('F' in sysType or isScaled):
            at.x-=math.floor(at.x)
            at.y-=math.floor(at.y)
            at.z-=math.floor(at.z)
        fo.write("%20.10E%20.10E%20.10E%5d\n" % (at.x,at.y,at.z,iel))
    fo.write("%endblock AtomicCoordinatesAndAtomicSpecies\n")
    fo.write("\n")
    fo.write("MaxSCFIterations      300\n")
    fo.write("SCF.Mix.First         T\n")
    fo.write("SCF.Mixer.Weight      0.25\n")
    fo.write("SCF.Mixer.History     6\n")
    fo.write("DM.UseSaveDM          T\n")
    fo.write("TS.MixH               yes\n")
    fo.write("TS.HS.Save            T\n")
    fo.write("TS.DE.Save            T\n")
    fo.write("\n")
    fo.write("MD.UseSaveXV          T\n")
    fo.write("\n")
    fo.write("WriteMullikenPop                1\n")
    fo.write("SaveElectrostaticPotential      T\n")
    fo.write("WriteCoorXmol                   T\n")
    fo.write("WriteMDXmol                     T\n")
    fo.write("WriteMDhistory                  F\n")
    fo.write("WriteEigenvalues                yes\n")
    fo.write("\n")
    fo.close()
    return

def writeCp2k(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire):
    ext=inName.split('.')[-1]
    if(args.cp2k_template is not None):
        writeCp2kTemplate(args.cp2k_template,outName,atoms,a,b,c,isScaled,hall_number,args,dire)
    elif((ext=='inp') or (ext=='restart')):
        writeCp2kTemplate(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire)
    else:
        writeCp2kDefault(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire)
    return

def writeCp2kTemplate(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire):
    lattice_list=['TRICLINIC','MONOCLINIC','MONOCLINIC_AB','MONOCLINIC_GAMMA_AB',
                 'ORTHORHOMBIC','TETRAGONAL','TETRAGONAL_AB','TETRAGONAL_AC',
                 'TETRAGONAL_BC','RHOMBOHEDRAL','HEXAGONAL','CUBIC','NONE']
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']
    kn=['1','2','3','4','3','4','5','6','7','8','9','10','3','4','5','6','7',
        '8','9','10','11','12','13','14','15','16','17','18','11','12','13',
        '4','5','6','7','8','9','10','11','12','13','14','15','16','17',
        '18','11','12','13','4','5','6','7','8','9','10','12','13','14','15',
        '16','17','18','11','12','13','4','5','6','7','8']
    lel=[]
    nel=[]
    for at in atoms:
        typeatom(at)
        if(at.el.strip() not in lel):
            lel.append(at.el.strip())
            nel.append(1)
        else:
            i=lel.index(at.el.strip())
            nel[i]+=1
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()
    wz(a,b,c)
    al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
    be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
    ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    tih=np.transpose(la.inv(hmat))
    if(((not args.cp2k_elastic_piezo) and (not args.cp2k_dielectric)) or (len(atoms)<=10000)):
        if(args.symmetry_excluded is not None):
            exclusionList=[]
            for al in args.symmetry_excluded:
                for i in range(al[0],al[1]+1):
                    exclusionList.append(i-1)
            syst=[]
            for i in range(len(atoms)):
                if i in exclusionList:
                    continue
                syst.append(Atom())
                cpAtom(syst[-1],atoms[i])
            dataset=get_dataset(syst,hmat,hall_number)
            del(syst)
        else:
            dataset=get_dataset(atoms,hmat,hall_number)
        print('Space Group:',dataset['number'])
        if(dataset['number']<=2):
            if( math.fabs(90.0-al)<=1e-2 and math.fabs(90.0-be)<=1e-2 and math.fabs(90.0-ga)<=1e-2 ):
                bravais_lattice='ORTHORHOMBIC'
            else:
                bravais_lattice='TRICLINIC'
        elif(dataset['number']<=15):
            if((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(90.0-ga)<=1e-2)):
                bravais_lattice='MONOCLINIC_GAMMA_AB'
            else:
                bravais_lattice='MONOCLINIC'
        elif(dataset['number']<=74):
            bravais_lattice='ORTHORHOMBIC'
        elif(dataset['number']<=142):
            if((math.fabs(c.norm-a.norm)<=1e-4)):
                bravais_lattice='TETRAGONAL_AC'
            elif((math.fabs(c.norm-b.norm)<=1e-4)):
                bravais_lattice='TETRAGONAL_BC'
            else:
                bravais_lattice='TETRAGONAL'
        elif(dataset['number']<=167):
            if((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(c.norm-a.norm)<=1e-4)):
                bravais_lattice='RHOMBOHEDRAL'
            else:
                if((math.fabs(60.0-ga)<=1e-2)):
                    bravais_lattice='HEXAGONAL'
                else:
                    bravais_lattice='MONOCLINIC_GAMMA_AB'
        elif(dataset['number']<=194):
            if((math.fabs(60.0-ga)<=1e-2)):
                bravais_lattice='HEXAGONAL'
            else:
                bravais_lattice='MONOCLINIC_GAMMA_AB'
        elif(dataset['number']<=230):
            bravais_lattice='CUBIC'
        else:
            bravais_lattice='NONE'
    else:
        bravais_lattice='TRICLINIC'
    isA=False
    isB=False
    isC=False
    fi=open(inName,'r')
    fo=open(outName,'w')
    basename=outName.split('.')[0]
    for i in range(1,len(outName.split('.'))-1):
        basename=basename+'.'+outName.split('.')[i]
    fi.seek(0,2)
    eof = fi.tell()
    fi.seek(0,0)
    isSer=False
    isMM=False
    isPrevLineDFT=False
    nAtoms=len(atoms)*math.ceil(18./a.norm)*math.ceil(18./b.norm)*math.ceil(18./c.norm)
    if(args.cp2k_opt_algo[0].strip()=='BFGS' and (nAtoms>=1000)):
        args.cp2k_opt_algo[0]='LBFGS'
    while (fi.tell() < eof):
        line=fi.readline()
        words=line.split()
        if(line.strip()[0]=='#'):
            continue
        if('&MM' in line):
            isMM=True
        if('PROJECT_NAME' in line):
            title='   PROJECT_NAME '+basename.strip()
            fo.write('%s\n' % (title) ) 
            continue
        if('RUN_TYPE' in line and (args.cp2k_elastic_piezo or args.strain or args.cp2k_dielectric or (args.cp2k_opt[0].strip()=='IONS'))):
            run='   RUN_TYPE GEO_OPT'
            fo.write('%s\n' % (run) ) 
            continue
        elif(args.cp2k_opt[0].strip()=='NONE'):
            run='   RUN_TYPE ENERGY_FORCE'
            fo.write('%s\n' % (run) ) 
            continue
        if('&END DFT' in line):
            isPrevLineDFT=False
        if(args.cp2k_molecular_orbitals and isPrevLineDFT):
            if('&MO_CUBES' in line):
                line=fi.readline()
                while('&END MO_CUBES' not in line):
                    line=fi.readline()
                    continue
                continue
            if('&PRINT' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                fo.write('        &MO_CUBES\n')
                fo.write('          NHOMO %d\n' % (args.cp2k_mo_numbers[0]))
                fo.write('          NLUMO %d\n' % (args.cp2k_mo_numbers[1]))
                fo.write('        &END MO_CUBES\n')
                continue
        if(args.cp2k_elastic_piezo or args.strain or args.cp2k_dielectric or (args.cp2k_opt[0].strip()=='IONS')):
            if('&MOTION' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                opt='BFGS'
                while('&END MOTION' not in line):
                    if( ('OPTIMIZER' in line) and (args.cp2k_opt_algo[0].strip())):
                        opt=args.cp2k_opt_algo[0].strip()
                    elif('OPTIMIZER' in line):
                        words=line.split()
                        opt=words[1]
                        line=fi.readline()
                        continue
                    if('SYMM_EXCLUDE_RANGE' in line):
                        words=line.split()
                        if(not isSer):
                            ser=[]
                        isSer=True
                        ser.append([int(words[1]),int(words[2])])
                    line=fi.readline()
                    continue
                fo.write('   &GEO_OPT\n')
                fo.write('     OPTIMIZER  %s\n' % (opt.strip()))
                fo.write('     MAX_ITER  10000\n')
                if(args.cp2k_opt_symmetry):
                    fo.write('     KEEP_SPACE_GROUP  T\n')
                    if(hall_number>0):
                        fo.write('     HALL_NUMBER  %d\n' % (hall_number))
                    elif(dataset is not None):
                        fo.write('     HALL_NUMBER  %d\n' % (dataset['hall_number']>0))
                else:
                    fo.write('     KEEP_SPACE_GROUP  F\n')
                fo.write('     EPS_SYMMETRY     1.0000000000000000E-04\n')
                if(isSer and (args.symmetry_excluded is not None)):
                    print('WARNING: Symmetry excluded atoms are defined both in the template and as an input option. The latter are used.')
                if(args.symmetry_excluded is not None):
                    for al in args.symmetry_excluded:
                        fo.write('     SYMM_EXCLUDE_RANGE %d %d\n' % (al[0],al[1]))
                elif(isSer):
                    for al in ser:
                        fo.write('     SYMM_EXCLUDE_RANGE %d %d\n' % (al[0],al[1]))
                if(dire is not None):
                    fo.write('     SYMM_REDUCTION %lf %lf %lf\n' % (dire[0],dire[1],dire[2]))
                fo.write('   &END GEO_OPT\n')
                fo.write('%s' % (line))
                continue
        elif(args.cp2k_opt[0].strip()=='NONE'):
            if('&MOTION' in line):
                line=fi.readline()
                while('&END MOTION' not in line):
                    line=fi.readline()
                    continue
                continue
        if(args.cp2k_dielectric and (dire is not None)):
            if('&PERIODIC_EFIELD' in line):
                line=fi.readline()
                while('&END PERIODIC_EFIELD' not in line):
                    line=fi.readline()
                    continue
                continue
            if(('STRESS_TENSOR' in line)):
                continue
            if('&DFT' in line):
                isPrevLineDFT=True
                fo.write( "%s" % (line))
                uvw=np.matmul(tih,dire)
                norm=la.norm(uvw)
                uvw=uvw/norm
                fo.write('     &PERIODIC_EFIELD\n')
                fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field[0]))
                fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
                fo.write('     &END PERIODIC_EFIELD\n')
            if('&MM' in line):
                fo.write( "%s" % (line))
                uvw=dire
                norm=la.norm(uvw)
                uvw=uvw/norm
                fo.write('     &PERIODIC_EFIELD\n')
                fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field[0]))
                fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
                fo.write('     &END PERIODIC_EFIELD\n')
                continue
        if( (args.cp2k_ot_algo[0].strip()!='STRICT') or (args.cpscf) ):
            if('&SCF' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                isScfGuess=False
                if(args.cp2k_ot_algo[0].strip()=='RESTART'):
                    fc=inName.split('.')[0]
                    for i in range(1,len(inName.split('.'))-1):
                        fc=fc+'.'+inName.split('.')[i]
                    fc=fc+'.wfn'
                    fp=outName.split('.')[0]
                    for i in range(1,len(outName.split('.'))-1):
                        fp=fp+'.'+outName.split('.')[i]
                    fp=fp+'.wfn'
                    command='cp '+fc+' '+fp
                    os.system(command)
                    fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo[0].strip()))
                while('&END SCF' not in line):
                    if('MAX_SCF' and args.cpscf):
                        fo.write('       MAX_SCF 100\n')
                        line=fi.readline()
                        continue
                    if('SCF_GUESS' in line and (args.cp2k_ot_algo[0].strip()=='RESTART')):
                        line=fi.readline()
                        continue
                    if('&OT' in line and not args.cpscf):
                        fo.write( "%s" % (line))
                        line=fi.readline()
                        if(args.cp2k_ot_algo[0].strip()=='IRAC'):
                            fo.write('         ALGORITHM %s\n' % (args.cp2k_ot_algo[0].strip()))
                        while('&END OT' not in line):
                            if('ALGORITHM' in line and (args.cp2k_ot_algo[0].strip()=='IRAC')):
                                line=fi.readline()
                                continue
                            fo.write('%s' % (line))
                            line=fi.readline()
                            continue
                        fo.write('%s' % (line))
                        line=fi.readline()
                        continue
                    elif('&OT' in line and args.cpscf):
                        line=fi.readline()
                        while('&END OT' not in line):
                            line=fi.readline()
                            continue
                        fo.write('       SCF_GUESS  ATOMIC\n')
                        fo.write('       ADDED_MOS 100')
                        fo.write('       &SMEAR ON\n')
                        fo.write('         METHOD FERMI_DIRAC\n')
                        fo.write('         ELECTRONIC_TEMPERATURE [K] 300\n')
                        fo.write('       &END SMEAR\n')
                        fo.write('       &DIAGONALIZATION\n')
                        fo.write('        ALGORITHM STANDARD\n')
                        fo.write('       &END DIAGONALIZATION\n')
                        fo.write('       &MIXING\n')
                        fo.write('         METHOD BROYDEN_MIXING\n')
                        fo.write('       &END MIXING\n')
                        line=fi.readline()
                        continue
                    fo.write('%s' % (line))
                    line=fi.readline()
                    continue
                fo.write('%s' % (line))
                continue
        if( (args.exchange_correlation[0].strip()!='DEFAULT') or args.d3):
            isXc=False
            isD3=False
            if('&XC' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                while(line.strip()!='&END XC'):
                    if('&XC_FUNCTIONAL' in line):
                        isXc=True
                        fo.write( "%s" % (line))
                        line=fi.readline()
                        while('&END XC_FUNCTIONAL' not in line):
                            if(args.exchange_correlation[0].strip()!='DEFAULT'):
                                xc='         &'+args.exchange_correlation[0].strip()+' T'
                                exc='         &END '+args.exchange_correlation[0].strip()
                                fo.write('%s\n' % (xc))
                                fo.write('%s\n' % (exc))
                            else:
                                fo.write('%s' % (line))
                            line=fi.readline()
                            continue
                        fo.write('%s' % (line))
                        line=fi.readline()
                        continue
                    if('&VDW_POTENTIAL' in line):
                        isD3=True
                    fo.write('%s' % (line))
                    line=fi.readline()
                    continue
                if(not isXc):
                    fo.write('      &XC_FUNCTIONAL\n')
                    if(args.exchange_correlation[0].strip()=='DEFAULT'):
                        xc='         &'+'PBE'+' T'
                        exc='         &END PBE'
                    else:
                        xc='         &'+args.exchange_correlation[0].strip()+' T'
                        exc='         &END '+args.exchange_correlation[0].strip()
                    fo.write('%s\n' % (xc))
                    fo.write('%s\n' % (exc))
                    fo.write('      &END XC_FUNCTIONAL\n')
                if(not isD3):
                    fo.write('       &VDW_POTENTIAL\n')
                    fo.write('         POTENTIAL_TYPE  PAIR_POTENTIAL\n')
                    fo.write('         &PAIR_POTENTIAL\n')
                    fo.write('           TYPE  DFTD3\n')
                    fo.write('           PARAMETER_FILE_NAME dftd3.dat\n')
                    fo.write('           REFERENCE_FUNCTIONAL %s\n' % (args.exchange_correlation[0].strip()))
                    fo.write('         &END PAIR_POTENTIAL\n')
                    fo.write('       &END VDW_POTENTIAL\n')
                fo.write('%s' % (line))
                continue
        if( (words[0]=='SYMMETRY') and (words[1] in lattice_list) ):
            fo.write('       SYMMETRY %s\n' % (bravais_lattice.strip()))
            continue
        if('NUMBER_OF_ATOMS' in line):
            fo.write('       NUMBER_OF_ATOMS %d\n' % (len(atoms)))
            continue
        if('MULTIPLE_UNIT_CELL' in line):
            fo.write('       MULTIPLE_UNIT_CELL %d %d %d\n' % (math.ceil(18./a.norm),math.ceil(18./b.norm),math.ceil(18./c.norm)))
            continue
        if(words[0]=='A'):
            isA=True
            if(not isB or not isC):
                continue
        if(words[0]=='B'):
            isB=True
            if(not isA or not isC):
                continue
        if(words[0]=='C'):
            isC=True
            if(not isA or not isB):
                continue
        if(isA and isB and isC):
            fo.write('       A  %25.16e %25.16e %25.16e\n' % (a.x,a.y,a.z) )
            fo.write('       B  %25.16e %25.16e %25.16e\n' % (b.x,b.y,b.z) )
            fo.write('       C  %25.16e %25.16e %25.16e\n' % (c.x,c.y,c.z) )
            isA=False
            isB=False
            isC=False
            continue
        if('&COORD' in line):
            fo.write( "%s" % (line))
            line=fi.readline()
            while('&END COORD' not in line):
                line=fi.readline()
                continue
            for at in atoms:
                if(isMM):
                    fo.write( '%6s %25.16e %25.16e %25.16e\n' % ( at.name,at.x,at.y,at.z ) )
                else:
                    fo.write( '%6s %25.16e %25.16e %25.16e\n' % ( at.el,at.x,at.y,at.z ) )
            fo.write("%s\n" % ('       UNIT angstrom'))
            if(isScaled):
                fo.write("%s\n" % ('       SCALED  T'))
            else:
                fo.write("%s\n" % ('       SCALED  F'))
            fo.write( "%s" % (line))
            continue
        if('COORD_FILE_NAME' in line):
            continue
        if('COORD_FILE_FORMAT' in line):
            continue
        if('&KIND' in line):
            line=fi.readline()
            while('&END KIND' not in line):
                line=fi.readline()
                continue
            continue
        if('&END SUBSYS' in line):
            if(isMM):
                lnm=[]
                for at in atoms:
                    if(at.name.strip() not in lnm):
                        fo.write("%s %s\n" % ('     &KIND',at.name.strip()))
                        fo.write("%s %s\n" % ('       ELEMENT',at.el.strip()))
                        fo.write("%s\n" % ('     &END KIND'))
                        lnm.append(at.name.strip())
            else:
                for el in lel:
                    k=ks.index(el.strip())
                    fo.write("%s %s\n" % ('     &KIND',el.strip()))
                    basisSet='       BASIS_SET '+args.basisset[0].strip()+'-MOLOPT-SR-GTH-q'
                    fo.write("%s%s\n" % (basisSet,kn[k]))
                    fo.write("%s%s\n" % ('       POTENTIAL GTH-PBE-q',kn[k]))
                    fo.write("%s\n" % ('     &END KIND'))
            fo.write( "%s" % (line))
            continue
        fo.write( "%s" % (line))
        continue
    fi.close()
    fo.close()
    return

def writeCp2kDefault(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']
    kn=['1','2','3','4','3','4','5','6','7','8','9','10','3','4','5','6','7',
        '8','9','10','11','12','13','14','15','16','17','18','11','12','13',
        '4','5','6','7','8','9','10','11','12','13','14','15','16','17',
        '18','11','12','13','4','5','6','7','8','9','10','12','13','14','15',
        '16','17','18','11','12','13','4','5','6','7','8']
    print(len(ks),len(kn))
    for i in range(len(ks)):
        print(i,ks[i],kn[i])
    lel=[]
    nel=[]
    for at in atoms:
        typeatom(at)
        if(at.el.strip() not in lel):
            lel.append(at.el.strip())
            nel.append(1)
        else:
            i=lel.index(at.el.strip())
            nel[i]+=1
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()
    wz(a,b,c)
    al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
    be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
    ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    tih=np.transpose(la.inv(hmat))
    if(args.symmetry_excluded is not None):
        exclusionList=[]
        for al in args.symmetry_excluded:
            for i in range(al[0],al[1]+1):
                exclusionList.append(i-1)
        syst=[]
        for i in range(len(atoms)):
            if i in exclusionList:
                continue
            syst.append(Atom())
            cpAtom(syst[-1],atoms[i])
        dataset=get_dataset(syst,hmat,hall_number)
        del(syst)
    else:
        print('Hall',hall_number)
        dataset=get_dataset(atoms,hmat,hall_number)
        print('Hall',hall_number)
    print(dataset['number'])
    if(dataset['number']<=2):
        if( math.fabs(90.0-al)<=1e-2 and math.fabs(90.0-be)<=1e-2 and math.fabs(90.0-ga)<=1e-2 ):
            bravais_lattice='ORTHORHOMBIC'
        else:
            bravais_lattice='TRICLINIC'
    elif(dataset['number']<=15):
        if((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(90.0-ga)<=1e-2)):
            bravais_lattice='MONOCLINIC_GAMMA_AB'
        else:
            bravais_lattice='MONOCLINIC'
    elif(dataset['number']<=74):
        bravais_lattice='ORTHORHOMBIC'
    elif(dataset['number']<=142):
        if((math.fabs(c.norm-a.norm)<=1e-4)):
            bravais_lattice='TETRAGONAL_AC'
        elif((math.fabs(c.norm-b.norm)<=1e-4)):
            bravais_lattice='TETRAGONAL_BC'
        else:
            bravais_lattice='TETRAGONAL'
    elif(dataset['number']<=167):
        if((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(c.norm-a.norm)<=1e-4)):
            bravais_lattice='RHOMBOHEDRAL'
        else:
            if((math.fabs(60.0-ga)<=1e-2)):
                bravais_lattice='HEXAGONAL'
            else:
                bravais_lattice='MONOCLINIC_GAMMA_AB'
    elif(dataset['number']<=194):
        if((math.fabs(60.0-ga)<=1e-2)):
            bravais_lattice='HEXAGONAL'
        else:
            bravais_lattice='MONOCLINIC_GAMMA_AB'
    elif(dataset['number']<=230):
        bravais_lattice='CUBIC'
    else:
        bravais_lattice='NONE'
    nAtoms=len(atoms)*math.ceil(18./a.norm)*math.ceil(18./b.norm)*math.ceil(18./c.norm)
    if(args.cp2k_opt_algo[0].strip()=='BFGS' and (nAtoms>=1000)):
        args.cp2k_opt_algo=['LBFGS']
    fo=open(outName,'w')
    fo.write(' &GLOBAL\n')
    fo.write('   PRINT_LEVEL  MEDIUM\n')
    basename=outName.split('.')[0]
    for i in range(1,len(outName.split('.'))-1):
        basename=basename+'.'+outName.split('.')[i]
    fo.write('   PROJECT_NAME %s\n' % (basename.strip()))
    if(args.cp2k_opt[0].strip()=='CELL'):
        fo.write('   RUN_TYPE  %s\n' % ('CELL_OPT'))
    elif(args.cp2k_opt[0].strip()=='IONS'):
        fo.write('   RUN_TYPE  %s\n' % ('GEO_OPT'))
    elif(args.cp2k_opt[0].strip()=='NONE'):
        fo.write('   RUN_TYPE  %s\n' % ('ENERGY_FORCE'))
    fo.write(' &END GLOBAL\n')
    if('NONE' not in args.cp2k_opt[0].strip()):
        fo.write(' &MOTION\n')
        if(args.cp2k_opt[0].strip()=='CELL'):
            fo.write('   &CELL_OPT\n')
            fo.write('     OPTIMIZER  %s\n' % (args.cp2k_opt_algo[0].strip()))
            fo.write('     MAX_ITER  1000\n')
            if(args.cp2k_opt_symmetry):
                fo.write('     KEEP_SPACE_GROUP  T\n')
                if(hall_number>0):
                    if(dataset['hall_number']==hall_number):
                        fo.write('     HALL_NUMBER  %d\n' % (hall_number))
                    else:
                        print("WARNING: The provided Hall Number: %d is different found by symmetry analysis: %d. Using the former. Check the structure!" % (hall_number,dataset['hall_number']))
                        fo.write('     HALL_NUMBER  %d\n' % (hall_number))
                #elif(dataset is not None):
                #    fo.write('     HALL_NUMBER  %d\n' % (dataset['hall_number']>0))
                fo.write('     EPS_SYMMETRY  1.0000000000000000E-04\n')
            if(args.cp2k_opt_angles):
                fo.write('     KEEP_ANGLES  T\n')
            else:
                fo.write('     KEEP_ANGLES  F\n')
            if(args.cp2k_opt_bravais):
                fo.write('     KEEP_SYMMETRY  T\n')
            else:
                fo.write('     KEEP_SYMMETRY  F\n')
            if(args.symmetry_excluded is not None):
                for al in args.symmetry_excluded:
                    fo.write('     SYMM_EXCLUDE_RANGE %d %d\n' % (al[0],al[1]))
            if(dire is not None):
                    fo.write('     SYMM_REDUCTION %lf %lf %lf\n' % (dire[0],dire[1],dire[2]))
            fo.write('   &END CELL_OPT\n')
        elif(args.cp2k_opt[0].strip()=='IONS'):
            fo.write('   &GEO_OPT\n')
            fo.write('     OPTIMIZER  %s\n' % (args.cp2k_opt_algo[0].strip()))
            fo.write('     MAX_ITER  1000\n')
            if(args.cp2k_opt_symmetry):
                fo.write('     KEEP_SPACE_GROUP  T\n')
                fo.write('     EPS_SYMMETRY  1.0000000000000000E-04\n')
            if(args.symmetry_excluded is not None):
                for al in args.symmetry_excluded:
                    fo.write('     SYMM_EXCLUDE_RANGE %d %d\n' % (al[0],al[1]))
            if(dire is not None):
                    fo.write('     SYMM_REDUCTION %lf %lf %lf\n' % (dire[0],dire[1],dire[2]))
            fo.write('   &END GEO_OPT\n')
        fo.write(' &END MOTION\n')
    fo.write(' &FORCE_EVAL\n')
    fo.write('   METHOD  QS\n')
    if(dire is None):
        fo.write('   STRESS_TENSOR  ANALYTICAL\n')
    fo.write('   &DFT\n')
    if(dire is not None):
        uvw=np.matmul(tih,dire)
        norm=la.norm(uvw)
        uvw=uvw/norm
        fo.write('     &PERIODIC_EFIELD\n')
        fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field[0]))
        fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
        fo.write('     &END PERIODIC_EFIELD\n')
    fo.write('     BASIS_SET_FILE_NAME BASIS_MOLOPT\n')
    fo.write('     POTENTIAL_FILE_NAME GTH_POTENTIALS\n')
    fo.write('     MULTIPLICITY  1\n')
    fo.write('     CHARGE  0\n')
    if(args.cpscf):
        if(args.cpkp):
            fo.write('     &KPOINTS\n')
            fo.write('       SCHEME MONKHORST-PACK %d %d %d\n' % (math.ceil(24./a.norm),math.ceil(24./b.norm),math.ceil(24./c.norm)))
            fo.write('     &END KPOINTS\n')
        fo.write('     &SCF\n')
        if(args.cp2k_ot_algo[0].strip()=='RESTART'):
            fc=inName.split('.')[0]
            for i in range(1,len(inName.split('.'))-1):
                fc=fc+'.'+inName.split('.')[i]
            fc=fc+'.wfn'
            fp=outName.split('.')[0]
            for i in range(1,len(outName.split('.'))-1):
                fp=fp+'.'+outName.split('.')[i]
            fp=fp+'.wfn'
            command='cp '+fc+' '+fp
            os.system(command)
            fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo[0].strip()))
        else:
            fo.write('       SCF_GUESS ATOMIC\n')
        fo.write('       ADDED_MOS 100\n')
        fo.write('       &SMEAR ON\n')
        fo.write('         METHOD FERMI_DIRAC\n')
        fo.write('         ELECTRONIC_TEMPERATURE [K] 300\n')
        fo.write('       &END SMEAR\n')
        fo.write('       &DIAGONALIZATION\n')
        fo.write('        ALGORITHM STANDARD\n')
        fo.write('       &END DIAGONALIZATION\n')
        fo.write('       &MIXING\n')
        fo.write('         METHOD BROYDEN_MIXING\n')
        fo.write('       &END MIXING\n')
        fo.write('     &END SCF\n')
    else:
        fo.write('     &SCF\n')
        fo.write('       MAX_SCF  20\n')
        fo.write('       EPS_SCF    1.e-6\n')
        if(args.cp2k_ot_algo[0].strip()=='RESTART'):
            fc=inName.split('.')[0]
            for i in range(1,len(inName.split('.'))-1):
                fc=fc+'.'+inName.split('.')[i]
            fc=fc+'.wfn'
            fp=outName.split('.')[0]
            for i in range(1,len(outName.split('.'))-1):
                fp=fp+'.'+outName.split('.')[i]
            fp=fp+'.wfn'
            command='cp '+fc+' '+fp
            os.system(command)
            fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo[0].strip()))
        else:
            fo.write('       SCF_GUESS ATOMIC\n')
        fo.write('       &OT  T\n')
        if(args.cp2k_ot_algo[0].strip()=='IRAC'):
            fo.write('         ALGORITHM %s\n' % (args.cp2k_ot_algo[0].strip()))
        fo.write('         MINIMIZER  CG\n')
        fo.write('         PRECONDITIONER  FULL_ALL\n')
        fo.write('       &END OT\n')
        fo.write('       &OUTER_SCF  T\n')
        fo.write('         EPS_SCF  1.e-6\n')
        fo.write('         MAX_SCF  100\n')
        fo.write('       &END OUTER_SCF\n')
        fo.write('     &END SCF\n')
    fo.write('     &QS\n')
    fo.write('       METHOD  GPW\n')
    fo.write('     &END QS\n')
    fo.write('     &MGRID\n')
    fo.write('       NGRIDS  5\n')
    fo.write('       CUTOFF     9.0000000000000000E+02\n')
    fo.write('       REL_CUTOFF     6.0000000000000000E+01\n')
    fo.write('     &END MGRID\n')
    fo.write('     &XC\n')
    fo.write('       DENSITY_CUTOFF    1.0000000000000000E-10\n')
    fo.write('       GRADIENT_CUTOFF   1.0000000000000000E-10\n')
    fo.write('       TAU_CUTOFF        1.0000000000000000E-10\n')
    fo.write('       &XC_FUNCTIONAL  NO_SHORTCUT\n')
    xc=args.exchange_correlation[0].strip()
    if(xc.strip()=='DEFAULT'):
        xc='PBE'
    fo.write('         &%s T\n' % (xc))
    fo.write('         &END %s\n' % (xc))
    fo.write('       &END XC_FUNCTIONAL\n')
    if(args.d3):
        fo.write('       &VDW_POTENTIAL\n')
        fo.write('         POTENTIAL_TYPE  PAIR_POTENTIAL\n')
        fo.write('         &PAIR_POTENTIAL\n')
        fo.write('           TYPE  DFTD3\n')
        fo.write('           PARAMETER_FILE_NAME dftd3.dat\n')
        fo.write('           REFERENCE_FUNCTIONAL %s\n' % (xc))
        fo.write('         &END PAIR_POTENTIAL\n')
        fo.write('       &END VDW_POTENTIAL\n')
    fo.write('     &END XC\n')
    fo.write('     &POISSON\n')
    fo.write('       POISSON_SOLVER  PERIODIC\n')
    fo.write('       PERIODIC  XYZ\n')
    fo.write('     &END POISSON\n')
    fo.write('     &PRINT\n')
    if(args.cp2k_molecular_orbitals):
        fo.write('        &MO_CUBES\n')
        fo.write('          NHOMO %d\n' % (args.cp2k_mo_numbers[0]))
        fo.write('          NLUMO %d\n' % (args.cp2k_mo_numbers[1]))
        fo.write('        &END MO_CUBES\n')
    fo.write('       &MOMENTS  SILENT\n')
    fo.write('         PERIODIC  T\n')
    fo.write('       &END MOMENTS\n')
    fo.write('     &END PRINT\n')
    if(args.cpscf):
        fo.write('     &LOCALIZE\n')
        fo.write('       &PRINT\n')
        fo.write('         &TOTAL_DIPOLE MEDIUM\n')
        fo.write('           REFERENCE ZERO\n')
        fo.write('         &END TOTAL_DIPOLE\n')
        fo.write('       &END PRINT\n')
        fo.write('     &END LOCALIZE\n')
    fo.write('   &END DFT\n')
    fo.write('   &SUBSYS\n')
    fo.write('     &CELL\n')
    fo.write('       A  %25.16e %25.16e %25.16e\n' % (a.x,a.y,a.z) )
    fo.write('       B  %25.16e %25.16e %25.16e\n' % (b.x,b.y,b.z) )
    fo.write('       C  %25.16e %25.16e %25.16e\n' % (c.x,c.y,c.z) )
    fo.write('       PERIODIC  XYZ\n')
    fo.write('       SYMMETRY %s\n' % (bravais_lattice.strip()))
    if(not args.cpkp):
        fo.write('       MULTIPLE_UNIT_CELL %d %d %d\n' % (math.ceil(18./a.norm),math.ceil(18./b.norm),math.ceil(18./c.norm)))
    fo.write('     &END CELL\n')
    fo.write('     &COORD\n')
    for at in atoms:
                fo.write( '%6s %25.16e %25.16e %25.16e\n' % ( at.el,at.x,at.y,at.z ) )
    fo.write('       UNIT angstrom\n')
    if(isScaled):
        fo.write('       SCALED  T\n')
    else:
        fo.write('       SCALED  F\n')
    fo.write('     &END COORD\n')
    fo.write('     &TOPOLOGY\n')
    fo.write('       NUMBER_OF_ATOMS %d\n' % (len(atoms)))
    if(not args.cpkp):
        fo.write('       MULTIPLE_UNIT_CELL %d %d %d\n' % (math.ceil(18./a.norm),math.ceil(18./b.norm),math.ceil(18./c.norm)))
    fo.write('     &END TOPOLOGY\n')
    for el in lel:
        k=ks.index(el.strip())
        fo.write("%s %s\n" % ('     &KIND',el.strip()))
        basisSet='       BASIS_SET '+args.basisset[0].strip()+'-MOLOPT-SR-GTH-q'
        fo.write("%s%s\n" % (basisSet,kn[k]))
        fo.write('       POTENTIAL GTH-%s-q%s\n' % (xc.strip(),kn[k]))
        fo.write("%s\n" % ('     &END KIND'))
    fo.write('   &END SUBSYS\n')
    fo.write('   &PRINT\n')
    fo.write('     &STRESS_TENSOR  ON\n')
    fo.write('     &END STRESS_TENSOR\n')
    fo.write('   &END PRINT\n')
    fo.write(' &END FORCE_EVAL\n')
    fo.close()
    return

def writeCry(outName,atoms,a,b,c,hall_number,args):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
        'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
        'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
        'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn']
    ra,rb,rc,vol=wz(a,b,c)
    al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
    be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
    ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    dataset=get_dataset(atoms,hmat,hall_number)
    print(dataset['number'])
    equivalent_atoms=dataset['equivalent_atoms']
    fo=open(outName,'w')
    basename=outName.split('.')[0]
    for i in range(1,len(outName.split('.'))-1):
        basename=basename+'.'+outName.split('.')[i]
    fo.write("%s\n" % (basename.strip()))
    fo.write('CRYSTAL\n')
    if(dataset['number']<=2):
        if( math.fabs(90.0-al)<=1e-2 and math.fabs(90.0-be)<=1e-2 and math.fabs(90.0-ga)<=1e-2 ):
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf\n" % (a.norm,b.norm,c.norm))
        else:
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,al,be,ga))
    elif(dataset['number']<=15):
        if((math.fabs(c.norm-a.norm)<=1e-4) and (math.fabs(90.0-al)<=1e-2) and (math.fabs(90.0-ga)<=1e-2)):
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,be))
        elif((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(90.0-al)<=1e-2) and (math.fabs(90.0-be)<=1e-2)):
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,ga))
        elif((math.fabs(c.norm-b.norm)<=1e-4) and (math.fabs(90.0-be)<=1e-2) and (math.fabs(90.0-ga)<=1e-2)):
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,al))
        else:
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,be))
    elif(dataset['number']<=74):
        fo.write('0 0 0\n')
        fo.write("%d\n" % (dataset['number']))
        fo.write("%lf %lf %lf\n" % (a.norm,b.norm,c.norm))
    elif(dataset['number']<=142):
        if((math.fabs(c.norm-a.norm)<=1e-4)):
            print('Not implemented yet')
            exit()
        elif((math.fabs(c.norm-b.norm)<=1e-4)):
            print('Not implemented yet')
            exit()
        else:
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf\n" % (a.norm,c.norm))
    elif(dataset['number']<=167):
        if((math.fabs(b.norm-a.norm)<=1e-4) and (math.fabs(c.norm-a.norm)<=1e-4)):
            fo.write('0 1 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf\n" % (a.norm,al))
        else:
            if((math.fabs(60.0-ga)<=1e-2)):
                fo.write('0 0 0\n')
                fo.write("%d\n" % (dataset['number']))
                fo.write("%lf %lf\n" % (a.norm,c.norm))
            else:
                fo.write('0 0 0\n')
                fo.write("%d\n" % (dataset['number']))
                fo.write("%lf %lf %lf\n" % (a.norm,c.norm,be))
    elif(dataset['number']<=194):
        if((math.fabs(60.0-ga)<=1e-2)):
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf\n" % (a.norm,c.norm))
        else:
            fo.write('0 0 0\n')
            fo.write("%d\n" % (dataset['number']))
            fo.write("%lf %lf %lf\n" % (a.norm,c.norm,be))
    elif(dataset['number']<=230):
        fo.write('0 0 0\n')
        fo.write("%d\n" % (dataset['number']))
        fo.write("%lf\n" % (a.norm,))
        bravais_lattice='CUBIC'
    else:
        print('Unknown Space Group:',dataset['number'])
        exit()
    neq=0
    for i in range(len(equivalent_atoms)):
        if(equivalent_atoms[i]!=i):
            continue
        neq+=1
    fo.write("%d\n" % (neq))
    for i in range(len(equivalent_atoms)):
        if(equivalent_atoms[i]!=i):
            continue
        atn=ks.index(atoms[i].el.strip())+1
        fo.write("%d %lf %lf %lf\n"%(atn,atoms[i].x,atoms[i].y,atoms[i].z))
    fo.write("ELAPIEZO\n")
    fo.write("NUMDERIV\n")
    fo.write("3\n")
    fo.write("PREOPTGEOM\n")
    fo.write("ENDELA\n")
    fo.write("BASISSET\n")
    basisSet='POB-'+args.basisset[0].strip()
    fo.write("%s\n" % (basisSet))
    fo.write("DFT\n")
    xc=args.exchange_correlation[0].strip()
    if((xc.strip()=='DEFAULT')):
        xc='PBE'
    if(args.d3):
        xc=xc.strip()+'-D3'
    fo.write("%s\n" % (xc))
    fo.write("ENDDFT\n")
    fo.write("SHRINK\n")
    is1=round(ra.norm/args.kgrid[0])
    is2=round(rb.norm/args.kgrid[0])
    is3=round(rc.norm/args.kgrid[0])
    print(is1,is2,is3,ra.norm,args.kgrid[0])
    isx=max(is1,is2,is3)
    isn=min(is1,is2,is3)
    if((isx-isn)<2):
        fo.write("%d %d\n" % (isx,isx))
    else:
        fo.write("0 %d\n" % (isx))
        fo.write("%d %d %d\n" % (is1,is2,is3))
    fo.write("END\n")
    fo.close()
    return

def undoSuperCell(atoms,a,b,c,isScaled,sc):
    ma=sc[0]
    mb=sc[1]
    mc=sc[2]

    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()

    ns=len(atoms)
    ne=int(len(atoms)/(ma*mb*mc))
    for i in range(ns-1,ne-1,-1):
        atoms.pop(-1)

    for at in atoms:
        at.x*=float(ma)
        at.y*=float(mb)
        at.z*=float(mc)

    if(a.norm>0. and b.norm>0. and c.norm>0.):
        a.x/=float(ma)
        a.y/=float(ma)
        a.z/=float(ma)
        b.x/=float(mb)
        b.y/=float(mb)
        b.z/=float(mb)
        c.x/=float(mc)
        c.y/=float(mc)
        c.z/=float(mc)
        wz(a,b,c)
    return(atoms,a,b,c)

def SuperCell(atoms,a,b,c,isScaled,sc):
    chainList=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    
    ma=sc[0]
    mb=sc[1]
    mc=sc[2]

    if(isScaled):
        frac2cart(atoms,a,b,c)
        isScaled=False

    m=0
    syst=[]
    ns=len(atoms)
    ne=ns*(ma*mb*mc)
    for l in range(mc):
        for k in range(mb):
            for j in range(ma):
                for i in range(ns):
                    syst.append(Atom())
                    cpAtom(syst[-1],atoms[i])
                    if(i==0 and j==0 and k==0 and l==0):
                        oldr=atoms[i].resIdx
                        nr=1
                    elif(atoms[i].resIdx!=oldr):
                        nr+=1
                    oldr=atoms[i].resIdx
                    chain=chainList[((nr-1) % int(len(chainList)/3))]
                    syst[-1].idx=m+1
                    syst[-1].resIdx=nr
                    syst[-1].chain=chain
                    syst[-1].x=atoms[i].x+j*a.x+k*b.x+l*c.x
                    syst[-1].y=atoms[i].y+j*a.y+k*b.y+l*c.y
                    syst[-1].z=atoms[i].z+j*a.z+k*b.z+l*c.z
                    m+=1

    a.x*=float(ma)
    a.y*=float(ma)
    a.z*=float(ma)
    b.x*=float(mb)
    b.y*=float(mb)
    b.z*=float(mb)
    c.x*=float(mc)
    c.y*=float(mc)
    c.z*=float(mc)
    wz(a,b,c)

    if(not isScaled):
        cart2frac(syst,a,b,c)
        isScaled=True

    return(syst,a,b,c)

def elastic_piezo_strain(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    numDev=[-1.0,1.0]
    voigt=[[0,0],[1,1],[2,2],[2,1],[2,0],[1,0]]
    
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()

    print(isScaled,atoms[0].x,atoms[15].x)
    basename=outName.split('.')[0]
    ext=outName.split('.')[-1]
    fname=basename+'.ref.'+ext.strip()
    io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args.hall_number[0],args,None)

    hmat=np.zeros((3,3))
    hmat[0,0]=a.x
    hmat[0,1]=a.y
    hmat[0,2]=a.z
    hmat[1,0]=b.x
    hmat[1,1]=b.y
    hmat[1,2]=b.z
    hmat[2,0]=c.x
    hmat[2,1]=c.y
    hmat[2,2]=c.z

    sa=lattice_vect()
    sb=lattice_vect()
    sc=lattice_vect()

    k=0
    for v in voigt:
        hall_number=args.hall_number_elastic_piezo[k]
        k+=1
        for n in numDev:
            e=np.identity(3)
            e[v[0],v[1]]+=args.cp2k_elastic_piezo_step[0]*n
            print(e)
            te=np.transpose(e)
            print(hmat)
            if('gro' in ext):
                hs=np.matmul(e,hmat)
            else:
                hs=np.matmul(hmat,te) # xyz axes, angles not conserved, VASP approach
            print(hs)
            sa.x=hs[0,0]
            sa.y=hs[0,1]
            sa.z=hs[0,2]
            sb.x=hs[1,0]
            sb.y=hs[1,1]
            sb.z=hs[1,2]
            sc.x=hs[2,0]
            sc.y=hs[2,1]
            sc.z=hs[2,2]
            wz(sa,sb,sc)
            fname=basename+'.strain_'+str(k)+'_'+str(int(n))+'.'+ext.strip()
            io_write(inName,fname,atoms,sa,sb,sc,isScaled,sysType,hall_number,args,None)
    return

def dielectric_field(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    numDev=[-1.0,1.0]
    ef=['x','y','z']
    basename=outName.split('.')[0]
    fname=basename+'.ref.inp'
    io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args.hall_number[0],args,None)
    for i in range(3):
        for n in numDev:
            fname=basename+'.efield_'+str(i+1)+'_'+str(int(n))+'.inp'
            dire=np.zeros((3))
            dire[i]=1.0*n
            io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args.hall_number[0],args,dire)
    return

def strain(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()

    hmat=np.zeros((3,3))
    hmat[0,0]=a.x
    hmat[0,1]=a.y
    hmat[0,2]=a.z
    hmat[1,0]=b.x
    hmat[1,1]=b.y
    hmat[1,2]=b.z
    hmat[2,0]=c.x
    hmat[2,1]=c.y
    hmat[2,2]=c.z

    sa=lattice_vect()
    sb=lattice_vect()
    sc=lattice_vect()

    ext=outName.split('.')[-1]
    basename=outName.split('.')[0]

    if(len(args.strain_list[0])>0):
        i=0
        fi=open(args.strain_list[0])
        for line in fi:
            w=line.split()
            if((len(line)==0) or (len(w)==0)):
                continue
            coord,a,b,c,tmp_isScaled,tmp_sysType,tmp_spg=io_read(w[0])
            if(i==0):
                h0=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
                ih0=la.inv(h0)
            else:
                h=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
                eta=np.matmul(h,ih0)
                hs=np.matmul(e,hmat)
                sa.x=hs[0,0]
                sa.y=hs[0,1]
                sa.z=hs[0,2]
                sb.x=hs[1,0]
                sb.y=hs[1,1]
                sb.z=hs[1,2]
                sc.x=hs[2,0]
                sc.y=hs[2,1]
                sc.z=hs[2,2]
                wz(sa,sb,sc)
                fname=basename+'.'+str(i+1)+'.'+ext
                io_write(inName,fname,atoms,sa,sb,sc,isScaled,sysType,0,args,None)
            i=i+1
    else:
        cax=['a','b','c','al','be','ga']
        fax=['x','y','z','yz','xz','xy']
        for s in args.strain_values:
            e=np.identity(3)
            if(args.strain_axis[0].strip()=='a' or args.strain_axis[0].strip()=='x'):
                e[0,0]+=float(s)
            elif(args.strain_axis[0].strip()=='b' or args.strain_axis[0].strip()=='y'):
                e[1,1]+=float(s)
            elif(args.strain_axis[0].strip()=='c' or args.strain_axis[0].strip()=='z'):
                e[2,2]+=float(s)
            elif(args.strain_axis[0].strip()=='al' or args.strain_axis[0].strip()=='yz'):
                e[2,1]+=float(s)
            elif(args.strain_axis[0].strip()=='be' or args.strain_axis[0].strip()=='xz'):
                e[2,0]+=float(s)
            elif(args.strain_axis[0].strip()=='ga' or args.strain_axis[0].strip()=='xy'):
                e[1,0]+=float(s)
            if(args.strain_axis[0].strip() in cax):
                hs=np.matmul(e,hmat) # abc axes, angles conserved
            elif(args.strain_axis[0].strip() in fax):
                te=np.transpose(e)
                hs=np.matmul(hmat,te) # xyz axes, angles not conserved
            sa.x=hs[0,0]
            sa.y=hs[0,1]
            sa.z=hs[0,2]
            sb.x=hs[1,0]
            sb.y=hs[1,1]
            sb.z=hs[1,2]
            sc.x=hs[2,0]
            sc.y=hs[2,1]
            sc.z=hs[2,2]
            wz(sa,sb,sc)
            fname=basename+'.'+args.strain_axis[0].strip()+'_'+s.strip()+'.'+ext
            io_write(inName,fname,atoms,sa,sb,sc,isScaled,sysType,0,args,None)
    return

def get_stress(inName,outName,args):
    cax=['a','b','c','al','be','ga']
    fax=['x','y','z','yz','xz','xy']
    axis=args.strain_axis[0].strip()

    basename=inName.split('.')[0]
    ext=inName.split('.')[-1]
    isVASP=False
    if('OUTCAR' in inName):
        isVASP=True
        stress=getStressTensorVASP(inName)
    elif(ext=='out' or ext=='log'):
        stress,u,v,w=getStressTensorCP2K(inName)
    else:
        print(ext,'unknown output file extension. This function is only available for VASP and CP2K')
        exit()

    fo=open(outName,'w')
    fo.write("# Step/Strain        XX              YY            ZZ               YZ            XZ               XY\n")
    fo.write("%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le\n" % (0.0,stress[0][0],stress[1][1],stress[2][2],stress[2][1],stress[2][0],stress[1][0]))
    if(len(args.stress_list[0])>0):
        fi=open(args.stress_list[0])
        i=0
        for line in fi:
            i=i+1
            w=line.split()
            if((len(line)==0) or (len(w)==0)):
                continue
            stress=np.zeros((3,3))
            if(isVASP):
                if os.path.isfile(w[0]):
                    stress=getStressTensorVASP(w[0])
            else:
                if os.path.isfile(w[0]):
                    stress,u,v,w=getStressTensorCP2K(w[0])
            fo.write("%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le\n" % (float(i),stress[0][0],stress[1][1],stress[2][2],stress[2][1],stress[2][0],stress[1][0]))
        fi.close()
    else:
        for s in args.strain_values:
            if(isVASP):
                fname=basename+'.'+axis.strip()+'_'+s.strip()
                if os.path.isfile(fname):
                    stress=getStressTensorVASP(fname)
            else:
                fname=basename+'.'+axis.strip()+'_'+s.strip()+'.'+ext
                if os.path.isfile(fname):
                    stress,u,v,w=getStressTensorCP2K(fname)
            fo.write("%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le%15.6le\n" % (float(s),stress[0][0],stress[1][1],stress[2][2],stress[2][1],stress[2][0],stress[1][0]))
    
    fo.close()
    return

def get_pz(fname):
    eci=np.zeros((3,6)) # clamped-ion contribution
    eri=np.zeros((3,6)) # ion relaxation contribution
    fi=open(fname,'r')
    isCI=False
    isRI=False
    for line in fi:
        if("PIEZOELECTRIC TENSOR (including local field effects)" in line):
            if("(C/m^2)" in line):
                isCI=True
                isRI=False
                continue
            continue
        if("PIEZOELECTRIC TENSOR IONIC CONTR" in line):
            if("(C/m^2)" in line):
                isRI=True
                isCI=False
                continue
            continue
        words=line.split()
        if(isCI and words[0].strip()=='x'):
            eci[0,0]=float(words[1])
            eci[0,1]=float(words[2])
            eci[0,2]=float(words[3])
            eci[0,3]=float(words[5])
            eci[0,4]=float(words[6])
            eci[0,5]=float(words[4])
            continue
        if(isCI and words[0].strip()=='y'):
            eci[1,0]=float(words[1])
            eci[1,1]=float(words[2])
            eci[1,2]=float(words[3])
            eci[1,3]=float(words[5])
            eci[1,4]=float(words[6])
            eci[1,5]=float(words[4])
            continue
        if(isCI and words[0].strip()=='z'):
            eci[2,0]=float(words[1])
            eci[2,1]=float(words[2])
            eci[2,2]=float(words[3])
            eci[2,3]=float(words[5])
            eci[2,4]=float(words[6])
            eci[2,5]=float(words[4])
            isCI=False
            continue
        if(isRI and words[0].strip()=='x'):
            eri[0,0]=float(words[1])
            eri[0,1]=float(words[2])
            eri[0,2]=float(words[3])
            eri[0,3]=float(words[5])
            eri[0,4]=float(words[6])
            eri[0,5]=float(words[4])
            continue
        if(isRI and words[0].strip()=='y'):
            eri[1,0]=float(words[1])
            eri[1,1]=float(words[2])
            eri[1,2]=float(words[3])
            eri[1,3]=float(words[5])
            eri[1,4]=float(words[6])
            eri[1,5]=float(words[4])
            continue
        if(isRI and words[0].strip()=='z'):
            eri[2,0]=float(words[1])
            eri[2,1]=float(words[2])
            eri[2,2]=float(words[3])
            eri[2,3]=float(words[5])
            eri[2,4]=float(words[6])
            eri[2,5]=float(words[4])
            isRI=False
            continue
    fi.close()
    return eci,eri

def get_elastic(inName):
    c=np.zeros((6,6))
    fi=open(inName,'r')
    isElastic=False
    for line in fi:
        if("TOTAL ELASTIC MODULI (kBar)" in line):
            isElastic=True
            continue
        words=line.split()
        if(isElastic and words[0].strip()=="XX"):
            c[0,0]=float(words[1])*0.1
            c[0,1]=float(words[2])*0.1
            c[0,2]=float(words[3])*0.1
            c[0,3]=float(words[5])*0.1
            c[0,4]=float(words[6])*0.1
            c[0,5]=float(words[4])*0.1
            continue
        if(isElastic and words[0].strip()=="YY"):
            c[1,0]=float(words[1])*0.1
            c[1,1]=float(words[2])*0.1
            c[1,2]=float(words[3])*0.1
            c[1,3]=float(words[5])*0.1
            c[1,4]=float(words[6])*0.1
            c[1,5]=float(words[4])*0.1
            continue
        if(isElastic and words[0].strip()=="ZZ"):
            c[2,0]=float(words[1])*0.1
            c[2,1]=float(words[2])*0.1
            c[2,2]=float(words[3])*0.1
            c[2,3]=float(words[5])*0.1
            c[2,4]=float(words[6])*0.1
            c[2,5]=float(words[4])*0.1
            continue
        if(isElastic and words[0].strip()=="YZ"):
            c[3,0]=float(words[1])*0.1
            c[3,1]=float(words[2])*0.1
            c[3,2]=float(words[3])*0.1
            c[3,3]=float(words[5])*0.1
            c[3,4]=float(words[6])*0.1
            c[3,5]=float(words[4])*0.1
            continue
        if(isElastic and words[0].strip()=="ZX"):
            c[4,0]=float(words[1])*0.1
            c[4,1]=float(words[2])*0.1
            c[4,2]=float(words[3])*0.1
            c[4,3]=float(words[5])*0.1
            c[4,4]=float(words[6])*0.1
            c[4,5]=float(words[4])*0.1
            isElastic=False
            continue
        if(isElastic and words[0].strip()=="XY"):
            c[5,0]=float(words[1])*0.1
            c[5,1]=float(words[2])*0.1
            c[5,2]=float(words[3])*0.1
            c[5,3]=float(words[5])*0.1
            c[5,4]=float(words[6])*0.1
            c[5,5]=float(words[4])*0.1
            continue
    fi.close()
    return c

def get_de(inName):
    dec=np.zeros((3,3)) # clamped-ion contribution
    der=np.zeros((3,3)) # ion relaxation contribution
    fi=open(inName,'r')
    isCI=False
    isRI=False
    i=0
    for line in fi:
        if("MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects" in line):
            i=0
            isCI=True
            isRI=False
            continue
        if("MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION" in line):
            i=0
            isRI=True
            isCI=False
            continue
        words=line.split()
        if(isCI and i<3):
            if('----' in line):
                continue
            dec[i,0]=float(words[0])
            dec[i,1]=float(words[1])
            dec[i,2]=float(words[2])
            i=i+1
            if(i==3):
                isCI=False
            continue
        if(isRI and i<3):
            if('----' in line):
                continue
            der[i,0]=float(words[0])
            der[i,1]=float(words[1])
            der[i,2]=float(words[2])
            i=i+1
            if(i==3):
                isRI=False
            continue
    fi.close()
    return dec,der

def io_read(inName):
    sysType='C'
    isScaled=False
    words=inName.split('.')
    ext=words[-1]
    spg='P1         '
    if(ext=='pdb'):
        atoms,a,b,c,spg=readPdb(inName)
        isScaled=False
    elif(ext=='gro'):
        atoms,a,b,c=readGro(inName,'A')
        isScaled=False
    elif(ext=='xyz'):
        atoms,a,b,c=readXyz(inName)
        isScaled=False
    elif(ext=='cry'):
        atoms,a,b,c=readCry(inName)
        isScaled=False
    elif(ext=='gen'):
        atoms,a,b,c,sysType=readGen(inName)
        isScaled=False
        if('F' in sysType):
            isScaled=True
    elif(ext=='cif'):
        atoms,a,b,c=readCif(inName)
        isScaled=False
    elif(ext=='POSCAR' or ext=="poscar" or ext=="vasp" or ('POSCAR' in inName) or ext=='CONTCAR' or ('CONTCAR' in inName)):
        atoms,a,b,c,isScaled=readPoscar(inName)
    elif(ext=='restart' or ext=='inp'):
        atoms,isScaled=getCoordCP2K(inName)
        a,b,c=getBoxCP2K(inName)
    wz(a,b,c)
    if(a.norm>0.0 and b.norm>0.0 and c.norm>0.0):
        sysType='F'
    return(atoms,a,b,c,isScaled,sysType,spg)

def io_write(inName,outName,atoms,a,b,c,isScaled,sysType,hall_number,args,dire):
    words=outName.split('.')
    ext=words[-1]
    if(args.asym):
        if(not isScaled):
            print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
            exit()
        hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
        dataset=get_dataset(atoms,hmat,hall_number)
        print(dataset['hall_number'],dataset['international'],dataset['pointgroup'])
        equivalent_atoms=dataset['equivalent_atoms']
        atmp=[]
        nat=0
        for i in range(len(equivalent_atoms)):
            if(equivalent_atoms[i]!=i):
                continue
            nat+=1
            atmp.append(Atom())
            cpAtom(atmp[-1],atoms[i])
            atmp[-1].idx=nat
        atoms=atmp
    if(ext=='POSCAR' or ext=="poscar" or ext=="vasp" or ('POSCAR' in inName)):
        if(isScaled):
            writePoscar(outName,atoms,a,b,c,args)
        else:
            isScaled=True
            cart2frac(atoms,a,b,c)
            writePoscar(outName,atoms,a,b,c,args)
    elif(ext=='pdb'):
        if(isScaled):
            isScaled=False
            frac2cart(atoms,a,b,c)
        writePdb(outName,atoms,a,b,c)
    elif(ext=='gro'):
        writeGro(outName,atoms,a,b,c,isScaled)
    elif(ext=='gen'):
        if(isScaled and (sysType=='F')):
            writeGen(outName,atoms,a,b,c,sysType)
        elif(sysType=='F'):
            isScaled=True
            cart2frac(atoms,a,b,c)
            writeGen(outName,atoms,a,b,c,sysType)
        else:
            writeGen(outName,atoms,a,b,c,sysType)
    elif(ext=='xyz'):
        if(isScaled):
            isScaled=False
            frac2cart(atoms,a,b,c)
        writeXyz(outName,atoms,a,b,c)
    elif(ext=='inp'):
        writeCp2k(inName,outName,atoms,a,b,c,isScaled,hall_number,args,dire)
    elif(ext=='cry' or ext=='d12'):
        if(not isScaled):
            isScaled=True
            cart2frac(atoms,a,b,c)
        writeCry(outName,atoms,a,b,c,hall_number,args)
    elif(ext=='fdf'):
        writeFdf(outName,atoms,a,b,c,sysType,isScaled)
    elif(ext=='cif'):
        data=ase.io.read(inName)
        ase.io.write(outName,data)
    else:
        print("%s format is not supported" % (ext.strip()) )
        print("Used PDB format instead.")
        if(isScaled):
            isScaled=False
            frac2cart(atoms,a,b,c)
        writePdb(outName,atoms,a,b,c)
    return

def build_crystal(atoms,a,b,c,isScaled,hall_number):
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    if(not isScaled):
        print("build_crystal: not scaled")
        isScaled=True
        cart2frac(atoms,a,b,c)
    dset=get_dataset(atoms,hmat,hall_number)
    ro=dset['rotations']
    tr=dset['translations']
    nRes=atoms[-1].resIdx
    syst=[]
    pn=0
    idx=0
    for i in range(len(tr)):
        r=ro[i]
        t=tr[i]
        pn+=1
        refs=''
        for at in atoms:
            idx+=1
            syst.append(Atom())
            cpAtom(syst[-1],at)
            u=np.array([at.x,at.y,at.z])
            u=np.matmul(r,u)
            u=u+t
            syst[-1].x=u[0]-math.floor(u[0])
            syst[-1].y=u[1]-math.floor(u[1])
            syst[-1].z=u[2]-math.floor(u[2])
            syst[-1].idx=idx
            syst[-1].chain='A'
            if(at.segName.strip()!=refs):
                refs=at.segName.strip()
                pn+=1
            syst[-1].segName='P'+str(pn)
    
    mask=np.zeros((len(syst)))
    for i in range(len(mask)):
        mask[i]=i

    for i in range(len(syst)-1):
        u=np.array([syst[i].x,syst[i].y,syst[i].z])
        for j in range(i+1,len(syst)):
            v=np.array([syst[j].x,syst[j].y,syst[j].z])
            w=v-u
            w[0]-=round(w[0])
            w[1]-=round(w[1])
            w[2]-=round(w[2])
            d=la.norm(w)
            if(d<1e-3 and (mask[j]==j)):
                mask[j]=i

    syst2=[]
    for i in range(len(syst)):
        if(mask[i]==i):
            syst2.append(Atom())
            cpAtom(syst2[-1],syst[i])

    return(syst2,isScaled)

def find_symmetry(atoms,a,b,c,isScaled,hall_number):
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()
    dset=get_dataset(atoms,hmat,hall_number)
    r=dset['rotations']
    t=dset['translations']
    nop=len(r)
    for i in range(nop):
        print(i+1,nop)
        for j in range(3):
            print("%2d %2d %2d %5.2f" % (r[i,j,0],r[i,j,1],r[i,j,2],t[i,j]))
    return

def get_dataset(atoms,hmat,hall_number):
    print(hall_number)
    if(hall_number>0):
        print('hall_number:',hall_number)
        dataset = sg.get_symmetry_from_database(hall_number)
        r=dataset['rotations']
        t=dataset['translations']
        nop=len(r)
        for i in range(nop):
            print(i+1,nop)
            for j in range(3):
                print("%2d %2d %2d %5.2f" % (r[i,j,0],r[i,j,1],r[i,j,2],t[i,j]))
    else:
        xyz=[]
        ele=[]
        kinds=[]
        for at in atoms:
            xyz.append([at.x,at.y,at.z])
            typeatom(at)
            if(at.el.strip() not in ele):
                ele.append(at.el.strip())
            ii=ele.index(at.el.strip())
            kinds.append(ii+1)
        cell=(hmat,xyz,kinds)
        dataset = sg.get_symmetry_dataset(cell, symprec=1e-2, hall_number=0)
        print("Hall number:",dataset['hall_number'])
        print("International number:",dataset['number'])
        print("International symbol:",dataset['international'])
        print("Point group:",dataset['pointgroup'])
        if(dataset==None):
            print("Spglib failed with the following error message:",sg.get_error_message())
            exit()
    return(dataset)

def get_rotations(atoms,hmat,hall_number):
    if(hall_number>0):
        print(hall_number)
        dataset = sg.get_symmetry_from_database(hall_number)
    else:
        xyz=[]
        ele=[]
        kinds=[]
        for at in atoms:
            xyz.append([at.x,at.y,at.z])
            typeatom(at)
            if(at.el.strip() not in ele):
                ele.append(at.el.strip())
            ii=ele.index(at.el.strip())
            kinds.append(ii+1)
        cell=(hmat,xyz,kinds)
        dataset = sg.get_symmetry_dataset(cell, symprec=1e-2, hall_number=0)
        if(dataset==None):
            print("Spglib failed with the following error message:",sg.get_error_message())
            exit()
        print(dataset['hall_number'],dataset['number'],dataset['international'],dataset['pointgroup'])
    return(dataset['rotations'])

def get_rotations_subset(atoms,hmat,hall_number):
    r=get_rotations(atoms,hmat,hall_number)
    nop=len(r)
    mask=np.ones((nop),int)
    for i in range(0,nop-1):
        if(not mask[i]):
            continue
        for j in range(i+1,nop):
            if(not mask[j]):
                continue
            d=r[j]-r[i]
            if(np.all(d==0)):
                mask[j]=0
    s=[]
    for i in range(len(r)):
        if(mask[i]):
            s.append(r[i])
    s=np.array(s)
    return(s)

def symm_coords(coords):
    return

def symm_forces(forces):
    return

def change_basis(roti,nop,h1,h2):
    ih1 = la.inv(h1)
    ih2 = la.inv(h2)
    tih1=np.transpose(ih1)
    tih2=np.transpose(ih2)
    h2ih1 = np.matmul(h2,tih1)
    h1ih2 = np.matmul(h1,tih2)
    rlist=[]
    for r in roti:
        s = np.matmul(h2ih1,r)
        r = np.matmul(s,h1ih2)
        rlist.append(r)
    roto=np.array(rlist)
    return roto

def sym_tensor(t,rotations,nop,hmat):
    t=np.array(t)
    rotations=np.array(rotations)
    hmat=np.array(hmat)
    rst=np.zeros((nop,3,3))
    ide=np.zeros((3,3))
    ide[0][0] = 1.e0
    ide[1][1] = 1.e0
    ide[2][2] = 1.e0

    rst=change_basis(rotations, nop, hmat, ide)
     
    tin=t
    t=np.zeros((3,3))
    for r in rst:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        t[i][j] = t[i][j] + ( r[k][i] * r[l][j] * tin[k][l] )
    t=t/float(nop)
    return t

def sym_tensor3_cleaner(t):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if(abs(t[i,j,k])<1.e-12):
                    t[i,j,k]=0.e00
    return t

def sym_tensor3(t,rotations,nop,hmat):
    t=np.array(t)
    rotations=np.array(rotations)
    hmat=np.array(hmat)
    rst=np.zeros((nop,3,3))
    ide=np.zeros((3,3))
    ide[0][0] = 1.e0
    ide[1][1] = 1.e0
    ide[2][2] = 1.e0

    rst=change_basis(rotations, nop, hmat, ide)
     
    tin=t
    t=np.zeros((3,3,3))
    for r in rst:
        to1=np.zeros((3,3,3))
        for i in range(3):
            to1[i,:,:]=r[0][i]*tin[0,:,:]+r[1][i]*tin[1,:,:]+r[2][i]*tin[2,:,:]
        to2=np.zeros((3,3,3))
        for i in range(3):
            to2[:,i,:]=r[0][i]*to1[:,0,:]+r[1][i]*to1[:,1,:]+r[2][i]*to1[:,2,:]
        to1=np.zeros((3,3,3))
        for i in range(3):
            to1[:,:,i]=r[0][i]*to2[:,:,0]+r[1][i]*to2[:,:,1]+r[2][i]*to2[:,:,2]
        t=t+to1
    t=t/float(nop)
    return t

def sym_tensor4_cleaner(t):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if(abs(t[i,j,k,l])<1.e-12):
                        t[i,j,k,l]=0.e00
    return t

def sym_tensor2_cleaner(t):
    for i in range(len(t)):
        for j in range(len(t[i])):
            if(abs(t[i,j])<1.e-12):
                t[i,j]=0.e00
    return t

def sym_tensor4_helper(t):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    tmp=0.5*(t[i,j,k,l]+t[k,l,i,j])
                    t[i,j,k,l]=tmp
                    t[k,l,i,j]=tmp
    return t

def sym_tensor4(t,rotations,nop,hmat):
    t=np.array(t)
    rotations=np.array(rotations)
    hmat=np.array(hmat)
    rst=np.zeros((nop,3,3))
    ide=np.zeros((3,3))
    ide[0][0] = 1.e0
    ide[1][1] = 1.e0
    ide[2][2] = 1.e0

    rst=change_basis(rotations, nop, hmat, ide)
     
    tin=t
    t=np.zeros((3,3,3,3))
    for r in rst:
        to1=np.zeros((3,3,3,3))
        for i in range(3):
            to1[i,:,:,:]=r[0][i]*tin[0,:,:,:]+r[1][i]*tin[1,:,:,:]+r[2][i]*tin[2,:,:,:]
        to2=np.zeros((3,3,3,3))
        for i in range(3):
            to2[:,i,:,:]=r[0][i]*to1[:,0,:,:]+r[1][i]*to1[:,1,:,:]+r[2][i]*to1[:,2,:,:]
        to1=np.zeros((3,3,3,3))
        for i in range(3):
            to1[:,:,i,:]=r[0][i]*to2[:,:,0,:]+r[1][i]*to2[:,:,1,:]+r[2][i]*to2[:,:,2,:]
        to2=np.zeros((3,3,3,3))
        for i in range(3):
            to2[:,:,:,i]=r[0][i]*to1[:,:,:,0]+r[1][i]*to1[:,:,:,1]+r[2][i]*to1[:,:,:,2]
        t=t+to2
    t=t/float(nop)
    return t

def dm_pbc(bf,tr):
    int2debye=1.602176634e-29/3.33564e-30
    ee=np.zeros((3,3,3))
    idx=[[0,0],[1,1],[2,2],[2,1],[2,0],[1,0]]
    for i in range(6):
        ii=idx[i][0]
        jj=idx[i][1]
        d=np.array([0.,0.,0.])
        d[:]=bf[i,1,:]-bf[i,0,:]
        for j in range(3):
            d[j]=d[j]-round(d[j])
        d=np.matmul(tr,d)
        ee[:,ii,jj]=d[:]
        ee[:,jj,ii]=d[:]
    return(ee)

def dm_pbc_dielec(bf,tr):
    xx=np.zeros((3,3))
    for i in range(3):
        d=np.array([0.,0.,0.])
        d[:]=bf[i,1,:]-bf[i,0,:]
        for j in range(3):
            d[j]=d[j]-round(d[j])
        d=np.matmul(tr,d)
        xx[:,i]=d[:]
        xx[:,i]=d[:]
    return(xx)

def get_tensors_vasp(args):
    e=np.zeros((3,6))
    eci=np.zeros((3,6))
    eri=np.zeros((3,6))
    cv=np.zeros((6,6))
    ci=np.zeros((6,6))
    d=np.zeros((3,6))
    g=np.zeros((3,6))
    de=np.zeros((3,3))
    dec=np.zeros((3,3))
    der=np.zeros((3,3))
    aa=0
    bb=0
    cc=0
    dd=0
    ee=0
    ff=0
    kv=0
    gv=0
    kr=0
    gr=0
    kh=0
    gh=0
    yv=0
    yr=0
    yh=0
    pv=0
    pr=0
    ph=0

    if(args.vasp_piezo_get and args.vasp_elastic_get):
        inputPiezo=args.input[0]
        inputElastic=args.input[1]
    elif(args.vasp_piezo_get):
        inputPiezo=args.input[0]
    elif(args.vasp_elastic_get):
        inputElastic=args.input[0]

    if(args.vasp_piezo_get):
        eci,eri=get_pz(inputPiezo)
        e=eci+eri
        dec,der=get_de(inputPiezo)
        de=dec+der

    if(args.vasp_elastic_get):
        cv=get_elastic(inputElastic)
        ci=la.inv(cv)
        aa=(cv[0][0] + cv[1][1] + cv[2][2])/3.
        bb=(cv[1][2] + cv[0][2] + cv[0][1])/3.
        cc=(cv[3][3] + cv[4][4] + cv[5][5])/3.
        dd=(ci[0][0] + ci[1][1] + ci[2][2])/3.
        ee=(ci[1][2] + ci[0][2] + ci[0][1])/3.
        ff=(ci[3][3] + ci[4][4] + ci[5][5])/3.
        kv=(aa + 2.*bb)/3.
        gv=(aa - bb + 3.*cc)/5.
        kr=1./(3.*dd + 6.*ee)
        gr=5./(4.*dd - 4.*ee + 3.*ff)
        kh=(kv + kr)/2.
        gh=(gv + gr)/2.
        yv=1./(1./(3.*gv) + 1./(9.*kv))
        yr=1./(1./(3.*gr) + 1./(9.*kr))
        yh=1./(1./(3.*gh) + 1./(9.*kh))
        pv=(1. - 3.*gv/(3.*kv+gv))/2.
        pr=(1. - 3.*gr/(3.*kr+gr))/2.
        ph=(1. - 3.*gh/(3.*kh+gh))/2.

    d=1.e3*np.matmul(e,ci)

    ide=np.identity(3)

    ele=1.602176634e-19
    hartree=4.3597447222071e-18
    a0=5.29177210903e-11
    eps=(ele*ele)/(4.*math.pi*a0*hartree)
    reps=1./eps

    es=(de*eps)+(1.e-12*np.matmul(e,np.transpose(d)))
    be=la.inv(es)
    
    g=1.e-9*np.matmul(be,d)

    print_tensors_vasp(args.output[0],args,e,eci,eri,cv,ci,d,de,dec,der,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph)

    return

def get_tensors(inName,outName,atoms,a0,b0,c0,isScaled,args):
    int2debye=1.602176634e-29/3.33564e-30
    debye2int=3.33564e-30/1.602176634e-29
    debye2au=3.33564

    ev=np.zeros((3,6))
    cv=np.zeros((6,6))
    ci=np.zeros((6,6))
    d=np.zeros((3,6))
    g=np.zeros((3,6))
    ep=np.zeros((3,3))
    aa=0
    bb=0
    cc=0
    dd=0
    ee=0
    ff=0
    kv=0
    gv=0
    kr=0
    gr=0
    kh=0
    gh=0
    yv=0
    yr=0
    yh=0
    pv=0
    pr=0
    ph=0

    basename=inName.split('.')[0]

    ra0,rb0,rc0,vol0=wz(a0,b0,c0)
    hmat=[[a0.x,a0.y,a0.z],[b0.x,b0.y,b0.z],[c0.x,c0.y,c0.z]]
    
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()
    
    th=np.transpose(hmat)
    tih=np.transpose(la.inv(hmat))
    if(args.symmetry_excluded is not None):
        exclusionList=[]
        for al in args.symmetry_excluded:
            for i in range(al[0],al[1]+1):
                exclusionList.append(i-1)
        syst=[]
        for i in range(len(atoms)):
            if i in exclusionList:
                continue
            syst.append(Atom())
            cpAtom(syst[-1],atoms[i])
        dataset=get_dataset(syst,hmat,args.hall_number[0])
        rotations=get_rotations_subset(syst,hmat,args.hall_number[0])
        del(syst)
    else:
        dataset=get_dataset(atoms,hmat,args.hall_number[0])
        rotations=get_rotations_subset(atoms,hmat,args.hall_number[0])

    nop=len(rotations)
    for i in range(nop):
        rotations[i]=np.transpose(rotations[i])

    if(args.cp2k_elastic_piezo_get):
        bf=np.zeros((6,2,3))
        ds=np.zeros((2,3,3,3,3))
        idx=[[0,0],[1,1],[2,2],[2,1],[2,0],[1,0]]
        for i in range(6):
            ii=idx[i][0]
            jj=idx[i][1]
            k=0
            for j in range(-1,2):
                if(j==0):
                    continue
                fname=basename+'.strain_'+str(i+1)+'_'+str(j)+'.out'
                print(fname)
                mu,a,b,c=getDipoleCP2K(fname)
                print(mu)
                stress,u,v,w=getStressTensorCP2K(fname)
                ra,rb,rc,vol=wz(a,b,c)
                h2=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
                tih2=np.transpose(la.inv(h2))
                bp=np.matmul(tih2,mu)
                bf[i,k,:]=bp*debye2int
                ds[k,ii,jj,:,:]=stress[:,:]
                ds[k,jj,ii,:,:]=stress[:,:]
                k=k+1
    
        h=args.cp2k_elastic_piezo_step[0]/debye2au
        print(args.cp2k_elastic_piezo_step[0],debye2au)
        e=dm_pbc(bf,th)
        e=e*int2debye/(2*h*vol0)
        e=sym_tensor3(e,rotations,nop,hmat)
        print(e)
    
        for i in range(3):
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                ev[i][k]=e[i][kk][ll]
    
        h=-args.cp2k_elastic_piezo_step[0]
        c=np.zeros((3,3,3,3))
        c[:,:,:,:]=( (ds[1,:,:,:,:]-ds[0,:,:,:,:]) ) / (2.0*h)
        c=sym_tensor4_helper(c)
        c=sym_tensor4(c,rotations,nop,hmat)
    
        idxy=[[0,0],[1,1],[2,2],[0,1],[1,2],[2,1]]
        for i in range(6):
            ii=idx[i][0]
            jj=idx[i][1]
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                cv[i][k]=c[ii][jj][kk][ll]
        
        ci=la.inv(cv)
    
        d=1.e3*np.matmul(ev,ci)
    
        aa=(cv[0][0] + cv[1][1] + cv[2][2])/3.
        bb=(cv[1][2] + cv[0][2] + cv[0][1])/3.
        cc=(cv[3][3] + cv[4][4] + cv[5][5])/3.
        dd=(ci[0][0] + ci[1][1] + ci[2][2])/3.
        ee=(ci[1][2] + ci[0][2] + ci[0][1])/3.
        ff=(ci[3][3] + ci[4][4] + ci[5][5])/3.
    
        kv=(aa + 2.*bb)/3.
        gv=(aa - bb + 3.*cc)/5.
    
        kr=1./(3.*dd + 6.*ee)
        gr=5./(4.*dd - 4.*ee + 3.*ff)
    
        kh=(kv + kr)/2.
        gh=(gv + gr)/2.
    
        yv=1./(1./(3.*gv) + 1./(9.*kv))
        yr=1./(1./(3.*gr) + 1./(9.*kr))
        yh=1./(1./(3.*gh) + 1./(9.*kh))
    
        pv=(1. - 3.*gv/(3.*kv+gv))/2.
        pr=(1. - 3.*gr/(3.*kr+gr))/2.
        ph=(1. - 3.*gh/(3.*kh+gh))/2.

    if(args.cp2k_dielectric_get):
        bf=np.zeros((6,2,3))
        for i in range(3):
            k=0
            for j in range(-1,2):
                if(j==0):
                    continue
                ir2=np.zeros((3,3))
                fname=basename+'.efield_'+str(i+1)+'_'+str(j)+'.out'
                mu,a,b,c=getDipoleCP2K(fname)
                ra,rb,rc,vol=wz(a,b,c)
                h2=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
                tih2=np.transpose(la.inv(h2))
                bp=np.matmul(tih2,mu)
                bf[i,k,:]=bp*debye2int
                k=k+1
    
        ef=5.14220674763e11
        ele=1.602176634e-19
        hartree=4.3597447222071e-18
        a0=5.29177210903e-11
        reps=4.*math.pi*a0*hartree/(ele*ele)
        h=(args.cp2k_dielectric_field[0]*ef)/(debye2au)
    
        ide=np.identity(3)
    
        x=np.zeros((3,3))
        x=dm_pbc_dielec(bf,th)
        x=x*int2debye/(2*h*vol0)
        x=np.transpose(x)
        for i in range(2):
            for j in range(i+1,3):
                offd=0.5*(x[i,j]+x[j,i])
                x[i,j]=offd
                x[j,i]=offd
    
        ep=((x*reps)+ide)
        ep=sym_tensor(ep,rotations,nop,hmat)
    
        xs=x+(1.e-12*np.matmul(ev,np.transpose(d)))
    
        es=((xs*reps)+ide)/reps
        be=la.inv(es)

    if(args.cp2k_elastic_piezo_get and args.cp2k_dielectric_get):
        g=1.e-9*np.matmul(be,d)

    print_tensors(outName,args,ev,cv,ci,d,ep,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph)

    return

def get_tensors_gmx(inName,outName,atoms,a0,b0,c0,isScaled,args):
    print('Start get_tensors_gmx:')
    int2debye=1.602176634e-29/3.33564e-30
    debye2int=3.33564e-30/1.602176634e-29
    debye2au=3.33564

    ev=np.zeros((3,6))
    cv=np.zeros((6,6))
    ci=np.zeros((6,6))
    d=np.zeros((3,6))
    g=np.zeros((3,6))
    ep=np.zeros((3,3))
    aa=0
    bb=0
    cc=0
    dd=0
    ee=0
    ff=0
    kv=0
    gv=0
    kr=0
    gr=0
    kh=0
    gh=0
    yv=0
    yr=0
    yh=0
    pv=0
    pr=0
    ph=0

    basename=inName.split('.')[0]

    ra0,rb0,rc0,vol0=wz(a0,b0,c0)
    hmat=[[a0.x,a0.y,a0.z],[b0.x,b0.y,b0.z],[c0.x,c0.y,c0.z]]
    
    if(not isScaled):
        print("Error: the coordinates should be fractional at this stage. Report issue to the developer.")
        exit()
    if(len(args.input)<2):
        print("Provide the file containing the pressure tensor in second position in the list of input files.")
        exit()
    if(len(args.psf_file[0])<0):
        print('A CHARMM PSF or GROMACS topology file is required to compute the piezoelectric response.')
        exit()
    
    ext=args.psf_file[0].split('.')[-1]
    if('psf' in ext):
        top=readPsf(args.psf_file[0])
    elif('top' in ext):
        top=readTop(args.psf_file[0])
        writeTop('test.top',top)
        #exit()
    
    th=np.transpose(hmat)
    tih=np.transpose(la.inv(hmat))
    if(args.symmetry_excluded is not None):
        exclusionList=[]
        for al in args.symmetry_excluded:
            for i in range(al[0],al[1]+1):
                exclusionList.append(i-1)
        syst=[]
        for i in range(len(atoms)):
            if i in exclusionList:
                continue
            syst.append(Atom())
            cpAtom(syst[-1],atoms[i])
        dataset=get_dataset(syst,hmat,args.hall_number[0])
        rotations=get_rotations_subset(syst,hmat,args.hall_number[0])
        #dataset=get_dataset(atoms[0:args.symmetry_excluded[0][0]],hmat,args.hall_number[0])
        #rotations=get_rotations_subset(atoms[0:args.symmetry_excluded[0][0]],hmat,args.hall_number[0])
        del(syst)
    else:
        dataset=get_dataset(atoms,hmat,args.hall_number[0])
        rotations=get_rotations_subset(atoms,hmat,args.hall_number[0])

    nop=len(rotations)
    for i in range(nop):
        rotations[i]=np.transpose(rotations[i])

    if(args.gmx_elastic_piezo_get):
        fs=open(args.input[1],'r')
        bf=np.zeros((6,2,3))
        ds=np.zeros((2,3,3,3,3))
        idx=[[0,0],[1,1],[2,2],[2,1],[2,0],[1,0]]
        for i in range(6):
            ii=idx[i][0]
            jj=idx[i][1]
            k=0
            for j in range(-1,2):
                if(j==0):
                    continue
                line=fs.readline()
                words=line.split()
                stl=[float(st) for st in words]
                stress=np.array(stl).reshape((3,3))
                stress=5e-5*(stress+np.transpose(stress))
                ds[k,ii,jj,:,:]=stress[:,:]
                ds[k,jj,ii,:,:]=stress[:,:]
                fname=basename+'.strain_'+str(i+1)+'_'+str(j)+'.gro'
                print(fname)
                syst,a,b,c,isScaled,sysType,spg=io_read(fname)
                h2=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
                tih2=np.transpose(la.inv(h2))
                bp=crystal_dipole(syst,hmat,top)
                #bp=np.matmul(tih2,mu)
                bf[i,k,:]=bp #*debye2int
                k=k+1
    
        h=args.cp2k_elastic_piezo_step[0]/debye2au
        print(args.cp2k_elastic_piezo_step[0],debye2au)
        e=dm_pbc(bf,th)
        e=e*int2debye/(2*h*vol0)
        e=sym_tensor3(e,rotations,nop,hmat)
        print(e)
    
        for i in range(3):
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                ev[i][k]=e[i][kk][ll]
        print(ev)
        h=-args.cp2k_elastic_piezo_step[0]
        c=np.zeros((3,3,3,3))
        c[:,:,:,:]=( (ds[1,:,:,:,:]-ds[0,:,:,:,:]) ) / (2.0*h)
        c=sym_tensor4_helper(c)
        c=sym_tensor4(c,rotations,nop,hmat)
    
        idxy=[[0,0],[1,1],[2,2],[0,1],[1,2],[2,1]]
        for i in range(6):
            ii=idx[i][0]
            jj=idx[i][1]
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                cv[i][k]=c[ii][jj][kk][ll]
        print(cv)
        ci=la.inv(cv)
    
        d=1.e3*np.matmul(ev,ci)
        print(d)
    
        aa=(cv[0][0] + cv[1][1] + cv[2][2])/3.
        bb=(cv[1][2] + cv[0][2] + cv[0][1])/3.
        cc=(cv[3][3] + cv[4][4] + cv[5][5])/3.
        dd=(ci[0][0] + ci[1][1] + ci[2][2])/3.
        ee=(ci[1][2] + ci[0][2] + ci[0][1])/3.
        ff=(ci[3][3] + ci[4][4] + ci[5][5])/3.
    
        kv=(aa + 2.*bb)/3.
        gv=(aa - bb + 3.*cc)/5.
    
        kr=1./(3.*dd + 6.*ee)
        gr=5./(4.*dd - 4.*ee + 3.*ff)
    
        kh=(kv + kr)/2.
        gh=(gv + gr)/2.
    
        yv=1./(1./(3.*gv) + 1./(9.*kv))
        yr=1./(1./(3.*gr) + 1./(9.*kr))
        yh=1./(1./(3.*gh) + 1./(9.*kh))
    
        pv=(1. - 3.*gv/(3.*kv+gv))/2.
        pr=(1. - 3.*gr/(3.*kr+gr))/2.
        ph=(1. - 3.*gh/(3.*kh+gh))/2.

    print_tensors(outName,args,ev,cv,ci,d,ep,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph)

    return

def crystal_dipole(atoms,hm,top):
    debye=1.0e+21*1.0E-10*299792458.0*1.602176487E-19
    thm=np.transpose(hm)
    ihm=la.inv(hm)
    tihm=np.transpose(ihm)
    ga=np.ones((3),complex)
    for j in range(len(atoms)):
        r=np.array([atoms[j].x,atoms[j].y,atoms[j].z])
        t=2.0*math.pi*np.matmul(tihm,r)
        for i in range(3):
            z=(complex(np.cos(t[i]),np.sin(t[i])))**(-top.atoms[j].q)
            ga[i]=ga[i]*z
    
    tmp=ga.imag/ga.real
    ci=np.arctan(tmp)
    #dm=np.matmul(thm,ci)*debye/(2.0*math.pi)
    dm=ci/(2.0*math.pi)

    return dm

def print_tensors(outName,args,e,c,ci,d,ep,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph):
    a1='     1    '
    a2='     2    '
    a3='     3    '
    a4='     4    '
    a5='     5    '
    a6='     6    '
    fo=open(outName,'w')
    if(args.cp2k_elastic_piezo_get or args.gmx_elastic_piezo_get):
        e=sym_tensor2_cleaner(e)
        fo.write('Piezoelectric Charge Constants [e] (C/m^2)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[0][0],e[0][1],e[0][2],e[0][3],e[0][4],e[0][5]) )
        fo.write("2 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[1][0],e[1][1],e[1][2],e[1][3],e[1][4],e[1][5]) )
        fo.write("3 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[2][0],e[2][1],e[2][2],e[2][3],e[2][4],e[2][5]) )
        fo.write('\n')
    
        c=sym_tensor2_cleaner(c)
        fo.write('Elastic Constants (GPa)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        for i in range(6):
            fo.write('%2d' % (i+1))
            for k in range(6):
                fo.write('%16.8lf' % (c[i][k]))
            fo.write('\n')
        fo.write('\n')
    
        ci=sym_tensor2_cleaner(ci)
        fo.write('Compliance Constants inv(elastic) (1/GPa)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        for i in range(6):
            fo.write('%2d' % (i+1))
            for k in range(6):
                fo.write('%10.3lf' % (ci[i][k]))
            fo.write('\n')
        fo.write('\n')
    
        d=sym_tensor2_cleaner(d)
        fo.write('Piezoelectric Strain Constants [d] (pC/N)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (d[0][0],d[0][1],d[0][2],d[0][3],d[0][4],d[0][5]) )
        fo.write("2 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (d[1][0],d[1][1],d[1][2],d[1][3],d[1][4],d[1][5]) )
        fo.write("3 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (d[2][0],d[2][1],d[2][2],d[2][3],d[2][4],d[2][5]) )
        fo.write('\n')
    
        fo.write(" GPa   |")
        fo.write("Bulk modulus |")
        fo.write("Young modulus|")
        fo.write("Shear modulus|")
        fo.write("Poisson ratio|\n")
        fo.write(" Voigt |%10.3lf   |%10.3lf   |%10.3lf   |%10.3lf   |\n" % (kv,yv,gv,pv))
        fo.write(" Reuss |%10.3lf   |%10.3lf   |%10.3lf   |%10.3lf   |\n" % (kr,yr,gr,pr))
        fo.write(" Hill  |%10.3lf   |%10.3lf   |%10.3lf   |%10.3lf   |\n" % (kh,yh,gh,ph))
        fo.write('\n')

    if(args.cp2k_dielectric_get):
        fo.write('Dielectric Constants \n')
        fo.write('  %10s%10s%10s\n' % (a1,a2,a3))
        fo.write("1 %10.3lf%10.3lf%10.3lf\n" % (ep[0][0],ep[0][1],ep[0][2]) )
        fo.write("2 %10.3lf%10.3lf%10.3lf\n" % (ep[1][0],ep[1][1],ep[1][2]) )
        fo.write("3 %10.3lf%10.3lf%10.3lf\n" % (ep[2][0],ep[2][1],ep[2][2]) )
        fo.write('\n')

    if(args.cp2k_elastic_piezo_get and args.cp2k_dielectric_get):
        g=sym_tensor2_cleaner(g)
        fo.write('Piezoelectric Voltage Constants [g] (mV m/N)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (g[0][0],g[0][1],g[0][2],g[0][3],g[0][4],g[0][5]) )
        fo.write("2 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (g[1][0],g[1][1],g[1][2],g[1][3],g[1][4],g[1][5]) )
        fo.write("3 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (g[2][0],g[2][1],g[2][2],g[2][3],g[2][4],g[2][5]) )
        fo.write('\n')
    fo.close()
    return

def print_tensors_vasp(outName,args,e,eci,eri,c,ci,d,de,dec,der,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph):
    a1='        1       '
    a2='        2       '
    a3='        3       '
    a4='        4       '
    a5='        5       '
    a6='        6       '
    fo=open(outName,'w')
    if(args.vasp_piezo_get):
        fo.write('Piezoelectric Charge Constants (clamped-ion) [C/m^2]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eci[0][0],eci[0][1],eci[0][2],eci[0][3],eci[0][4],eci[0][5]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eci[1][0],eci[1][1],eci[1][2],eci[1][3],eci[1][4],eci[1][5]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eci[2][0],eci[2][1],eci[2][2],eci[2][3],eci[2][4],eci[2][5]) )
        fo.write('\n')
        
        fo.write('Piezoelectric Charge Constants (ion contribution) [C/m^2]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eri[0][0],eri[0][1],eri[0][2],eri[0][3],eri[0][4],eri[0][5]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eri[1][0],eri[1][1],eri[1][2],eri[1][3],eri[1][4],eri[1][5]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (eri[2][0],eri[2][1],eri[2][2],eri[2][3],eri[2][4],eri[2][5]) )
        fo.write('\n')
        
        fo.write('Piezoelectric Charge Constants [C/m^2]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (e[0][0],e[0][1],e[0][2],e[0][3],e[0][4],e[0][5]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (e[1][0],e[1][1],e[1][2],e[1][3],e[1][4],e[1][5]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (e[2][0],e[2][1],e[2][2],e[2][3],e[2][4],e[2][5]) )
        fo.write('\n')
    
    if(args.vasp_elastic_get):
        fo.write('Elastic Constants [GPa]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        for i in range(6):
            fo.write('%2d' % (i+1))
            for k in range(6):
                fo.write('%16.8lf' % (c[i][k]))
            fo.write('\n')
        fo.write('\n')
    
        fo.write('Compliance Constants inv(elastic) [1/GPa]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        for i in range(6):
            fo.write('%2d' % (i+1))
            for k in range(6):
                fo.write('%16.8lf' % (ci[i][k]))
            fo.write('\n')
        fo.write('\n')
        
        fo.write("  GPa  |")
        fo.write("  Bulk modulus   |")
        fo.write("  Young modulus  |")
        fo.write("  Shear modulus  |")
        fo.write("  Poisson ratio  |\n")
        fo.write(" Voigt |%12.4lf     |%12.4lf     |%12.4lf     |%12.4lf     |\n" % (kv,yv,gv,pv))
        fo.write(" Reuss |%12.4lf     |%12.4lf     |%12.4lf     |%12.4lf     |\n" % (kr,yr,gr,pr))
        fo.write(" Hill  |%12.4lf     |%12.4lf     |%12.4lf     |%12.4lf     |\n" % (kh,yh,gh,ph))
        fo.write('\n')

    if(args.vasp_piezo_get and args.vasp_elastic_get):
        fo.write('Piezoelectric Strain Constants [pC/N]\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (d[0][0],d[0][1],d[0][2],d[0][3],d[0][4],d[0][5]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (d[1][0],d[1][1],d[1][2],d[1][3],d[1][4],d[1][5]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (d[2][0],d[2][1],d[2][2],d[2][3],d[2][4],d[2][5]) )
        fo.write('\n')

    if(args.vasp_piezo_get):
        fo.write('Dielectric Constants (clamped-ion)\n')
        fo.write('  %16s%16s%16s\n' % (a1,a2,a3))
        fo.write("1 %16.8lf%16.8lf%16.8lf\n" % (dec[0][0],dec[0][1],dec[0][2]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf\n" % (dec[1][0],dec[1][1],dec[1][2]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf\n" % (dec[2][0],dec[2][1],dec[2][2]) )
        fo.write('\n')

        fo.write('Dielectric Constants (ion contribution)\n')
        fo.write('  %16s%16s%16s\n' % (a1,a2,a3))
        fo.write("1 %16.8lf%16.8lf%16.8lf\n" % (der[0][0],der[0][1],der[0][2]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf\n" % (der[1][0],der[1][1],der[1][2]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf\n" % (der[2][0],der[2][1],der[2][2]) )
        fo.write('\n')
    
        fo.write('Dielectric Constants\n')
        fo.write('  %16s%16s%16s\n' % (a1,a2,a3))
        fo.write("1 %16.8lf%16.8lf%16.8lf\n" % (de[0][0],de[0][1],de[0][2]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf\n" % (de[1][0],de[1][1],de[1][2]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf\n" % (de[2][0],de[2][1],de[2][2]) )
        fo.write('\n')

    if(args.vasp_piezo_get and args.vasp_elastic_get):
        fo.write('Piezoelectric Voltage Constants [g] (mV m/N)\n')
        fo.write('  %16s%16s%16s%16s%16s%16s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (g[0][0],g[0][1],g[0][2],g[0][3],g[0][4],g[0][5]) )
        fo.write("2 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (g[1][0],g[1][1],g[1][2],g[1][3],g[1][4],g[1][5]) )
        fo.write("3 %16.8lf%16.8lf%16.8lf%16.8lf%16.8lf%16.8lf\n" % (g[2][0],g[2][1],g[2][2],g[2][3],g[2][4],g[2][5]) )
        fo.write('\n')
    
    fo.close()
    return

def box_padding(atoms,a,b,c,isScaled,args):
    if(isScaled):
        frac2cart(atoms,a,b,c)
        isScaled=False
    for at in atoms:
        at.x+=args.padding[0]*0.5
        at.y+=args.padding[1]*0.5
        at.z+=args.padding[2]*0.5
    fa=1.0+(args.padding[0]/a.norm)
    fb=1.0+(args.padding[1]/b.norm)
    fc=1.0+(args.padding[0]/c.norm)
    a.x*=fa
    a.y*=fa
    a.z*=fa
    b.x*=fb
    b.y*=fb
    b.z*=fb
    c.x*=fc
    c.y*=fc
    c.z*=fc
    wz(a,b,c)
    return(atoms,a,b,c,isScaled)

@nb.jit(nopython=True, parallel=True, fastmath=True)
def rlist(xyz,rMin):
    for i in range(len(xyz)-1):
        #ri=np.array([xyz[i][0],xyz[i][1],xyz[i][2]])
        for j in range(i+1,len(xyz)):
            r=xyz[j]-xyz[0]
            r[0]-=round(r[0])
            r[1]-=round(r[1])
            r[2]-=round(r[2])
            if(la.norm(r)<rMin):
                print(i+1,j+1,la.norm(r),rMin)
            r=None
    return

def cg(coords,forces,energy):
    emin=0.0
    xold=np.zeros(len(coords))
    gradient=np.zeros(len(coords))

    return

# Program begins here:

parser=ap.ArgumentParser(prog='crystools',description='Convert between various structure and input file formats. Generate perturbations for piezoelectric and elastic properties. When the requested output file is in CP2K format, if a template is provided together with an input file in CP2K format, the keywords found in the tenplate are used with the structure cointained in the input. Any additional argumnet like -d3 will overwrite what is present in the template',epilog='Please repport any bugs to P.-A. Cazade at pierre.cazade@ul.ie.')
parser.add_argument('-i','--input',nargs='+',required=True,help='Input file(s). Format: CIF(.cif), PDB (.pdb), GROMACS (.gro), Cartesian (.xyz), DFTB+ (.gen), VASP (POSCAR, .poscar, .vasp), CP2K (.inp, .restart), Crystal23 (.cry, .d12)')
parser.add_argument('-o','--output',nargs='+',required=True,help='Output file(s). Format: CIF(.cif), PDB (.pdb), GROMACS (.gro), Cartesian (.xyz), DFTB+ (.gen), VASP (POSCAR, .poscar, .vasp), CP2K (.inp, .restart) ,Crystal23 (.cry, .d12), Siesta (.fdf)')
parser.add_argument('-pbc',action='store_true',help='Use Periodic Boundary Conditions to warp atoms within the box.')
parser.add_argument('-pad', '--padding',type=float,nargs=3,default=[0.0,0.0,0.0],help='Increase the size of the crystal box by increasing the a, b, and c parameters. This breaks the symmetry of the crystal.')
parser.add_argument('-sc', '--super_cell',type=int,nargs=3,default=[1,1,1],help='Number of replicas in each direction to make or reverse a supercell.')
parser.add_argument('-msc', '--make_super_cell',nargs=1,choices=['no','do','undo'],default=['no'],help='Whether to do nothing (no), make (do), or reverse (undo) a supercell.')
parser.add_argument('-sd',action='store_true',help='Selective Dynamics for VASP.')
parser.add_argument('-d3',action='store_true',help='Use Grimme D3 corrections in DFT input files.')
parser.add_argument('-asym',action='store_true',help='Print only the aymmetric unit.')
parser.add_argument('-bsym',action='store_true',help='Build the crystal unit cell from the aymmetric unit. A Hall number must be provided')
parser.add_argument('-fsym',action='store_true',help='Find and print the space group number, Hall number, and symmetry operations.')
parser.add_argument('-hn','--hall_number',nargs=1,type=int,default=[-1],help='Space group number used by Spglib, aka Hall number: 1-530. If provided it prevents the seach for the space group speeding up the symmetry section.')
parser.add_argument('-hnelpi','--hall_number_elastic_piezo',nargs=6,type=int,default=[-1,-1,-1,-1,-1,-1],help='Space group number used by Spglib (aka Hall number: 1-530), for each of the 6 strains. If provided it prevents the seach for the space group speeding up the symmetry section.')
parser.add_argument('-strain',action='store_true',help='Generate as series of strained systems along one of the crystallographic or cartesian axis.')
parser.add_argument('-sl','--strain_list',nargs=1,default=[''],help='File name containg the list of strcuture files from which strain is exctracted.')
parser.add_argument('-sv','--strain_values',nargs='+',default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.04, 0.05,0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25],help='Values for straining the system. The values are actually stored as a string type to facilitate the file naming.')
parser.add_argument('-sa','--strain_axis',nargs=1,choices=['a','b','c','al','be','ga','x','y','z','yz','xz','xy'],default=['c'],help='Axis along which to generate as series of strained systems.')
parser.add_argument('-getstress',action='store_true',help='Read the stress from the output files of as series of strained systems along one of the crystallographic or cartesian axis. The results are stored in outName and the stress tensor for the unstrained system is read in inName. For VASP, input file must be name OUTCAR, for the series of strains, files are expected to be named OUTCAR.[strained value]. E.g. OUTCAR.0.1. Alternatively, the output files can be listed in a file passed with the -ssl option, without naming convention.')
parser.add_argument('-ssl','--stress_list',nargs=1,default=[''],help='File name containg the list of output files from which to extract the stress tensors for the stress-strain curves.')
parser.add_argument('-cpmo','--cp2k_molecular_orbitals',action='store_true',help='Whether MOs should be printed in the output file. Usefull for band gap. Use together with -cpmon to control the number of HOMO and LUMO to be printed.')
parser.add_argument('-cpmon','--cp2k_mo_numbers',nargs=2,type=int,default=[3,3],help='Takes two type int arguments: number of HOMO and LUMO to be printed.')
parser.add_argument('-cpot','--cp2k_ot_algo',nargs=1,choices=['STRICT','IRAC','RESTART'],default=['STRICT'],help='Algorithm to ensure convergence of the Choleski decomposition. For difficult systems, use IRAC or RESTART. For RESTART, you need to have the wavefunction file from a previous calculation for the system of interest. This file should have the same basename as the input file with the extension .wfn.')
parser.add_argument('-cpopt','--cp2k_opt',nargs=1,choices=['CELL','IONS','NONE'],default=['CELL'],help='Optimization approach: full cell and ionic positions (CELL), or only the ionic positions (IONS), or no geometry optimization (NONE).')
parser.add_argument('-cpoptal','--cp2k_opt_algo',nargs=1,choices=['BFGS','CG'],default=['BFGS'],help='Optimization algorithm. If the number of atoms exceeds 999, BFGS is changed to its linearized version LBFGS.')
parser.add_argument('-cpoptan','--cp2k_opt_angles',action='store_true',help='Whether the angles of the lattice are allowed to relax.')
parser.add_argument('-cpoptbr','--cp2k_opt_bravais',action='store_true',help='Whether the bravais lattice is conserved or not.')
parser.add_argument('-cpoptsy','--cp2k_opt_symmetry',action='store_true',help='Whether the space group is conserved or not.')
parser.add_argument('-cptp','--cp2k_template',help='Template file for CP2K software if interested in different options.')
parser.add_argument('-cpscf',action='store_true',help='Use standard diagonalization instead of Orbital Transformation in CP2K calculations.')
parser.add_argument('-cpkp',action='store_true',help='Use k-points instead of supercell.')
parser.add_argument('-xc','--exchange_correlation',nargs=1,choices=['BLYP','B3LYP','PBE','PBE0','DEFAULT'],default=['DEFAULT'],help='Exchange-correlation functional. This is a snall selection of the most common functional found in all DFT software. Check the manual of your DFT software for the full rannge of its capabilities, and edit the input file accordingly. DEFAULT corresponds to whatever is provided in the input/template or PBE.')
parser.add_argument('-bs','--basisset',nargs=1,choices=['DZVP','TZVP'],default=['DZVP'],help='Basis set. This is a snall selection of the most basis sets found in all DFT software. Check the manual of your DFT software for the full rannge of its capabilities, and edit the input file accordingly.')
parser.add_argument('-kgrid',nargs=1,type=float,default=[4.e-2],help='Point spacing for the k-point grid in the reciprocal space.')
parser.add_argument('-potcar',action='store_true',help='Create a POTCAR file compatible with the POSCAR being written.')
parser.add_argument('-potsrc','--potcar_source',nargs=1,default=['/home/cazade/Documents/vasp_pp'],help='Directory where are the POTCARs for each atom.')
parser.add_argument('-kpoints',action='store_true',help='Create a KPOINTS file compatible with the POSCAR being written.')
parser.add_argument('-incar',action='store_true',help='Create a template INCAR file. It uses some of the cp2k options to set the value of ISIF')
parser.add_argument('-cpelpi','--cp2k_elastic_piezo',action='store_true',help='Generate as series of strain of the system to obtain the elastic and piezolectric tensors with CP2K.')
parser.add_argument('-cpdie','--cp2k_dielectric',action='store_true',help='Generate as series of variations of the electric field to obtain the dielectric tensor with CP2K.')
parser.add_argument('-cpelpig','--cp2k_elastic_piezo_get',action='store_true',help='Get the piezoelectric and elastic tensors from a series of strain of the system.')
parser.add_argument('-cpdieg','--cp2k_dielectric_get',action='store_true',help='Get the dielectric tensor from a series of variations of the electric field.')
parser.add_argument('-cpst','--cp2k_elastic_piezo_step',nargs=1,type=float,default=[1.e-2],help='Strain for the calculation of the elastic and piezolectric tensors with CP2K.')
parser.add_argument('-cpef','--cp2k_dielectric_field',nargs=1,type=float,default=[2.e-4],help='Field step for the calculation of dielectric tensor with CP2K.')
parser.add_argument('-se','--symmetry_excluded',nargs=2,type=int,action='append',help='Range of atoms excluded from symmetry identification and enforcement. Keywords is repeatable.')
parser.add_argument('-vaspelg','--vasp_elastic_get',action='store_true',help='Get the elastic tensor from VASP output file. This file is provided as an input')
parser.add_argument('-vasppig','--vasp_piezo_get',action='store_true',help='Get the piezoelectric and dielectric tensors from VASP output file. This file is provided as an input. If both --vasp_elastic_get and --vasp_piezo_get are requested, the files are provided as a list of inputs with the piezoelectric file first.')
parser.add_argument('-gmxelpig','--gmx_elastic_piezo_get',action='store_true',help='Get the piezoelectric and elastic tensors from a series of strain of the system. Needs a PSF or TOP file provided via the option -psf')
parser.add_argument('-psf','--psf_file',nargs=1,default=[''],help='CHARMM PSF or GROMACS TOP file providing the topology of the system. Required for --gmx_elastic_piezo_get.')
parser.add_argument('-sam','--select_atom_names',nargs='+',help='List of the selected atom names to write in the output files.')
parser.add_argument('-check',action='store_true',help='Check that the distance bewteen atoms is at least 0.5 A.')

args = parser.parse_args()

print(args)

#exit()

if(args.vasp_elastic_get or args.vasp_piezo_get):
    get_tensors_vasp(args)
    exit()
elif(args.getstress):
    get_stress(args.input[0],args.output[0],args)
    exit()
else:
    atoms,a,b,c,isScaled,sysType,spg=io_read(args.input[0])

if(not isScaled and a.norm>0.0 and b.norm>0.0 and c.norm>0.0):
    isScaled=True
    cart2frac(atoms,a,b,c)

if(args.check):
    if(not isScaled):
        print("Error, the coordinates should be fractional.")
        exit()
    i=0
    xyz=np.zeros((len(atoms),3))
    for at in atoms:
        xyz[i][0]=at.x
        xyz[i][1]=at.y
        xyz[i][2]=at.z
        i=i+1
    norm=max(a.norm,b.norm,c.norm)
    rMin=0.8/norm
    print(norm,rMin)
    rlist(xyz,rMin)
    exit()

if(args.select_atom_names is not None):
    tmp=[]
    for at in atoms:
        tmp.append(Atom())
        cpAtom(tmp[-1],at)
    atoms=[]
    for at in tmp:
        if(at.name.strip() in args.select_atom_names):
            atoms.append(Atom())
            cpAtom(atoms[-1],at)
    del tmp

if(args.fsym):
    find_symmetry(atoms,a,b,c,isScaled,args.hall_number[0])
    exit()

if(args.bsym):
    if(args.hall_number[0]<=0):
        print("A Hall number (1-530) is required to obtain the symmetry operations and build the crystal.")
        exit()
    atoms,isScaled=build_crystal(atoms,a,b,c,isScaled,args.hall_number[0])

print(isScaled)

if(args.make_super_cell[0].strip()=='undo'):
    atoms,a,b,c=undoSuperCell(atoms,a,b,c,isScaled,args.super_cell)
elif(args.make_super_cell[0].strip()=='do'):
    atoms,a,b,c=SuperCell(atoms,a,b,c,isScaled,args.super_cell)

if(args.strain):
    strain(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_elastic_piezo and args.cp2k_dielectric):
    elastic_piezo_strain(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
    dielectric_field(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_elastic_piezo):
    elastic_piezo_strain(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_dielectric):
    dielectric_field(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_elastic_piezo_get or args.cp2k_dielectric_get):
    get_tensors(args.input[0],args.output[0],atoms,a,b,c,isScaled,args)
elif(args.gmx_elastic_piezo_get):
    get_tensors_gmx(args.input[0],args.output[0],atoms,a,b,c,isScaled,args)
else:
    if( (args.padding[0]>1e-8) or (args.padding[1]>1e-8) or (args.padding[1]>1e-8) ):
        atoms,a,b,c,isScaled=box_padding(atoms,a,b,c,isScaled,args)
    if(args.pbc):
        pbc(atoms,a,b,c,isScaled)
    for outName in args.output:
        io_write(args.input[0],outName,atoms,a,b,c,isScaled,sysType,args.hall_number[0],args,None)

exit()

