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
# crystools.py written by P.-A. Cazade                                          #
# Copyright (C) 2019 - 2024  P.-A. Cazade                                       #
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

class Atom(object):
    header="ATOM  "
    idx=1
    name=" H  "
    loc=' '
    resName="DUM"
    chain='A'
    resIdx=1
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
    resName="ALA"
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
    resName="ALA"
    segName="P1  "

class Site(object):
    at1=0
    at2=0
    at3=0
    resIdx=0
    resName="ALA"
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

def readPdb(fName):
    chainList=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    fi=open(fName,'r')
    isCryst=False
    isTer=False
    atoms=[]
    i=0
    nter=1
    for line in fi:
        if(line[0:6]=="REMARK"):
            continue
        elif(line[0:6]=="ANISOU"):
            continue
        elif(line[0:6]=="HETATM"):
            continue
        elif(line[0:6]=="CRYST1"):
            words=line.split()
            a,b,c=crystobox(words)
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
        atoms[i].resName=line[17:21]
        #atoms[i].chain=line[21]
        atoms[i].chain=chainList[(nter-1) % len(chainList)]
        if(i==0):
            oldr=int(line[22:26])
            nr=1
        elif(int(line[22:26])!=oldr):
            nr=nr+1
        oldr=int(line[22:26])
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
    return atoms,a,b,c

def writePdb(fName,atoms,a,b,c):
    fo=open(fName,'w')
    if(a.norm>0. and b.norm>0. and c.norm>0.):
        al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
        be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
        ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
        fo.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d\n" % (a.norm,b.norm,c.norm,al,be,ga,'P1',1))
    oldChain=atoms[0].chain
    oldResIdx=atoms[0].resIdx
    for at in atoms:
        typeatom(at)
        if(at.idx<=99999):
            fo.write("%6s%5d %4s %4s%c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%-2s\n" % (at.header,at.idx,at.name,at.resName,at.chain,at.resIdx,at.x,at.y,at.z,at.occ,at.beta,at.segName,at.el))
        else:
            fo.write("%6s%5x %4s %4s%c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%-2s\n" % (at.header,at.idx,at.name,at.resName,at.chain,at.resIdx,at.x,at.y,at.z,at.occ,at.beta,at.segName,at.el))
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
            nr=1
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

def writeGro(fName,atoms,a,b,c):
    fo=open(fName,'w')
    fo.write('File written by cryst.py from P.-A. Cazade\n')
    fo.write('%d\n' % (len(atoms)))
    for at in atoms:
        typeatom(at)
        resIdx=at.resIdx%100000
        idx=at.idx%100000
        fo.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n' % (resIdx,at.resName,at.name,idx,at.x*0.1,at.y*0.1,at.z*0.1))
    if(b.x>1.e-6 or c.x>1.e-6 or c.y>1.e-6):
        fo.write('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n' % (a.x*0.1,b.y*0.1,c.z*0.1,a.y*0.1,a.z*0.1,b.x*0.1,b.z*0.1,c.x*0.1,c.y*0.1))
    else:
        fo.write('%10.5f%10.5f%10.5f\n' % (a.x*0.1,b.y*0.1,c.z*0.1))
    fo.close()
    return

def readGzmat(fName,fPdb):
    atoms=readPdb(fPdb)
    
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

def pbc(atoms,a,b,c):
    ra,rb,rc,vol=wz(a,b,c)
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
    ra.x/=det
    ra.y/=det
    ra.z/=det
    rb.x/=det
    rb.y/=det
    rb.z/=det
    rc.x/=det
    rc.y/=det
    rc.z/=det
    vol=math.fabs(det)
    return(ra,rb,rc,vol)

def writePoscar(fName,atoms,a,b,c):
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
    fo.write("Direct\n")
    for el in lel:
        for at in atoms:
            if(at.el.strip()==el):
                at.x-=math.floor(at.x)
                at.y-=math.floor(at.y)
                at.z-=math.floor(at.z)
                fo.write("%20.16f%20.16f%20.16f\n" % (at.x,at.y,at.z))
    fo.close()
    return

def readPoscar(fName):
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
    al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
    be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
    ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    fo.write("%lf %lf %lf %lf %lf %lf\n" % (a.norm,b.norm,c.norm,al,be,ga))
    #fo.write("%s\n" % ("system"))
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

def writeCp2k(inName,outName,atoms,a,b,c,isScaled,args,dire):
    ext=inName.split('.')[-1]
    if(args.cp2k_template is not None):
        writeCp2kTemplate(args.cp2k_template,outName,atoms,a,b,c,isScaled,args,dire)
    elif((ext=='inp') or (ext=='restart')):
        writeCp2kTemplate(inName,outName,atoms,a,b,c,isScaled,args,dire)
    else:
        writeCp2kDefault(inName,outName,atoms,a,b,c,isScaled,args,dire)
    return

def writeCp2kTemplate(inName,outName,atoms,a,b,c,isScaled,args,dire):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']
    kn=['1','2','3','4','3','4','5','6','7','8','9','10','3','4','5','6','7',
        '8','9','10','11','12','13','14','15','16','17','18','11','12','13',
        '4','5','6','7','8','9','10','11','12','13','14','15','16','17','9',
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
        isScaled=True
        cart2frac(atoms,a,b,c)
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
        dataset=get_dataset(syst,hmat)
        del(syst)
    else:
        dataset=get_dataset(atoms,hmat)
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
    if(args.cp2k_opt_algo.strip()=='BFGS' and (len(atoms)>=1000)):
        args.cp2k_opt_algo='LBFGS'
    while (fi.tell() < eof):
        line=fi.readline()
        words=line.split()
        if('PROJECT_NAME' in line):
            title='   PROJECT_NAME '+basename.strip()
            fo.write('%s\n' % (title) ) 
            continue
        if('RUN_TYPE' in line and (args.cp2k_elastic_piezo or args.strain or (args.cp2k_opt.strip()=='IONS'))):
            run='   RUN_TYPE GEO_OPT'
            fo.write('%s\n' % (run) ) 
            continue
        if(args.cp2k_elastic_piezo or args.strain or (args.cp2k_opt.strip()=='IONS')):
            if('&MOTION' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                opt='BFGS'
                while('&END MOTION' not in line):
                    line=fi.readline()
                    if( ('OPTIMIZER' in line) and (args.cp2k_opt_algo.strip())):
                        opt=args.cp2k_opt_algo.strip()
                    elif('OPTIMIZER' in line):
                        words=line.split()
                        opt=words[1]
                        continue
                    if('SYMM_EXCLUDE_RANGE' in line):
                        words=line.split()
                        if(not isSer):
                            ser=[]
                        isSer=True
                        ser.append([int(words[1]),int(words[2])])
                    continue
                fo.write('   &GEO_OPT\n')
                fo.write('     OPTIMIZER  %s\n' % (opt.strip()))
                fo.write('     MAX_ITER  10000\n')
                if(args.cp2k_opt_symmetry):
                    fo.write('     KEEP_SPACE_GROUP  T\n')
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
        if(args.cp2k_elastic_piezo and (dire is not None)):
            if(('STRESS_TENSOR' in line)):
                continue
            if('&DFT' in line):
                fo.write( "%s" % (line))
                uvw=np.matmul(tih,dire)
                norm=la.norm(uvw)
                uvw=uvw/norm
                fo.write('     &PERIODIC_EFIELD\n')
                fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field))
                fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
                fo.write('     &END PERIODIC_EFIELD\n')
                continue
            if('&MM' in line):
                fo.write( "%s" % (line))
                uvw=dire
                norm=la.norm(uvw)
                uvw=uvw/norm
                print(dire)
                print(uvw)
                fo.write('     &PERIODIC_EFIELD\n')
                fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field))
                fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
                fo.write('     &END PERIODIC_EFIELD\n')
                continue
        if( (args.cp2k_ot_algo.strip()!='STRICT') or (args.cpscf) ):
            if('&SCF' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                while('&END SCF' not in line):
                    line=fi.readline()
                    if('MAX_SCF' and args.cpscf):
                        fo.write('       MAX_SCF 100\n')
                        continue
                    if('SCF_GUESS' in line and (args.cp2k_ot_algo.strip()!='RESTART')):
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
                        fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo.strip()))
                        continue
                    elif(args.cp2k_ot_algo.strip()!='RESTART'):
                        fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo.strip()))
                        continue
                    if('&OT' in line and not args.cpscf):
                        fo.write( "%s" % (line))
                        line=fi.readline()
                        while('&END OT' not in line):
                            line=fi.readline()
                            if('ALGORITHM' in line and (args.cp2k_ot_algo.strip()!='IRAC')):
                                fo.write('         ALGORITHM %s\n' % (args.cp2k_ot_algo.strip()))
                                continue
                            elif(args.cp2k_ot_algo.strip()!='IRAC'):
                                fo.write('         ALGORITHM %s\n' % (args.cp2k_ot_algo.strip()))
                                continue
                            fo.write('%s' % (line))
                            continue
                        fo.write('%s' % (line))
                        continue
                    elif('&OT' in line and args.cpscf):
                        fo.write( "%s" % (line))
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
                        continue
                    fo.write('%s' % (line))
                    continue
                fo.write('%s' % (line))
                continue
        if( (args.exchange_correlation.strip()!='DEFAULT') or args.d3):
            isXc-False
            isD3=False
            if('&XC' in line):
                fo.write( "%s" % (line))
                line=fi.readline()
                while('&END XC' not in line):
                    line=fi.readline()
                    if('&XC_FUNCTIONAL' in line):
                        isXc=True
                        fo.write( "%s" % (line))
                        line=fi.readline()
                        xc='         &'+args.exchange_correlation.strip()+' T'
                        fo.write('%s\n' % (xc))
                        exc='         &END '+args.exchange_correlation.strip()
                        fo.write('%s\n' % (exc))
                        while('&END XC_FUNCTIONAL' not in line):
                            line=fi.readline()
                            continue
                        fo.write('%s' % (line))
                        continue
                    if('&VDW_POTENTIAL' in line):
                        isD3=True
                    fo.write('%s' % (line))
                    continue
                if(not isXc):
                    fo.write('      &XC_FUNCTIONAL\n')
                    xc='         &'+args.exchange_correlation.strip()+' T'
                    fo.write('%s\n' % (xc))
                    exc='         &END '+args.exchange_correlation.strip()
                    fo.write('%s\n' % (exc))
                    fo.write('      &END XC_FUNCTIONAL\n')
                if(not isD3):
                    fo.write('       &VDW_POTENTIAL\n')
                    fo.write('         POTENTIAL_TYPE  PAIR_POTENTIAL\n')
                    fo.write('         &PAIR_POTENTIAL\n')
                    fo.write('           TYPE  DFTD3\n')
                    fo.write('           PARAMETER_FILE_NAME dftd3.dat\n')
                    fo.write('           REFERENCE_FUNCTIONAL %s\n' % (args.exchange_correlation.strip()))
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
                fo.write( '%6s %25.16e %25.16e %25.16e\n' % ( at.el,at.x,at.y,at.z ) )
            fo.write("%s\n" % ('       UNIT angstrom'))
            if(isScaled):
                fo.write("%s\n" % ('       SCALED  T'))
            else:
                fo.write("%s\n" % ('       SCALED  F'))
            fo.write( "%s" % (line))
            continue
        if('&KIND' in line):
            line=fi.readline()
            while('&END KIND' not in line):
                line=fi.readline()
                continue
            continue
        if('&END SUBSYS' in line):
            for el in lel:
                k=ks.index(el.strip())
                fo.write("%s %s\n" % ('     &KIND',el.strip()))
                basisSet='       BASIS_SET '+args.basisset.strip()+'-MOLOPT-SR-GTH-q'
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

def writeCp2kDefault(inName,outName,atoms,a,b,c,isScaled,args,dire):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']
    kn=['1','2','3','4','3','4','5','6','7','8','9','10','3','4','5','6','7',
        '8','9','10','11','12','13','14','15','16','17','18','11','12','13',
        '4','5','6','7','8','9','10','11','12','13','14','15','16','17','9',
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
        isScaled=True
        cart2frac(atoms,a,b,c)
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
        dataset=get_dataset(syst,hmat)
        del(syst)
    else:
        dataset=get_dataset(atoms,hmat)
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
    if(args.cp2k_opt_algo.strip()=='BFGS' and (len(atoms)>=1000)):
        args.cp2k_opt_algo='LBFGS'
    fo=open(outName,'w')
    fo.write(' &GLOBAL\n')
    fo.write('   PRINT_LEVEL  MEDIUM\n')
    basename=outName.split('.')[0]
    for i in range(1,len(outName.split('.'))-1):
        basename=basename+'.'+outName.split('.')[i]
    title='   PROJECT_NAME '+basename.strip()
    fo.write('   PROJECT_NAME %s\n' % (title))
    fo.write('   RUN_TYPE  %s\n' % (args.cp2k_opt.strip()))
    fo.write(' &END GLOBAL\n')
    fo.write(' &MOTION\n')
    if(args.cp2k_opt.strip()=='CELL'):
        fo.write('   &CELL_OPT\n')
        fo.write('     OPTIMIZER  %s\n' % (args.cp2k_opt_algo.strip()))
        fo.write('     MAX_ITER  1000\n')
        if(args.cp2k_opt_symmetry):
            fo.write('     KEEP_SPACE_GROUP  T\n')
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
    elif(args.cp2k_opt.strip()=='IONS'):
        fo.write('   &GEO_OPT\n')
        fo.write('     OPTIMIZER  %s\n' % (args.cp2k_opt_algo.strip()))
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
        fo.write('       INTENSITY %lg\n' % (args.cp2k_dielectric_field))
        fo.write('       POLARISATION %lf %lf %lf\n' % (uvw[0],uvw[1],uvw[2]))
        fo.write('     &END PERIODIC_EFIELD\n')
    fo.write('     BASIS_SET_FILE_NAME BASIS_MOLOPT\n')
    fo.write('     POTENTIAL_FILE_NAME GTH_POTENTIALS\n')
    fo.write('     MULTIPLICITY  1\n')
    fo.write('     CHARGE  0\n')
    if(args.cpscf):
        fo.write('     &SCF\n')
        if(args.cp2k_ot_algo.strip()=='RESTART'):
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
            fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo.strip()))
        else:
            fo.write('       SCF_GUESS ATOMIC\n')
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
        fo.write('     &END SCF\n')
    else:
        fo.write('     &SCF\n')
        fo.write('       MAX_SCF  20\n')
        fo.write('       EPS_SCF    1.e-6\n')
        if(args.cp2k_ot_algo.strip()=='RESTART'):
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
            fo.write('       SCF_GUESS %s\n' % (args.cp2k_ot_algo.strip()))
        else:
            fo.write('       SCF_GUESS ATOMIC\n')
        fo.write('       &OT  T\n')
        if(args.cp2k_ot_algo.strip()=='IRAC'):
            fo.write('         ALGORITHM %s\n' % (args.cp2k_ot_algo.strip()))
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
    xc=args.exchange_correlation.strip()
    if(xc.strip()=='DEFAULT'):
        xc='PBE'
    fo.write('         & %s T\n' % (xc))
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
    fo.write('       &MOMENTS  SILENT\n')
    fo.write('         PERIODIC  T\n')
    fo.write('       &END MOMENTS\n')
    fo.write('     &END PRINT\n')
    fo.write('   &END DFT\n')
    fo.write('   &SUBSYS\n')
    fo.write('     &CELL\n')
    fo.write('       A  %25.16e %25.16e %25.16e\n' % (a.x,a.y,a.z) )
    fo.write('       B  %25.16e %25.16e %25.16e\n' % (b.x,b.y,b.z) )
    fo.write('       C  %25.16e %25.16e %25.16e\n' % (c.x,c.y,c.z) )
    fo.write('       PERIODIC  XYZ\n')
    fo.write('       SYMMETRY %s\n' % (bravais_lattice.strip()))
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
    fo.write('       MULTIPLE_UNIT_CELL %d %d %d\n' % (math.ceil(18./a.norm),math.ceil(18./b.norm),math.ceil(18./c.norm)))
    fo.write('     &END TOPOLOGY\n')
    for el in lel:
        k=ks.index(el.strip())
        fo.write("%s %s\n" % ('     &KIND',el.strip()))
        basisSet='       BASIS_SET '+args.basisset.strip()+'-MOLOPT-SR-GTH-q'
        fo.write("%s%s\n" % (basisSet,kn[k]))
        fo.write('       POTENTIAL GTH-%s-q%s' % (xc.strip(),kn[k]))
        fo.write("%s\n" % ('     &END KIND'))
    fo.write('   &END SUBSYS\n')
    fo.write('   &PRINT\n')
    fo.write('     &STRESS_TENSOR  ON\n')
    fo.write('     &END STRESS_TENSOR\n')
    fo.write('   &END PRINT\n')
    fo.write(' &END FORCE_EVAL\n')
    fo.close()
    return

def get_dataset(atoms,hmat):
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
    dataset = sg.get_symmetry_dataset(cell, symprec=1e-4, hall_number=0)
    return(dataset)

def writeCry(outName,atoms,a,b,c,args):
    ks=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
        'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
        'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
        'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
        'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
        'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
        'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
        'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn']
    wz(a,b,c)
    al=math.acos((b.x*c.x+b.y*c.y+b.z*c.z)/(b.norm*c.norm))*180./math.pi
    be=math.acos((a.x*c.x+a.y*c.y+a.z*c.z)/(a.norm*c.norm))*180./math.pi
    ga=math.acos((a.x*b.x+a.y*b.y+a.z*b.z)/(a.norm*b.norm))*180./math.pi
    hmat=[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]
    dataset=get_dataset(atoms,hmat)
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
    basisSet='POB-'+args.basisset.strip()
    fo.write("%s\n" % (basisSet))
    fo.write("DFT\n")
    xc=args.exchange_correlation.strip()
    if((xc.strip()=='DEFAULT') or (xc.strip()=='PBE')):
        xc='PBEXC'
    if(args.d3):
        xc=xc.strip()+'-D3'
    fo.write("%s\n" % (xc))
    fo.write("ENDDFT\n")
    fo.write("SHRINK\n")
    fo.write("6 6\n")
    fo.write("END\n")
    fo.close()
    return

def undoSuperCell(atoms,a,b,c,sc):
    ma=sc[0]
    mb=sc[1]
    mc=sc[2]

    if(not isScaled):
        cart2frac(atoms,a,b,c)
        isScaled=True

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

def SuperCell(atoms,a,b,c,sc):
    chainList=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    
    ma=sc[0]
    mb=sc[1]
    mc=sc[2]

    if(not isScaled):
        cart2frac(atoms,a,b,c)
        isScaled=True

    m=0
    ns=len(atoms)
    ne=ns*(ma*mb*mc)
    for i in range(ma):
        for j in range(mb):
            for k in range(mc):
                for l in range(ns):
                    atoms.append(Atom())
                    cpAtom(atoms[-1],atoms[l])
                    if(i==0 and j==0 and k==0 and l==0):
                        oldr=atoms[l].resIdx
                        nr=1
                    elif(atoms[l].resIdx!=oldr):
                        nr+=1
                    oldr=atoms[l].resIdx
                    chain=chainList[((nr-1) % int(len(chainList)/3))]
                    atoms[-1].idx=m+1
                    atoms[-1].resIdx=nr
                    atoms[-1].chain=chain
                    atoms[-1].x=atoms[l].x+i
                    atoms[-1].y=atoms[l].y+j
                    atoms[-1].z=atoms[l].z+k
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
    return(atoms,a,b,c)

def elastic_piezo_strain(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    numDev=[-1.0,1.0]
    voigt=[[0,0],[1,1],[2,2],[2,1],[2,0],[1,0]]
    
    if(not isScaled):
        cart2frac(atoms,a,b,c)
        isScaled=True

    basename=outName.split('.')[0]
    fname=basename+'.ref.inp'
    io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args,None)

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

    for v in voigt:
        for n in numDev:
            e=np.identity(3)
            e[v[0],v[1]]+=args.cp2k_elastic_piezo_step
            te=np.transpose(e)
            hs=np.matmul(hmat,te) # xyz axes, angles not conserved, VASP approach
            a.x=hs[0,0]
            a.y=hs[0,1]
            a.z=hs[0,2]
            b.x=hs[1,0]
            b.y=hs[1,1]
            b.z=hs[1,2]
            c.x=hs[2,0]
            c.y=hs[2,1]
            c.z=hs[2,2]
            fname=basename+'.strain_'+str(k+1)+'_'+str(int(n))+'.inp'
            io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args,None)
    return

def dielectric_field(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    numDev=[-1.0,1.0]
    ef=['x','y','z']
    basename=outName.split('.')[0]
    fname=basename+'.ref.inp'
    io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args,None)
    for i in range(3):
        for n in numDev:
            fname=basename+'.efield_'+str(i+1)+'_'+str(int(n))+'.inp'
            dire=np.zeros((3))
            dire[i]=1.0*n
            io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args,dire)
    return

def strain(inName,outName,atoms,a,b,c,isScaled,sysType,args):
    if(not isScaled):
        cart2frac(atoms,a,b,c)
        isScaled=True

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

    ext=outName.split('.')[-1]
    basename=outName.split('.')[0]

    cax=['a','b','c']
    fax=['x','y','z']
    for s in args.strain_values:
        e=np.identity(3)
        if(args.strain_axis.strip()=='a' or args.strain_axis.strip()=='x'):
            e[0,0]+=s
        elif(args.strain_axis.strip()=='b' or args.strain_axis.strip()=='y'):
            e[1,1]+=s
        elif(args.strain_axis.strip()=='c' or args.strain_axis.strip()=='z'):
            e[2,2]+=s
        if(args.strain_axis.strip() in cax):
            hs=np.matmul(e,hmat) # abc axes, angles conserved
        elif(args.strain_axis.strip() in fax):
            hs=np.matmul(hmat,e) # xyz axes, angles not conserved
        a.x=hs[0,0]
        a.y=hs[0,1]
        a.z=hs[0,2]
        b.x=hs[1,0]
        b.y=hs[1,1]
        b.z=hs[1,2]
        c.x=hs[2,0]
        c.y=hs[2,1]
        c.z=hs[2,2]
        fname=basename+'.'+args.strain_axis.strip()+'_'+str(s)+'.'+ext
        io_write(inName,fname,atoms,a,b,c,isScaled,sysType,args,None)
    return

def get_stress(inName,outName,args):
    cax=['a','b','c']
    fax=['x','y','z']
    axis=args.strain_axis.strip()
    if(axis in cax):
        sigma=np.zeros((len(st)+1,5))
    else:
        sigma=np.zeros((len(st)+1,3))

    basename=inName.split('.')[0]
    ext=inName.split('.')[-1]
    isVASP=False
    if('OUTCAR' in inName):
        isVASP=True
        stress=getStressTensorVASP(inName)
    elif(ext=='.out' or ext=='.log'):
        stress,u,v,w=getStressTensorCP2K(inName)
    else:
        print(ext,'unknown output file extension. This function is only available for VASP and CP2K')
        exit()
    
    if(axis=='a'):
        si0=stress[0,0]
        sig0=stress[0]
    elif(axis=='b'):
        si0=stress[1,1]
        sig0=stress[1]
    elif(axis=='c'):
        si0=stress[2,2]
        sig0=stress[2]
    elif(axis=='x'):
        si0=stress[0,0]
    elif(axis=='y'):
        si0=stress[1,1]
    elif(axis=='z'):
        si0=stress[2,2]

    sigma[0,1]=si0
    sigma[0,2]=si0-si0
    if(axis in cax):
        sigma[0,3]=la.norm(sig0)
        sigma[0,4]=la.norm(sig0-sig0)

    i=0
    for s in st:
        i+=1
        if(isVASP):
            fname=basename+'.'+axis.strip()+'_'+str(s)
        else:
            fname=basename+'.'+axis.strip()+'_'+str(s)+'.'+ext
            stress,u,v,w=getStressTensorCP2K(fname)
        if(axis=='a'):
            si=stress[0,0]
            sig=stress[0]
        elif(axis=='b'):
            si=stress[1,1]
            sig=stress[1]
        elif(axis=='c'):
            si=stress[2,2]
            sig=stress[2]
        elif(axis=='x'):
            si=stress[0,0]
        elif(axis=='y'):
            si=stress[1,1]
        elif(axis=='z'):
            si=stress[2,2]
        sigma[i,0]=s
        sigma[i,1]=si
        sigma[i,2]=si-si0
        if(axis in cax):
            sigma[i,3]=la.norm(sig)
            sigma[i,4]=la.norm(sig-sig0)
    
    fo=open(outName,'w')
    for sig in sigma:
        for si in sig:
            fo.write("%lf " % (si))
        fo.write("\n")
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
    sysType=''
    isScaled=False
    words=inName.split('.')
    ext=words[-1]
    if(ext=='pdb'):
        atoms,a,b,c=readPdb(inName)
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
    elif(ext=='POSCAR' or ext=="poscar" or ext=="vasp" or ('POSCAR' in inName)):
        atoms,a,b,c,isScaled=readPoscar(inName)
    elif(ext=='restart' or ext=='inp'):
        atoms,isScaled=getCoordCP2K(inName)
        a,b,c=getBoxCP2K(inName)
    wz(a,b,c)
    return(atoms,a,b,c,isScaled,sysType)

def io_write(inName,outName,atoms,a,b,c,isScaled,sysType,args,dire):
    words=outName.split('.')
    ext=words[-1]
    if(ext=='POSCAR' or ext=="poscar" or ext=="vasp" or ('POSCAR' in inName)):
        if(isScaled):
            writePoscar(outName,atoms,a,b,c)
        else:
            cart2frac(atoms,a,b,c)
            writePoscar(outName,atoms,a,b,c)
    elif(ext=='pdb'):
        if(isScaled):
            frac2cart(atoms,a,b,c)
        writePdb(outName,atoms,a,b,c)
    elif(ext=='gro'):
        if(isScaled):
            frac2cart(atoms,a,b,c)
        writeGro(outName,atoms,a,b,c)
    elif(ext=='gen'):
        if(isScaled):
            sysType='F'
            writeGen(outName,atoms,a,b,c,sysType)
        else:
            sysType='F'
            cart2frac(atoms,a,b,c)
            writeGen(outName,atoms,a,b,c,sysType)
    elif(ext=='xyz'):
        if(isScaled):
            frac2cart(atoms,a,b,c)
        writeXyz(outName,atoms,a,b,c)
    elif(ext=='inp'):
        writeCp2k(inName,outName,atoms,a,b,c,isScaled,args,dire)
    elif(ext=='cry' or ext=='d12'):
        if(not isScaled):
            cart2frac(atoms,a,b,c)
        writeCry(outName,atoms,a,b,c,args)
    elif(ext=='cif'):
        data=ase.io.read(inName)
        ase.io.write(outName,data)
    else:
        print("%s format is not supported" % (ext.strip()) )
        print("Used PDB format instead.")
        if(isScaled):
            frac2cart(atoms,a,b,c)
        writePdb(outName,atoms,a,b,c)
    return

def get_rotations(atoms,hmat):
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
    dataset = sg.get_symmetry_dataset(cell, symprec=1e-4, hall_number=0)
    print(dataset['hall_number'],dataset['international'],dataset['pointgroup'])
    return(dataset['rotations'])

def get_rotations_subset(atoms,hmat):
    r=get_rotations(atoms,hmat)
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
        cart2frac(atoms,a0,b0,c0)
        isScaled=True
    
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
        dataset=get_dataset(syst,hmat)
        rotations=get_rotations_subset(syst,hmat)
        del(syst)
    else:
        dataset=get_dataset(atoms,hmat)
        rotations=get_rotations_subset(atoms,hmat)

    nop=len(rotations)
    for i in range(nop):
        rotations[i]=np.transpose(rotations[i])

    if(args.cp2k_elastic_piezo):
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
                mu,a,b,c=getDipoleCP2K(fname)
                stress,u,v,w=getStressTensorCP2K(fname)
                ra,rb,rc,vol=wz(a,b,c)
                h2=np.array([[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]])
                tih2=np.transpose(la.inv(h2))
                bp=np.matmul(tih2,mu)
                bf[i,k,:]=bp*debye2int
                ds[k,ii,jj,:,:]=stress[:,:]
                ds[k,jj,ii,:,:]=stress[:,:]
                k=k+1
    
        h=args.cp2k_elastic_piezo_step/debye2au
        e=dm_pbc(bf,tr1)
        e=e*int2debye/(2*h*vol0)
        e=sym_tensor3(e,rotations,nop,hmat)
    
        for i in range(3):
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                ev[i][k]=e[i][kk][ll]
    
        h=-args.cp2k_elastic_piezo_step
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

    if(args.cp2k_dielectric):
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
        h=(args.cp2k_dielectric_field*ef)/(debye2au)
    
        ide=np.identity(3)
    
        x=np.zeros((3,3))
        x=dm_pbc_dielec(bf,th1)
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

    if(args.cp2k_elastic_piezo and args.cp2k_dielectric):
        g=1.e-9*np.matmul(be,d)

    print_tensors(outName,args,e,cv,ci,d,ep,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph)

    return

def print_tensors(outName,args,e,c,ci,d,ep,g,kv,yv,gv,pv,kr,yr,gr,pr,kh,yh,gh,ph):
    a1='     1    '
    a2='     2    '
    a3='     3    '
    a4='     4    '
    a5='     5    '
    a6='     6    '
    fo=open(outName,'w')
    if(args.cp2k_elastic_piezo_get):
        e=sym_tensor3_cleaner(e)
        fo.write('Piezoelectric Charge Constants [e] (C/m^2)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        fo.write("1 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[0][0][0],e[0][1][1],e[0][2][2],e[0][1][2],e[0][2][0],e[0][0][1]) )
        fo.write("2 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[1][0][0],e[1][1][1],e[1][2][2],e[1][1][2],e[1][2][0],e[1][0][1]) )
        fo.write("3 %10.3lf%10.3lf%10.3lf%10.3lf%10.3lf%10.3lf\n" % (e[2][0][0],e[2][1][1],e[2][2][2],e[2][1][2],e[2][2][0],e[2][0][1]) )
        fo.write('\n')
    
        c=sym_tensor4_cleaner(c)
        fo.write('Elastic Constants (GPa)\n')
        fo.write('  %10s%10s%10s%10s%10s%10s\n' % (a1,a2,a3,a4,a5,a6))
        for i in range(6):
            ii=idx[i][0]
            jj=idx[i][1]
            fo.write('%2d' % (i+1))
            for k in range(6):
                kk=idx[k][0]
                ll=idx[k][1]
                fo.write('%10.3lf' % (c[ii][jj][kk][ll]))
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
                fo.write('%16.8lf' % (cv[i][k]))
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

# Program begins here:

parser=ap.ArgumentParser(prog='crystools',description='Convert between various structure and input file formats. Generate perturbations for piezoelectric and elastic properties. When the requested output file is in CP2K format, if a template is provided together with an input file in CP2K format, the keywords found in the tenplate are used with the structure cointained in the input. Any additional argumnet like -d3 will overwrite what is present in the template',epilog='Please repport any bugs to P.-A. Cazade at pierre.cazade@ul.ie.')
parser.add_argument('-i','--input',nargs='+',required=True,help='Input file(s). Format: CIF(.cif), PDB (.pdb), GROMACS (.gro), Cartesian (.xyz), DFTB+ (.gen), VASP (POSCAR, .poscar, .vasp), CP2K (.inp, .restart),Crystal23 (.cry, .d12)')
parser.add_argument('-o','--output',nargs='+',required=True,help='Output file(s). Format: CIF(.cif), PDB (.pdb), GROMACS (.gro), Cartesian (.xyz), DFTB+ (.gen), VASP (POSCAR, .poscar, .vasp), CP2K (.inp, .restart),Crystal23 (.cry, .d12)')
parser.add_argument('-sc', '--super_cell',type=int,nargs=3,default=[1,1,1],help='Number of replicas in each direction to make or reverse a supercell.')
parser.add_argument('-msc', '--make_super_cell',nargs=1,choices=['no','do','undo'],default='no',help='Whether to do nothing (no), make (do), or reverse (undo) a supercell.')
parser.add_argument('-d3',action='store_true',help='Use Grimme D3 corrections in DFT input files.')
parser.add_argument('-asym',action='store_true',help='Print only the aymmetric unit.')
parser.add_argument('-strain',action='store_true',help='Generate as series of strained systems along one of the crystallographic or cartesian axis.')
parser.add_argument('-sv','--strain_values',type=float,nargs='+',default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.04, 0.05,0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25],help='Values for straining the system.')
parser.add_argument('-sa','--strain_axis',nargs=1,choices=['a','b','c','x','y','z'],default='c',help='Axis along which to generate as series of strained systems.')
parser.add_argument('-getstress',action='store_true',help='Read the stress from the output files of as series of strained systems along one of the crystallographic or cartesian axis. The results are stored in outName and the stress tensor for the unstrained system is read in inName. For VASP, input file must be name OUTCAR, for the series of strains, files are expected to be named OUTCAR.[strained value]. E.g. OUTCAR.0.1')
parser.add_argument('-cpot','--cp2k_ot_algo',nargs=1,choices=['STRICT','IRAC','RESTART'],default='STRICT',help='Algorithm to ensure convergence of the Choleski decomposition. For difficult systems, use IRAC or RESTART. For RESTART, you need to have the wavefunction file from a previous calculation for the system of interest. This file should have the same basename as the input file with the extension .wfn.')
parser.add_argument('-cpopt','--cp2k_opt',nargs=1,choices=['CELL','IONS'],default='CELL',help='Optimization approach: full cell and ionic positions (CELL), or only the ionic positions (IONS).')
parser.add_argument('-cpoptal','--cp2k_opt_algo',nargs=1,choices=['BFGS','CG'],default='BFGS',help='Optimization algorithm. If the number of atoms exceeds 999, BFGS is changed to its linearized version LBFGS.')
parser.add_argument('-cpoptan','--cp2k_opt_angles',action='store_true',help='Whether the angles of the lattice are allowed to relax.')
parser.add_argument('-cpoptbr','--cp2k_opt_bravais',action='store_true',help='Whether the bravais lattice is conserved or not.')
parser.add_argument('-cpoptsy','--cp2k_opt_symmetry',action='store_true',help='Whether the space group is conserved or not.')
parser.add_argument('-cptp','--cp2k_template',help='Template file for CP2K software if interested in different options.')
parser.add_argument('-cpscf',action='store_true',help='Use standard diagonalization instead of Orbital Transformation in CP2K calculations.')
parser.add_argument('-xc','--exchange_correlation',nargs=1,choices=['BLYP','B3LYP','PBE','PBE0','DEFAULT'],default='DEFAULT',help='Exchange-correlation functional. This is a snall selection of the most common functional found in all DFT software. Check the manual of your DFT software for the full rannge of its capabilities, and edit the input file accordingly. DEFAULT corresponds to whatever is provided in the input/template or PBE.')
parser.add_argument('-bs','--basisset',nargs=1,choices=['DZVP','TZVP'],default='DZVP',help='Basis set. This is a snall selection of the most basis sets found in all DFT software. Check the manual of your DFT software for the full rannge of its capabilities, and edit the input file accordingly.')
parser.add_argument('-cpelpi','--cp2k_elastic_piezo',action='store_true',help='Generate as series of strain of the system to obtain the elastic and piezolectric tensors with CP2K.')
parser.add_argument('-cpdie','--cp2k_dielectric',action='store_true',help='Generate as series of variations of the electric field to obtain the dielectric tensor with CP2K.')
parser.add_argument('-cpelpig','--cp2k_elastic_piezo_get',action='store_true',help='Get the piezoelectric and elastic tensors from a series of strain of the system.')
parser.add_argument('-cpdieg','--cp2k_dielectric_get',action='store_true',help='Get the dielectric tensor from a series of variations of the electric field.')
parser.add_argument('-cpst','--cp2k_elastic_piezo_step',nargs=1,type=float,default=1.e-2,help='Strain percentage for the calculation of the elastic and piezolectric tensors with CP2K.')
parser.add_argument('-cpef','--cp2k_dielectric_field',nargs=1,type=float,default=2.e-4,help='Field step for the calculation of dielectric tensor with CP2K.')
parser.add_argument('-se','--symmetry_excluded',nargs=2,type=int,action='append',help='Range of atoms excluded from symmetry identification and enforcement. Keywords is repeatable.')
parser.add_argument('-vaspelg','--vasp_elastic_get',action='store_true',help='Get the elastic tensor from VASP output file. This file is provided as an input')
parser.add_argument('-vasppig','--vasp_piezo_get',action='store_true',help='Get the piezoelectric and dielectric tensors from VASP output file. This file is provided as an input. If both --vasp_elastic_get and --vasp_piezo_get are requested, the files are provided as a list of inputs with the piezoelectric file first.')

args = parser.parse_args()

print(args)

#exit()

if(args.vasp_elastic_get or args.vasp_piezo_get):
    get_tensors_vasp(args)
    exit()
else:
    atoms,a,b,c,isScaled,sysType=io_read(args.input[0])

if(args.make_super_cell.strip()=='undo'):
    atoms,a,b,c=undoSuperCell(atoms,a,b,c,args.super_cell)
elif(args.make_super_cell.strip()=='do'):
    atoms,a,b,c=superCell(atoms,a,b,c,args.super_cell)

if(args.getstress):
    get_stress(args.input[0],args.output[0],args)
elif(args.strain):
    strain(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_elastic_piezo):
    elastic_piezo_strain(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_dielectric):
    dielectric_field(args.input[0],args.output[0],atoms,a,b,c,isScaled,sysType,args)
elif(args.cp2k_elastic_piezo_get or args.cp2k_dielectric_get):
    get_tensors(inName,outName,atoms,a,b,c,isScaled,args)
else:
    for outName in args.output:
        io_write(args.input[0],outName,atoms,a,b,c,isScaled,sysType,args,None)

exit()

