from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Atom:
    header: str= "ATOM  " #to write pdb file (pdb starts with ATOM)
    idx: int= 1 #index/position of the atom in crystal
    name: str= " H  " #name of atom in a residue. Each atom in a residue has unique name. In between multiple residue there may remain atoms of same name.
    loc: str= " " #needed for pdb file format
    resName: str= "DUM" #residue name. atoms ⊂ residue
    chain: str= "A" #residue ⊂ chain
    resIdx: int= 1 #residue index
    aType: str= "DUM" #atom type
    inser: str= " " #needed for pdb file format
    x: float= 0.0 #x coordinate
    y: float= 0.0 #y coordinate
    z: float= 0.0 #z coordinate
    vx: float = 0.0 #velocity along x axis
    vy: float= 0.0 #velocity along y axis
    vz: float= 0.0 #velocity along z axis
    occ: float= 1.0 #needed for pdb file format
    beta: float= 0.0 #needed for pdb file format
    segName: str= "P1  " #segment name. segment is a constituent of crystals as like atoms, residue, chain. Used while molecule assembly or simulation setup.
    # element = " H" #same as el (to be removed)
    chg: str= "1 " #needed for pdb file format
    q: float= 0.0 #charge of atom
    m: float= 0.0 #mass of atom
    # nAt = 0 #n atoms. not used anywhere (to be removed)
    el: str= " " #name of the element
    sig: float= 2.0 #sigma = van der Waals radius