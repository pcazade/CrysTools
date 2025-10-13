# === Refactor for crystools.py: Atom-centric OOP methods ===
from dataclasses import dataclass

@dataclass
class Atom(object):
    header = "ATOM  "
    idx = 1
    name = " H  "
    loc = ' '
    resName = "DUM"
    chain = 'A'
    resIdx = 1
    aType = "DUM"
    inser = ' '
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    occ = 1.0
    beta = 0.0
    segName = "P1  "
    element = " H"
    chg = "1 "
    q = 0.0
    m = 0.0
    nAt = 0
    el = ' '
    sig = 2.0

    def copyAtom(self, other: "Atom") -> "Atom":
        """Copy all public fields from other into self.
        Same as cpAtom(at1, at2).
        """
        self.header = other.header
        self.idx = other.idx
        self.name = other.name
        self.loc = other.loc
        self.resName = other.resName
        self.aType = other.aType
        self.chain = other.chain
        self.resIdx = other.resIdx
        self.inser = other.inser
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.vx = other.vx
        self.vy = other.vy
        self.vz = other.vz
        self.occ = other.occ
        self.beta = other.beta
        self.segName = other.segName
        self.element = other.element
        self.el = other.el
        self.chg = other.chg
        self.q = other.q
        self.m = other.m
        self.nAt = other.nAt
        self.sig = other.sig
        return self