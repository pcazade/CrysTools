# === Refactor for crystools.py: Atom-centric OOP methods ===
from dataclasses import dataclass
from ase.data import atomic_masses, vdw_radii, atomic_numbers

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
        Mirrors legacy cpAtom(at1, at2).
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


    def inferAtom(self) -> "Atom":
        """Infer element (self.el), mass (self.m) if missing, and sigma (self.sig)
        from the atom's name/element. Mirrors legacy typeatom(atom).
        """
        if (self.name.strip() == 'CAL' or self.name.strip() == 'Ca'):
            self.el = 'Ca'
        elif (self.name.strip() == 'CLA' or self.name.strip()[0:3] == 'CLG' or self.name.strip()[0:3] == 'Cl'):
            self.el = 'Cl'
        elif (self.name.strip() == 'SOD' or self.name.strip() == 'NA' or self.name.strip() == 'Na'):
            self.el = 'Na'
        elif (self.name.strip() == 'MGA' or self.name.strip() == 'MG' or self.name.strip() == 'Mg'):
            self.el = 'Mg'
        elif (self.name.strip() == 'ZN' or self.name.strip() == 'Zn'):
            self.el = 'Zn'
        elif (self.name.strip() == 'POT' or self.name.strip() == 'K'):
            self.el = 'K'
        elif (self.name.strip() == 'RUB'):
            self.el = 'Rb'
        elif (self.name.strip() == 'FE' or self.name.strip() == 'Fe'):
            self.el = 'Fe'
        elif (self.name.strip() == 'CES' or self.name.strip() == 'Ce'):
            self.el = 'Ce'
        elif (self.name.strip() == 'CAD' or self.name.strip() == 'Cd'):
            self.el = 'Cd'
        elif (self.name.strip() == 'ALG1' or self.name.strip() == 'Al'):
            self.el = 'Al'
        elif (self.name.strip()[0:2] == 'BR' or self.name.strip() == 'Br'):
            self.el = 'Br'
        elif (self.name.strip()[0:2] == 'AU' or self.name.strip() == 'Au'):
            self.el = 'Au'
        elif (self.name.strip() == 'BAR'):
            self.el = 'Ba'
        elif (self.name.strip() == 'LIT'):
            self.el = 'Li'
        elif (len(self.name.strip()) > 1 and self.name.strip()[1].islower()):
            self.el = self.name.strip()[0:2]
        else:
            self.el = ' ' + self.name.strip()[0]

        Z = atomic_numbers[self.el]

        if (self.m < 1e-6):
            self.m = float(atomic_masses[Z])

        self.sig = float(vdw_radii[Z])

        return self