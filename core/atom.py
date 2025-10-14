# === Refactor for crystools.py: Atom-centric OOP methods ===
from dataclasses import dataclass
from ase.data import atomic_masses, vdw_radii, atomic_numbers


@dataclass
class Atom():
    header:str = "ATOM  "
    idx:int = 1
    name:str = " H  "
    loc:str = ' '
    resName:str = "DUM"
    chain:str = 'A'
    resIdx:int = 1
    aType:str = "DUM"
    inser:str = ' '
    x:float = 0.0
    y:float = 0.0
    z:float = 0.0
    vx:float = 0.0
    vy:float = 0.0
    vz:float = 0.0
    occ:float = 1.0
    beta:float = 0.0
    segName:str = "P1  "
    element:str = " H"
    chg:str = "1 "
    q:float = 0.0
    m:float = 0.0
    nAt:int = 0
    el:str = ' '
    sig:str = 2.0


    def copyFromAtom(self, other: "Atom") -> "Atom":
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


    def copyFromZmat(self, zmat: "Zmat") -> "Atom":
        """Copy common fields from *zmat* into self.
        Mirrors legacy cpzmat(zmat, atom).
        """
        self.idx = zmat.idx
        self.name = zmat.name
        self.loc = zmat.loc
        self.resName = zmat.resName
        self.chain = zmat.chain
        self.resIdx = zmat.resIdx
        self.inser = zmat.inser
        self.occ = zmat.occ
        self.beta = zmat.beta
        self.segName = zmat.segName
        self.element = zmat.element
        self.chg = zmat.chg
        self.nAt = zmat.nAt
        self.el = zmat.el
        self.sig = zmat.sig
        self.q = zmat.q
        self.m = zmat.m
        return self


    def inferAtom(self) -> "Atom":
        """Infer element (self.el), mass (self.m) if missing, and sigma (self.sig)
        from the atom's name/element. Mirrors legacy typeatom(atom).
        """
        aliases = {
            "CAL": "Ca", "Ca": "Ca",
            "CLA": "Cl", "CLG": "Cl", "Cl": "Cl",
            "SOD": "Na", "NA": "Na", "Na": "Na",
            "MGA": "Mg", "MG": "Mg", "Mg": "Mg",
            "ZN": "Zn", "Zn": "Zn",
            "POT": "K", "K": "K",
            "RUB": "Rb",
            "FE": "Fe", "Fe": "Fe",
            "CES": "Ce", "Ce": "Ce",
            "CAD": "Cd", "Cd": "Cd",
            "ALG1": "Al", "Al": "Al",
            "BR": "Br", "Br": "Br",
            "AU": "Au", "Au": "Au",
            "BAR": "Ba",
            "LIT": "Li",
        }
        if self.name.strip() in aliases:
            self.el=aliases[self.name.strip()]
        elif (len(self.name.strip()) > 1 and self.name.strip()[1].islower()):
            self.el = self.name.strip()[0:2]
        else:
            self.el = ' ' + self.name.strip()[0]

        Z = atomic_numbers[self.el]

        if (self.m < 1e-6):
            self.m = float(atomic_masses[Z])

        self.sig = float(vdw_radii[Z])

        return self


