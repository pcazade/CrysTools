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
        sig = [['H', 1.2],
               ['C', 1.7],
               ['N', 1.55],
               ['O', 1.52],
               ['F', 1.47],
               ['P', 1.8],
               ['S', 1.8],
               ['Cl', 1.75],
               ['Cu', 1.4],
               ['Li', 1.82],
               ['Na', 2.27],
               ['K', 2.75],
               ['I', 1.98]]
        mass = {'H': 1.00790,
                'He': 4.00260,
                'Li': 6.94100,
                'Be': 9.01220,
                'B': 10.81100,
                'C': 12.01070,
                'N': 14.00670,
                'O': 15.99940,
                'F': 18.99840,
                'Ne': 20.17970,
                'Na': 22.98970,
                'Mg': 24.30500,
                'Al': 26.98150,
                'Si': 28.08550,
                'P': 30.97380,
                'S': 32.06500,
                'Cl': 35.45300,
                'Ar': 39.94800,
                'K': 39.09830,
                'Ca': 40.07800,
                'Sc': 44.95590,
                'Ti': 47.86700,
                'V': 50.94150,
                'Cr': 51.99610,
                'Mn': 54.93800,
                'Fe': 55.84500,
                'Co': 58.93320,
                'Ni': 58.69340,
                'Cu': 63.54600,
                'Zn': 65.39000,
                'Ga': 69.72300,
                'Ge': 72.64000,
                'As': 74.92160,
                'Se': 78.96000,
                'Br': 79.90400,
                'Kr': 83.80000,
                'Rb': 85.46780,
                'Sr': 87.62000,
                'Y': 88.90590,
                'Zr': 91.22400,
                'Nb': 92.90640,
                'Mo': 95.94000,
                'Tc': 98.00000,
                'Ru': 101.07000,
                'Rh': 102.90550,
                'Pd': 106.42000,
                'Ag': 107.86820,
                'Cd': 112.41100,
                'In': 114.81800,
                'Sn': 118.71000,
                'Sb': 121.76000,
                'Te': 127.60000,
                'I': 126.90450,
                'Xe': 131.29300,
                'Cs': 132.90550,
                'Ba': 137.32700,
                'La': 138.90550,
                'Ce': 140.11600,
                'Pr': 140.90770,
                'Nd': 144.24000,
                'Pm': 145.00000,
                'Sm': 150.36000,
                'Eu': 151.96400,
                'Gd': 157.25000,
                'Tb': 158.92530,
                'Dy': 162.50000,
                'Ho': 164.93030,
                'Er': 167.25900,
                'Tm': 168.93420,
                'Yb': 173.04000,
                'Lu': 174.96700,
                'Hf': 178.49000,
                'Ta': 180.94790,
                'W': 183.84000,
                'Re': 186.20700,
                'Os': 190.23000,
                'Ir': 192.21700,
                'Pt': 195.07800,
                'Au': 196.96650,
                'Hg': 200.59000,
                'Tl': 204.38330,
                'Pb': 207.20000,
                'Bi': 208.98040,
                'Po': 209.00000,
                'At': 210.00000,
                'Rn': 222.00000,
                'Fr': 223.00000,
                'Ra': 226.00000,
                'Ac': 227.00000,
                'Th': 232.03810,
                'Pa': 231.03590,
                'U': 238.02890,
                'Np': 237.00000,
                'Pu': 244.00000,
                'Am': 243.00000,
                'Cm': 247.00000,
                'Bk': 247.00000,
                'Cf': 251.00000,
                'Es': 252.00000,
                'Fm': 257.00000,
                'Md': 258.00000,
                'No': 259.00000,
                'Lr': 262.00000,
                'Rf': 261.00000,
                'Db': 262.00000,
                'Sg': 266.00000,
                'Bh': 264.00000,
                'Hs': 277.00000,
                'Mt': 268.00000}
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

        if (self.m < 1e-6):
            self.m = mass[self.el.strip()]

        for i in range(len(sig)):
            if (self.el == sig[i][0]):
                self.sig = sig[i][1]
                break
        return self