# === Refactor for crystools.py: Topology-centric OOP methods ===
from dataclasses import dataclass, field
from typing import List
from .atom import Atom

@dataclass
class Topology():
    ff:str = ''
    molName:str = ''
    nrexcl:int=0
    atoms:List[Atom] = field(default_factory=list)
    bonds:List[List[int]] = field(default_factory=list)
    pairs:List[List[int]] = field(default_factory=list)
    angles:List[List[int]] = field(default_factory=list)
    dihedrals:List[List[int]] = field(default_factory=list)
    impropers:List[List[int]] = field(default_factory=list)
    cmap:List[List[int]] = field(default_factory=list)

    nAtoms:int = 0
    nBonds:int = 0
    nAngles:int = 0
    nDihedrals:int = 0
    nImpropers:int = 0
    nCmap:int = 0

    def __post_init__(self) -> None:
        self.n_atoms = len(self.atoms)
        self.n_bonds = len(self.bonds)
        self.n_angles = len(self.angles)
        self.n_dihedrals = len(self.dihedrals)
        self.n_impropers = len(self.impropers)
        self.n_cmap = len(self.cmap)