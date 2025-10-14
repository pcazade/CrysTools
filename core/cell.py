# === Refactor for crystools.py: Cell-centric OOP methods ===
from dataclasses import dataclass,field
import numpy as np
from numpy.typing import NDArray
from ase.cell import Cell as ASECell
import numpy.linalg as la

@dataclass
class Cell:
    a:float = 0.0
    b:float = 0.0
    c:float = 0.0
    alpha:float = 0.0
    beta:float = 0.0
    gamma:float = 0.0
    hmat: NDArray[np.float64] = field(default_factory=lambda: np.zeros((3, 3)))
    gmat: NDArray[np.float64] = field(default_factory=lambda: np.zeros((3, 3)))


    def lp2box(self):
        cell = ASECell.fromcellpar([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
        self.hmat = cell.array
        self.gmat = la.inv(self.hmat)

    def mat2box(self):
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = ASECell(self.hmat).cellpar()