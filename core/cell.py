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


    def __post_init__(self):
        """
        Initialization rules:
        - If hmat has any non-zero entries: trust it and back-fill (a,b,c,alpha,beta,gamma).
        - Else if a,b,c look set (>0): build hmat from these params.
        - Else: leave uninitialized; conversions will raise until set.
        """
        # Prefer an explicitly provided matrix if it looks non-empty
        if np.any(self.hmat):
            params = ASECell(self.hmat).cellpar()
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma = params
            self.gmat = la.inv(self.hmat)
        # Otherwise, build from parameters if they look meaningful
        elif (self.a > 0.0) or (self.b > 0.0) or (self.c > 0.0):
            cell = ASECell.fromcellpar([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
            self.hmat = cell.array
            self.gmat = la.inv(self.hmat)
        # else: keep zero matrices/params; user must call set_* before converting


    # def lp2box(self):
    #     cell = ASECell.fromcellpar([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
    #     self.hmat = cell.array
    #     self.gmat = la.inv(self.hmat)
    #
    #
    # def mat2box(self):
    #     self.a, self.b, self.c, self.alpha, self.beta, self.gamma = ASECell(self.hmat).cellpar()

