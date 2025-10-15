# === Refactor for crystools.py: Cell-centric OOP methods ===
from dataclasses import dataclass,field
import numpy as np
from numpy.typing import NDArray
from ase.cell import Cell as ASECell
import numpy.linalg as la
import sys, math


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


    def cart2frac(self, atoms):
        r = self.hmat
        ir = la.inv(r)
        tir = np.transpose(ir)
        for at in atoms:
            xyz = np.array([at.x, at.y, at.z])
            uvw = np.matmul(tir, xyz)
            at.x = uvw[0]
            at.y = uvw[1]
            at.z = uvw[2]


    def frac2cart(self, atoms):
        r = self.hmat
        tr = np.transpose(r)
        for at in atoms:
            uvw = np.array([at.x, at.y, at.z])
            xyz = np.matmul(tr, uvw)
            at.x = xyz[0]
            at.y = xyz[1]
            at.z = xyz[2]


    # wz function for 1)reciprocal lattice vector & 2)unit cell volume calculation
    def wz(self):
        r = self.hmat  # rows: a, b, c
        a = r[0]
        b = r[1]
        c = r[2]

        # Cross products (numerators)
        ra = np.array([b[1] * c[2] - c[1] * b[2],
                       b[2] * c[0] - c[2] * b[0],
                       b[0] * c[1] - c[0] * b[1]], dtype=float)  # b × c
        rb = np.array([c[1] * a[2] - a[1] * c[2],
                       c[2] * a[0] - a[2] * c[0],
                       c[0] * a[1] - a[0] * c[1]], dtype=float)  # c × a
        rc = np.array([a[1] * b[2] - b[1] * a[2],
                       a[2] * b[0] - b[2] * a[0],
                       a[0] * b[1] - b[0] * a[1]], dtype=float)  # a × b

        # Determinant = volume (signed)
        det = float(a[0] * ra[0] + a[1] * ra[1] + a[2] * ra[2])

        if det > sys.float_info.min:
            ra /= det
            rb /= det
            rc /= det

        vol = abs(det)

        return ra, rb, rc, vol


    def pbc(self, atoms, isScaled: bool):
        """
        Apply periodic boundary conditions in-place.
        - If isScaled is True: atoms are fractional and are wrapped to [0,1).
        - If isScaled is False: atoms are Cartesian; convert -> wrap -> convert back.
        """
        if not np.any(self.hmat):
            raise ValueError("Cell.hmat is not set; cannot apply PBC.")

        # Lattice (direct) vectors from the cell matrix (rows are a, b, c)
        a = self.hmat[0]
        b = self.hmat[1]
        c = self.hmat[2]

        # Reciprocal vectors (and volume) from your existing method
        ra, rb, rc, vol = self.wz()

        if isScaled:
            for at in atoms:
                at.x -= math.floor(at.x)
                at.y -= math.floor(at.y)
                at.z -= math.floor(at.z)
        else:
            for at in atoms:
                # Cartesian -> fractional via dot with reciprocal vectors
                xt = at.x * ra[0] + at.y * ra[1] + at.z * ra[2]
                yt = at.x * rb[0] + at.y * rb[1] + at.z * rb[2]
                zt = at.x * rc[0] + at.y * rc[1] + at.z * rc[2]

                # Wrap to [0,1)
                xt -= math.floor(xt)
                yt -= math.floor(yt)
                zt -= math.floor(zt)

                # Fractional -> Cartesian via linear combo of direct vectors
                at.x = xt * a[0] + yt * b[0] + zt * c[0]
                at.y = xt * a[1] + yt * b[1] + zt * c[1]
                at.z = xt * a[2] + yt * b[2] + zt * c[2]