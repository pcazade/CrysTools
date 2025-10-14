from core.atom import Atom
from core.cell import Cell
from crystools import typeatom
import numpy as np

a=Cell(a=5, b=6, c=7, alpha=90, beta=100, gamma=120)
a.lp2box()
# print(a.hmat)
# print(a.gmat)

b = np.array([
                 [5.,          0.,          0.],
                 [-3.,          5.19615242,  0.],
             [-1.21553724, - 0.70179075, 6.85783923],
])
c=Cell(hmat=b)
c.mat2box()
print(c.a,c.b,c.c,c.alpha,c.beta,c.gamma)
