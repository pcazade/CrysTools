from numpy.linalg import det

from core.atom import Atom
from core.cell import Cell
from crystools import typeatom, cart2frac
import numpy as np


obj=Cell(a=5,b=6,c=7,alpha=90,beta=100,gamma=120)
print(obj.hmat)
print(obj.gmat)
