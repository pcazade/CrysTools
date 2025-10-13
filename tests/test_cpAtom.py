from core.atom import Atom

a1 = Atom()
a1.x, a1.y, a1.z = 1.0, 2.0, 3.0
a1.name = "H"
a2 = Atom()
a2.x, a2.y, a2.z = 0.0, 0.0, 0.0
a2.name = "O"

print("Before:", vars(a2))
a2.copyAtom(a1)
print("After:", vars(a2))
