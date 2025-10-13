from crystools import Atom, cpAtom

if __name__ == '__main__':
    a1 = Atom()
    a2 = Atom()
    a1.x, a1.y, a1.z = 1, 2, 3
    cpAtom(a2, a1)
    print(a2.x, a2.y, a2.z)
