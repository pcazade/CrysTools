from typing import List, Tuple
import numpy as np
from core.atom import Atom
from core.cell import Cell


def readGro(
    fname: str,
    chain: str,
    to_angstrom: bool = True,
) -> Tuple[List[Atom], Cell]:
    """
    Read a GROMACS .gro coordinate file.

    Parameters
    ----------
    fname : str
        Path to the .gro file.
    chain : str
        Chain identifier to assign to all atoms (e.g. 'A').
    to_angstrom : bool, default True
        If True, convert coordinates and box from nm to Å (×10).
        If False, keep everything in nm.

    Returns
    -------
    atoms : list[Atom]
        List of Atom objects with positions (and velocities if present).
    cell : Cell
        Simulation cell with hmat/gmat and (a,b,c,alpha,beta,gamma) set.
    """
    atoms: List[Atom] = []
    scale = 10.0 if to_angstrom else 1.0

    with open(fname, "r") as fi:
        # ---- Header ----
        title = fi.readline()  # currently unused

        n_atom_line = fi.readline()
        if not n_atom_line:
            raise ValueError(f"{fname!r}: unexpected EOF when reading number of atoms.")

        try:
            n_atom = int(n_atom_line.strip())
        except ValueError as exc:
            raise ValueError(
                f"{fname!r}: invalid number-of-atoms line: {n_atom_line!r}"
            ) from exc

        # ---- Atom lines ----
        current_res_idx = 0
        last_res_num = None

        for i in range(n_atom):
            line = fi.readline()
            if not line:
                raise ValueError(
                    f"{fname!r}: unexpected EOF while reading atom {i+1}/{n_atom}."
                )

            # Must at least contain up to z (col 44)
            if len(line) < 44:
                raise ValueError(
                    f"{fname!r}: atom line {i+1} too short ({len(line)} chars): {line!r}"
                )

            # Fixed-width fields (standard .gro layout)
            #  1–5   : residue number
            #  6–10  : residue name
            # 11–15  : atom name
            # 16–20  : atom number
            # 21–28  : x (nm)
            # 29–36  : y (nm)
            # 37–44  : z (nm)
            # 45–52  : vx (nm/ps, optional)
            # 53–60  : vy (nm/ps, optional)
            # 61–68  : vz (nm/ps, optional)

            res_num = int(line[0:5])
            res_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_num = int(line[15:20])

            # residue index: compact 1..Nres, increments when residue number changes
            if i == 0:
                current_res_idx = 1
                last_res_num = res_num
            elif res_num != last_res_num:
                current_res_idx += 1
                last_res_num = res_num

            # coordinates in nm in the file
            x_nm = float(line[20:28])
            y_nm = float(line[28:36])
            z_nm = float(line[36:44])

            atom = Atom()

            # Atom attributes
            atom.resIdx = current_res_idx
            atom.resName = res_name
            atom.name = atom_name
            atom.idx = i + 1
            atom.x = x_nm * scale
            atom.y = y_nm * scale
            atom.z = z_nm * scale
            atom.chain = chain

            # velocities (optional); stored in nm/ps
            if len(line) >= 68:
                vx = float(line[44:52])
                vy = float(line[52:60])
                vz = float(line[60:68])
            else:
                vx = vy = vz = 0.0

            atom.vx = vx
            atom.vy = vy
            atom.vz = vz

            atoms.append(atom)

        # ---- Box line ----
        box_line = fi.readline()
        if not box_line:
            raise ValueError(f"{fname!r}: unexpected EOF when reading box line.")

        parts = box_line.split()
        if len(parts) not in (3, 9):
            raise ValueError(
                f"{fname!r}: box line must have 3 or 9 floats, got {len(parts)}: {parts!r}"
            )

        # convert nm → Å if requested
        box_vals = [float(v) * scale for v in parts]

        # GROMACS triclinic order: xx yy zz xy xz yx yz zx zy
        if len(box_vals) == 3:
            # Orthorhombic: just lengths along x, y, z
            xx, yy, zz = box_vals
            hmat = np.array(
                [
                    [xx, 0.0, 0.0],
                    [0.0, yy, 0.0],
                    [0.0, 0.0, zz],
                ],
                dtype=float,
            )
        else:
            xx, yy, zz, xy, xz, yx, yz, zx, zy = box_vals

            # reconstruct the three cell vectors
            a_vec = (xx, xy, xz)
            b_vec = (yx, yy, yz)
            c_vec = (zx, zy, zz)

            hmat = np.array(
                [
                    [a_vec[0], b_vec[0], c_vec[0]],
                    [a_vec[1], b_vec[1], c_vec[1]],
                    [a_vec[2], b_vec[2], c_vec[2]],
                ],
                dtype=float,
            )

    # Cell class handle (a,b,c,angles,gmat) in __post_init__
    cell = Cell(hmat=hmat)

    return atoms, cell



