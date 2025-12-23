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



def writeGro(
    fname: str,
    atoms: List[Atom],
    cell: Cell,
    is_scaled: bool,
    title: str = "File written by cryst.py from P.-A. Cazade",
) -> None:
    """
    Write a GROMACS .gro file from a list of atoms and a Cell.

    Parameters
    ----------
    fname : str
        Output .gro filename.
    atoms : list[Atom]
        Atoms with coordinates. If `is_scaled` is True, their (x,y,z) are
        fractional coordinates; otherwise they are Cartesian (Å).
    cell : Cell
        Simulation cell. Must have a valid `hmat` in Å.
    is_scaled : bool
        If True, treat atom.x/y/z as fractional (u,v,w) and convert using
        hmat; if False, treat them as Cartesian coordinates in Å.
    title : str, optional
        Title line to write at the top of the .gro file.
    """
    # hmat is 3x3 with columns = a, b, c (in Å)
    hmat = np.asarray(cell.hmat, dtype=float)
    if hmat.shape != (3, 3):
        raise ValueError("Cell.hmat must be a 3x3 matrix.")

    with open(fname, "w") as fo:
        # Header
        fo.write(f"{title}\n")
        fo.write(f"{len(atoms):d}\n")

        # Atom lines
        for at in atoms:
            at.inferAtom()

            # GROMACS supports up to 5 digits for residue & atom index
            res_idx = at.resIdx % 100000
            atm_idx = at.idx % 100000

            res_name = at.resName
            atm_name = at.name

            if is_scaled:
                # Treat (x,y,z) as fractional (u,v,w)
                uvw = np.array([at.x, at.y, at.z], dtype=float)
                # Cartesian in Å: r = H * s   (H columns = a,b,c)
                xyz = hmat @ uvw
            else:
                # Already Cartesian in Å
                xyz = np.array([at.x, at.y, at.z], dtype=float)

            # Convert Å → nm for .gro
            x_nm, y_nm, z_nm = xyz * 0.1

            # Standard .gro fixed-width fields
            #  1–5   : residue number
            #  6–10  : residue name
            # 11–15  : atom name
            # 16–20  : atom number
            # 21–28  : x (nm)
            # 29–36  : y (nm)
            # 37–44  : z (nm)
            fo.write(
                "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                % (res_idx, res_name, atm_name, atm_idx, x_nm, y_nm, z_nm)
            )

        # ---- Box line ----
        # Extract a, b, c as column vectors from hmat (Å)
        a_vec = hmat[:, 0]
        b_vec = hmat[:, 1]
        c_vec = hmat[:, 2]

        # GROMACS triclinic order (all in nm):
        # xx yy zz xy xz yx yz zx zy
        xx = a_vec[0]
        yy = b_vec[1]
        zz = c_vec[2]
        xy = a_vec[1]
        xz = a_vec[2]
        yx = b_vec[0]
        yz = b_vec[2]
        zx = c_vec[0]
        zy = c_vec[1]

        # Convert Å → nm
        xx *= 0.1
        yy *= 0.1
        zz *= 0.1
        xy *= 0.1
        xz *= 0.1
        yx *= 0.1
        yz *= 0.1
        zx *= 0.1
        zy *= 0.1

        # Check if we need triclinic (any tilt non-zero)
        tilt_max = max(abs(xy), abs(xz), abs(yx), abs(yz), abs(zx), abs(zy))

        if tilt_max > 1e-6:
            # triclinic: 9 values
            fo.write(
                "%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n"
                % (xx, yy, zz, xy, xz, yx, yz, zx, zy)
            )
        else:
            # orthorhombic: only box lengths
            fo.write("%10.5f%10.5f%10.5f\n" % (xx, yy, zz))