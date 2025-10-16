# === Refactor for crystools.py: Read Write ITP file - OOP methods ===
from __future__ import annotations
import re
from dataclasses import dataclass, field  # only needed here if you keep local Atom/Topol; otherwise remove
from typing import List, Optional
from core.topology import Topology
from core.atom import Atom


# --- helpers (unchanged) ---
_section_re = re.compile(r"\[\s*([A-Za-z_]+)\s*\]", re.IGNORECASE)


def _strip_comment(line: str) -> str:
    line = line.split(';', 1)[0]
    if line.lstrip().startswith('#') and not line.lstrip().lower().startswith('#include'):
        return ""
    return line


# def _ints(words, n):
#     if len(words) < n:
#         raise ValueError(f"Expected {n} integers, got {len(words)}: {words}")
#     return [int(words[i]) for i in range(n)]


class ITP:
    """
    Reader for GROMACS .itp molecule templates.

    Usage (mutating a provided list):
        tops: list[Topol] = []
        ITP.readItp("molecule.itp", tops)
    """


    @staticmethod
    def itpReader(fname: str, itp: List[Topology]) -> None:
        """
        Parse a GROMACS .itp molecule template and append a Topology() to 'itp'.
        Handles: moleculetype (incl. nrexcl), atoms, pairs, bonds, angles, dihedrals, impropers, cmap.
        """
        current_section: Optional[str] = None
        topology: Optional[Topology] = None
        dihedrals_block_count  = 0  # fallback for "second [ dihedrals ] means impropers"

        with open(fname, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = _strip_comment(raw).strip()
                if not line:
                    continue  # blank/comment

                # Section header?
                m = _section_re.match(line)
                if m:
                    section = m.group(1).lower()  # e.g. 'atoms', 'pairs', 'moleculetype'
                    current_section = section

                    # On a new [ moleculetype ] start a fresh template
                    if section == "moleculetype":
                        topology = Topology()
                        itp.append(topology)
                        dihedrals_block_count = 0
                    elif section == "dihedrals": # to keep track of how many times we are getting dihedrals to refer to improper dihedrals
                        dihedrals_block_count += 1  # <-- count blocks, not lines
                    continue

                # If we don't have a section yet, skip
                if current_section is None or topology is None:
                    continue

                # Tokenize for data lines
                words = line.split()

                # --- per-section handlers ---
                if current_section == "moleculetype":
                    # Format: molName  nrexcl
                    topology.molName = words[0]
                    if len(words) > 1:
                        try:
                            topology.nrexcl = int(words[1])
                        except ValueError:
                            topology.nrexcl = None
                    continue

                if current_section == "atoms":
                    # Expected: nr type resnr res name cgnr charge mass
                    a = Atom()
                    a.idx     = int(words[0])
                    a.aType   = words[1]
                    a.resIdx  = int(words[2])
                    a.resName = words[3]
                    a.name    = words[4]
                    a.q       = float(words[6])
                    a.m = float(words[7])
                    topology.atoms.append(a)
                    continue

                if current_section == "pairs":
                    # i j funct
                    i, j, funct = [int(words[i]) for i in range(3)]
                    topology.pairs.append([i, j, funct])
                    continue

                if current_section == "bonds":
                    # i j funct
                    i, j, funct = [int(words[i]) for i in range(3)]
                    topology.bonds.append([i, j, funct])
                    continue

                if current_section == "angles":
                    # i j k funct
                    i, j, k, funct = [int(words[i]) for i in range(4)]
                    topology.angles.append([i, j, k, funct])
                    continue

                if current_section == "dihedrals":
                    i, j, k, l, funct = [int(words[i]) for i in range(5)]
                    # Some topologies misuse a second [ dihedrals ] block for impropers
                    if dihedrals_block_count>=2:
                        topology.impropers.append([i, j, k, l, funct])
                    else:
                        topology.dihedrals.append([i, j, k, l, funct])
                    continue

                if current_section == "cmap":
                    # i j k l m funct
                    i, j, k, l, m, funct = [int(words[i]) for i in range(6)]
                    topology.cmap.append([i, j, k, l, m, funct])
                    continue


# Optional: a functional wrapper
def readItp(fname: str) -> List[Topology]:
    tops: List[Topology] = []
    ITP.readItp(fname, tops)
    return tops