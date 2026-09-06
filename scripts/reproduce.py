"""Regenerate paper artifacts from source calculations.

Run from the repo root. Cached outputs in paper/ (analysis.json, the .dat
table, derivative_reference.txt) already reproduce the figures and tables.
The flags below rebuild those caches, or only the figures.

    python scripts/reproduce.py --versions
    python scripts/reproduce.py --figures          # seconds; reads caches
    python scripts/reproduce.py --quadratures      # V_H, V_1, reduced-model a_{3/2}
    python scripts/reproduce.py --closure          # beta/c_2 derivation, L2 parity, V_2^{(U_2)}, V_2^{red}, a_2
    python scripts/reproduce.py --galois           # L1 Galois groups
    python scripts/reproduce.py --analysis         # rebuild analysis.json (minutes)
    python scripts/reproduce.py --reference        # 73-point table, npts=300 (hours)
    python scripts/reproduce.py --derivative       # D(q_E) table (hours)

Nelder-Mead settings for the minimax of eqs. (27)--(28) are in analysis.py
(xatol=1e-11, fatol=1e-13, maxiter=40000, least-squares start). The
reference volumes use 80-bit longdouble and 70 bisection steps
(reference.py). Quadratures use mpmath dps=40 (hill3.py) and dps=30 (a1.py).
Closure of c_2 and a_2: c2_analytic.py (sympy + mpmath dps=40; c2b.py is the
dps=60 Richardson check), a2_symbolic.py (sympy), v2u2.py (dps=25),
fp.py/fp2.py (dps=28, exact subtraction constants c_-2 = 4E(1/4)/9, c_-1 = 0).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys

from paths import SCRIPTS

HERE = SCRIPTS


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(HERE / script), *(extra or [])]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def versions() -> None:
    import matplotlib
    import numpy
    import scipy

    print("python", sys.version.split()[0], platform.platform())
    print("numpy", numpy.__version__)
    print("scipy", scipy.__version__)
    print("matplotlib", matplotlib.__version__)
    try:
        import mpmath
        print("mpmath", mpmath.__version__)
    except ImportError:
        print("mpmath (not installed; needed for --quadratures)")
    try:
        import sympy
        print("sympy", sympy.__version__)
    except ImportError:
        print("sympy (not installed; needed for --galois)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--versions", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--quadratures", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--galois", action="store_true")
    p.add_argument("--analysis", action="store_true")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--derivative", action="store_true")
    args = p.parse_args()
    if not any(vars(args).values()):
        p.print_help()
        return
    if args.versions:
        versions()
    if args.galois:
        run("checks.py")
    if args.quadratures:
        run("hill3.py")
        run("a1.py")
        run("hillpert.py")
        run("limit_q0.py")
    if args.closure:
        run("c2_analytic.py")
        run("c2b.py")
        run("a2_symbolic.py")
        run("v2u2.py")
        run("fp.py", ["red"])
        run("fp.py", ["shift"])
        run("fp2.py")
    if args.reference:
        run("reference.py", ["table"])
    if args.derivative:
        run("make_derivative_reference.py")
    if args.analysis:
        run("analysis.py")
    if args.figures:
        run("figures.py")
        run("derivative_analysis.py")


if __name__ == "__main__":
    main()
