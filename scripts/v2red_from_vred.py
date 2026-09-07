"""Extract V_2^red from the reduced-model volumes in paper/vred.txt (n=90)
and paper/vred_n200.txt (n=200).

Richardson extrapolation of S(eps) in powers of sqrt(eps) over
1e-3 >= eps >= 1e-5 supports four digits of the finite-part value
-1.260674456, not six. Cancellation in S grows below eps ~ 2e-6.
"""
import numpy as np
from numpy.linalg import lstsq
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
VH = 1.50854985613111314
V1 = -1.7280570857676475595
V32 = 8 * np.pi / (9 * np.sqrt(3))
TARGET = -1.260674456


def remainder(eps, V):
    return (V - VH - V1 * eps - V32 * eps**1.5) / eps**2


def report(path, eps, S):
    print(f"==== {path.name} ====")
    print(f"{'eps':>10} {'S':>14} {'S-target':>12}")
    for e, s in zip(eps, S):
        print(f"  {e:8.0e} {s:14.8f} {s - TARGET:12.6f}")
    mask = (eps <= 1e-3) & (eps >= 1e-5)
    e2, s2 = eps[mask], S[mask]
    x = np.sqrt(e2)
    print("Richardson on 1e-3 >= eps >= 1e-5:")
    for deg in (1, 2, 3):
        A = np.vander(x, deg + 1, increasing=True)
        coef, *_ = lstsq(A, s2, rcond=None)
        print(f"  poly deg {deg}: a={coef[0]:.8f}")
    b = (s2[-1] - s2[-2]) / (np.sqrt(e2[-1]) - np.sqrt(e2[-2]))
    a = s2[-2] - b * np.sqrt(e2[-2])
    print(f"  pair small-end: a={a:.8f}")
    print()


def main():
    print(f"V32 = {V32:.12f}")
    print(f"target V2red = {TARGET}")
    print()
    e90, v90 = np.loadtxt(PAPER / "vred.txt").T[:2]
    report(PAPER / "vred.txt", e90, remainder(e90, v90))
    d200 = np.loadtxt(PAPER / "vred_n200.txt")
    report(PAPER / "vred_n200.txt", d200[:, 0], d200[:, 2])


if __name__ == "__main__":
    main()
