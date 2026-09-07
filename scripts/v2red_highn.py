"""High-n reduced-model volumes to extract V_2^red independently of the finite-part prescription."""
import time
import numpy as np
from scipy.optimize import brentq
from paths import PAPER

LD = np.longdouble


def panels(edges, n):
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = g.astype(LD), w.astype(LD)
    X, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo, hi = LD(lo), LD(hi)
        X.append((hi - lo) / 2 * g + (hi + lo) / 2)
        W.append((hi - lo) / 2 * w)
    return np.concatenate(X), np.concatenate(W)


def V_red(eps, n=200):
    eps = LD(eps)
    r1 = LD(brentq(lambda r: r**3 + float(eps) * r**4 - 1, 0.5, 1.0, xtol=1e-16))
    for _ in range(8):
        r1 -= (r1**3 + eps * r1**4 - 1) / (3 * r1**2 + 4 * eps * r1**3)
    Uc = 3 / r1 + LD(1.5) * r1**2 + eps * r1**3
    se = float(np.sqrt(eps)) if eps > 0 else 1e-3
    ph_edges = [0.0, 0.3, np.pi / 2, np.pi - 0.3]
    ph_edges += [np.pi - k * se for k in (64, 16, 4, 1) if k * se < 0.3]
    ph_edges += [np.pi]
    th_edges = [0.0, np.pi / 2 - 0.3]
    th_edges += [np.pi / 2 - k * se for k in (64, 16, 4, 1) if k * se < 0.3]
    th_edges += [np.pi / 2]
    th, thw = panels(th_edges, n)
    ph, phw = panels(ph_edges, n)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(thw * np.sin(th), phw)
    nx = (np.sin(TH) * np.cos(PH)).ravel()
    nz = np.cos(TH).ravel()
    A = (3 * nx**2 - nz**2) / 2
    B = (5 * nx**3 - 3 * nx) / 2

    def U(r):
        return 3 / r + A * r**2 + eps * B * r**3

    Rg = np.linspace(LD(0.3), r1, 320)
    Z = 3 / Rg[None, :] + A[:, None] * Rg[None, :] ** 2 + eps * B[:, None] * Rg[None, :] ** 3
    below = Z < Uc
    anyb = below.any(axis=1)
    idx = np.maximum(np.where(anyb, np.argmax(below, axis=1), 319), 1)
    lo, hi = Rg[idx - 1].copy(), Rg[idx].copy()
    for _ in range(80):
        m = (lo + hi) / 2
        up = U(m) >= Uc
        lo = np.where(up, m, lo)
        hi = np.where(up, hi, m)
    R = np.where(anyb, (lo + hi) / 2, r1).reshape(TH.shape)
    return LD(4) / 3 * np.sum(W * R**3)


if __name__ == "__main__":
    VH = LD("1.508549856131113140040991")
    V1 = LD("-1.7280570857676475595")
    V32 = 8 * np.pi / (9 * np.sqrt(3))
    target = -1.260674456
    print("eps=0:", V_red(0.0), "VH", VH, "diff", V_red(0.0) - VH, flush=True)
    rows = []
    for e in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
        t0 = time.time()
        v = V_red(e, n=200)
        R = float(v - VH - V1 * e - V32 * e**1.5)
        S = R / e**2
        rows.append((e, float(v), S))
        print(
            f"{e:.0e} V={float(v):.18f} S={S:.10f} S-tgt={S - target:.6f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    np.savetxt(
        PAPER / "vred_n200.txt",
        np.array(rows),
        fmt="%.6e %.18f %.12f",
        header="eps V S",
    )
    print("wrote paper/vred_n200.txt")
    S = np.array([r[2] for r in rows])
    eps = np.array([r[0] for r in rows])
    print("pair two-term a + b sqrt(eps):")
    for i in range(len(eps) - 1):
        e1, e2 = eps[i], eps[i + 1]
        s1, s2 = S[i], S[i + 1]
        b = (s2 - s1) / (np.sqrt(e2) - np.sqrt(e1))
        a = s1 - b * np.sqrt(e1)
        print(f"  {e1:.0e}-{e2:.0e}: a={a:.10f}  b={b:.5f}  a-tgt={a - target:.8f}")
    from numpy.linalg import lstsq

    A = np.column_stack([np.ones_like(eps), np.sqrt(eps), eps, eps**1.5])
    coef, *_ = lstsq(A, S, rcond=None)
    print("4-term fit a =", coef[0], coef)
