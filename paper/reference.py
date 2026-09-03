"""High-accuracy reference Roche-lobe volumes.
Units G=M1=a=1, q = M2/M1 (companion / lobe-star), omega^2 = 1+q.  Output uses q_E = M1/M2 = 1/q.
Extended precision (np.longdouble, 80-bit) throughout the root finding; composite Gauss-Legendre in angle."""
import numpy as np, time, sys
from scipy.optimize import brentq
LD = np.longdouble

def zeta(R, lam, gam, q):
    return (1/R + q*(1/np.sqrt(1 - 2*lam*R + R*R) - R*lam) + (1+q)/2*R*R*(1 - gam*gam))

def xL1(q):
    # double-precision bracket, then polish in longdouble with Newton
    f = lambda x: -1/x**2 + q/(1-x)**2 + (1+q)*x - q
    x = LD(brentq(f, 1e-15, 1-1e-15, xtol=1e-17, rtol=8.9e-16)); qL = LD(q)
    for _ in range(6):
        F = -1/x**2 + qL/(1-x)**2 + (1+qL)*x - qL
        dF = 2/x**3 + 2*qL/(1-x)**3 + (1+qL)
        x -= F/dF
    return x

def panels(edges, n):
    g, w = np.polynomial.legendre.leggauss(n)
    g = g.astype(LD); w = w.astype(LD)
    X, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo, hi = LD(lo), LD(hi)
        X.append((hi-lo)/2*g + (hi+lo)/2); W.append((hi-lo)/2*w)
    return np.concatenate(X), np.concatenate(W)

def solve_dirs(lam, gam, q, zc, a, b, nscan=160, nbis=70, chunk=30000):
    out = np.empty(lam.size, dtype=LD)
    Rg = np.linspace(a, b, nscan, dtype=LD)
    for s in range(0, lam.size, chunk):
        e = min(s+chunk, lam.size); L, G = lam[s:e], gam[s:e]
        Z = zeta(Rg[None, :], L[:, None], G[:, None], q)
        below = Z < zc; anyb = below.any(axis=1)
        idx = np.maximum(np.where(anyb, np.argmax(below, axis=1), nscan-1), 1)
        lo, hi = Rg[idx-1].copy(), Rg[idx].copy()
        for _ in range(nbis):
            m = (lo+hi)/2; up = zeta(m, L, G, q) >= zc
            lo = np.where(up, m, lo); hi = np.where(up, hi, m)
        out[s:e] = np.where(anyb, (lo+hi)/2, b)
    return out

def grad_zeta(X, Y, Z, q):
    r1 = np.sqrt(X*X+Y*Y+Z*Z); r2 = np.sqrt((X-1)**2+Y*Y+Z*Z)
    return (-X/r1**3 - q*(X-1)/r2**3 - q + (1+q)*X,
            -Y/r1**3 - q*Y/r2**3 + (1+q)*Y,
            -Z/r1**3 - q*Z/r2**3)

def lobe(q, npts):
    """returns V (direct), V (von Zeipel surface identity), x1, and the R(theta,phi) grid"""
    q = LD(q); x1 = xL1(float(q)); zc = zeta(x1, LD(1), LD(0), q)
    eps = min(float(q/3)**(1/3), 0.4); d = min(4*eps, 0.9*np.pi/2)
    th, thw = panels([0, np.pi/2-d, np.pi/2], npts)
    ph, phw = panels([0, d, np.pi], npts)
    TH, PH = np.meshgrid(th, ph, indexing='ij'); W = np.outer(thw*np.sin(th), phw)
    lam = (np.sin(TH)*np.cos(PH)).ravel(); gam = np.cos(TH).ravel()
    R = solve_dirs(lam, gam, q, zc, LD(0.3)*x1, x1*(1-LD(1e-17))).reshape(TH.shape)
    V = LD(4)/3*np.sum(W*R**3)
    # independent functional: surface integral of |grad zeta|  ->  V = (4 pi - Sigma)/(2(1+q))
    mu = np.sin(TH)*np.sin(PH)
    X = R*lam.reshape(TH.shape); Y = R*mu; Zc = R*gam.reshape(TH.shape)
    gx, gy, gz = grad_zeta(X, Y, Zc, q)
    g2 = gx*gx+gy*gy+gz*gz; zr = (gx*X+gy*Y+gz*Zc)/R
    Sigma = 4*np.sum(W*g2*R*R/(-zr))
    Vz = (4*LD(np.pi) - Sigma)/(2*(1+q))
    return V, Vz, x1, R, TH, PH

Req = lambda V: (3*V/(4*LD(np.pi)))**(LD(1)/3)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "converge":
        print("# convergence ladder: Req(npts) and |diff| to next level; also von Zeipel check")
        for qE in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10., 1e2, 1e4, 1e6]:
            vals = []
            for n in (100, 200, 400):
                V, Vz, x1, *_ = lobe(1/qE, n); vals.append((Req(V), Req(Vz)))
            r = [v[0] for v in vals]
            print(f"qE={qE:8.0e}  R(100)={r[0]:.18f}  d(100-200)={abs(r[0]-r[1]):.1e}  d(200-400)={abs(r[1]-r[2]):.1e}  |R-Rz|(400)={abs(vals[2][0]-vals[2][1]):.1e}", flush=True)
    elif mode == "table":
        lq = np.linspace(-6, 6, 73); t0 = time.time()
        with open("reference_table.txt", "w") as f:
            f.write("# q_E=M1/M2   R_L/a (extended precision, npts=300)   R_L/a via von Zeipel identity   x_L1/a\n")
            for l in lq:
                qE = 10.0**l; V, Vz, x1, *_ = lobe(1/qE, 300)
                f.write(f"{qE:.6e}  {float(Req(V)):.16f}  {float(Req(Vz)):.16f}  {float(x1):.16f}\n"); f.flush()
        print("table done", time.time()-t0)
