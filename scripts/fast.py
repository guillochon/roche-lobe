import numpy as np
from scipy.optimize import brentq

def zeta(R, lam, gam, q):
    return (1.0/R + q*(1.0/np.sqrt(1.0 - 2.0*lam*R + R*R) - R*lam)
            + 0.5*(1.0+q)*R*R*(1.0 - gam*gam))

def xL1(q):
    f = lambda x: -1.0/x**2 + q/(1.0-x)**2 + (1.0+q)*x - q
    return brentq(f, 1e-15, 1.0-1e-15, xtol=1e-17, rtol=8.9e-16)

def Rsurf(lam, gam, q, zc, a, b, nscan=120, nbis=64, chunk=400000):
    """first crossing of zeta=zc on [a,b]; assumes zeta(a)>zc>zeta(b) generically"""
    out = np.empty_like(lam)
    Rg = np.linspace(a, b, nscan)
    for s in range(0, lam.size, chunk):
        e = min(s+chunk, lam.size)
        L, G = lam[s:e], gam[s:e]
        Z = zeta(Rg[None, :], L[:, None], G[:, None], q)
        below = Z < zc
        anyb = below.any(axis=1)
        idx = np.maximum(np.where(anyb, np.argmax(below, axis=1), nscan-1), 1)
        lo, hi = Rg[idx-1].copy(), Rg[idx].copy()
        for _ in range(nbis):
            m = 0.5*(lo+hi)
            up = zeta(m, L, G, q) >= zc
            lo = np.where(up, m, lo); hi = np.where(up, hi, m)
        out[s:e] = np.where(anyb, 0.5*(lo+hi), b)
    return out

def panels(edges, npts):
    """composite Gauss-Legendre nodes/weights over the given panel edges"""
    g, w = np.polynomial.legendre.leggauss(npts)
    X, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        X.append(0.5*(hi-lo)*g + 0.5*(hi+lo)); W.append(0.5*(hi-lo)*w)
    return np.concatenate(X), np.concatenate(W)

def volume(q, npts=240, kpinch=4.0):
    x1 = xL1(q); zc = zeta(x1, 1.0, 0.0, q)
    eps = min((q/3.0)**(1/3.0), 0.4)          # angular size of the pinch region
    d = min(kpinch*eps, 0.9*np.pi/2)
    th, thw = panels([0.0, np.pi/2 - d, np.pi/2], npts)
    ph, phw = panels([0.0, d, np.pi], npts)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    W = np.outer(thw*np.sin(th), phw)
    lam = (np.sin(TH)*np.cos(PH)).ravel(); gam = np.cos(TH).ravel()
    R = Rsurf(lam, gam, q, zc, 0.35, x1*(1-1e-14)).reshape(TH.shape)
    return 4.0/3.0*np.sum(W*R**3)

def Req(q, **kw):
    return (3*volume(q, **kw)/(4*np.pi))**(1/3)

if __name__ == "__main__":
    import sys
    R0 = 0.8148856767823898493630111
    print("resolution check (Req):")
    for q in [1e-2, 1e-4]:
        print(f"  q={q:.0e}", ["%.10f" % Req(q, npts=n) for n in (120, 240, 360)])
    sys.stdout.flush()
