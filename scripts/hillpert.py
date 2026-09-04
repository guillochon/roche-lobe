import numpy as np
from scipy.optimize import brentq
from fast import panels
# Perturbed Hill model: U = 3/rho + A rho^2 + eps*B rho^3,  A=(3nx^2-nz^2)/2, B=(5nx^3-3nx)/2
V0 = 1.50854985613111314; V1 = -1.7280570857676475595; V32 = 8*np.pi/(9*np.sqrt(3))

def volume(eps, npts=300):
    r1 = brentq(lambda r: r**3 + eps*r**4 - 1, 0.5, 1.0, xtol=1e-16)  # perturbed L1 distance
    Uc = 3/r1 + 1.5*r1**2 + eps*r1**3
    d = 0.25 if eps==0 else max(3*np.sqrt(eps), 0.05)
    th, thw = panels([0, np.pi/2-0.6, np.pi/2-d, np.pi/2], npts)
    ph, phw = panels([0, 0.6, np.pi-0.6, np.pi-d, np.pi], npts)
    TH, PH = np.meshgrid(th, ph, indexing='ij'); W = np.outer(thw*np.sin(th), phw)
    nx = (np.sin(TH)*np.cos(PH)).ravel(); nz = np.cos(TH).ravel()
    A = (3*nx**2 - nz**2)/2; B = (5*nx**3 - 3*nx)/2
    U = lambda r: 3/r + A*r**2 + eps*B*r**3
    Rg = np.linspace(0.3, r1, 150)
    Z = 3/Rg[None,:] + A[:,None]*Rg[None,:]**2 + eps*B[:,None]*Rg[None,:]**3
    below = Z < Uc; anyb = below.any(axis=1)
    idx = np.maximum(np.where(anyb, np.argmax(below,axis=1), 149), 1)
    lo, hi = Rg[idx-1].copy(), Rg[idx].copy()
    for _ in range(64):
        m = 0.5*(lo+hi); up = U(m) >= Uc
        lo = np.where(up, m, lo); hi = np.where(up, hi, m)
    R = np.where(anyb, 0.5*(lo+hi), r1).reshape(TH.shape)
    return 4/3*np.sum(W*R**3)

print("eps=0 check:", "%.12f"%volume(0.0), " V0 =", "%.12f"%V0)
print("\n eps        V(eps)          [V-V0-eps*V1]/eps^1.5   predicted 8pi/(9 sqrt3) = %.5f" % V32)
for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
    V = volume(eps)
    print(f"{eps:<8.0e}  {V:.12f}   {(V-V0-eps*V1)/eps**1.5:10.5f}")
