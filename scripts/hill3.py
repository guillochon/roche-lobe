import mpmath as mp
mp.mp.dps = 40

# Exact radius of the Hill lobe:  rho = (2/m) sin(arcsin(m)/3),  m^2 = n_xi^2 - n_zeta^2/3
# (analytically continued to sinh/arcsinh when m^2 < 0)
def rho_exact(nx, nz):
    m2 = nx**2 - nz**2/3
    if m2 > 0:
        m = mp.sqrt(m2);  return 2/m*mp.sin(mp.asin(m)/3)
    elif m2 == 0:
        return mp.mpf(2)/3
    else:
        m = mp.sqrt(-m2); return 2/m*mp.sinh(mp.asinh(m)/3)

# verify against the defining equation 3/rho + (3nx^2-nz^2)/2 rho^2 = 9/2
for (nx, nz) in [(0.3,0.2),(0.9,0.1),(0.0,1.0),(1.0,0.0),(0.1,0.9),(0.5,0.5)]:
    n = mp.sqrt(nx**2+nz**2); nx_, nz_ = mp.mpf(nx), mp.mpf(nz)
    r = rho_exact(nx_, nz_)
    res = 3/r + (3*nx_**2-nz_**2)/2*r**2 - mp.mpf(9)/2
    print("nx=%.2f nz=%.2f rho=%s residual=%s" % (nx, nz, mp.nstr(r,15), mp.nstr(res,4)))

# volume with the exact radius (no root finding) -> high accuracy
def inner(t):
    s, c = mp.sin(t), mp.cos(t)
    return mp.quad(lambda p: rho_exact(s*mp.cos(p), c)**3, [0, mp.pi/2, mp.pi])
V = mp.mpf(4)/3*mp.quad(lambda t: mp.sin(t)*inner(t), [0, mp.pi/3, mp.pi/2])
kap = (3*V/(4*mp.pi))**(mp.mpf(1)/3)
print("\nV_Hill =", mp.nstr(V, 25))
print("kappa  =", mp.nstr(kap, 25))
print("C = kappa*3^(-1/3) =", mp.nstr(kap/mp.cbrt(3), 25))
for basis in (None, ['pi'], ['pi','log(2)','log(3)','sqrt(3)'], ['pi','sqrt(2)','sqrt(3)','sqrt(5)']):
    print("identify V:", mp.identify(V, basis) if basis else mp.identify(V))
