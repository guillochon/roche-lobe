import mpmath as mp
mp.mp.dps = 30
# Hill lobe, exact radius rho(n) on U0 = 3/rho + (3 xi^2 - chi^2)/2 = 9/2
def rho(nx, nz):
    m2 = nx**2 - nz**2/3
    if m2 > 0:  m = mp.sqrt(m2);  return 2/m*mp.sin(mp.asin(m)/3)
    if m2 == 0: return mp.mpf(2)/3
    m = mp.sqrt(-m2); return 2/m*mp.sinh(mp.asinh(m)/3)

# V0 = (1/3) oint rho^3 dOmega ;  dV/dc = oint rho^2 (d rho/dc) dOmega with d rho/dc = rho^2/(9(rho-1))
def sph(f):
    inner = lambda t: mp.quad(lambda p: f(mp.sin(t)*mp.cos(p), mp.cos(t)), [0, mp.pi/2, mp.pi])
    return 4*mp.quad(lambda t: mp.sin(t)*inner(t), [0, mp.pi/3, mp.pi/2])
V0 = sph(lambda nx,nz: rho(nx,nz)**3)/3
I  = sph(lambda nx,nz: (lambda r: r**4/(1-r) if r<1 else mp.mpf(0))(rho(nx,nz)))
V1 = -I/9
a1 = 3**(-mp.mpf(1)/3)*V1/(3*V0)
print("V0 =", mp.nstr(V0,20)); print("I  =", mp.nstr(I,20)); print("V1 = dV/dc =", mp.nstr(V1,20))
print("a1 = 3^(-1/3) V1/(3 V0) =", mp.nstr(a1,20), "   (-1/4 = -0.25)")
print("V1/V0 =", mp.nstr(V1/V0,20), "   I/V0 =", mp.nstr(I/V0,20))
print("identify a1:", mp.identify(a1), " identify I/V0:", mp.identify(I/V0, ['pi','sqrt(3)','log(2)','log(3)']))
