import mpmath as mp
mp.mp.dps=25
def rho(nx,nz):
    m2=nx**2-nz**2/3
    if m2>0: m=mp.sqrt(m2); return 2/m*mp.sin(mp.asin(m)/3)
    if m2==0: return mp.mpf(2)/3
    m=mp.sqrt(-m2); return 2/m*mp.sinh(mp.asinh(m)/3)
def integrand(nx,nz):
    r=rho(nx,nz)
    if r>=1: return mp.mpf(0)
    U2=r**4*(35*nx**4-30*nx**2+3)/8
    return (U2-1)*r**4/(1-r)
def sph(f):
    inner=lambda t: mp.quad(lambda p: f(mp.sin(t)*mp.cos(p),mp.cos(t)),[0,mp.pi/6,mp.pi/2,5*mp.pi/6,mp.pi])
    return 4*mp.quad(lambda t: mp.sin(t)*inner(t),[0,mp.pi/3,mp.pi/2])
I=sph(integrand); V2U2=I/9
print("V2^(U2) = (1/9) oint (U2-1) rho^4/(1-rho) dOmega =", mp.nstr(V2U2,18))
# sanity: same machinery reproduces V1
V1=-sph(lambda nx,nz:(lambda r: r**4/(1-r) if r<1 else mp.mpf(0))(rho(nx,nz)))/9
print("V1 check =", mp.nstr(V1,18), " (expected -1.72805708577)")
