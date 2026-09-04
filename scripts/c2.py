import mpmath as mp
mp.mp.dps = 40
pi = mp.pi
# axisymmetric potential zeta0 = 1/r + r^2 sin^2(th)/2 ; level c = 3/2 + delta
def r_of(th, c):
    s = mp.sin(th)
    if s == 0: return 1/c
    m = (3/(2*c))**mp.mpf(1.5)*s
    return 2/s*mp.sqrt(2*c/3)*mp.sin(mp.asin(m)/3)
def dVdc(c):
    f = lambda th: r_of(th,c)**4*mp.sin(th)/(2*c*r_of(th,c)-3)
    return 4*pi*mp.quad(f, [0, pi/3, pi/2-mp.mpf('1e-3'), pi/2-mp.mpf('1e-6'), pi/2])
alpha = 2*pi/mp.sqrt(3)
R0 = (3*mp.sqrt(3)-4+3*mp.log(6-3*mp.sqrt(3)))**(mp.mpf(1)/3)
D0 = mp.mpf(9)/2*mp.mpf(3)**(-mp.mpf(2)/3)
print("check c=3/2 volume via r_of:", mp.nstr(4*pi/3*mp.quad(lambda th: r_of(th,mp.mpf(1.5))**3*mp.sin(th),[0,pi/2]),20), " exact", mp.nstr(4*pi/3*R0**3,20))
print("\n delta        dV/ddelta         g = dV/dd - alpha ln d     (alpha+beta) estimate")
prev=None
for k in [4,6,8,10,12,14]:
    d = mp.mpf(10)**(-k); c = mp.mpf(3)/2 + d
    g = dVdc(c) - alpha*mp.log(d)
    print(f"1e-{k:<3}  {mp.nstr(dVdc(c),15):>18}  {mp.nstr(g,20)}")
    prev=g
beta = prev - alpha
c2 = -D0*(alpha*mp.log(D0)+beta)/(4*pi*R0**2)
print("\nalpha =", mp.nstr(alpha,15), " beta =", mp.nstr(beta,15))
print("c2 =", mp.nstr(c2,15), "   (fits gave -1.1 +/- 0.1)")
print("identify beta:", mp.identify(beta, ['pi','pi*sqrt(3)','sqrt(3)','log(2)','log(3)','pi*log(2)','pi*log(3)','pi*sqrt(3)*log(2)','pi*sqrt(3)*log(3)','pi*sqrt(3)*log(2-sqrt(3))']))
