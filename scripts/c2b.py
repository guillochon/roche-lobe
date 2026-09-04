import mpmath as mp
mp.mp.dps = 60
pi=mp.pi
def V(c):
    kap=(2*c/3)**3; k12=1/mp.sqrt(kap)
    def f(ph):
        u=mp.asin(k12*mp.sin(ph))/3
        return mp.sin(ph)/(3-4*mp.sin(u)**2)**3
    return 32*pi/(3*kap)*mp.quad(f,[0,pi/4,pi/2-mp.mpf('1e-2'),pi/2-mp.mpf('1e-4'),pi/2-mp.mpf('1e-6'),pi/2])
R0=(3*mp.sqrt(3)-4+3*mp.log(6-3*mp.sqrt(3)))**(mp.mpf(1)/3); V0=4*pi/3*R0**3
print("V(3/2) check:", mp.nstr(V(mp.mpf(3)/2)-V0,5))
alpha=2*pi/mp.sqrt(3)
# model: V-V0 = alpha d ln d + beta d + g2 d^2 ln d + h2 d^2 + g3 d^3 ln d + h3 d^3 ; solve for beta,g2,h2,g3,h3 from 5 deltas after subtracting alpha term
ds=[mp.mpf(10)**(-k) for k in (6,7,8,9,10)]
ys=[(V(mp.mpf(3)/2+d)-V0-alpha*d*mp.log(d)) for d in ds]
A=mp.matrix([[d, d**2*mp.log(d), d**2, d**3*mp.log(d), d**3] for d in ds]); sol=mp.lu_solve(A, mp.matrix(ys))
beta=sol[0]; print("beta (5-term Richardson) =", mp.nstr(beta,25))
ds2=[mp.mpf(10)**(-k) for k in (8,9,10,11,12,13)]
ys2=[(V(mp.mpf(3)/2+d)-V0-alpha*d*mp.log(d)) for d in ds2]
A2=mp.matrix([[d, d**2*mp.log(d), d**2, d**3*mp.log(d), d**3, d**4*mp.log(d)] for d in ds2]); sol2=mp.lu_solve(A2, mp.matrix(ys2))
print("beta (6-term, smaller d)  =", mp.nstr(sol2[0],25))
beta=sol2[0]
D0=mp.mpf(9)/2*mp.mpf(3)**(-mp.mpf(2)/3)
c2=-D0*(alpha*mp.log(D0)+beta)/(4*pi*R0**2)
print("c2 =", mp.nstr(c2,20))
# try to identify beta / (pi/sqrt3)
x=beta/(pi/mp.sqrt(3)); print("beta/(pi/sqrt3) =", mp.nstr(x,20))
L2,L3,Lt=mp.log(2),mp.log(3),mp.log(2+mp.sqrt(3))
for basis,names in [([1,L2,L3,Lt],"1,ln2,ln3,ln(2+r3)"),([1,L2,L3,Lt,mp.sqrt(3),mp.sqrt(3)*L2,mp.sqrt(3)*L3,mp.sqrt(3)*Lt],"+sqrt3 mixes"),([1,L2,L3,Lt,mp.pi,mp.pi*mp.sqrt(3)],"+pi")]:
    rel=mp.pslq([x]+basis, maxcoeff=2000, maxsteps=10**6, tol=mp.mpf(10)**(-18))
    print(names, "->", rel)
