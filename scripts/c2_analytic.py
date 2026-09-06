"""Analytic derivation of beta (hence c2) for the rotating-star volume function V(c).

Cylindrical coordinates: phi = 1/r + w^2/2 >= c, w = cylindrical radius, x = w^2/2.
Bounded component: z^2 = 1/(c-x)^2 - 2x  for 0 <= x <= x_-(c), x_-(c) the smallest
positive root of g(x;c) = 1 - 2x(c-x)^2.
   V(c)   = 4 pi int_0^{x_-} sqrt(g)/(c-x) dx
   dV/dc  = -4 pi int_0^{x_-} dx / [ (c-x)^2 sqrt(g) ]          (exact)
Near c=3/2: g has a double root at x=1/2 that splits into x_-, x_+ ~ 1/2 -/+ sqrt(2 delta/3).
Subtraction: with f_c(x) = 1/((c-x)^2 sqrt(2(x3-x))),
   I = int_0^{x_-} [f_c(x)-f_c(x_m)]/sqrt((x_- - x)(x_+ - x)) dx + f_c(x_m) ln[(sqrt x_- + sqrt x_+)^2/(x_+ - x_-)]
   -> J + (1/sqrt3)(-1/2 ln delta + 1/2 ln(3/2)),  J = int_0^{1/2} [f(x)-f(1/2)]/(1/2-x) dx  (elementary)
so dV/dc = (2 pi/sqrt3) ln delta - 4 pi J - (2 pi/sqrt3) ln(3/2) + o(1)  and
   V = V0 + alpha delta ln delta + beta delta,   alpha = 2 pi/sqrt3,   beta = -4 pi J - alpha ln(3/2) - alpha.
With t = sqrt(2(2-x)) the integrand of J is -2/(sqrt3 (t+sqrt3)) - 2/(t^2-1) - 4/(t^2-1)^2 on [sqrt3, 2], giving
   J = 4/3 - sqrt3 + (2/sqrt3) ln( 2 sqrt3/(2+sqrt3) ).
"""
import mpmath as mp, sympy as sp
mp.mp.dps = 40
pi = mp.pi; s3 = mp.sqrt(3)

# ---- 0. symbolic J -------------------------------------------------------
t = sp.symbols('t', positive=True)
f_of_t = 4/((t**2-1)**2*t)                    # f(x(t)),  x = 2 - t^2/2
integrand = (f_of_t - 1/sp.sqrt(3))*2*t/(t**2-3)  # dx/(1/2-x) = -t dt/((t^2-3)/2) -> 2t dt/(t^2-3) with limits swapped
reduced = -2/(sp.sqrt(3)*(t+sp.sqrt(3))) - 2/(t**2-1) - 4/(t**2-1)**2
print("integrand == reduced form:", sp.simplify(integrand - reduced) == 0)
J_sym = sp.integrate(reduced, (t, sp.sqrt(3), 2))
J_closed = sp.Rational(4,3) - sp.sqrt(3) + 2/sp.sqrt(3)*sp.log(2*sp.sqrt(3)/(2+sp.sqrt(3)))
print("J (sympy)  =", sp.N(J_sym, 30)); print("J (closed) =", sp.N(J_closed, 30))
print("J symbolic difference simplifies to:", sp.simplify(sp.expand_log(J_sym - J_closed, force=True)))
alpha_s = 2*sp.pi/sp.sqrt(3)
beta_closed = -4*sp.pi*J_closed - alpha_s*sp.log(sp.Rational(3,2)) - alpha_s
beta_pslq = sp.pi/sp.sqrt(3)*(10 + 8*sp.log(2+sp.sqrt(3)) - 6*sp.log(6)) - 16*sp.pi/3
print("beta derived =", sp.N(beta_closed, 30)); print("beta (paper) =", sp.N(beta_pslq, 30))
print("derived - paper simplifies to:", sp.simplify(sp.expand_log(beta_closed - beta_pslq, force=True)))

# ---- 1. numerical checks of the representation -------------------------
def g(x, c): return 1 - 2*x*(c-x)**2
def roots(c):
    if c == mp.mpf(3)/2: return [mp.mpf(1)/2, mp.mpf(1)/2, mp.mpf(2)]
    r = mp.polyroots([-2, 4*c, -2*c**2, 1], maxsteps=200, extraprec=60)   # -2x^3 + 4c x^2 - 2c^2 x + 1
    r = sorted([mp.re(z) for z in r]); return r
def V_cyl(c):
    xm, xp, x3 = roots(c)
    return 4*pi*mp.quad(lambda x: mp.sqrt(abs(g(x,c)))/(c-x), [0, xm])
def V_paper(c):
    kap=(2*c/3)**3; k12=1/mp.sqrt(kap)
    f=lambda ph: mp.sin(ph)/(3-4*mp.sin(mp.asin(k12*mp.sin(ph))/3)**2)**3
    return 32*pi/(3*kap)*mp.quad(f,[0,pi/4,pi/2-mp.mpf('1e-2'),pi/2-mp.mpf('1e-4'),pi/2-mp.mpf('1e-6'),pi/2])
def dVdc(c):
    xm, xp, x3 = roots(c)
    return -4*pi*mp.quad(lambda x: 1/((c-x)**2*mp.sqrt(abs(g(x,c)))), [0, xm/2, xm])
for c in (mp.mpf('1.6'), mp.mpf(2), mp.mpf(3)):
    print(f"c={c}: V_cyl={mp.nstr(V_cyl(c),20)}  V_eq19={mp.nstr(V_paper(c),20)}")
c=mp.mpf(2); h=mp.mpf('1e-8')
print("dV/dc analytic vs finite difference at c=2:", mp.nstr(dVdc(c),15), mp.nstr((V_cyl(c+h)-V_cyl(c-h))/(2*h),15))
R0=(3*s3-4+3*mp.log(6-3*s3))**(mp.mpf(1)/3); V0=4*pi/3*R0**3
print("V_cyl(3/2) - V0 =", mp.nstr(V_cyl(mp.mpf(3)/2)-V0, 5))

alpha = 2*pi/s3
beta = mp.mpf(sp.N(beta_closed, 40))
print("\nbeta =", mp.nstr(beta, 30))
print("check dV/dc - alpha ln(delta) - alpha  ->  beta :")
for k in (4, 6, 8, 10):
    d = mp.mpf(10)**(-k); c = mp.mpf(3)/2 + d
    print(f"  delta=1e-{k}: {mp.nstr(dVdc(c) - alpha*mp.log(d) - alpha, 20)}")
print("check [V - V0 - alpha d ln d]/d -> beta :")
for k in (4, 6, 8):
    d = mp.mpf(10)**(-k); c = mp.mpf(3)/2 + d
    print(f"  delta=1e-{k}: {mp.nstr((V_cyl(c)-V0-alpha*d*mp.log(d))/d, 20)}")
D0=mp.mpf(9)/2*mp.mpf(3)**(-mp.mpf(2)/3)
c2=-D0*(alpha*mp.log(D0)+beta)/(4*pi*R0**2)
print("\nc2 =", mp.nstr(c2, 20), "  (paper: -0.983192566)")
