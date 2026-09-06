"""Symbolic structure of the reduced Hill model U0 + eps U1 = 9/2 + eps - eps^2/2 near L2, in ray coordinates.
rho = 1 - x, vartheta = angle from the -xi axis, phi = azimuth; Q = 3 cos^2 phi + 4 sin^2 phi = 3 + sin^2 phi.
Exact ray equation (times 2 rho):  E = 2[3 + A rho^3 + eps B rho^4 - (9/2 + eps - eps^2/2) rho] = 0,
A = (3 cos^2 th - sin^2 th sin^2 phi)/2,  B = -(5 cos^3 th - 3 cos th)/2.
(1) weighted expansion x = s X, th = s Th, eps = s^2: coefficient of s^k has parity (-1)^k in X  ->  X1 polynomial in Th.
(2) outer expansion x = xi0(th) + eps xi1 + eps^2 xi2: the eps^2 coefficient of (1/3)(1-x)^3 sin(th) has NO th^{-1} term.
"""
import sympy as sp
x, th, e, s, X, Th = sp.symbols('x theta epsilon s X Theta')
Q = sp.symbols('Q', positive=True)
rho = 1 - x
A = (3*sp.cos(th)**2 - sp.sin(th)**2*(Q-3))/2
B = -(5*sp.cos(th)**3 - 3*sp.cos(th))/2
E = sp.expand(2*(3 + A*rho**3 + e*B*rho**4 - (sp.Rational(9,2) + e - e**2/2)*rho))

# ---------- (1) inner (weighted) expansion ----------
N = 7
Es = sp.series(E.subs({x: s*X, th: s*Th, e: s**2}), s, 0, N+1).removeO()
Es = sp.expand(Es/s**2)
print("weight-2 (leading) part:", sp.factor(Es.coeff(s, 0)))
ok = True
for k in range(1, N-1):
    ck = sp.expand(Es.coeff(s, k))
    par = sp.simplify(ck.subs(X, -X) - (-1)**k*ck)
    print(f"  weight {k+2}: parity (-1)^{k} in X holds:", par == 0); ok &= (par == 0)
# perturbative solution X = X0 + s X1 + s^2 X2 + s^3 X3, X0 = R/3 with R^2 = 4 + Q Th^2
R = sp.symbols('R', positive=True)
def red(expr):  # reduce powers of R using R^2 = 4 + Q Th^2 (negative powers allowed)
    expr = sp.expand(expr)
    num, den = sp.fraction(sp.together(expr))
    out = 0
    for (k,), c in sp.Poly(sp.expand(num), R).terms():
        out += c * (4 + Q*Th**2)**(k//2) * R**(k % 2)
    return sp.expand(out)/den
X1, X2, X3 = sp.symbols('X1 X2 X3')
Xser = R/3 + s*X1 + s**2*X2 + s**3*X3
F = sp.expand(sp.series(Es.subs(X, Xser), s, 0, 4).removeO())
sol = {}
for k, sym in ((1, X1), (2, X2), (3, X3)):
    ck = red(F.coeff(s, k).subs(sol))
    val = sp.solve(ck, sym)[0]
    sol[sym] = sp.simplify(val)
    print(f"X{k} =", sp.factor(sp.simplify(sol[sym])))
print("X1 is R-free (polynomial in Theta):", not sp.simplify(sol[X1]).has(R))

# ---------- (2) outer expansion ----------
r = sp.symbols('r', positive=True)      # r = sqrt(Q)
M = 8                                     # order in theta
E0 = sp.series(E.subs(Q, r**2), th, 0, M+2).removeO()   # E as polynomial in x, e, series in th
E0 = sp.expand(E0)
def trunc(expr, n=M):   # truncate series in th at O(th^n), allowing negative powers
    expr = sp.expand(expr)
    return sum(expr.coeff(th, k)*th**k for k in range(-6, n))
# xi0 by Newton iteration in truncated series (e = 0)
Ez = E0.subs(e, 0)
xi0 = r*th/3
for it in range(6):
    Fv = trunc(Ez.subs(x, xi0)); Fp = trunc(sp.diff(Ez, x).subs(x, xi0))
    # divide series: Fv/Fp with Fp ~ 6 r th (1+...)
    lead = Fp.coeff(th, 1)
    inv = sp.series(1/(Fp/(lead*th)), th, 0, M).removeO()
    xi0 = trunc(sp.expand(xi0 - Fv*inv/(lead*th)))
print("xi0 =", sp.nsimplify(trunc(xi0, 5)))
print("residual of xi0:", trunc(Ez.subs(x, xi0), 7))
Ex = trunc(sp.diff(E0, x).subs({x: xi0, e: 0})); Ee = trunc(sp.diff(E0, e).subs({x: xi0, e: 0}))
Exx = trunc(sp.diff(E0, x, 2).subs({x: xi0, e: 0})); Exe = trunc(sp.diff(E0, x, e).subs({x: xi0, e: 0})); Eee = trunc(sp.diff(E0, e, 2).subs({x: xi0, e: 0}))
lead = Ex.coeff(th, 1); invEx = sp.expand(sp.series(1/(Ex/(lead*th)), th, 0, M).removeO()/(lead*th))
xi1 = trunc(sp.expand(-Ee*invEx))
xi2 = trunc(sp.expand(-(Exx*xi1**2/2 + Exe*xi1 + Eee/2)*invEx))
print("xi1 =", trunc(xi1, 3)); print("xi2 =", trunc(xi2, 1))
Jint = trunc(sp.expand((-(1-xi0)**2*xi2 + (1-xi0)*xi1**2)*sp.series(sp.sin(th), th, 0, M).removeO()), 2)
print("\n(1/3)(1-x)^3 sin(th), eps^2 coefficient, Laurent series in theta:")
for k in range(-3, 2):
    print(f"   theta^{k}: ", sp.factor(Jint.coeff(th, k)))
