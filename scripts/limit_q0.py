import sympy as sp

th, u, c = sp.symbols('theta u c', positive=True)

# Claim: surface  1/r + r^2 sin^2(th)/2 = 3/2  has exact root r = 2 sin(th/3)/sin(th)
r = 2*sp.sin(th/3)/sp.sin(th)
expr = sp.simplify(1/r + r**2*sp.sin(th)**2/2 - sp.Rational(3,2))
print("residual of claimed root:", sp.simplify(sp.trigsimp(expr)))

# Volume  V = (2*pi/3) int_0^pi r^3 sin th dth = (4pi/3) int_0^{pi/2} r^3 sin th dth
integrand = sp.simplify(r**3*sp.sin(th))
print("r^3 sin th =", sp.simplify(sp.trigsimp(integrand)))

# substitute u = th/3
I = sp.integrate((8/(3-4*sp.sin(u)**2)**3)*(sp.sin(3*u))*3, (u, 0, sp.pi/6))
V = sp.Rational(4,3)*sp.pi*sp.simplify(I)
V = sp.simplify(V)
print("V exact =", V)
print("V num   =", sp.N(V, 30))
Req = (3*V/(4*sp.pi))**sp.Rational(1,3)
print("Req exact =", sp.simplify(Req))
print("Req num   =", sp.N(Req, 30))

# alternative form via c = cos u substitution: V = 32 pi int_{sqrt3/2}^1 dc/(4c^2-1)^2
J = sp.integrate(1/(4*c**2-1)**2, (c, sp.sqrt(3)/2, 1))
print("\nJ =", sp.simplify(J))
V2 = sp.simplify(32*sp.pi*J)
print("V2 =", sp.simplify(V2), "=", sp.N(V2, 30))
print("Req2 =", sp.N((3*V2/(4*sp.pi))**sp.Rational(1,3), 30))
