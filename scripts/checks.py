import sympy as sp
x, q = sp.symbols('x q')
P = x**5*(q+1) + x**4*(-3*q-2) + x**3*(3*q+1) - x**2 + 2*x - 1

print("=== Galois / solvability of the L1 quintic ===")
p2 = sp.Poly(sp.expand(P.subs(q, 2)), x, domain=sp.QQ)
print("q=2 irreducible over Q:", p2.is_irreducible)
print("q=2 rational roots:", sp.roots(p2, x, filter="Q"))
print("q=2 factor_list:", sp.factor_list(p2.as_expr()))
G2, alt2 = sp.galois_group(p2)
print("q=2 galois_group:", G2, "order", G2.order(), "solvable", G2.is_solvable, "alt", alt2)
for qq in [sp.Integer(2), sp.Rational(1, 2), sp.Integer(3), sp.Rational(104, 19), sp.Integer(1)]:
    p = sp.Poly(sp.expand(P.subs(q, qq) * sp.denom(sp.nsimplify(qq))), x)
    fl = sp.factor_list(p.as_expr())[1]
    degs = [(sp.Poly(f, x).degree(), m) for f, m in fl]
    line = f"q={qq}: factors {degs}"
    for f, m in fl:
        pf = sp.Poly(f, x)
        if pf.degree() >= 2:
            G, alt = sp.galois_group(pf)
            line += f" | deg{pf.degree()} group order {G.order()} solvable={G.is_solvable}"
    print(line)

print("\n=== exact rational L1 points ===")
qx = (1-x)**3*(1+x+x**2)/(x**3*(x**2-3*x+3))
for xv in [sp.Rational(1,2), sp.Rational(1,3), sp.Rational(2,3), sp.Rational(1,4), sp.Rational(3,4), sp.Rational(1,5)]:
    print(f"  x_L1 = {xv}  <=>  q = {sp.nsimplify(qx.subs(x, xv))} = {float(qx.subs(x,xv)):.6f}")
