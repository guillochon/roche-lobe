import mpmath as mp, sys
mp.mp.dps = 28
pi = mp.pi

# ---------- exact Hill surface at level c (default critical 9/2) ----------
def rho0(A, c=mp.mpf(9)/2):
    if abs(A) < mp.mpf(10)**(-30): return 3/c
    m = (9/(2*c))*mp.sqrt(abs(3*A/c))
    if A > 0:  return 2*mp.sqrt(c/(3*A))*mp.sin(mp.asin(m)/3)
    else:      return 2*mp.sqrt(c/(3*(-A)))*mp.sinh(mp.asinh(m)/3)

# direction from polar angle vartheta about the -xi axis (L2 direction) and azimuth phi
def nvec(vt, ph):
    return (-mp.cos(vt), mp.sin(vt)*mp.cos(ph), mp.sin(vt)*mp.sin(ph))

def J_red(vt, ph):
    """second-order integrand rho0^2 rho2 + rho0 rho1^2 for U0 + eps*U1, level 9/2 + eps - eps^2/2"""
    nx, ny, nz = nvec(vt, ph)
    A = (3*nx**2 - nz**2)/2; B = (5*nx**3 - 3*nx)/2
    r = rho0(A)
    Up = -3/r**2 + 2*A*r; Upp = 6/r**3 + 2*A
    r1 = -(B*r**3 - 1)/Up
    r2 = -(Upp*r1**2/2 + 3*B*r**2*r1 + mp.mpf(1)/2)/Up
    return r**2*r2 + r*r1**2

def J_shift(vt, ph):
    """same for pure level shift: U0 >= 9/2 + Delta  (rho1 = 1/U0', rho2 = -U0'' rho1^2/(2 U0'))"""
    nx, ny, nz = nvec(vt, ph)
    A = (3*nx**2 - nz**2)/2
    r = rho0(A)
    Up = -3/r**2 + 2*A*r; Upp = 6/r**3 + 2*A
    r1 = 1/Up; r2 = -Upp*r1**2/(2*Up)
    return r**2*r2 + r*r1**2

def g(J, vt):   # azimuthal integral times sin(vt); J even in eta and zeta -> 4 x quarter
    return 4*mp.quad(lambda ph: J(vt, ph), [0, pi/4, pi/2])*mp.sin(vt)

def laurent(J):
    """fit g = c_-2/vt^2 + c_-1/vt + c0 + c1 vt + c2 vt^2 from small-vt samples"""
    vts = [mp.mpf(10)**(-k) for k in (2, 2.5, 3, 3.5, 4)]
    vals = [g(J, v)*v**2 for v in vts]
    A = mp.matrix([[1, v, v**2, v**3, v**4] for v in vts]); sol = mp.lu_solve(A, mp.matrix(vals))
    return sol

def finite_part(J, vt_hi, cm2, cm1, vt_c=mp.mpf('0.25')):
    reg = mp.quad(lambda v: g(J, v) - cm2/v**2 - cm1/v, [0, vt_c/8, vt_c], method='gauss-legendre')
    tail = mp.quad(lambda v: g(J, v), [vt_c, pi/2, vt_hi - mp.mpf('0.2'), vt_hi], method='gauss-legendre')
    return reg - cm2/vt_c + cm1*mp.log(vt_c) + tail

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "shift":
        # ---- validation: pure level shift. FP at both poles (symmetric): 2 x integral over [0, pi/2]
        c = laurent(J_shift); print("shift Laurent: c-2=%s  c-1=%s  c0=%s" % tuple(mp.nstr(x, 10) for x in c[:3]), flush=True)
        reg = mp.quad(lambda v: g(J_shift, v) - c[0]/v**2 - c[1]/v, [0, mp.mpf('0.03'), mp.mpf('0.25')], method='gauss-legendre')
        tail = mp.quad(lambda v: g(J_shift, v), [mp.mpf('0.25'), pi/2])
        FP = 2*(reg - c[0]/mp.mpf('0.25') + c[1]*mp.log(mp.mpf('0.25')) + tail)
        print("shift: FP prediction for T =", mp.nstr(FP, 15), flush=True)
        # ---- independent: exact V_H(c) expansion
        def VH(cc):
            def f(t, p):
                nx, nz = mp.sin(t)*mp.cos(p), mp.cos(t); A = (3*nx**2 - nz**2)/2
                return rho0(A, cc)**3
            sd = mp.sqrt(cc - mp.mpf(9)/2)
            inner = lambda t: mp.quad(lambda p: f(t, p), [0, 4*sd, pi/2, pi - 4*sd, pi])
            return mp.mpf(4)/3*mp.quad(lambda t: mp.sin(t)*inner(t), [0, pi/2 - 4*sd, pi/2])
        VH0 = mp.mpf('1.508549856131113140040991'); V1 = mp.mpf('-1.7280570857676475595')
        S = 2*(pi/(3*mp.sqrt(3)))*(2*mp.sqrt(2))/3
        Ds = [mp.mpf(10)**(-k) for k in (3, 3.5, 4, 4.5, 5)]
        ys = [VH(mp.mpf(9)/2 + D) - VH0 - V1*D - S*D**mp.mpf(1.5) for D in Ds]
        A = mp.matrix([[D**2, D**mp.mpf(2.5), D**3, D**mp.mpf(3.5), D**4] for D in Ds]); sol = mp.lu_solve(A, mp.matrix(ys))
        print("shift: T from exact V_H(c) =", mp.nstr(sol[0], 12), "  (next coeffs", mp.nstr(sol[1], 6), mp.nstr(sol[2], 6), ")")
        # also with a log term allowed
        A2 = mp.matrix([[D**2, D**2*mp.log(D), D**mp.mpf(2.5), D**3, D**mp.mpf(3.5)] for D in Ds]); sol2 = mp.lu_solve(A2, mp.matrix(ys))
        print("shift: with D^2 ln D allowed: T =", mp.nstr(sol2[0], 10), " log coeff =", mp.nstr(sol2[1], 4))
    else:
        c = laurent(J_red); print("red Laurent (numerical fit): c-2=%s  c-1=%s  c0=%s" % tuple(mp.nstr(x, 12) for x in c[:3]), flush=True)
        # exact subtraction constants (Appendix C): c-2 = (4/9) E(m=1/4), c-1 = 0 identically
        cm2 = mp.mpf(4)/9*mp.ellipe(mp.mpf(1)/4); cm1 = mp.mpf(0)
        print("red Laurent (exact):           c-2=%s  c-1=0" % mp.nstr(cm2, 12), flush=True)
        FP = finite_part(J_red, pi, cm2, cm1)
        print("V2_red = FP =", mp.nstr(FP, 15), "   (numerical reduced model: -1.2606 +/- 0.0003)")
