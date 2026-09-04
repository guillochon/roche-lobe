import mpmath as mp
from fp import *
mp.mp.dps = 28
c = laurent(J_red)
for vc in (mp.mpf('0.12'), mp.mpf('0.25'), mp.mpf('0.5')):
    print("vt_c =", vc, " FP =", mp.nstr(finite_part(J_red, pi, c[0], c[1], vt_c=vc), 14), flush=True)
V2red = finite_part(J_red, pi, c[0], c[1])
VH = mp.mpf('1.508549856131113140040991'); V1 = mp.mpf('-1.7280570857676475595'); V2U2 = mp.mpf('-1.35624034152162966')
a2 = mp.mpf(3)**(-mp.mpf(2)/3)*((V2U2+V2red)/(3*VH) - V1**2/(9*VH**2))
print("\nV2_red =", mp.nstr(V2red, 13)); print("V2 = V2red + V2U2 =", mp.nstr(V2red+V2U2, 13)); print("a2 =", mp.nstr(a2, 12), "  (full-Roche fit: -0.346; semi-analytic estimate: -0.3481 +/- 0.0002)")
