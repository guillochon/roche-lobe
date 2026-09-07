"""Minimax coefficients, published-formula errors, and endpoint fits.

Reads the 73-point reference table and writes analysis.json. Small-q_E points
below 10^{-6} are recomputed with the double-precision solver (npts=200).
Run:  python scripts/analysis.py
"""
import json

import numpy as np
from scipy.optimize import least_squares, minimize

from fast import Rsurf, panels, xL1, zeta
from paths import PAPER

HERE = PAPER

R0 = 0.8148856767823898
K = 3 ** (-1 / 6) / (2 * R0**2)
C = 0.4933062311526333
A0 = R0**2 / (3 * K)
E0 = A0 / C
V0 = 1.50854985613111314
V1 = -1.7280570857676475595
a1 = 3 ** (-1 / 3) * V1 / (3 * V0)
a32 = 8 * np.pi / (81 * V0)
c2 = -(3 ** (1 / 3)) / (8 * R0**2) * (
    np.sqrt(3) * (10 - 8 * np.log(2) - (10 / 3) * np.log(3) + 8 * np.log(2 + np.sqrt(3))) - 16
)
V2U2 = -1.35624034152162966
V2red = -1.26067445581712
a2 = 3 ** (-2 / 3) * ((V2U2 + V2red) / (3 * V0) - V1**2 / (9 * V0**2))


def volume(q, npts=200):
    x1 = xL1(q)
    zc = zeta(x1, 1.0, 0.0, q)
    eps = min((q / 3.0) ** (1 / 3.0), 0.4)
    d = min(4 * eps, 0.9 * np.pi / 2)
    th, thw = panels([0.0, np.pi / 2 - d, np.pi / 2], npts)
    ph, phw = panels([0.0, d, np.pi], npts)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(thw * np.sin(th), phw)
    lam = (np.sin(TH) * np.cos(PH)).ravel()
    gam = np.cos(TH).ravel()
    Rg = Rsurf(lam, gam, q, zc, 0.3 * x1, x1 * (1 - 1e-14)).reshape(TH.shape)
    return 4 / 3 * np.sum(W * Rg**3)


def Req(q, **k):
    return (3 * volume(q, **k) / (4 * np.pi)) ** (1 / 3)


def F1(p, t):
    H, J = p
    return R0 * A0 * t**2 / (A0 * t**2 + R0 * np.log(1 + E0 * t) + H * t**2 / (1 + J * t**2))


def F3(p, t):
    H, J, H2, J2, H3, J3 = p
    return R0 * A0 * t**2 / (
        A0 * t**2
        + R0 * np.log(1 + E0 * t)
        + H * t**2 / (1 + J * t**2)
        + H2 * t**3 / (1 + J2 * t**2) ** 1.5
        + H3 * t**4 / (1 + J3 * t**2) ** 2
    )


def egg(q):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))


def qi26(q):
    return 0.49 * q**0.665 / (0.602 * q**0.665 + np.log(1 + q**0.329))


def leahy(qE):
    q = 1 / qE
    a1_, a2_, a3_, a4_, a5_ = 0.64334, 0.86907, 1.2809, -0.74303, -0.73103
    return a1_ * q**a4_ / (a2_ * q**a5_ + np.log(1 + a3_ * q ** (a4_ + 1 / 3)))


def minimax(F, p0, t, R, rounds=6):
    p = least_squares(lambda p: F(p, t) / R - 1, p0).x
    obj = lambda p: np.max(np.abs(F(p, t) / R - 1))
    for _ in range(rounds):
        p = minimize(
            obj, p, method="Nelder-Mead",
            options=dict(xatol=1e-11, fatol=1e-13, maxiter=40000),
        ).x
    return p


def main():
    table = np.loadtxt(HERE / "roche_lobe_reference_table.dat")
    qE, R = table[:, 0], table[:, 1]
    t = qE ** (1 / 3)

    p1 = minimax(F1, [0.1, 1.0], t, R)
    p3 = minimax(F3, [-0.0157, 0.0869, -0.0622, 1.03, 0.0, 1.0], t, R)
    errs = {}
    for name, f in [
        ("Eggleton", egg(qE)),
        ("Qi", qi26(qE)),
        ("Leahy", leahy(qE)),
        ("F1", F1(p1, t)),
        ("F3", F3(p3, t)),
    ]:
        e = f / R - 1
        m = (qE >= 0.01) & (qE <= 100)
        errs[name] = dict(
            max=float(np.abs(e).max()),
            rms=float(np.sqrt(np.mean(e**2))),
            max_classic=float(np.abs(e[m]).max()),
            err=e.tolist(),
        )
        print(
            f"{name:10s} max={100 * errs[name]['max']:.4f}%  "
            f"classic-range max={100 * errs[name]['max_classic']:.4f}%  "
            f"rms={100 * errs[name]['rms']:.4f}%"
        )
    print("p1 =", p1)
    print("p3 =", p3)
    c2_F1 = (R0**2 * np.log(E0) + R0 * p1[0] / p1[1]) / A0
    c2_F3 = (
        R0**2 * np.log(E0)
        + R0 * (p3[0] / p3[1] + p3[2] / p3[3] ** 1.5 + p3[4] / p3[5] ** 2)
    ) / A0
    print("implied c2: F1", c2_F1, " F3", c2_F3)

    qs = np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7])
    Rs = np.array([Req(1 / x, npts=200) for x in qs])
    small = HERE / "roche_lobe_small_qE.dat"
    np.savetxt(
        small,
        np.column_stack([qs, Rs]),
        header="q_E=M1/M2   R_L/a\nSix points below the 73-point n=300 table, used in §3.2 (19 points over 1e-9..1e-4).\nDouble precision, npts=200 (scripts/analysis.py).",
        comments="# ",
    )
    print("wrote", small)
    m = qE >= 1e2
    d = R0 - R[m]
    rem = d - K * qE[m] ** (-2 / 3) * np.log(qE[m]) - c2 * qE[m] ** (-2 / 3)
    A = np.column_stack(
        [qE[m] ** (-1), qE[m] ** (-1) * np.log(qE[m])]
    )
    cc = np.linalg.lstsq(A, rem, rcond=None)[0]
    print("log side: exact c2 =", c2, " (q^-1, q^-1 ln q coeffs)", cc)
    m3 = qE >= 1e3
    rem3 = (R0 - R[m3]) * qE[m3] ** (2 / 3) - K * np.log(qE[m3]) - c2
    A3 = np.column_stack(
        [
            qE[m3] ** (-1 / 3) * np.log(qE[m3]),
            qE[m3] ** (-1 / 3),
            qE[m3] ** (-2 / 3) * np.log(qE[m3]),
        ]
    )
    pred3 = A3 @ np.linalg.lstsq(A3, rem3, rcond=None)[0]
    print("log remainder rms (y, qE>=1e3) =", float(np.sqrt(np.mean((rem3 - pred3) ** 2))))
    qq = np.concatenate([qs, qE[qE <= 1e-4]])
    RR = np.concatenate([Rs, R[qE <= 1e-4]])
    tt = qq ** (1 / 3)
    y = RR / (C * tt) - 1 - a1 * tt - a32 * tt**1.5 - a2 * tt**2
    A = np.column_stack([tt**2.5, tt**3, tt**3.5])
    ch = np.linalg.lstsq(A, y, rcond=None)[0]
    print("Hill side: exact a2 =", a2, " a5/2 =", ch[0])
    out = HERE / "analysis.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            dict(
                R0=R0, K=K, C=C, A0=A0, E0=E0, V0=V0, V1=V1, a1=a1, a32=a32,
                p1=p1.tolist(), p3=p3.tolist(),
                c2=float(c2), c3=float(cc[0]), c3L=float(cc[1]),
                c2_F1=float(c2_F1), c2_F3=float(c2_F3),
                a2=float(a2), a52=float(ch[0]),
                errs=errs, qE=qE.tolist(), R=R.tolist(),
                qs=qs.tolist(), Rs=Rs.tolist(),
            ),
            fh,
            indent=1,
        )
    print("saved", out)


if __name__ == "__main__":
    main()
