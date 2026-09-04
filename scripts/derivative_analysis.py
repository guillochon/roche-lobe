"""Formula derivatives, stability quantities, figure and table."""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PAPER

HERE = str(PAPER)
D = json.load(open(os.path.join(HERE, "analysis.json"), encoding="utf-8"))
R0, K, C, A0, E0 = D["R0"], D["K"], D["C"], D["A0"], D["E0"]
A1, A32 = D["a1"], D["a32"]
C2 = D["c2"]
P1, P3 = np.array(D["p1"]), np.array(D["p3"])

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
})


def egg(q):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))


def qi26(q):
    return 0.49 * q ** 0.665 / (0.602 * q ** 0.665 + np.log(1 + q ** 0.329))


def leahy(qE):
    q = 1 / qE
    a1_, a2_, a3_, a4_, a5_ = 0.64334, 0.86907, 1.2809, -0.74303, -0.73103
    return a1_ * q ** a4_ / (a2_ * q ** a5_ + np.log(1 + a3_ * q ** (a4_ + 1 / 3)))


def F1(p, t):
    H, J = p
    return R0 * A0 * t ** 2 / (A0 * t ** 2 + R0 * np.log(1 + E0 * t) + H * t ** 2 / (1 + J * t ** 2))


def F3(p, t):
    H, J, H2, J2, H3, J3 = p
    return R0 * A0 * t ** 2 / (
        A0 * t ** 2 + R0 * np.log(1 + E0 * t)
        + H * t ** 2 / (1 + J * t ** 2)
        + H2 * t ** 3 / (1 + J2 * t ** 2) ** 1.5
        + H3 * t ** 4 / (1 + J3 * t ** 2) ** 2
    )


def R_formula(name, qE):
    qE = np.asarray(qE, dtype=float)
    if name == "Eggleton":
        return egg(qE)
    if name == "Qi":
        return qi26(qE)
    if name == "Leahy":
        return leahy(qE)
    if name == "F1":
        return F1(P1, qE ** (1 / 3))
    if name == "F3":
        return F3(P3, qE ** (1 / 3))
    raise KeyError(name)


def D_formula(name, qE, delta=1e-5):
    qE = np.asarray(qE, dtype=float)
    Rp = R_formula(name, qE * np.exp(+delta))
    Rm = R_formula(name, qE * np.exp(-delta))
    return (np.log(Rp) - np.log(Rm)) / (2.0 * delta)


def D_large(qE, c2=C2):
    qE = np.asarray(qE, dtype=float)
    return (K * qE ** (-2 / 3) * (2 / 3 * np.log(qE) - 1) + (2 / 3) * c2 * qE ** (-2 / 3)) / R0


def D_small(qE):
    t = np.asarray(qE, dtype=float) ** (1 / 3)
    return 1 / 3 + (1 / 3) * A1 * t + 0.5 * A32 * t ** 1.5


def zeta_L(qE, D):
    qE = np.asarray(qE, dtype=float)
    D = np.asarray(D, dtype=float)
    return (1 + qE) * D + 2 * (qE - 1)


def load_reference():
    qE, Dref = [], []
    with open(os.path.join(HERE, "derivative_reference.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()
            qE.append(float(a))
            Dref.append(float(b))
    extra = {}
    partial = os.path.join(HERE, "derivative_reference_partial.json")
    if os.path.exists(partial):
        raw = json.load(open(partial, encoding="utf-8"))
        for v in raw.values():
            extra[v["qE"]] = v["D"]
    return np.array(qE), np.array(Dref), extra


def stats(err, mask=None):
    e = err if mask is None else err[mask]
    return float(np.max(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def qcrit_from(lnq, z, zad):
    spl = CubicSpline(lnq, z - zad)
    # bracket: zeta_L rises with qE
    try:
        return float(np.exp(brentq(spl, lnq[0], lnq[-1])))
    except ValueError:
        return float("nan")


def fmt_q(q):
    if abs(q - 10 ** round(np.log10(q))) < 1e-12 * max(q, 1):
        e = int(round(np.log10(q)))
        if e == 0:
            return r"$1$"
        if e == 1:
            return r"$10$"
        return rf"$10^{{{e}}}$"
    if abs(q - 0.1) < 1e-12:
        return r"$0.1$"
    if abs(q - 0.3) < 1e-12:
        return r"$0.3$"
    if abs(q - 3) < 1e-12:
        return r"$3$"
    return rf"${q:g}$"


def fmt_pct(x):
    ax = abs(x)
    if ax >= 0.1:
        return f"${x:+.2f}$"
    if ax >= 0.01:
        return f"${x:+.3f}$"
    return f"${x:+.4f}$"


def main():
    qE, Dref, extra = load_reference()
    lnq = np.log(qE)
    names = ["Eggleton", "Leahy", "Qi", "F1", "F3"]
    Dform = {n: D_formula(n, qE) for n in names}
    rel = {n: Dform[n] / Dref - 1 for n in names}

    classic = (qE >= 0.01) & (qE <= 100)
    print("=== relative error of D ===")
    for n in names:
        mx, rms = stats(rel[n])
        mxc, rmsc = stats(rel[n], classic)
        print(f"{n:10s}  full max={100*mx:.4f}% rms={100*rms:.4f}%   "
              f"classic max={100*mxc:.4f}% rms={100*rmsc:.4f}%")

    # asymptotic checks
    i4 = np.argmin(np.abs(qE - 1e4))
    im4 = np.argmin(np.abs(qE - 1e-4))
    r_large = Dref[i4] - D_large(qE[i4])
    r_small = Dref[im4] - D_small(qE[im4])
    r_large_fit = Dref[i4] - D_large(qE[i4], c2=D["c2"])
    print("=== asymptotic residuals (D_ref - D_asym) ===")
    print(f"qE=1e4  D_ref={Dref[i4]:.8f}  asym(c2 exact)={float(D_large(qE[i4])):.8f}  residual={r_large:.3e}")
    print(f"qE=1e4  asym(c2={D['c2']:.4f}) residual={r_large_fit:.3e}")
    print(f"qE=1e-4 D_ref={Dref[im4]:.8f}  asym={float(D_small(qE[im4])):.8f}  residual={r_small:.3e}")
    if abs(r_large) > 2e-3 or abs(r_small) > 5e-4:
        raise SystemExit("asymptotic check failed; stop and fix delta/sign")

    zref = zeta_L(qE, Dref)
    zform = {n: zeta_L(qE, Dform[n]) for n in names}
    zads = [-1 / 3, 1.0, 2.0, 4.0, 6.85]
    zad_labels = [r"$-1/3$", r"$1$", r"$2$", r"$4$", r"$6.85$"]
    qcrit = {"ref": []}
    for n in names:
        qcrit[n] = []
    print("=== q_crit ===")
    print(f"{'zeta_ad':>8s}  {'ref':>10s}  {'Egg':>10s}  {'dEgg%':>8s}  {'F3':>10s}  {'dF3%':>8s}  {'F1':>10s}")
    for zad in zads:
        qr = qcrit_from(lnq, zref, zad)
        qcrit["ref"].append(qr)
        row = f"{zad:8.3f}  {qr:10.6f}"
        for n in names:
            qf = qcrit_from(lnq, zform[n], zad)
            qcrit[n].append(qf)
        for n, lab in (("Eggleton", "Egg"), ("F3", "F3")):
            qf = qcrit[n][-1]
            shift = 100 * (qf / qr - 1)
            row += f"  {qf:10.6f}  {shift:+8.3f}"
        row += f"  {qcrit['F1'][-1]:10.6f}"
        print(row)
        if zad == -1 / 3 and not (0.6 < qr < 1.0):
            raise SystemExit(f"q_crit(zeta_ad=-1/3)={qr} looks inverted")

    # fine grid for plot
    qfine = np.logspace(-4, 4, 1000)
    Dfine = {n: D_formula(n, qfine) for n in names}
    Dref_fine = CubicSpline(lnq, Dref)(np.log(qfine))

    sty = {
        "Eggleton": ("C0", "-", "Eggleton (1983)"),
        "Leahy": ("C2", "-.", "Leahy & Leahy (2015)"),
        "Qi": ("C1", "--", "Qi et al. (2026)"),
        "F1": ("C4", ":", "this work, 2 par."),
        "F3": ("C3", "-", "this work, 6 par."),
    }
    legend_kw = dict(
        frameon=True, facecolor="white", framealpha=0.8,
        edgecolor="none", fontsize=6.4, labelspacing=0.32,
    )

    qE_R = np.array(D["qE"])
    R = np.array(D["R"])
    t_R = qE_R ** (1 / 3)
    lqR = np.log10(qE_R)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.1, 4.7), sharex="col",
        gridspec_kw=dict(height_ratios=[1, 1.4], hspace=0.08, wspace=0.32),
    )
    axD0, axR0 = axes[0]
    axD1, axR1 = axes[1]
    lq = np.log10(qE)
    lqf = np.log10(qfine)

    axD0.plot(lq, Dref, "k.", ms=3, label="reference (this work)")
    axD0.plot(lqf, Dfine["F3"], "C3-", lw=0.8, label="Eq. (fit), 6 parameters")
    mhi = qfine > 1e2
    mlo = qfine < 1e-2
    axD0.plot(lqf[mhi], D_large(qfine[mhi]), color="0.5", ls=":", lw=0.8)
    axD0.plot(lqf[mlo], D_small(qfine[mlo]), color="0.5", ls=":", lw=0.8)
    axD0.set_ylabel(r"$D=\partial\ln(R_L/a)/\partial\ln q_E$")
    axD0.set_ylim(0, 0.40)

    axR0.plot(lqR, R, "k.", ms=3, label="reference (this work)")
    axR0.plot(lqR, F3(P3, t_R), "C3-", lw=0.8, label="Eq. (fit), 6 parameters")
    axR0.axhline(R0, color="0.5", ls=":", lw=0.7)
    axR0.text(
        3.4, R0 + 0.035, r"$R_0=0.81489$", fontsize=7, color="0.4",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.15),
    )
    lq2 = np.linspace(-6, -3, 50)
    axR0.plot(lq2, C * 10 ** (lq2 / 3), color="0.5", ls=":", lw=0.7)
    axR0.text(-5.5, 0.09, r"$C\,q_E^{1/3}$", fontsize=7, color="0.4")
    axR0.set_ylabel(r"$R_L/a$")
    axR0.set_ylim(0, 0.9)
    axR0.legend(loc="lower right", **legend_kw)

    for n, (c, ls, lab) in sty.items():
        axD1.plot(lqf, 100 * (Dfine[n] / Dref_fine - 1), color=c, ls=ls, lw=1.0, label=lab)
        axR1.plot(lqR, 100 * np.array(D["errs"][n]["err"]), color=c, ls=ls, lw=1.0, label=lab)
    axD1.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    axD1.axhline(0, color="k", lw=0.4)
    axD1.axvspan(-2, 2, color="0.92", zorder=0)
    axD1.set_ylim(-8, 30)
    axD1.set_yticks([-4, -1, -0.1, 0, 0.1, 1, 4, 20])
    axD1.set_yticklabels(["$-4$", "$-1$", "$-0.1$", "0", "0.1", "1", "4", "20"])
    axD1.set_xlabel(r"$\log_{10} q_E$,  $q_E=M_1/M_2$")
    axD1.set_ylabel("relative error of $D$ (%)")

    axR1.set_yscale("symlog", linthresh=0.05, linscale=0.8)
    axR1.axhline(0, color="k", lw=0.4)
    axR1.set_ylim(-5, 8)
    axR1.set_yticks([-4, -1, -0.1, 0, 0.1, 1, 4])
    axR1.set_yticklabels(["$-4$", "$-1$", "$-0.1$", "0", "0.1", "1", "4"])
    axR1.axvspan(-2, 2, color="0.92", zorder=0)
    axR1.text(-1.9, 4.5, "classical\nfit range", fontsize=6, color="0.4")
    axR1.set_xlabel(r"$\log_{10} q_E$,  $q_E=M_1/M_2$")
    axR1.set_ylabel("relative error (%)")
    axR1.legend(loc="lower right", **legend_kw)

    for ax, lab in zip((axD0, axR0, axD1, axR1), "abcd"):
        ax.text(
            0.03, 0.96, f"({lab})", transform=ax.transAxes,
            va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.15),
        )

    fig.savefig(os.path.join(HERE, "fig_fits_deriv.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fits_deriv.pdf")

    # table rows
    qtab = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 1e2, 1e3]
    Dspl = CubicSpline(lnq, Dref)
    zspl = CubicSpline(lnq, zref)
    lines = []
    lines.append(r"\begin{deluxetable}{lccccccc}")
    lines.append(r"\tablewidth{\textwidth}")
    lines.append(
        r"\tablecaption{Logarithmic derivative $D=\partial\ln(\RL/a)/\partial\ln\qE$ "
        r"of the reference solution and percentage errors of five fitting formulae. "
        r"The last column is the conservative-transfer lobe response "
        r"$\zeta_L=(1+\qE)\,D+2(\qE-1)$.\label{tab:deriv}}"
    )
    lines.append(
        r"\tablehead{\colhead{$\qE$} & \colhead{$D_{\rm ref}$} & "
        r"\colhead{Eggleton} & \colhead{Leahy} & \colhead{Qi} & "
        r"\colhead{new 2} & \colhead{new 6} & \colhead{$\zeta_L^{\rm ref}$} \\ "
        r"\colhead{} & \colhead{} & \colhead{(\%)} & \colhead{(\%)} & "
        r"\colhead{(\%)} & \colhead{(\%)} & \colhead{(\%)} & \colhead{}}"
    )
    lines.append(r"\startdata")
    for q in qtab:
        # prefer a computed extra / grid point over the spline
        Dr = None
        for qg, Dg in zip(qE, Dref):
            if abs(qg / q - 1) < 1e-10:
                Dr = Dg
                break
        if Dr is None:
            for qg, Dg in extra.items():
                if abs(qg / q - 1) < 1e-8:
                    Dr = Dg
                    break
        if Dr is None:
            Dr = float(Dspl(np.log(q)))
        zr = float(zeta_L(q, Dr))
        pcts = []
        for n in names:
            pcts.append(100 * (float(D_formula(n, q)) / Dr - 1))
        if abs(zr) < 10:
            zfmt = f"{zr:.4f}"
        elif abs(zr) < 100:
            zfmt = f"{zr:.2f}"
        else:
            zfmt = f"{zr:.1f}"
        if abs(q - 1e3) < 1e-12:
            pct_strs = []
            for i, p in enumerate(pcts):
                s = fmt_pct(p)
                if names[i] == "Leahy":
                    s = s + r"\tablenotemark{a}"
                pct_strs.append(s)
        else:
            pct_strs = [fmt_pct(p) for p in pcts]
        lines.append(
            f"{fmt_q(q)} & {Dr:.6f} & "
            + " & ".join(pct_strs)
            + f" & {zfmt} \\\\"
        )
    lines.append(r"\enddata")
    lines.append(
        r"\tablenotetext{a}{Outside the Leahy \& Leahy (2015) calibration "
        r"$0.01\le q\le 100$ ($10^{-2}\le\qE\le10^{2}$); the $+22\%$ is the "
        r"onset of the large-$\qE$ divergence, not a mid-range residual.}"
    )
    lines.append(r"\end{deluxetable}")
    tex = os.path.join(HERE, "table_derivative.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", tex)

    qlines = []
    qlines.append(r"\begin{deluxetable}{lccccc}")
    qlines.append(
        r"\tablecaption{Critical mass ratio $\qE^{\rm crit}$ at which $\zeta_L=\zeta_{\rm ad}$ "
        r"for conservative transfer, from the reference $D$, the Eggleton formula, "
        r"and equation~(\ref{eq:fit}) with (\ref{eq:h6}). "
        r"Shifts are $(q_{\rm formula}/q_{\rm ref}-1)$.\label{tab:qcrit}}"
    )
    qlines.append(
        r"\tablehead{\colhead{$\zeta_{\rm ad}$} & \colhead{$q_{\rm ref}$} & "
        r"\colhead{$q_{\rm Egg}$} & \colhead{shift} & "
        r"\colhead{$q_{\rm 6}$} & \colhead{shift}}"
    )
    qlines.append(r"\startdata")
    for lab, zad, qr, qe, q6 in zip(zad_labels, zads, qcrit["ref"], qcrit["Eggleton"], qcrit["F3"]):
        se = 100 * (qe / qr - 1)
        s6 = 100 * (q6 / qr - 1)

        def sh(x):
            return f"{x:+.3f}\\%" if abs(x) < 0.05 else f"{x:+.2f}\\%"

        qlines.append(
            f"{lab} & {qr:.4f} & {qe:.4f} & {sh(se)} & {q6:.4f} & {sh(s6)} \\\\"
        )
    qlines.append(r"\enddata")
    qlines.append(r"\end{deluxetable}")

    qtex = os.path.join(HERE, "table_qcrit.tex")
    with open(qtex, "w", encoding="utf-8") as f:
        f.write("\n".join(qlines) + "\n")
    print("wrote", qtex)

    report = {
        "rel_err": {
            n: {
                "full_max": stats(rel[n])[0],
                "full_rms": stats(rel[n])[1],
                "classic_max": stats(rel[n], classic)[0],
                "classic_rms": stats(rel[n], classic)[1],
            }
            for n in names
        },
        "asym": {
            "large_c2_m11": float(r_large),
            "large_c2_fit": float(r_large_fit),
            "small": float(r_small),
            "D_ref_1e4": float(Dref[i4]),
            "D_ref_1em4": float(Dref[im4]),
        },
        "qcrit": {
            "zeta_ad": zads,
            "ref": qcrit["ref"],
            "Eggleton": qcrit["Eggleton"],
            "F3": qcrit["F3"],
            "F1": qcrit["F1"],
            "Leahy": qcrit["Leahy"],
            "Qi": qcrit["Qi"],
        },
        "Dref_mid": {
            "0.1": float(Dref[np.argmin(np.abs(qE - 0.1))]),
            "1": float(Dref[np.argmin(np.abs(qE - 1))]),
            "10": float(Dref[np.argmin(np.abs(qE - 10))]),
        },
        "zref_zero": float(np.exp(brentq(CubicSpline(lnq, zref), lnq[0], lnq[-1]))),
    }
    with open(os.path.join(HERE, "derivative_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote derivative_report.json")
    print("zeta_L=0 at qE =", report["zref_zero"])


if __name__ == "__main__":
    main()
