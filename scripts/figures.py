import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq

from fast import xL1, zeta, Rsurf
from paths import PAPER

HERE = PAPER
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif', 'axes.linewidth': 0.6, 'lines.linewidth': 1.0})
D = json.load(open(HERE / 'analysis.json'))
R0, K, C, A0, E0 = D['R0'], D['K'], D['C'], D['A0'], D['E0']
a1, a32, a2 = D['a1'], D['a32'], D['a2']
c2 = D['c2']
qE = np.array(D['qE']); R = np.array(D['R']); t = qE**(1/3)

def surf(q, TH, PH):
    x1 = xL1(q); zc = zeta(x1, 1.0, 0.0, q)
    lam = (np.sin(TH)*np.cos(PH)).ravel(); gam = np.cos(TH).ravel()
    return Rsurf(lam, gam, q, zc, 0.3*x1, x1*(1-1e-14)).reshape(TH.shape), x1

def grad(X, Y, Z, q):
    r1 = np.sqrt(X**2+Y**2+Z**2); r2 = np.sqrt((X-1)**2+Y**2+Z**2)
    return (-X/r1**3 - q*(X-1)/r2**3 - q + (1+q)*X, -Y/r1**3 - q*Y/r2**3 + (1+q)*Y, -Z/r1**3 - q*Z/r2**3)

def rho_hill(nx, nz):
    m2 = nx**2 - nz**2/3; out = np.empty_like(m2)
    p = m2 > 1e-14; n = m2 < -1e-14; z = ~(p | n)
    mp_ = np.sqrt(m2[p]); out[p] = 2/mp_*np.sin(np.arcsin(mp_)/3)
    mn = np.sqrt(-m2[n]); out[n] = 2/mn*np.sinh(np.arcsinh(mn)/3)
    out[z] = 2/3; return out

def r_inf(th):
    thp = np.minimum(th, np.pi - th)
    return 2/(1 + 2*np.cos(2*thp/3))

def rH_of(qe):
    return (qe/(3*(qe + 1)))**(1/3)

# ---------------- Fig 1: cross sections ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))
phi = np.linspace(0, 2*np.pi, 1441); thm = np.linspace(0, np.pi, 1441)
cols = cm.viridis(np.linspace(0.05, 0.9, 6))
for c, qe in zip(cols, [0.01, 0.1, 1, 10, 100, 1e4]):
    q = 1/qe
    Rp, x1 = surf(q, np.full_like(phi, np.pi/2), phi)              # orbital plane
    ax[0].plot(Rp*np.cos(phi), Rp*np.sin(phi), color=c, label=rf'$q_E={qe:g}$')
    Rm1, _ = surf(q, thm, np.zeros_like(thm)); Rm2, _ = surf(q, thm, np.full_like(thm, np.pi))
    ax[1].plot(np.concatenate([Rm1*np.sin(thm), -Rm2*np.sin(thm)[::-1]]),
               np.concatenate([Rm1*np.cos(thm), Rm2*np.cos(thm)[::-1]]), color=c, label=rf'$q_E={qe:g}$')
# exact q_E -> infinity limit
th = np.linspace(0, np.pi, 800); rl = r_inf(th)
ax[0].plot(np.cos(phi), np.sin(phi), 'k--', lw=0.8, label=r'$q_E\to\infty$ (exact)')
ax[1].plot(np.concatenate([rl*np.sin(th), -rl*np.sin(th)[::-1]]),
           np.concatenate([rl*np.cos(th), rl*np.cos(th)[::-1]]), 'k--', lw=0.8, label=r'$q_E\to\infty$ (exact)')
# exact Hill surface, scaled to r_H at the smallest plotted q_E
rH0 = rH_of(0.01)
hx, hy = rH0*rho_hill(np.cos(phi), 0*phi)*np.cos(phi), rH0*rho_hill(np.cos(phi), 0*phi)*np.sin(phi)
hm1, hm2 = rH0*rho_hill(np.sin(thm), np.cos(thm)), rH0*rho_hill(-np.sin(thm), np.cos(thm))
hx_m = np.concatenate([hm1*np.sin(thm), -hm2*np.sin(thm)[::-1]])
hz_m = np.concatenate([hm1*np.cos(thm), hm2*np.cos(thm)[::-1]])
ax[0].plot(hx, hy, 'k:', lw=0.9, label=r'$q_E\to0$ (Hill, exact)')
ax[1].plot(hx_m, hz_m, 'k:', lw=0.9, label=r'$q_E\to0$ (Hill, exact)')
for a, ttl, yl in zip(ax, ['orbital plane ($z=0$)', 'meridional plane ($y=0$)'], ['$y/a$', '$z/a$']):
    a.plot(0, 0, 'k.', ms=4); a.plot(1, 0, 'k+', ms=6, mew=0.8)
    a.set_aspect('equal'); a.set_xlabel('$x/a$'); a.set_ylabel(yl); a.set_title(ttl, fontsize=9)
    a.set_xlim(-1.1, 1.15); a.set_ylim(-1.1, 1.1); a.axhline(0, color='0.85', lw=0.5, zorder=0); a.axvline(0, color='0.85', lw=0.5, zorder=0)
h,l = ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=4,frameon=False,handlelength=1.6,bbox_to_anchor=(0.5,-0.01),fontsize=6.4)
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(HERE / 'fig_sections.pdf'); plt.close(fig)

# ---------------- Fig 2: 3-D surfaces ----------------
from matplotlib.colors import LightSource, Normalize
fig = plt.figure(figsize=(7.0, 3.1))
TH, PH = np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2*np.pi, 361), indexing='ij')
ls = LightSource(azdeg=315, altdeg=45)
for k, qe in enumerate([0.01, 1, 100]):
    q = 1/qe; Rg, x1 = surf(q, TH, PH)
    X = Rg*np.sin(TH)*np.cos(PH); Y = Rg*np.sin(TH)*np.sin(PH); Z = Rg*np.cos(TH)
    RL = R[np.argmin(np.abs(qE - qe))]
    pct = 100*(Rg - RL)/RL
    vmax = np.nanmax(np.abs(pct))
    print(f'qE={qe:g}: RL={RL:.6f}, pct=[{np.nanmin(pct):.2f}, {np.nanmax(pct):.2f}]')
    axs = fig.add_subplot(1, 3, k+1, projection='3d')
    norm = Normalize(-vmax, vmax)
    rgb = cm.RdBu_r(norm(pct)); rgb = ls.shade_rgb(rgb[..., :3], Z, blend_mode='soft', vert_exag=1)
    axs.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    L = 1.02*x1; axs.set_xlim(-L, L); axs.set_ylim(-L, L); axs.set_zlim(-L, L)
    axs.set_box_aspect((1, 1, 1)); axs.view_init(elev=18, azim=-60)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r); mappable.set_array([])
    cb = fig.colorbar(mappable, ax=axs, shrink=0.55, pad=0.08, orientation='horizontal')
    cb.set_label(r'$(R-R_{\rm L})/R_{\rm L}$ (\%)', fontsize=6); cb.ax.tick_params(labelsize=5)
    axs.set_title(rf'$q_E=M_1/M_2={qe:g}$', fontsize=8, pad=0)
    axs.set_xlabel('$x/a$', labelpad=-4); axs.set_ylabel('$y/a$', labelpad=-4); axs.set_zlabel('$z/a$', labelpad=-4)
    axs.tick_params(labelsize=5, pad=-2)
fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0, wspace=0.0)
fig.savefig(HERE / 'fig_surface3d.png', dpi=300); plt.close(fig)

# ---------------- Fig 3: exact limiting shapes ----------------
def rho_pert(nx, nz, eps):
    A = (3*nx**2 - nz**2)/2; B = (5*nx**3 - 3*nx)/2
    r1 = brentq(lambda r: r**3 + eps*r**4 - 1, 0.5, 1.0); Uc = 3/r1 + 1.5*r1**2 + eps*r1**3
    out = np.empty_like(nx)
    for i in range(nx.size):
        f = lambda r: 3/r + A[i]*r**2 + eps*B[i]*r**3 - Uc
        rg = np.linspace(0.3, r1, 400); v = f(rg); j = np.argmax(v < 0)
        out[i] = brentq(f, rg[j-1], rg[j]) if (v < 0).any() and j > 0 else r1
    return out
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
th = np.linspace(0, np.pi, 801); thp = np.minimum(th, np.pi-th); r = 2/(1+2*np.cos(2*thp/3))
ax[0].plot(r*np.sin(th), r*np.cos(th), 'k-'); ax[0].plot(-r*np.sin(th), r*np.cos(th), 'k-'); ax[0].plot(0, 0, 'k.', ms=4)
ax[0].plot([1, 1+0.25*np.cos(np.radians(120))], [0, 0.25*np.sin(np.radians(120))], color='C3', lw=0.7)
ax[0].plot([1, 1+0.25*np.cos(np.radians(-120))], [0, 0.25*np.sin(np.radians(-120))], color='C3', lw=0.7)
ax[0].text(0.62, 0.22, r'$60^\circ$', color='C3', fontsize=7)
ax[0].set_title(r'$q_E\to\infty$: $r=2a/(1+2\cos\frac{2\theta}{3})$', fontsize=8)
ax[0].set_xlabel(r'$\varpi/a$'); ax[0].set_ylabel('$z/a$')
phi = np.linspace(0, 2*np.pi, 1200)
for a, (lab, nxf, nzf, yl) in zip(ax[1:], [(r'$\xi$–$\eta$ plane', np.cos(phi), 0*phi, r'$\eta/r_H$'), (r'$\xi$–$\zeta$ plane', np.cos(phi), np.sin(phi), r'$\zeta/r_H$')]):
    rh = rho_hill(nxf, nzf); a.plot(rh*np.cos(phi), rh*np.sin(phi), 'k-', label=r'Hill lobe ($\epsilon=0$)')
    for eps, c in [(0.1, 'C0'), (0.3, 'C3')]:
        rp = rho_pert(nxf, nzf, eps); a.plot(rp*np.cos(phi), rp*np.sin(phi), color=c, lw=0.8, ls='--', label=rf'octupole, $\epsilon={eps}$')
    a.plot(0, 0, 'k.', ms=4); a.set_xlabel(r'$\xi/r_H$'); a.set_ylabel(yl); a.set_title(lab, fontsize=8)
    a.annotate('$L_1$', (1.0, 0), xytext=(0.78, 0.55), fontsize=7, arrowprops=dict(arrowstyle='-', lw=0.5))
    a.annotate('$L_2$', (-1.0, 0), xytext=(-0.95, 0.55), fontsize=7, arrowprops=dict(arrowstyle='-', lw=0.5))
h2,l2 = ax[1].get_legend_handles_labels(); fig.legend(h2,l2,loc='lower center',ncol=3,frameon=False,fontsize=6.5,bbox_to_anchor=(0.62,-0.01))
for a in ax: a.set_aspect('equal'); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.0, 1.0)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(HERE / 'fig_limits.pdf'); plt.close(fig)

# ---------------- Fig 4: fits (standalone; the paper uses fig_fits_deriv.pdf) ----------------
def F3(p, t):
    H, J, H2, J2, H3, J3 = p
    return R0*A0*t**2/(A0*t**2 + R0*np.log(1+E0*t) + H*t**2/(1+J*t**2) + H2*t**3/(1+J2*t**2)**1.5 + H3*t**4/(1+J3*t**2)**2)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4], hspace=0.06))
lq = np.log10(qE)
ax[0].plot(lq, R, 'k.', ms=3, label='reference (this work)')
ax[0].plot(lq, F3(D['p3'], t), 'C3-', lw=0.8, label='Eq. (fit), 6 parameters')
ax[0].axhline(R0, color='0.5', ls=':', lw=0.7)
ax[0].text(3.4, R0 + 0.035, r'$R_0=0.81489$', fontsize=7, color='0.4',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.15))
lq2 = np.linspace(-6, -3, 50); ax[0].plot(lq2, C*10**(lq2/3), color='0.5', ls=':', lw=0.7)
ax[0].text(-5.5, 0.09, r'$C\,q_E^{1/3}$', fontsize=7, color='0.4')
ax[0].set_ylabel('$R_L/a$'); ax[0].legend(frameon=False, loc='lower right'); ax[0].set_ylim(0, 0.9)
sty = {'Eggleton': ('C0', '-', 'Eggleton (1983)'), 'Leahy': ('C2', '-.', 'Leahy & Leahy (2015)'), 'Qi': ('C1', '--', 'Qi et al. (2026)'),
       'F1': ('C4', ':', 'this work, 2 par.'), 'F3': ('C3', '-', 'this work, 6 par.')}
for k, (c, ls, lab) in sty.items():
    ax[1].plot(lq, 100*np.array(D['errs'][k]['err']), color=c, ls=ls, lw=1.0, label=lab)
ax[1].set_yscale('symlog', linthresh=0.05, linscale=0.8); ax[1].axhline(0, color='k', lw=0.4)
ax[1].set_ylim(-5, 8); ax[1].set_yticks([-4, -1, -0.1, 0, 0.1, 1, 4]); ax[1].set_yticklabels(['$-4$', '$-1$', '$-0.1$', '0', '0.1', '1', '4'])
ax[1].axvspan(-2, 2, color='0.92', zorder=0); ax[1].text(-1.9, 4.5, 'classical\nfit range', fontsize=6, color='0.4')
ax[1].set_xlabel(r'$\log_{10} q_E$,  $q_E=M_1/M_2$'); ax[1].set_ylabel('relative error (%)'); ax[1].legend(frameon=False, fontsize=6, loc='lower right', ncol=1)
fig.savefig(HERE / 'fig_fits.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Fig 2: asymptotics ----------------
legend_kw = dict(frameon=True, facecolor='white', framealpha=0.8,
                 edgecolor='none', fontsize=6.4, labelspacing=0.32)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
# points above the 73-point n=300 table (npts=120)
hi = np.loadtxt(HERE / "roche_lobe_large_qE.dat")
qe_ext, R_ext = hi[:, 0], hi[:, 1]
m = qE >= 30
qe_a = np.concatenate([qE[m], qe_ext]); R_a = np.concatenate([R[m], R_ext])
S = (R0 - R_a) * qe_a**(2 / 3) - K * np.log(qe_a) - c2
m_s = qe_a >= 1e3
xx = qe_a[m_s]**(-1 / 3) * np.log(qe_a[m_s])
ax[0].plot(xx, S[m_s], 'k.', ms=4, label='reference')
ax[0].axhline(0, color='C3', lw=0.7, label=r'$S\to0$')
ax[0].set_xlim(left=0)
ax[0].set_xlabel(r'$q_E^{-1/3}\ln q_E$')
ax[0].set_ylabel(r'$S(q_E)$')
ax[0].legend(loc='upper right', **legend_kw)
ax[0].set_title(r'dominant-star limit $q_E\to\infty$', fontsize=8)
# inset: linear combination with exact K, c2
x = np.log(qe_a)
y = (R0 - R_a) * qe_a**(2 / 3)
axins = ax[0].inset_axes([0.38, 0.18, 0.55, 0.42])
axins.plot(x, y, 'k.', ms=3)
xs = np.linspace(x.min(), x.max(), 80)
two = K * xs + c2
axins.plot(xs, two, 'C3-', lw=0.8, label=r'$K\ln q_E+c_2$')
axins.plot(xs, two + (D['c3'] + D['c3L'] * xs) * np.exp(-xs / 3), 'C3--', lw=0.8)
axins.set_xlabel(r'$\ln q_E$', fontsize=5.5, labelpad=0.5)
axins.set_ylabel(r'$(R_0-R_L/a)\,q_E^{2/3}$', fontsize=5.5, labelpad=0.5)
axins.tick_params(labelsize=5, pad=1.0, direction='in')
for sp in axins.spines.values():
    sp.set_linewidth(0.6)
axins.patch.set_alpha(0.95)
axins.legend(loc='upper left', frameon=True, facecolor='white',
             framealpha=0.8, edgecolor='none', fontsize=5.5, labelspacing=0.2)
lo = np.loadtxt(HERE / "roche_lobe_small_qE.dat")
qs, Rs = lo[:, 0], lo[:, 1]
qq = np.concatenate([qs, qE[qE <= 1e-3]]); RR = np.concatenate([Rs, R[qE <= 1e-3]]); tt = qq**(1/3)
yy = (RR/(C*tt) - 1)/tt
ax[1].plot(tt, yy, 'k.', ms=4, label='reference')
ts = np.linspace(0, 0.06, 200)
ax[1].axhline(a1, color='C0', ls='--', lw=0.9, label=r'$a_1$ (analytic)')
ax[1].plot(ts, a1 + a32*np.sqrt(ts), 'C2-.', lw=0.9, label=r'$a_1+a_{3/2}\sqrt{t}$ (analytic)')
ax[1].plot(ts, a1 + a32*np.sqrt(ts) + a2*ts + D['a52']*ts**1.5, 'C3-', lw=0.9,
           label=r'$+\,a_2 t$ (quadrature) $+\,a_{5/2}t^{3/2}$ (fitted)')
ax[1].set_xlabel(r'$t=q_E^{1/3}$'); ax[1].set_ylabel(r'$[R_L/(C a t)-1]/t$'); ax[1].set_xlim(0, 0.06); ax[1].set_ylim(-0.27, -0.228)
ax[1].legend(loc='lower right', **legend_kw); ax[1].set_title(r'Hill limit $q_E\to0$', fontsize=8)
fig.tight_layout(); fig.savefig(HERE / 'fig_asymptotics.pdf'); plt.close(fig)
print("figures done")
