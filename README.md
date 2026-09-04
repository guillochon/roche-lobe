# Exact Asymptotics of the Roche-Lobe Radius

LaTeX source and figures for Guillochon & Prochaska on the exact analytic
structure of the volume-equivalent Roche-lobe radius $R_{\rm L}(q_E)$ and a
six-coefficient fit constrained by those asymptotics.

## Build

```
pdflatex roche_lobe && bibtex roche_lobe && pdflatex roche_lobe && pdflatex roche_lobe
```

from `paper/`. `aastex702.cls`, `orcid-ID.png`, and `aasjournalv7.bst` are included.

## Files

- `paper/roche_lobe.tex`, `paper/roche.bib` — manuscript
- `paper/table_reference.tex`, `paper/table_derivative.tex`, `paper/table_qcrit.tex` — appendix tables
- `paper/fig_limits.pdf` — exact limiting shapes (dominant-star and Hill)
- `paper/fig_asymptotics.pdf` — large- and small-$q_E$ expansions
- `paper/fig_sections.pdf` — orbital- and meridional-plane cross sections
- `paper/fig_fits_deriv.pdf` — $R_{\rm L}/a$ and $D=\partial\ln(R_{\rm L}/a)/\partial\ln q_E$ errors
- `paper/fig_surface3d.png` — three-dimensional lobe surfaces
- `paper/roche_lobe_reference_table.dat` — 73-point reference solution
- `paper/analysis.json`, `paper/derivative_reference.txt` — cached reference used by the figure scripts
- `scripts/fast.py` — $L_1$, surface, and volume solver used by the figures
- `scripts/reference.py` — extended-precision reference volumes
- `scripts/figures.py` — writes `fig_sections.pdf`, `fig_surface3d.png`, `fig_limits.pdf`, `fig_asymptotics.pdf`
- `scripts/derivative_analysis.py` — writes `fig_fits_deriv.pdf` and the derivative tables
- `scripts/c2b.py`, `scripts/v2u2.py`, `scripts/fp.py`, `scripts/fp2.py`, `scripts/hillred.py` — closed-form $c_2$ and $a_2$ quadratures
- `scripts/reproduce.py` — driver for the commands below

## Regenerating figures from cached data

```
pip install -r requirements.txt
python scripts/reproduce.py --versions
python scripts/reproduce.py --figures
```

from the repository root. Both figure scripts read `paper/analysis.json`; `derivative_analysis.py` also reads `paper/derivative_reference.txt`. This is the path that rebuilds every figure in the paper.

## Regenerating coefficients, quadratures, and tables

`python scripts/reproduce.py` documents the end-to-end path from source calculations. Cached outputs in `paper/` are the published numbers; the scripts below are what produced them.

| command | what it rebuilds | runtime |
|---|---|---|
| `--quadratures` | $V_H$ (`hill3.py`), $V_1$ (`a1.py`), reduced-model $8\pi/(9\sqrt{3})$ (`hillpert.py`), $R_0$ (`limit_q0.py`) | minutes |
| `--closure` | $c_2$ (`c2b.py`), $V_2^{(U_2)}$ (`v2u2.py`), $V_2^{\rm red}$ and $a_2$ (`fp.py`, `fp2.py`) | ~1 hour |
| `--galois` | $L_1$ Galois groups (`checks.py`) | seconds |
| `--analysis` | minimax coefficients, published-formula errors, $q_E<10^{-6}$ points → `analysis.json` | minutes |
| `--reference` | 73-point table, `npts=300` (`reference.py table`) | hours |
| `--derivative` | $D=\partial\ln(R_L/a)/\partial\ln q_E$ (`make_derivative_reference.py`) | hours |
| `--figures` | all manuscript figures from the caches | minutes |

Nelder–Mead settings: least-squares start, then `method='Nelder-Mead'` on the maximum relative error, `xatol=1e-11`, `fatol=1e-13`, `maxiter=40000` (`analysis.py`). Reference volumes: 80-bit `longdouble`, 70 bisection steps, panel corner at angular distance $4\epsilon$ from $L_1$ (`reference.py`). Hill quadratures: `mpmath` `dps=40` (`hill3.py`) and `dps=30` (`a1.py`). Closure quadratures: `mpmath` `dps=60` (`c2b.py`), `dps=25` (`v2u2.py`), `dps=28` (`fp.py`, `fp2.py`); reduced-model volumes in `hillred.py` / `paper/vred.txt`. High-$q_E$ points $10^{7,8,9}$ used in Figure 2a are listed in `figures.py`.

Python ≥3.10; see `--versions` for numpy/scipy/matplotlib/mpmath/sympy. The 73-point table and derivative file are platform-sensitive at the $10^{-15}$ level because of `longdouble`.
