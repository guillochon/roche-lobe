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
- `paper/fast.py` — $L_1$, surface, and volume solver used by the figures
- `paper/reference.py` — extended-precision reference volumes
- `paper/figures.py` — writes `fig_sections.pdf`, `fig_surface3d.png`, `fig_limits.pdf`, `fig_asymptotics.pdf`
- `paper/derivative_analysis.py` — writes `fig_fits_deriv.pdf` and the derivative tables
- `paper/analysis.json`, `paper/derivative_reference.txt` — cached reference used by the figure scripts

## Regenerating figures

```
pip install -r requirements.txt
python figures.py
python derivative_analysis.py
```

from `paper/`. Both scripts read `analysis.json`; `derivative_analysis.py` also reads `derivative_reference.txt`.
