"""Repo layout for paper-generating scripts.

Data, figures, and TeX live in paper/; the generators live in scripts/.
"""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
PAPER = ROOT / "paper"
