"""Filesystem layout for this repository.

RAW_ZIP must point at the public deposit (Ciecierski & Mandat, Mendeley Data,
doi:10.17632/tr93krswn2.1) as downloaded, i.e. the zip that contains
<patient>/<inside|outside>/*_processed.npy together with selected_files.csv.
Override it with the environment variable NEDBS_RAW_ZIP.

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGOUT = ROOT / "figures_out"
DATA.mkdir(exist_ok=True)
FIGOUT.mkdir(exist_ok=True)

RAW_ZIP = Path(os.environ.get(
    "NEDBS_RAW_ZIP",
    ROOT / "raw" / "drive-download-20261002T191830Z-1-001.zip",
))
