#!/usr/bin/env bash
# Compute the full 6x6 effective stiffness tensor (Voigt notation) for the three datasets used in the
# paper: RTP, TD, and ATTD. Uses scripts/compute_stiffness_tensor.py, which implements the FFT-based
# homogenization solver described in main.tex, Section 2 ("Estimation of the elastic stiffness tensor").
#
# Runs are resumable: compute_stiffness_tensor.py skips structures already present in the output CSV,
# so re-running this script after an interruption picks up where it left off instead of starting over.
#
# --workers is machine-specific (roughly the number of physical cores) -- adjust the value below to
# match the machine this runs on.
# --E_void_ratio 1e-2 (void/solid stiffness contrast K=100) is the production setting validated in the
# paper; see scripts/compute_stiffness_tensor.py for details.
set -euo pipefail

# RTP: Au0.30Ag0.70 alloy, E=81.5 GPa, nu=0.39
python scripts/compute_stiffness_tensor.py \
    --inputdir structures/rtp \
    --output   structures/stiffness_rtp.csv \
    --material AuAg \
    --E_void_ratio 1e-2 \
    --workers 22

# TD: aluminium, E=70.0 GPa, nu=0.33
python scripts/compute_stiffness_tensor.py \
    --inputdir structures/td \
    --output   structures/stiffness_td.csv \
    --material Al \
    --E_void_ratio 1e-2 \
    --workers 22

# ATTD: aluminium, E=70.0 GPa, nu=0.33
python scripts/compute_stiffness_tensor.py \
    --inputdir structures/attd \
    --output   structures/stiffness_attd.csv \
    --material Al \
    --E_void_ratio 1e-2 \
    --workers 22
