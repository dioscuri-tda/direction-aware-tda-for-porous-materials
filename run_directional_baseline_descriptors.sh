#!/usr/bin/env bash
# Compute directional baseline descriptors (porosity profile, two-point correlation function, fabric
# tensor profile) for RTP, TD, and ATTD, using the same 7 loading directions as the direction-aware TDA
# descriptors (3 coordinate axes, 3 face diagonals, 1 body diagonal). These are the non-topological
# directional baselines compared against TDA in the paper.
set -euo pipefail

DIRECTIONS="--direction 1 0 0 --direction 0 1 0 --direction 0 0 1 \
            --direction 1 1 0 --direction 1 0 1 --direction 0 1 1 --direction 1 1 1"

# ── Porosity profiles (n_out=80) ─────────────────────────────────────────────
python scripts/compute_directional_porosity_profiles.py \
    --database structures/stiffness_rtp.csv --npy_col path \
    --output   database/baseline.porosity.profiles/rtp/database.csv \
    $DIRECTIONS --n_out 80 --workers 8

python scripts/compute_directional_porosity_profiles.py \
    --database structures/stiffness_td.csv --npy_col path \
    --output   database/baseline.porosity.profiles/td/database.csv \
    $DIRECTIONS --n_out 80 --workers 8

python scripts/compute_directional_porosity_profiles.py \
    --database structures/stiffness_attd.csv --npy_col path \
    --output   database/baseline.porosity.profiles/attd/database.csv \
    $DIRECTIONS --n_out 80 --workers 8

# ── Two-point correlation function ────────────────────────────────
python scripts/compute_directional_twopoint_correlation.py \
    --database structures/stiffness_rtp.csv --npy_col path \
    --output   database/baseline.tpc.directional/rtp/database.csv \
    $DIRECTIONS --n_out 41 --workers 8

python scripts/compute_directional_twopoint_correlation.py \
    --database structures/stiffness_td.csv --npy_col path \
    --output   database/baseline.tpc.directional/td/database.csv \
    $DIRECTIONS --n_out 41 --workers 8

python scripts/compute_directional_twopoint_correlation.py \
    --database structures/stiffness_attd.csv --npy_col path \
    --output   database/baseline.tpc.directional/attd/database.csv \
    $DIRECTIONS --n_out 41 --workers 8

# ── Fabric tensor profile  ─────────────────────────────────────────
python scripts/compute_directional_fabric_profiles.py \
    --database structures/stiffness_rtp.csv --npy_col path \
    --output   database/baseline.fabric.directional/rtp/database.csv \
    $DIRECTIONS --n_out 80 --workers 8

python scripts/compute_directional_fabric_profiles.py \
    --database structures/stiffness_td.csv --npy_col path \
    --output   database/baseline.fabric.directional/td/database.csv \
    $DIRECTIONS --n_out 80 --workers 8

python scripts/compute_directional_fabric_profiles.py \
    --database structures/stiffness_attd.csv --npy_col path \
    --output   database/baseline.fabric.directional/attd/database.csv \
    $DIRECTIONS --n_out 80 --workers 8
