#!/usr/bin/env bash
# Baseline runs: TPC, porosity profiles, fabric → all target subsets.
set -euo pipefail

ALL7="1_0_0 0_1_0 0_0_1 1_1_0 1_0_1 0_1_1 1_1_1"
TARGETS="uniaxial poisson shear offdiagonal full_tensor"

# Dataset to run by default: rtp. To also run td/attd, add them to the list below
# (e.g. "for DATASET in rtp td attd; do").
for DATASET in rtp; do
    OUT="results/final/${DATASET}"
    COMMON="--path_col path --outputdir $OUT --cv_folds 5 --folds 0 1 2 3 4 --directions $ALL7"

    # ── TPC ───────────────────────────────────────────────────────────────────
    DB="database/baseline.tpc.directional/${DATASET}/database.csv"
    for TARGET in $TARGETS; do
        python scripts/train_catboost_directional.py $COMMON \
            --database "$DB" --features tpc \
            --targets "$TARGET" --suffix "tpc_all7_${TARGET}"
    done

    # ── Porosity profiles ─────────────────────────────────────────────────────
    DB="database/baseline.porosity.profiles/${DATASET}/database.csv"
    for TARGET in $TARGETS; do
        python scripts/train_catboost_directional.py $COMMON \
            --database "$DB" --features por \
            --targets "$TARGET" --suffix "por_all7_${TARGET}"
    done

    # ── Fabric ────────────────────────────────────────────────────────────────
    DB="database/baseline.fabric.directional/${DATASET}/database.csv"
    for TARGET in $TARGETS; do
        python scripts/train_catboost_directional.py $COMMON \
            --database "$DB" --features fab \
            --targets "$TARGET" --suffix "fab_all7_${TARGET}"
    done
done
