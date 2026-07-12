#!/usr/bin/env bash
# Final TDA runs: ECP + PH + porosity → all target subsets, RTP + TD
set -euo pipefail

FILT_TAG="radius=3.coneheight=6.coneradius=3"
ALL7="1_0_0 0_1_0 0_0_1 1_1_0 1_0_1 0_1_1 1_1_1"
TARGETS="uniaxial poisson shear offdiagonal full_tensor"

# Dataset to run by default: rtp. To also run td/attd, add them to the list below
# (e.g. "for DATASET in rtp td attd; do").
for DATASET in rtp; do
    OUT="results/final/${DATASET}"
    COMMON="--path_col path --outputdir $OUT --cv_folds 5 --folds 0 1 2 3 4 --directions $ALL7"

    DB_ECP="database/directional.tda.${FILT_TAG}/${DATASET}/ecp_gridres=8/database_ecp.csv"
    DB_PH="database/directional.tda.${FILT_TAG}/${DATASET}/ph_res=12x12/database_ph_cone.csv"

    for TARGET in $TARGETS; do
        python scripts/train_catboost_directional.py $COMMON \
            --database "$DB_ECP" "$DB_PH" --features ecp ph_cone \
            --with_porosity \
            --targets "$TARGET" --suffix "tda_all7_${TARGET}"
    done
done
