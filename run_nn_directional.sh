#!/bin/bash
# CNN baseline experiments — one experiment per (dataset, target-group).
# Mirrors the catboost directional experiments for direct comparison.
# Results land in results/cnn/<dataset>/ and are notebook-compatible.
set -euo pipefail

SEED=100
OUTDIR_RTP="results/cnn/rtp"
OUTDIR_ATTD="results/cnn/attd"
OUTDIR_TD="results/cnn/td"

DB_RTP="structures/stiffness_rtp.csv"
DB_ATTD="structures/stiffness_attd.csv"
DB_TD="structures/stiffness_td.csv"

# Shared CNN hyperparameters (match existing successful runs)
MODEL="densenet-121"
LR=0.001
BATCH=24
EPOCHS=200
CV_FOLDS=5
FOLDS="0 1 2 3 4"
GRADIENT_CLIP=0.5
P_FLIP=0.3
ROLL=0.3

# ── Helper ───────────────────────────────────────────────────────────────────
run() {
    local db="$1"
    local outdir="$2"
    local targets="$3"
    local suffix="$4"

    python scripts/train_nn.py \
        --database       "$db" \
        --outputdir      "$outdir" \
        --path_column    path \
        --targets        $targets \
        --model_name     "$MODEL" \
        --lr             "$LR" \
        --batch_size     "$BATCH" \
        --n_epochs       "$EPOCHS" \
        --cv_folds       "$CV_FOLDS" \
        --folds          $FOLDS \
        --gradient_clip  "$GRADIENT_CLIP" \
        --p_flip         "$P_FLIP" \
        --aug_roll_ratio "$ROLL" \
        --optimizer      Adam \
        --size           80 \
        --seed           "$SEED" \
        --suffix         "$suffix" \
        --remove_model
}

# ── RTP (default) ────────────────────────────────────────────────────────────
echo "════════════════════════════════  RTP  ═══════════════════════════════════"

run "$DB_RTP" "$OUTDIR_RTP" "uniaxial"                  "rtp_uniaxial"
run "$DB_RTP" "$OUTDIR_RTP" "poisson"                   "rtp_poisson"
run "$DB_RTP" "$OUTDIR_RTP" "shear"                     "rtp_shear"
run "$DB_RTP" "$OUTDIR_RTP" "offdiagonal"               "rtp_offdiagonal"
run "$DB_RTP" "$OUTDIR_RTP" "uniaxial+poisson"          "rtp_uniaxialpluspoisson"
run "$DB_RTP" "$OUTDIR_RTP" "uniaxial+poisson+shear"    "rtp_uniaxialpluspoissonplusshear"
run "$DB_RTP" "$OUTDIR_RTP" "full_tensor"               "rtp_full_tensor"

# UNCOMMENT IF NEEDED
# ── ATTD ─────────────────────────────────────────────────────────────────────
# echo "════════════════════════════════  ATTD  ══════════════════════════════════"

# run "$DB_ATTD" "$OUTDIR_ATTD" "uniaxial"                "attd_uniaxial"
# run "$DB_ATTD" "$OUTDIR_ATTD" "poisson"                 "attd_poisson"
# run "$DB_ATTD" "$OUTDIR_ATTD" "shear"                   "attd_shear"
# run "$DB_ATTD" "$OUTDIR_ATTD" "offdiagonal"             "attd_offdiagonal"
# run "$DB_ATTD" "$OUTDIR_ATTD" "uniaxial+poisson"        "attd_uniaxialpluspoisson"
# run "$DB_ATTD" "$OUTDIR_ATTD" "uniaxial+poisson+shear"  "attd_uniaxialpluspoissonplusshear"
# run "$DB_ATTD" "$OUTDIR_ATTD" "full_tensor"             "attd_full_tensor"

# UNCOMMENT IF NEEDED
# ── TD ───────────────────────────────────────────────────────────────────────
# echo "════════════════════════════════  TD  ══════════════════════════════════"

# run "$DB_TD" "$OUTDIR_TD" "uniaxial"                "td_uniaxial"
# run "$DB_TD" "$OUTDIR_TD" "poisson"                 "td_poisson"
# run "$DB_TD" "$OUTDIR_TD" "shear"                   "td_shear"
# run "$DB_TD" "$OUTDIR_TD" "offdiagonal"             "td_offdiagonal"
# run "$DB_TD" "$OUTDIR_TD" "uniaxial+poisson"        "td_uniaxialpluspoisson"
# run "$DB_TD" "$OUTDIR_TD" "uniaxial+poisson+shear"  "td_uniaxialpluspoissonplusshear"
# run "$DB_TD" "$OUTDIR_TD" "full_tensor"             "td_full_tensor"

