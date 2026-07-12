#!/bin/bash
set -euo pipefail

# ── Directions ──────────────────────────────────────────────────────────────
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 0 1" "0 1 1" "1 1 1")

# ── Filtration hyperparameters ──────────────────────────────────────────────
RADIUS=3
CONE_RADIUS=3.0
CONE_HEIGHT=6.0

# ── ECP / PH hyperparameters (values used throughout the paper) ─────────────
ECP_GRID_RES=8
PH_RESOLUTION="12 12"
PH_TAG="12x12"
PH_IM_RANGE="0.0 1.25 0.0 1.25"

# ── Directory roots (filtration params encoded in name) ─────────────────────
# Strip trailing .0 from floats for cleaner tag (6.0 → 6, 3.0 → 3)
FILT_TAG="radius=${RADIUS}.coneheight=${CONE_HEIGHT%.0}.coneradius=${CONE_RADIUS%.0}"
FILT_ROOT="directional_filtrations.${FILT_TAG}"
DESC_ROOT="directional_descriptors.${FILT_TAG}"
DB_ROOT="database/directional.tda.${FILT_TAG}"

process_dataset() {
    local name="$1"
    local stiffness_csv="$2"

    echo ""
    echo "════════════════════════════════════════"
    echo "  Dataset : $name"
    echo "  Labels  : $stiffness_csv"
    echo "  FiltTag : $FILT_TAG"
    echo "════════════════════════════════════════"

    local filt_base="${FILT_ROOT}/${name}"
    local ecp_base="${DESC_ROOT}/${name}/ecp_gridres=${ECP_GRID_RES}"
    local ph_base="${DESC_ROOT}/${name}/ph_res=${PH_TAG}"

    # ── Steps 1-3: filtration + ECP + PH, per direction ────────────────────
    for dir_str in "${DIRECTIONS[@]}"; do
        local dir_tag="${dir_str// /_}"
        echo ""
        echo "  ── Direction [$dir_str] ──"

        # Step 1: filtration
        python scripts/directional_multifiltration.py \
            --database  "$stiffness_csv" \
            --outputdir "${filt_base}/${dir_tag}" \
            --radius        "$RADIUS" \
            --cone-radius   "$CONE_RADIUS" \
            --cone-height   "$CONE_HEIGHT" \
            --direction     $dir_str

        # Step 2: ECP
        mkdir -p "${ecp_base}/${dir_tag}"
        echo "    ECP gridres=${ECP_GRID_RES} → ${ecp_base}/${dir_tag}"
        julia --project="./" --threads auto scripts/directional_ecp.jl \
            "${filt_base}/${dir_tag}" \
            "${ecp_base}/${dir_tag}" \
            --grid-res "$ECP_GRID_RES"

        # Step 3: PH
        mkdir -p "${ph_base}/${dir_tag}"
        echo "    PH res=${PH_TAG} → ${ph_base}/${dir_tag}"
        python scripts/ph_calc.py \
            --inputdir  "${filt_base}/${dir_tag}" \
            --outputdir "${ph_base}/${dir_tag}" \
            --resolution $PH_RESOLUTION \
            --im_range   $PH_IM_RANGE

        # Filtration is intermediate — delete after ECP/PH computed
        rm -rf "${filt_base}/${dir_tag}"
        echo "  Deleted filtrations: ${filt_base}/${dir_tag}"
    done

    # ── Step 4: collect all directions into CSVs ────────────────────────────
    echo ""
    echo "  ── Collecting descriptors → CSVs ──"

    local ecp_csv="${DB_ROOT}/${name}/ecp_gridres=${ECP_GRID_RES}/database_ecp.csv"
    local ph_csv="${DB_ROOT}/${name}/ph_res=${PH_TAG}/database_ph_cone.csv"
    mkdir -p "$(dirname "$ecp_csv")" "$(dirname "$ph_csv")"

    local first=1
    for dir_str in "${DIRECTIONS[@]}"; do
        local dir_tag="${dir_str// /_}"
        local append_flag=""
        [ "$first" -eq 0 ] && append_flag="--append"

        python scripts/collect_tda_descriptors.py \
            --database  "$stiffness_csv" \
            --desc_dir  "${ecp_base}/${dir_tag}" \
            --output    "$ecp_csv" \
            --desc_type ecp \
            --direction $dir_str \
            --porosity \
            $append_flag

        python scripts/collect_tda_descriptors.py \
            --database  "$stiffness_csv" \
            --desc_dir  "${ph_base}/${dir_tag}" \
            --output    "$ph_csv" \
            --desc_type ph_cone \
            --direction $dir_str \
            --porosity \
            $append_flag

        first=0
    done
    echo "  ECP → ${ecp_csv}"
    echo "  PH  → ${ph_csv}"

    echo "  Done → ${DB_ROOT}/${name}/"
}

# ── RTP ─────────────────────────────────────────────────────────────────────
process_dataset "rtp" "structures/stiffness_rtp.csv"

# UNCOMMENT IF NEEDED
# ── TD ──────────────────────────────────────────────────────────────────────
# process_dataset "td" "structures/stiffness_td.csv"

# UNCOMMENT IF NEEDED
# ── ATTD ────────────────────────────────────────────────────────────────────
# process_dataset "attd" "structures/stiffness_attd.csv"
