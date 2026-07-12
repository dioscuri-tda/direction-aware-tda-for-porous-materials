"""
Collect per-structure TDA descriptor .npy files into a single CSV database,
with direction-tagged column names matching the baseline convention.

Column naming: {descriptor_type}_{h}_{k}_{l}_{i}
  e.g. direction [1,0,0], type ecp  → ecp_1_0_0_0 … ecp_1_0_0_342
  e.g. direction [1,1,0], type ph_cone → ph_cone_1_1_0_0 … ph_cone_1_1_0_299

The descriptor directory must contain one .npy file per structure, named
identically to the basename of the corresponding npy_path in the database CSV.
If a file is missing the row's descriptor columns are left as NaN.

Usage
-----
# First direction — creates output CSV
python scripts/collect_tda_descriptors.py \\
    --database  results/stiffness_rtp.csv \\
    --desc_dir  descriptors/directional/rtp/ecp/1_0_0 \\
    --output    database/directional.tda/rtp/database_ecp.csv \\
    --desc_type ecp \\
    --direction 1 0 0

# Subsequent directions — append columns to existing CSV
python scripts/collect_tda_descriptors.py \\
    --database  results/stiffness_rtp.csv \\
    --desc_dir  descriptors/directional/rtp/ecp/0_1_0 \\
    --output    database/directional.tda/rtp/database_ecp.csv \\
    --desc_type ecp \\
    --direction 0 1 0 \\
    --append
"""

import argparse
import os

import numpy as np
import pandas as pd


def direction_tag(d):
    return "_".join(str(int(v)) if v == int(v) else str(v) for v in d)


def collect(
    database: str,
    desc_dir: str,
    output: str,
    desc_type: str,
    direction: tuple,
    append: bool,
    npy_col: str,
    porosity: bool,
    basedir: str,
) -> None:
    tag = direction_tag(direction)
    prefix = f"{desc_type}_{tag}_"
    porosity_col = f"{desc_type}_porosity"

    # If appending, start from existing output CSV; otherwise from the database
    if append and os.path.isfile(output):
        df = pd.read_csv(output)
    else:
        df = pd.read_csv(database)

    new_cols: dict[str, list] = {}
    n_found = 0

    for _, row in df.iterrows():
        filename = os.path.basename(str(row[npy_col]))
        desc_path = os.path.join(desc_dir, filename)

        if os.path.isfile(desc_path):
            vec = np.load(desc_path).flatten()
            n_found += 1
            for i, v in enumerate(vec):
                new_cols.setdefault(f"{prefix}{i}", []).append(float(v))
        else:
            # Keep any existing descriptor length if we already have one key
            existing_len = next(
                (len(v) for v in new_cols.values()), None
            )
            fill_len = existing_len if existing_len is not None else 0
            for i in range(fill_len):
                new_cols.setdefault(f"{prefix}{i}", []).append(float("nan"))
            if fill_len == 0:
                # First row is missing — append NaN placeholder; will be backfilled
                new_cols.setdefault("__missing__", []).append(True)

    # Remove helper key if present
    new_cols.pop("__missing__", None)

    print(f"Direction [{' '.join(str(v) for v in direction)}]: "
          f"found {n_found}/{len(df)} descriptor files  →  {len(new_cols)} columns")

    # Solid volume fraction from structure files — stored as {desc_type}_porosity.
    # np.mean(arr) where solid=1 gives solid VF; only added on first direction call
    # (col absent) to avoid redundant recomputation on subsequent --append calls.
    if porosity and porosity_col not in df.columns:
        por_vals = []
        for _, row in df.iterrows():
            struct_path = os.path.join(basedir, str(row[npy_col]))
            if os.path.isfile(struct_path):
                arr = np.load(struct_path)
                por_vals.append(float(np.mean(arr.astype(np.float32))))
            else:
                por_vals.append(float("nan"))
        new_cols[porosity_col] = por_vals
        n_ok = sum(1 for v in por_vals if not np.isnan(v))
        print(f"  {porosity_col}: computed for {n_ok}/{len(df)} structures")

    for col, vals in new_cols.items():
        if len(vals) < len(df):
            vals.extend([float("nan")] * (len(df) - len(vals)))
        df[col] = vals

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-structure TDA descriptor files into a direction-tagged CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--database",  required=True,
                        help="Input CSV with structure paths and target values.")
    parser.add_argument("--desc_dir",  required=True,
                        help="Directory containing per-structure descriptor .npy files.")
    parser.add_argument("--output",    required=True,
                        help="Output CSV path.")
    parser.add_argument("--desc_type", required=True,
                        choices=["ecp", "ph_cone"],
                        help="Descriptor type — used as column prefix.")
    parser.add_argument("--direction", type=float, nargs=3, required=True,
                        metavar=("H", "K", "L"),
                        help="Direction vector (e.g. 1 0 0).")
    parser.add_argument("--append",    action="store_true",
                        help="Append columns to existing output CSV instead of "
                             "starting from --database.")
    parser.add_argument("--npy_col",   default="path",
                        help="Column in --database holding structure .npy paths "
                             "(default: path).")
    parser.add_argument("--porosity",  action="store_true",
                        help="Compute solid volume fraction from each structure .npy "
                             "and add a single '{desc_type}_porosity' column. "
                             "Only written on the first direction call (skipped if "
                             "column already exists when --append is used).")
    parser.add_argument("--basedir",   default=".",
                        help="Base directory prepended to relative structure paths "
                             "when --porosity is used (default: '.').")
    args = parser.parse_args()

    collect(
        database=args.database,
        desc_dir=args.desc_dir,
        output=args.output,
        desc_type=args.desc_type,
        direction=tuple(args.direction),
        append=args.append,
        npy_col=args.npy_col,
        porosity=args.porosity,
        basedir=args.basedir,
    )


if __name__ == "__main__":
    main()
