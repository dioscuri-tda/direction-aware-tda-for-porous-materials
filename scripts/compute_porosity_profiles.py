"""
Compute porosity-profile descriptors for porous structures.

The descriptor is the sequence of per-slice solid fractions along the loading
axis (numpy axis 0).  For an 80-voxel grid this gives 80 values.

All datasets store .npy files pre-permuted so that the loading direction maps
to numpy axis 0 — the same convention used by the cone-filtration code.

Usage
-----
python scripts/compute_porosity_profiles.py \\
    --database database/directional/rtp/database_both.csv \\
    --output   database/directional/rtp/database_porosity.csv \\
    --workers  8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def porosity_profile(grid: np.ndarray) -> np.ndarray:
    """Solid fraction at each slice along axis 0 (loading direction)."""
    return grid.astype(float).mean(axis=(1, 2))


def _worker(args):
    path, basedir = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = np.load(full)
    return path, porosity_profile(grid)


def main():
    parser = argparse.ArgumentParser(
        description="Compute porosity-profile descriptors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--database", required=True, help="Input database CSV.")
    parser.add_argument("--output",   required=True, help="Output CSV path.")
    parser.add_argument("--basedir",  default=".",
                        help="Base directory prepended to relative npy_path values (default '.').")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel worker processes (default 4).")
    args = parser.parse_args()

    df = pd.read_csv(args.database)
    meta_cols = [c for c in ("npy_path", "stress_axis", "cii") if c in df.columns]
    paths = df["npy_path"].tolist()

    results: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, (p, args.basedir)): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                _, profile = fut.result()
                results[p] = profile
                print(f"[{i:5d}/{len(paths)}] {os.path.basename(p)}")
            except Exception as e:
                print(f"[{i:5d}/{len(paths)}] ERROR {p}: {e}")

    N = next(iter(results.values())).shape[0]
    rows = []
    for _, row in df.iterrows():
        p = row["npy_path"]
        meta = {c: row[c] for c in meta_cols}
        if p in results:
            meta.update({f"porosity_{k}": v for k, v in enumerate(results[p])})
        rows.append(meta)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"Descriptor: {N} values  (porosity_0 … porosity_{N - 1})")


if __name__ == "__main__":
    main()
