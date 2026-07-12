"""
Compute directional porosity-profile descriptors for arbitrary directions.

For each requested direction (h k l), the descriptor is the sequence of mean
solid fractions in slices perpendicular to that direction, resampled to a fixed
number of output values (default 80) regardless of direction.

The script works in the canonical coordinate frame (numpy axis 0 = physical x,
axis 1 = physical y, axis 2 = physical z).  Input .npy files may be in any of
the three pre-permuted forms used by the pipeline:

    *_axis-x.npy  →  canonical as-is
    *_axis-y.npy  →  transpose(2,0,1) to canonical
    *_axis-z.npy  →  transpose(1,2,0) to canonical
    (no suffix)   →  assumed canonical

Column naming convention: por_{h}_{k}_{l}_{i}
  e.g. direction [1,1,0] → por_1_1_0_0 … por_1_1_0_79

Usage
-----
python scripts/compute_directional_porosity_profiles.py \\
    --database /path/to/stiffness_rtp.csv \\
    --npy_col  path \\
    --output   database/baseline.porosity.directional/rtp/database.csv \\
    --direction 1 0 0 \\
    --direction 0 1 0 \\
    --direction 0 0 1 \\
    --direction 1 1 0 \\
    --direction 1 0 1 \\
    --direction 0 1 1 \\
    --direction 1 1 1 \\
    --n_out 80 \\
    --workers 8
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical-frame loading
# ---------------------------------------------------------------------------

_AXIS_PERM = {
    "x": None,          # already canonical
    "y": (2, 0, 1),
    "z": (1, 2, 0),
}


def load_canonical(path: str) -> np.ndarray:
    """Load .npy and return in canonical (x,y,z) frame."""
    grid = np.load(path)
    m = re.search(r"_axis-([xyz])\.npy$", path)
    if m:
        perm = _AXIS_PERM[m.group(1)]
        if perm is not None:
            grid = grid.transpose(perm)
    return grid


# ---------------------------------------------------------------------------
# Directional profile
# ---------------------------------------------------------------------------

def directional_porosity_profile(
    grid: np.ndarray,
    direction: tuple[float, float, float],
    n_out: int = 80,
) -> np.ndarray:
    """
    Solid-fraction profile along an arbitrary direction.

    Projects each voxel onto the direction vector, groups voxels by their
    (rounded) projection value — one group per natural "plane" perpendicular
    to the direction — computes mean solid fraction per group, then resamples
    to exactly n_out evenly-spaced points.

    For axis-aligned directions like (1,0,0) the groups correspond exactly to
    grid slices, so the result matches the existing porosity_profile function
    with no interpolation error.
    """
    dx, dy, dz = direction
    nx, ny, nz = grid.shape

    ii = np.arange(nx, dtype=float)
    jj = np.arange(ny, dtype=float)
    kk = np.arange(nz, dtype=float)
    proj = (
        dx * ii[:, None, None]
        + dy * jj[None, :, None]
        + dz * kk[None, None, :]
    )

    p_min, p_max = proj.min(), proj.max()
    if p_max == p_min:
        return np.full(n_out, grid.mean())

    # Round projections to avoid floating-point noise when grouping natural planes.
    # Scale by 1e6 and cast to int64 so np.unique can group them exactly.
    proj_int = np.round(proj * 1_000_000).astype(np.int64)
    unique_int, inv = np.unique(proj_int.ravel(), return_inverse=True)

    flat_solid = grid.ravel().astype(np.float32)
    grp_sum = np.bincount(inv, weights=flat_solid, minlength=len(unique_int))
    grp_cnt = np.bincount(inv, minlength=len(unique_int))
    grp_mean = grp_sum / grp_cnt          # all groups non-empty by construction
    grp_pos = unique_int / 1_000_000      # recover float projection positions

    if len(grp_pos) == n_out:
        return grp_mean

    # Interpolate to n_out evenly-spaced positions across the profile
    out_pos = np.linspace(grp_pos[0], grp_pos[-1], n_out)
    return np.interp(out_pos, grp_pos, grp_mean)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    path, basedir, directions, n_out = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = load_canonical(full)
    profiles = {d: directional_porosity_profile(grid, d, n_out) for d in directions}
    return path, profiles


def direction_tag(d):
    return "_".join(str(int(v)) if v == int(v) else str(v) for v in d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute directional porosity-profile descriptors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--database", required=True,
                        help="Input CSV containing structure paths.")
    parser.add_argument("--npy_col", default="path",
                        help="Column name holding .npy paths (default 'path').")
    parser.add_argument("--output", required=True,
                        help="Output CSV path.")
    parser.add_argument("--basedir", default=".",
                        help="Base directory prepended to relative paths (default '.').")
    parser.add_argument("--direction", nargs=3, type=float, action="append",
                        metavar=("H", "K", "L"), dest="directions",
                        help="Direction vector; repeat for multiple directions. "
                             "E.g. --direction 1 0 0 --direction 1 1 1")
    parser.add_argument("--n_out", type=int, default=80,
                        help="Number of output samples per direction (default 80).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4).")
    args = parser.parse_args()

    if not args.directions:
        parser.error("Specify at least one --direction H K L")

    directions = [tuple(d) for d in args.directions]

    df = pd.read_csv(args.database)
    if args.npy_col not in df.columns:
        raise ValueError(f"Column '{args.npy_col}' not found in {args.database}. "
                         f"Available: {list(df.columns)}")

    paths = df[args.npy_col].tolist()

    # Describe what we're about to compute
    dir_strs = ["[" + " ".join(str(int(v)) if v == int(v) else str(v) for v in d) + "]"
                for d in directions]
    print(f"Directions: {', '.join(dir_strs)}")
    print(f"n_out per direction: {args.n_out}")
    print(f"Total features per structure: {len(directions) * args.n_out}")
    print(f"Structures: {len(paths)}")

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_worker, (p, args.basedir, directions, args.n_out)): p
            for p in paths
        }
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                _, profiles = fut.result()
                results[p] = profiles
                print(f"[{i:5d}/{len(paths)}] {os.path.basename(p)}")
            except Exception as e:
                print(f"[{i:5d}/{len(paths)}] ERROR {p}: {e}")

    # Build output dataframe — keep all original columns, append profile features
    rows = []
    for _, row in df.iterrows():
        p = row[args.npy_col]
        rec = row.to_dict()
        if p in results:
            for d in directions:
                tag = direction_tag(d)
                for k, v in enumerate(results[p][d]):
                    rec[f"por_{tag}_{k}"] = v
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"New columns: por_{direction_tag(directions[0])}_0 … "
          f"por_{direction_tag(directions[-1])}_{args.n_out - 1}")


if __name__ == "__main__":
    main()
