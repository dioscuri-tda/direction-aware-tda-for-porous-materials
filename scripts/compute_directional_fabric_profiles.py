"""
Compute directional fabric-profile descriptors for porous structures.

For each direction d̂ the descriptor is a 1-D spatial profile of interface
normal alignment:

    F_d(r) = mean_{boundary voxels in slab r} [ (n̂ · d̂)² ]

where n̂ is the unit outward normal (normalised gradient) at each solid/void
interface voxel, and slab r contains voxels whose projection onto d̂ falls in
the r-th position along the direction.

Interpretation
--------------
  F_d(r) ≈ 1    interfaces in slab r predominantly face along d
  F_d(r) ≈ 0    interfaces in slab r predominantly perpendicular to d
  F_d(r) ≈ 1/3  isotropic distribution

The profile captures how interface orientation varies spatially along d, which
the single global fabric tensor cannot resolve.

For axis-aligned directions the slabs coincide exactly with grid planes.
For diagonal directions the same projection+binning approach as
compute_directional_porosity_profiles.py is used.

Column naming convention: fab_{h}_{k}_{l}_{i}
  e.g. direction [1,1,0] → fab_1_1_0_0 … fab_1_1_0_79

Usage
-----
python scripts/compute_directional_fabric_profiles.py \\
    --database /path/to/stiffness_rtp.csv \\
    --npy_col  path \\
    --output   database/baseline.fabric.directional/rtp/database.csv \\
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
# Canonical-frame loading  (same convention as other directional scripts)
# ---------------------------------------------------------------------------

_AXIS_PERM = {"x": None, "y": (2, 0, 1), "z": (1, 2, 0)}


def load_canonical(path: str) -> np.ndarray:
    """Load .npy and return in canonical (x, y, z) frame."""
    grid = np.load(path)
    m = re.search(r"_axis-([xyz])\.npy$", path)
    if m:
        perm = _AXIS_PERM[m.group(1)]
        if perm is not None:
            grid = grid.transpose(perm)
    return grid


# ---------------------------------------------------------------------------
# Directional fabric profile
# ---------------------------------------------------------------------------

def directional_fabric_profile(
    grid: np.ndarray,
    direction: tuple[float, float, float],
    n_out: int = 80,
) -> np.ndarray:
    """
    Spatial profile of interface-normal alignment along an arbitrary direction.

    For each projection slab perpendicular to d̂:
        F_d(r) = mean [ (n̂ · d̂)² ]  over boundary voxels in that slab

    Empty slabs (no boundary voxels) are filled by linear interpolation.
    The result is resampled to n_out evenly-spaced positions.
    """
    dx, dy, dz = direction
    d = np.array([dx, dy, dz], dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return np.full(n_out, 1.0 / 3.0)
    d_hat = d / norm

    g = grid.astype(float)
    nx, ny, nz = g.shape

    # Gradient of solid indicator in canonical (x,y,z) frame
    gx = np.gradient(g, axis=0)
    gy = np.gradient(g, axis=1)
    gz = np.gradient(g, axis=2)

    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    boundary = mag > 1e-10

    if not boundary.any():
        return np.full(n_out, 1.0 / 3.0)

    # (n̂ · d̂)² at every boundary voxel
    nx_n = gx[boundary] / mag[boundary]
    ny_n = gy[boundary] / mag[boundary]
    nz_n = gz[boundary] / mag[boundary]
    cos2 = (nx_n * d_hat[0] + ny_n * d_hat[1] + nz_n * d_hat[2]) ** 2

    # Projection of every voxel onto d̂ (for slab assignment)
    ii = np.arange(nx, dtype=float)
    jj = np.arange(ny, dtype=float)
    kk = np.arange(nz, dtype=float)
    proj = (
        d_hat[0] * ii[:, None, None]
        + d_hat[1] * jj[None, :, None]
        + d_hat[2] * kk[None, None, :]
    )

    # Use integer-rounded projections to define exact slab positions
    # (same approach as directional_porosity_profile — no binning artifacts)
    proj_int = np.round(proj * 1_000_000).astype(np.int64)
    unique_int, all_inv = np.unique(proj_int.ravel(), return_inverse=True)
    n_slabs = len(unique_int)

    # Accumulate cos2 only for boundary voxels; count boundary voxels per slab
    cos2_all = np.zeros(nx * ny * nz, dtype=np.float64)
    cos2_all[boundary.ravel()] = cos2
    bdry_mask = boundary.ravel().astype(np.float64)

    grp_cos2 = np.bincount(all_inv, weights=cos2_all, minlength=n_slabs)
    grp_bdry = np.bincount(all_inv, weights=bdry_mask, minlength=n_slabs)

    with np.errstate(invalid="ignore"):
        grp_mean = np.where(grp_bdry > 0, grp_cos2 / grp_bdry, np.nan)

    grp_pos = unique_int / 1_000_000

    # Fill empty slabs (no boundary voxels) by linear interpolation
    valid = np.isfinite(grp_mean)
    if valid.sum() == 0:
        return np.full(n_out, 1.0 / 3.0)
    if valid.sum() < n_slabs:
        grp_mean[~valid] = np.interp(
            grp_pos[~valid], grp_pos[valid], grp_mean[valid]
        )

    if n_slabs == n_out:
        return grp_mean

    out_pos = np.linspace(grp_pos[0], grp_pos[-1], n_out)
    return np.interp(out_pos, grp_pos, grp_mean)


# ---------------------------------------------------------------------------
# Worker / helpers
# ---------------------------------------------------------------------------

def _worker(args):
    path, basedir, directions, n_out = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = load_canonical(full)
    profiles = {d: directional_fabric_profile(grid, d, n_out) for d in directions}
    return path, profiles


def direction_tag(d):
    return "_".join(str(int(v)) if v == int(v) else str(v) for v in d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute directional fabric-profile descriptors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--npy_col",  default="path",
                        help="Column holding .npy paths (default 'path').")
    parser.add_argument("--output",   required=True)
    parser.add_argument("--basedir",  default=".")
    parser.add_argument("--direction", nargs=3, type=float, action="append",
                        metavar=("H", "K", "L"), dest="directions",
                        help="Direction vector; repeat for multiple. E.g. --direction 1 1 0")
    parser.add_argument("--n_out", type=int, default=80,
                        help="Output profile points per direction (default 80).")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not args.directions:
        parser.error("Specify at least one --direction H K L")

    directions = [tuple(d) for d in args.directions]

    df = pd.read_csv(args.database)
    if args.npy_col not in df.columns:
        raise ValueError(f"Column '{args.npy_col}' not found. Available: {list(df.columns)}")
    paths = df[args.npy_col].tolist()

    dir_strs = ["[" + " ".join(str(int(v)) if v == int(v) else str(v) for v in d) + "]"
                for d in directions]
    print(f"Directions : {', '.join(dir_strs)}")
    print(f"n_out      : {args.n_out}")
    print(f"Structures : {len(paths)}")

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, (p, args.basedir, directions, args.n_out)): p
                   for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                _, profiles = fut.result()
                results[p] = profiles
                print(f"[{i:5d}/{len(paths)}] {os.path.basename(p)}")
            except Exception as e:
                print(f"[{i:5d}/{len(paths)}] ERROR {p}: {e}")

    rows = []
    for _, row in df.iterrows():
        p = row[args.npy_col]
        rec = row.to_dict()
        if p in results:
            for d in directions:
                tag = direction_tag(d)
                for k, v in enumerate(results[p][d]):
                    rec[f"fab_{tag}_{k}"] = v
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"New columns: fab_{direction_tag(directions[0])}_0 … "
          f"fab_{direction_tag(directions[-1])}_{args.n_out - 1}")


if __name__ == "__main__":
    main()
