"""
Compute directional two-point correlation (S2) descriptors for arbitrary directions.

    S2(r, d) = P(voxel at x is solid  AND  voxel at x + r·d is solid)

computed via periodic shifts (np.roll) for integer Miller-index direction vectors.
The physical lag range is kept fixed at [0, N/2] voxels regardless of direction,
then resampled to n_out points.

For axis-aligned directions (1,0,0) / (0,1,0) / (0,0,1) the result matches the
existing compute_twopoint_correlation.py exactly (no interpolation needed when
n_out = N//2 + 1).

Column naming convention: tpc_{h}_{k}_{l}_{r}
  e.g. direction [1,1,0] → tpc_1_1_0_0 … tpc_1_1_0_40

Usage
-----
python scripts/compute_directional_twopoint_correlation.py \\
    --database /path/to/stiffness_rtp.csv \\
    --npy_col  path \\
    --output   database/baseline.tpc.directional/rtp/database.csv \\
    --direction 1 0 0 \\
    --direction 0 1 0 \\
    --direction 0 0 1 \\
    --direction 1 1 0 \\
    --direction 1 0 1 \\
    --direction 0 1 1 \\
    --direction 1 1 1 \\
    --n_out 41 \\
    --workers 8
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical-frame loading  (same convention as directional porosity script)
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
# Directional S2
# ---------------------------------------------------------------------------

def directional_twopoint_correlation(
    grid: np.ndarray,
    direction: tuple[float, float, float],
    n_out: int = 41,
) -> np.ndarray:
    """
    Two-point correlation S2 along an arbitrary integer Miller-index direction.

    Uses periodic roll: S2(r) = mean_x [ g(x) · g(x + r·d) ]

    The physical lag range is 0 … N/2 voxels (same as axis-aligned S2).
    The result is resampled to n_out evenly-spaced lag values.
    """
    h, k, l = int(round(direction[0])), int(round(direction[1])), int(round(direction[2]))
    N = grid.shape[0]
    g = grid.astype(np.float32)

    step_len = np.sqrt(h * h + k * k + l * l)
    if step_len == 0:
        vf = g.mean()
        return np.full(n_out, vf * vf)

    # Number of integer steps to reach physical distance N/2
    r_max = max(int(round(N / 2 / step_len)), 1)

    S2 = np.empty(r_max + 1, dtype=np.float64)
    S2[0] = float(np.mean(g * g))
    shifted = g
    for r in range(1, r_max + 1):
        # Shift by one more step each iteration (cheaper than shifting by r from scratch)
        if h != 0:
            shifted = np.roll(shifted, -h, axis=0)
        if k != 0:
            shifted = np.roll(shifted, -k, axis=1)
        if l != 0:
            shifted = np.roll(shifted, -l, axis=2)
        S2[r] = float(np.mean(g * shifted))

    if len(S2) == n_out:
        return S2

    lags = np.arange(len(S2), dtype=float)
    out_lags = np.linspace(0.0, float(r_max), n_out)
    return np.interp(out_lags, lags, S2)


# ---------------------------------------------------------------------------
# Worker / helpers
# ---------------------------------------------------------------------------

def _worker(args):
    path, basedir, directions, n_out = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = load_canonical(full)
    profiles = {d: directional_twopoint_correlation(grid, d, n_out) for d in directions}
    return path, profiles


def direction_tag(d):
    return "_".join(str(int(v)) if v == int(v) else str(v) for v in d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute directional two-point correlation descriptors.",
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
    parser.add_argument("--n_out", type=int, default=41,
                        help="Output lag points per direction (default 41 = N//2+1 for N=80).")
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
                for r, v in enumerate(results[p][d]):
                    rec[f"tpc_{tag}_{r}"] = v
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"New columns: tpc_{direction_tag(directions[0])}_0 … "
          f"tpc_{direction_tag(directions[-1])}_{args.n_out - 1}")


if __name__ == "__main__":
    main()
