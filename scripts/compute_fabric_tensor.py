"""
Compute gradient-based fabric-tensor descriptors for porous structures.

The fabric tensor captures the orientation distribution of solid-void interfaces:

    F_ij = < n̂_i · n̂_j >_boundary

where n̂ is the unit outward normal at each interface voxel, estimated as the
normalised gradient of the solid indicator field (central differences).

Output per structure
--------------------
6 independent tensor components (symmetric 3×3, loading direction = axis 0):
  fabric_00   along loading axis (axis 0)
  fabric_11   first transverse axis (axis 1)
  fabric_22   second transverse axis (axis 2)
  fabric_01   off-diagonal loading/transverse
  fabric_02   off-diagonal loading/transverse
  fabric_12   off-diagonal transverse/transverse

3 eigenvalues in ascending order:
  fabric_eig1 ≤ fabric_eig2 ≤ fabric_eig3

For an isotropic structure F ≈ I/3 and all eigenvalues ≈ 1/3.
For a structure elongated along axis 0: fabric_11 ≈ fabric_22 > fabric_00.

All datasets store .npy files pre-permuted so that the loading direction maps
to numpy axis 0 — the same convention used by the cone-filtration code.

Usage
-----
python scripts/compute_fabric_tensor.py \\
    --database database/directional/rtp/database_both.csv \\
    --output   database/directional/rtp/database_fabric.csv \\
    --workers  8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def fabric_tensor(grid: np.ndarray):
    """
    Gradient-based fabric tensor and its eigenvalues.

    Returns
    -------
    F        : (3, 3) ndarray — symmetric fabric tensor (rows/cols: axis 0,1,2)
    eigvals  : (3,) ndarray  — eigenvalues in ascending order
    """
    g = grid.astype(float)
    d0 = np.gradient(g, axis=0)   # gradient along loading axis
    d1 = np.gradient(g, axis=1)
    d2 = np.gradient(g, axis=2)

    mag = np.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2)
    mask = mag > 1e-10
    if not mask.any():
        F = np.eye(3) / 3.0
        return F, np.full(3, 1.0 / 3.0)

    n0 = d0[mask] / mag[mask]
    n1 = d1[mask] / mag[mask]
    n2 = d2[mask] / mag[mask]

    F = np.array([
        [np.mean(n0 * n0), np.mean(n0 * n1), np.mean(n0 * n2)],
        [np.mean(n1 * n0), np.mean(n1 * n1), np.mean(n1 * n2)],
        [np.mean(n2 * n0), np.mean(n2 * n1), np.mean(n2 * n2)],
    ])
    eigvals = np.sort(np.linalg.eigvalsh(F))
    return F, eigvals


def _worker(args):
    path, basedir = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = np.load(full)
    F, eigvals = fabric_tensor(grid)
    return path, F, eigvals


def main():
    parser = argparse.ArgumentParser(
        description="Compute fabric-tensor descriptors.",
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

    results: dict[str, tuple] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, (p, args.basedir)): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                _, F, eigvals = fut.result()
                results[p] = (F, eigvals)
                print(f"[{i:5d}/{len(paths)}] {os.path.basename(p):60s} "
                      f"λ={eigvals[0]:.3f},{eigvals[1]:.3f},{eigvals[2]:.3f}")
            except Exception as e:
                print(f"[{i:5d}/{len(paths)}] ERROR {p}: {e}")

    rows = []
    for _, row in df.iterrows():
        p = row["npy_path"]
        meta = {c: row[c] for c in meta_cols}
        if p in results:
            F, eigvals = results[p]
            meta.update({
                "fabric_00": F[0, 0], "fabric_11": F[1, 1], "fabric_22": F[2, 2],
                "fabric_01": F[0, 1], "fabric_02": F[0, 2], "fabric_12": F[1, 2],
                "fabric_eig1": eigvals[0], "fabric_eig2": eigvals[1], "fabric_eig3": eigvals[2],
            })
        rows.append(meta)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print("Descriptor: 9 values  "
          "(fabric_00, fabric_11, fabric_22, fabric_01, fabric_02, fabric_12, "
          "fabric_eig1, fabric_eig2, fabric_eig3)")


if __name__ == "__main__":
    main()
