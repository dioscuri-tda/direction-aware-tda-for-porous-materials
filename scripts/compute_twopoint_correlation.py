"""
Compute two-point correlation (S2) descriptors along the loading axis.

    S2(r) = P(voxel at x is solid  AND  voxel at x + r·ê₀ is solid)

computed via circular FFT autocorrelation along axis 0, averaged over the
perpendicular (axis-1, axis-2) plane.  Only lags r = 0 … N/2 are stored
(the correlation is symmetric under r → N-r due to periodic boundary).

Interpretation
--------------
  S2(0) = φ           solid volume fraction
  S2(r) → φ²          uncorrelated limit (large r)
  decay rate encodes   characteristic length scale along loading axis

All datasets store .npy files pre-permuted so that the loading direction maps
to numpy axis 0 — the same convention used by the cone-filtration code.

Usage
-----
python scripts/compute_twopoint_correlation.py \\
    --database database/directional/rtp/database_both.csv \\
    --output   database/directional/rtp/database_tpc.csv \\
    --workers  8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def twopoint_correlation(grid: np.ndarray) -> np.ndarray:
    """
    Two-point correlation along axis 0 for lags 0 … N//2.

    Uses the FFT identity:
        IFFT(|FFT(g, axis=0)|²)[r, i, j] = Σ_k g[k,i,j] · g[(k+r)%N, i,j]
    then averages over (i, j) and normalises by N.
    """
    N = grid.shape[0]
    G = np.fft.rfft(grid.astype(float), axis=0)
    autocorr = np.fft.irfft(np.abs(G) ** 2, n=N, axis=0).real  # (N, N, N)
    S2 = autocorr.mean(axis=(1, 2)) / N                         # normalised probability
    return S2[: N // 2 + 1]


def _worker(args):
    path, basedir = args
    full = os.path.join(basedir, path) if not os.path.isabs(path) else path
    grid = np.load(full)
    return path, twopoint_correlation(grid)


def main():
    parser = argparse.ArgumentParser(
        description="Compute two-point correlation descriptors.",
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
                _, s2 = fut.result()
                results[p] = s2
                print(f"[{i:5d}/{len(paths)}] {os.path.basename(p)}")
            except Exception as e:
                print(f"[{i:5d}/{len(paths)}] ERROR {p}: {e}")

    n_lags = next(iter(results.values())).shape[0]
    rows = []
    for _, row in df.iterrows():
        p = row["npy_path"]
        meta = {c: row[c] for c in meta_cols}
        if p in results:
            meta.update({f"tpc_{r}": v for r, v in enumerate(results[p])})
        rows.append(meta)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")
    print(f"Descriptor: {n_lags} values  (tpc_0 … tpc_{n_lags - 1})")


if __name__ == "__main__":
    main()
