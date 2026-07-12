"""
Compute the full 6x6 effective stiffness tensor for voxelized porous structures
using FFT-based homogenization
"""

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.fft import fftn, ifftn, fftfreq


# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------

MATERIALS = {
    "AuAg": dict(E=81.5, nu=0.39,  description="Au0.30Ag0.70 alloy (RTP dataset)"),
    "Al":   dict(E=70.0, nu=0.33,  description="Aluminium (TD/ATTD datasets)"),
}


# ---------------------------------------------------------------------------
# Core FFT homogenization
# ---------------------------------------------------------------------------

def _lame(E, nu):
    """Lamé constants from Young's modulus and Poisson's ratio."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    return lam, mu


def _green_operator(tau_hat, n_vec, c1, c2):
    """Apply Green's operator Gamma_0 to tau_hat in Fourier space."""
    q = np.einsum('...ij,...j->...i', tau_hat, n_vec)
    s = np.einsum('...i,...i->...', n_vec, q)
    return (
        c1 * (n_vec[..., :, None] * q[..., None, :]
              + q[..., :, None] * n_vec[..., None, :])
        - c2 * s[..., None, None]
             * (n_vec[..., :, None] * n_vec[..., None, :])
    )


def _compatible_projection(f, lam_ref, mu_ref, c1_ref, c2_ref, n_vec, E_mac, N):
    """
    Project field f onto compatible strains with mean E_mac (C_ref-weighted).

    phi = sym(n ⊗ v),  v = A^{-1} (n . C_ref : f),
    A = (lam_ref+mu_ref) n⊗n + mu_ref I  (acoustic tensor of C_ref),
    A^{-1} = (1/mu_ref) I - c2_ref n⊗n.
    DC component enforced: phi_hat[0] = E_mac * N^3.
    """
    tr_f = f[..., 0, 0] + f[..., 1, 1] + f[..., 2, 2]
    C_ref_f = (lam_ref * tr_f)[..., None, None] * np.eye(3) + 2.0 * mu_ref * f
    C_ref_f_hat = fftn(C_ref_f, axes=(0, 1, 2))
    # b = n . (C_ref:f)_hat  — shape (..., 3)
    b = np.einsum('...i,...ij->...j', n_vec, C_ref_f_hat)
    # v = A^{-1} b  (A^{-1} = 2*c1_ref * I - c2_ref * n⊗n)
    s = np.einsum('...i,...i->...', n_vec, b)
    v = 2.0 * c1_ref * b - c2_ref * s[..., None] * n_vec
    # phi_hat = sym(n ⊗ v)
    phi_hat = 0.5 * (n_vec[..., :, None] * v[..., None, :] +
                     v[..., :, None] * n_vec[..., None, :])
    phi_hat[0, 0, 0] = E_mac * N**3
    return np.real(ifftn(phi_hat, axes=(0, 1, 2)))


def compute_stiffness_tensor(grid, E_solid, nu_solid,
                              E_void_ratio=1e-3,
                              max_iter=600, tol=1e-6):
    """
    Compute the 6x6 effective stiffness tensor (Voigt notation) for a single
    binary voxelized structure using ADMM-accelerated FFT homogenization with
    geometric-mean reference medium (Michel, Moulinec & Suquet 2001).

    Voigt ordering: [xx, yy, zz, yz, xz, xy]

    Parameters
    ----------
    grid : (N, N, N) bool/int array  —  True / 1 = solid phase
    E_solid : float  — Young's modulus of the solid phase (any unit, e.g. GPa)
    nu_solid : float — Poisson's ratio of the solid phase
    E_void_ratio : float — E_void / E_solid (default 1e-3).
        Contrast K = 1/E_void_ratio.
        ADMM convergence rate: rho = (sqrt(K)-1)/(sqrt(K)+1).
        For K=1000 (ratio=1e-3): rho≈0.94, converges in ~380 iterations.
        For K=100  (ratio=1e-2): rho≈0.82, converges in ~120 iterations.
    max_iter : int  — max iterations per load case (default 600)
    tol : float     — relative convergence tolerance on compatible strain change

    Returns
    -------
    C_eff : (6, 6) ndarray — effective stiffness tensor in Voigt notation,
            same units as E_solid.
    n_iters : list[int]   — number of iterations per load case (length 6)
    """
    N = grid.shape[0]
    solid = grid.astype(bool)

    lam_s, mu_s = _lame(E_solid, nu_solid)
    lam_v, mu_v = _lame(E_solid * E_void_ratio, nu_solid)

    # Scalar Lamé fields over the grid
    lam_loc = np.where(solid, lam_s, lam_v)   # (N, N, N)
    mu_loc  = np.where(solid, mu_s,  mu_v)

    # Reference medium: geometric mean of phase moduli (optimal for ADMM).
    # Gives ADMM convergence rate rho = (sqrt(K)-1)/(sqrt(K)+1), which
    # is the square root of the basic scheme's (K-1)/(K+1).
    mu_ref  = np.sqrt(mu_s  * mu_v)
    C11_ref = np.sqrt((lam_s + 2*mu_s) * (lam_v + 2*mu_v))
    lam_ref = C11_ref - 2.0 * mu_ref

    c1_ref = 1.0 / (2.0 * mu_ref)
    c2_ref = (lam_ref + mu_ref) / (mu_ref * (lam_ref + 2.0 * mu_ref))

    # Unit wavevectors — built once, reused for all 6 load cases
    freq = fftfreq(N)
    KX, KY, KZ = np.meshgrid(freq, freq, freq, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    n_vec = np.stack([KX, KY, KZ], axis=-1) / np.sqrt(K2)[..., None]  # (N,N,N,3)

    # Per-voxel local-solve prefactors for (C_loc + C_ref)^{-1}
    # (C_loc + C_ref):eps = rhs  →  eps = (rhs - lam_eff * tr(rhs)/M_eff * I) / (2*mu_eff)
    mu_eff  = mu_loc  + mu_ref   # (N, N, N)
    lam_eff = lam_loc + lam_ref
    M_eff   = 3.0 * lam_eff + 2.0 * mu_eff

    I3 = np.eye(3)

    # 6 unit macroscopic strain tensors (Voigt order: xx,yy,zz,yz,xz,xy)
    # Engineering convention: shear Voigt strain = 1  →  tensor strain = 0.5
    voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    E_macros = []
    for a, b in voigt_pairs:
        E = np.zeros((3, 3))
        if a == b:
            E[a, b] = 1.0
        else:
            E[a, b] = E[b, a] = 0.5
        E_macros.append(E)

    C_eff   = np.zeros((6, 6))
    n_iters = []

    for J, E_mac in enumerate(E_macros):
        # ADMM initialisation: compatible strain phi = E_mac, dual variable u = 0
        phi = np.broadcast_to(E_mac, (N, N, N, 3, 3)).copy()
        u   = np.zeros((N, N, N, 3, 3))

        converged_at = max_iter
        eps = phi.copy()
        norm_mac = np.linalg.norm(E_mac) * N**1.5 + 1e-14
        for it in range(max_iter):
            # --- Step 1: local solve ---
            # eps_new = (C_loc + C_ref)^{-1} : C_ref : (phi - u)
            pmu = phi - u
            tr_pmu = pmu[..., 0, 0] + pmu[..., 1, 1] + pmu[..., 2, 2]
            rhs = (lam_ref * tr_pmu)[..., None, None] * I3 + 2.0 * mu_ref * pmu
            tr_rhs  = rhs[..., 0, 0] + rhs[..., 1, 1] + rhs[..., 2, 2]
            tr_eps  = tr_rhs / M_eff
            eps_new = (rhs - (lam_eff * tr_eps)[..., None, None] * I3) \
                      / (2.0 * mu_eff[..., None, None])

            # --- Step 2: compatible projection ---
            # phi_new = P_{C_ref}(eps_new + u)
            phi_new = _compatible_projection(
                eps_new + u, lam_ref, mu_ref, c1_ref, c2_ref, n_vec, E_mac, N)

            # --- Step 3: dual update ---
            u = u + eps_new - phi_new

            # Convergence: both primal (||eps - phi||) and dual (||phi_new - phi||)
            # residuals must be small.  Using only the dual residual underestimates
            # primal infeasibility at high contrast, causing stress errors.
            primal_res = np.linalg.norm(eps_new - phi_new) / norm_mac
            dual_res   = np.linalg.norm(phi_new - phi)     / norm_mac
            phi = phi_new
            eps = eps_new

            if primal_res < tol and dual_res < tol:
                converged_at = it + 1
                break

        n_iters.append(converged_at)

        # Average stress from the LOCAL strain eps, which satisfies the
        # constitutive law exactly (C_loc:eps = sigma at each voxel).
        # Using phi (compatible strain) instead causes systematic underestimation
        # at high contrast because the primal residual ||eps - phi|| is larger
        # than the dual residual at declared convergence.
        tr_eps = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
        sig_avg = (
            (lam_loc * tr_eps)[..., None, None] * I3
            + 2.0 * mu_loc[..., None, None] * eps
        ).mean(axis=(0, 1, 2))

        for I, (a, b) in enumerate(voigt_pairs):
            C_eff[I, J] = sig_avg[a, b]

    return C_eff, n_iters


def elastic_properties(C_eff):
    """
    Extract engineering elastic constants from a Voigt stiffness matrix.

    Returns a dict with:
      E_x, E_y, E_z          — uniaxial Young's moduli
      G_yz, G_xz, G_xy       — shear moduli
      nu_xy, nu_xz, nu_yz    — Poisson ratios
      A_zener                 — Zener anisotropy ratio 2 G_xy / (C11 - C12)
                                (= 1 for isotropic response)
      bulk_modulus            — effective bulk modulus (isotropic approx)
    """
    S = np.linalg.inv(C_eff)

    E_x = 1.0 / S[0, 0]
    E_y = 1.0 / S[1, 1]
    E_z = 1.0 / S[2, 2]
    G_yz = 1.0 / S[3, 3]
    G_xz = 1.0 / S[4, 4]
    G_xy = 1.0 / S[5, 5]

    nu_xy = -S[0, 1] / S[0, 0]
    nu_xz = -S[0, 2] / S[0, 0]
    nu_yz = -S[1, 2] / S[1, 1]

    # Zener anisotropy ratio (cubic symmetry approximation)
    dC = C_eff[0, 0] - C_eff[0, 1]
    A_zener = (2.0 * C_eff[5, 5] / dC) if abs(dC) > 1e-12 else float('nan')

    # Voigt bulk modulus K = (C11+C22+C33 + 2(C12+C13+C23)) / 9
    K_voigt = (C_eff[0,0] + C_eff[1,1] + C_eff[2,2]
               + 2*(C_eff[0,1] + C_eff[0,2] + C_eff[1,2])) / 9.0

    return dict(
        E_x=E_x, E_y=E_y, E_z=E_z,
        G_yz=G_yz, G_xz=G_xz, G_xy=G_xy,
        nu_xy=nu_xy, nu_xz=nu_xz, nu_yz=nu_yz,
        A_zener=A_zener, K_voigt=K_voigt,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _is_axis_variant(filename):
    """Return True if the filename contains an axis tag like _axis-x."""
    return bool(re.search(r'_axis-[xyz]\.npy$', filename))


def _axis_tag(filename):
    """Extract axis tag: 'x', 'y', or 'z'. Returns None if not present."""
    m = re.search(r'_axis-([xyz])\.npy$', filename)
    return m.group(1) if m else None


def collect_files(inputdir):
    """
    Return list of .npy paths to process.

    For directories containing axis-variant files (e.g. RTP, ATTD), only the
    *_axis-z.npy variant is kept — it is the canonical orientation (no axis
    permutation applied).  The full stiffness tensor computed from it gives all
    six independent Young's and shear moduli.

    For directories without axis variants (TD), every file
    is returned.
    """
    all_npy = sorted(
        os.path.join(inputdir, f)
        for f in os.listdir(inputdir)
        if f.endswith('.npy')
    )

    has_variants = any(_is_axis_variant(p) for p in all_npy)

    if has_variants:
        # Keep only _axis-z.npy (canonical, no rotation)
        files = [p for p in all_npy if _axis_tag(p) == 'z']
        print(f"Axis-variant directory detected.  "
              f"Using {len(files)} *_axis-z.npy files out of {len(all_npy)} total.")
    else:
        files = all_npy
        print(f"Found {len(files)} structure files.")

    return files


# ---------------------------------------------------------------------------
# Worker (runs in subprocess for parallel execution)
# ---------------------------------------------------------------------------

def _worker(args):
    path, E_solid, nu_solid, E_void_ratio, max_iter, tol = args
    t0 = time.time()
    grid = np.load(path)
    C, n_iters = compute_stiffness_tensor(
        grid, E_solid, nu_solid,
        E_void_ratio=E_void_ratio,
        max_iter=max_iter, tol=tol,
    )
    props = elastic_properties(C)
    elapsed = time.time() - t0

    row = {"path": path, "elapsed_s": round(elapsed, 1)}
    row.update(props)

    # Store all 21 independent elements of the upper triangle
    labels = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    for i in range(6):
        for j in range(i, 6):
            row[f"C_{labels[i]}_{labels[j]}"] = C[i, j]

    row["converged"] = all(n < max_iter for n in n_iters)
    row["max_iters_used"] = max(n_iters)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute full 6x6 stiffness tensor for porous structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--inputdir", required=True,
                   help="Directory containing .npy structure files.")
    p.add_argument("--output", required=True,
                   help="Output CSV file path.")
    p.add_argument("--material", choices=list(MATERIALS), default=None,
                   help=(f"Material preset. Options: "
                         + ", ".join(f"{k} ({v['description']})"
                                     for k, v in MATERIALS.items())))
    p.add_argument("--E",  type=float, default=None,
                   help="Young's modulus of solid phase (GPa or any consistent unit). "
                        "Overrides --material.")
    p.add_argument("--nu", type=float, default=None,
                   help="Poisson's ratio of solid phase. Overrides --material.")
    p.add_argument("--E_void_ratio", type=float, default=1e-3,
                   help="E_void / E_solid ratio (default 1e-3). "
                        "ADMM convergence: rho=(sqrt(K)-1)/(sqrt(K)+1). "
                        "K=1000 (ratio=1e-3): ~220 iters. K=100 (ratio=1e-2): ~35 iters.")
    p.add_argument("--max_iter", type=int, default=600,
                   help="Max iterations per load case (default 600). "
                        "~380 needed for E_void_ratio=1e-3 (K=1000, rho=0.94).")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Relative convergence tolerance (default 1e-6).")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel worker processes (default 4).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N files (0 = all). Useful for testing.")
    return p.parse_args()


def _load_done(output_path):
    """Return set of already-processed absolute paths from an existing output CSV."""
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_csv(output_path)
        return {os.path.abspath(p) for p in existing["path"].tolist()}
    except Exception:
        return set()


def _append_row(output_path, row, write_header):
    """Append one result row to the output CSV (creates file on first call)."""
    pd.DataFrame([row]).to_csv(output_path, mode="a", header=write_header, index=False)


def main():
    args = parse_args()

    # --- resolve material constants ---
    if args.E is not None and args.nu is not None:
        E_solid, nu_solid = args.E, args.nu
        print(f"Material: E={E_solid}, nu={nu_solid} (user-specified)")
    elif args.material is not None:
        m = MATERIALS[args.material]
        E_solid, nu_solid = m["E"], m["nu"]
        print(f"Material: {args.material} — {m['description']} (E={E_solid}, nu={nu_solid})")
    else:
        p = argparse.ArgumentParser()
        p.error("Specify either --material or both --E and --nu.")

    # --- collect files ---
    files = collect_files(args.inputdir)
    if args.limit > 0:
        files = files[:args.limit]
        print(f"Limiting to first {len(files)} files.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # --- resume: skip already-done structures ---
    done_paths = _load_done(os.path.abspath(args.output))
    if done_paths:
        n_before = len(files)
        files = [f for f in files if os.path.abspath(f) not in done_paths]
        print(f"Resuming: {len(done_paths)} already done, "
              f"{n_before - len(files)} skipped, {len(files)} remaining.")

    if not files:
        print("Nothing to do — all structures already computed.")
        return

    # header is needed only when starting a fresh output file
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    worker_args = [
        (f, E_solid, nu_solid, args.E_void_ratio, args.max_iter, args.tol)
        for f in files
    ]

    print(f"\nRunning {len(files)} structures with {args.workers} workers "
          f"(void ratio={args.E_void_ratio}, tol={args.tol}) ...\n")

    new_results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, a): a[0] for a in worker_args}
        for i, fut in enumerate(as_completed(futures), 1):
            path = futures[fut]
            try:
                row = fut.result()
                new_results.append(row)
                _append_row(args.output, row, write_header)
                write_header = False
                elapsed_total = time.time() - t_start
                rate = i / elapsed_total
                eta  = (len(files) - i) / rate if rate > 0 else 0
                print(f"[{i:4d}/{len(files)}] {os.path.basename(path):50s}  "
                      f"E_z={row['E_z']:6.3f}  A={row['A_zener']:5.3f}  "
                      f"t={row['elapsed_s']:.0f}s  ETA={eta/60:.1f}min")
            except Exception as e:
                print(f"[{i:4d}/{len(files)}] ERROR {os.path.basename(path)}: {e}")

    # sort the full output file (existing + new rows) by path
    df = pd.read_csv(args.output).sort_values("path").reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df)} rows total → {args.output}  "
          f"({len(new_results)} new this run)")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")

    # Quick summary
    numeric_cols = ['E_x','E_y','E_z','G_yz','G_xz','G_xy','A_zener']
    print("\nSummary (mean ± std):")
    for col in numeric_cols:
        if col in df:
            print(f"  {col:12s}: {df[col].mean():.3f} ± {df[col].std():.3f}")


if __name__ == "__main__":
    main()
