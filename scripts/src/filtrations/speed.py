import numpy as np

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    # fall back to non-numba version if needed


def make_sphere_offsets(radius: int) -> np.ndarray:
    """All (dx,dy,dz) with dx^2+dy^2+dz^2 <= r^2."""
    r2 = radius * radius
    offs = []
    for dx in range(-radius, radius + 1):
        dx2 = dx * dx
        for dy in range(-radius, radius + 1):
            dy2 = dy * dy
            for dz in range(-radius, radius + 1):
                if dx2 + dy2 + dz * dz <= r2:
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=np.int16)


def make_cone_offsets(radius: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute offsets in the cone bounding box and their weights.
    Weight is 1.0 if inside cone at that dz, else 0.0 (matches Julia get_impact).
    """
    radius = float(radius)
    height = float(abs(height))
    max_x = int(np.ceil(radius))
    max_y = int(np.ceil(radius))
    max_z = int(height)

    offs = []
    ws = []
    if height == 0.0:
        return np.zeros((0, 3), dtype=np.int16), np.zeros((0,), dtype=np.float32)

    for dx in range(-max_x, max_x + 1):
        for dy in range(-max_y, max_y + 1):
            orth = (dx * dx + dy * dy) ** 0.5
            for dz in range(-max_z, max_z + 1):
                z_abs = abs(dz)
                if z_abs > height:
                    w = 0.0
                else:
                    # inside radius proportional to z_abs
                    w = 1.0 if orth <= (radius * z_abs / height) else 0.0
                if w != 0.0:
                    offs.append((dx, dy, dz))
                    ws.append(w)

    return np.asarray(offs, dtype=np.int16), np.asarray(ws, dtype=np.float32)


if NUMBA_OK:
    @njit(inline="always")
    def _mod_index(i: int, n: int) -> int:
        return i % n

    @njit(inline="always")
    def _grid_get(grid: np.ndarray, x: int, y: int, z: int) -> int:
        sx, sy, sz = grid.shape
        return grid[_mod_index(x, sx), _mod_index(y, sy), _mod_index(z, sz)]

    @njit
    def _power_iteration_top_evec_3x3(C: np.ndarray, iters: int = 12) -> np.ndarray:
        """
        Fast top eigenvector approximation for symmetric 3x3 matrix.
        Good enough for PCA direction; deterministic and numba-friendly.
        """
        v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # normalize
        n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
        if n != 0.0:
            v /= n

        for _ in range(iters):
            w0 = C[0, 0]*v[0] + C[0, 1]*v[1] + C[0, 2]*v[2]
            w1 = C[1, 0]*v[0] + C[1, 1]*v[1] + C[1, 2]*v[2]
            w2 = C[2, 0]*v[0] + C[2, 1]*v[1] + C[2, 2]*v[2]
            n = (w0*w0 + w1*w1 + w2*w2) ** 0.5
            if n == 0.0:
                break
            v[0], v[1], v[2] = w0/n, w1/n, w2/n
        return v

    @njit(inline="always")
    def _local_direction_yz_fast(grid: np.ndarray, x: int, y: int, z: int, sphere_offsets: np.ndarray) -> tuple[float, float]:
        """
        Computes PCA direction using running sums (no dataset build).
        Returns (dir_y, dir_z) mapped as 1 - abs(component).
        """
        # Running sums for mean and second moment
        n = 0.0
        sx = sy = sz = 0.0
        sxx = sxy = sxz = 0.0
        syy = syz = 0.0
        szz = 0.0

        for i in range(sphere_offsets.shape[0]):
            dx = int(sphere_offsets[i, 0])
            dy = int(sphere_offsets[i, 1])
            dz = int(sphere_offsets[i, 2])

            if _grid_get(grid, x + dx, y + dy, z + dz) == 0:
                continue

            n += 1.0
            fx = float(dx); fy = float(dy); fz = float(dz)

            sx += fx; sy += fy; sz += fz
            sxx += fx*fx; sxy += fx*fy; sxz += fx*fz
            syy += fy*fy; syz += fy*fz
            szz += fz*fz

        if n < 2.0:
            return 1.0, 1.0

        mx = sx / n
        my = sy / n
        mz = sz / n

        # Covariance = E[vv^T] - m m^T
        C = np.empty((3, 3), dtype=np.float32)
        exx = sxx / n; exy = sxy / n; exz = sxz / n
        eyy = syy / n; eyz = syz / n
        ezz = szz / n

        C[0, 0] = exx - mx*mx
        C[0, 1] = exy - mx*my
        C[0, 2] = exz - mx*mz
        C[1, 0] = C[0, 1]
        C[1, 1] = eyy - my*my
        C[1, 2] = eyz - my*mz
        C[2, 0] = C[0, 2]
        C[2, 1] = C[1, 2]
        C[2, 2] = ezz - mz*mz

        v = _power_iteration_top_evec_3x3(C)

        # Normalize (power iter already normalized, but keep safe)
        nn = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
        if nn != 0.0:
            v /= nn

        vy = 1.0 - abs(v[1])
        vz = 1.0 - abs(v[2])
        return vy, vz

    @njit(parallel=True)
    def cone_pca_filtration_fast(
        grid: np.ndarray,
        sphere_offsets: np.ndarray,
        cone_offsets: np.ndarray,
        cone_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Output: (X,Y,Z,3) => [cone, dir_y, dir_z]
        """
        X, Y, Z = grid.shape
        out = np.empty((X, Y, Z, 3), dtype=np.float32)

        total_cone_w = 0.0
        for i in range(cone_weights.shape[0]):
            total_cone_w += cone_weights[i]

        for x in prange(X):
            for y in range(Y):
                for z in range(Z):
                    if grid[x, y, z] == 0:
                        out[x, y, z, 0] = 1.25
                        out[x, y, z, 1] = 1.25
                        out[x, y, z, 2] = 1.25
                        continue

                    # Cone channel
                    full = 0.0
                    for i in range(cone_offsets.shape[0]):
                        dx = int(cone_offsets[i, 0])
                        dy = int(cone_offsets[i, 1])
                        dz = int(cone_offsets[i, 2])
                        if _grid_get(grid, x + dx, y + dy, z + dz) != 0:
                            full += cone_weights[i]

                    if total_cone_w == 0.0:
                        cone_val = 1.0
                    else:
                        cone_val = 1.0 - (full / total_cone_w)

                    out[x, y, z, 0] = cone_val

                    # PCA directional channels
                    vy, vz = _local_direction_yz_fast(grid, x, y, z, sphere_offsets)
                    out[x, y, z, 1] = vy
                    out[x, y, z, 2] = vz

        return out


def compute_cone_pca(grid: np.ndarray, radius: int, cone_radius: float, cone_height: float) -> np.ndarray:
    """
    Public API: uses numba fast path if available, else raises with guidance.
    """
    if not NUMBA_OK:
        raise RuntimeError(
            "Numba is not available. Install it (pip install numba) to use the fast implementation."
        )

    sphere_offsets = make_sphere_offsets(radius)
    cone_offsets, cone_weights = make_cone_offsets(cone_radius, cone_height)

    # Ensure a numba-friendly dtype (bool -> uint8 is often faster)
    g = np.asarray(grid)
    if g.dtype != np.uint8:
        g = g.astype(np.uint8, copy=False)

    return cone_pca_filtration_fast(g, sphere_offsets, cone_offsets, cone_weights)
