from pathlib import Path
import numpy as np
import os

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as ScipyRotation

from .speed import compute_cone_pca, NUMBA_OK


# ---------------------------------------------------------------------------
# Grid rotation helpers
# ---------------------------------------------------------------------------

# For axis-aligned directions: exact numpy transpose (no interpolation).
# np.transpose(grid, perm) means: new axis i = old axis perm[i].
# We want direction's axis to become axis 2 (z) in the new grid.
_AXIS_PERMS = {
    (1, 0, 0): (1, 2, 0),   # old x → new z
    (0, 1, 0): (0, 2, 1),   # old y → new z
    (0, 0, 1): None,         # already z, no-op
}


def rotate_grid_to_z(grid: np.ndarray, direction: tuple) -> np.ndarray:
    """
    Return a (possibly permuted/rotated) view of `grid` such that
    `direction` aligns with axis 2 (z).  The cone filtration always
    runs along axis 2, so this makes it direction-aware.

    Axis-aligned directions use exact numpy permutation (lossless).
    Diagonal directions use nearest-neighbour scipy rotation (binary grid).
    """
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    d = d / norm

    # Axis-aligned shortcut
    d_int = tuple(int(round(x)) for x in direction)
    if d_int in _AXIS_PERMS:
        perm = _AXIS_PERMS[d_int]
        return np.transpose(grid, perm) if perm is not None else grid

    # General case: rotate so d → z = (0,0,1)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(d, z)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.dot(d, z))

    if sin_a < 1e-10:
        # d already parallel to z (same or opposite direction)
        return grid if cos_a > 0 else grid[:, :, ::-1].copy()

    axis = cross / sin_a
    angle = np.arctan2(sin_a, cos_a)
    R = ScipyRotation.from_rotvec(angle * axis).as_matrix()

    # affine_transform maps output coords → input coords, so use R^T = R^{-1}
    center = (np.array(grid.shape, dtype=float) - 1.0) / 2.0
    offset = center - R.T @ center

    rotated = affine_transform(
        grid.astype(np.float32), R.T,
        offset=offset, order=0, cval=0.0,
        output_shape=grid.shape,
    )
    return (rotated > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-file and per-directory processing
# ---------------------------------------------------------------------------

def cone_pca_filtration(
    grid: np.ndarray,
    radius: int,
    cone_radius: float = 3.0,
    cone_height: float = 6.0,
) -> np.ndarray:
    if not NUMBA_OK:
        raise RuntimeError("Numba not available. Install with: pip install numba")
    return compute_cone_pca(grid, radius, cone_radius, cone_height)


def process_file(
    input_path: Path,
    output_path: Path,
    radius: int,
    cone_radius: float,
    cone_height: float,
    direction: tuple = (0, 0, 1),
) -> None:
    grid = np.load(input_path)
    if direction != (0, 0, 1):
        grid = rotate_grid_to_z(grid, direction)
    filt = cone_pca_filtration(grid, radius, cone_radius, cone_height)
    np.save(output_path, filt)


def process_files(
    input_paths: list,
    output_dir: Path,
    radius: int = 4,
    cone_radius: float = 3.0,
    cone_height: float = 6.0,
    direction: tuple = (0, 0, 1),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        output_path = output_dir / Path(input_path).name
        process_file(
            input_path=Path(input_path),
            output_path=output_path,
            radius=radius,
            cone_radius=cone_radius,
            cone_height=cone_height,
            direction=direction,
        )
