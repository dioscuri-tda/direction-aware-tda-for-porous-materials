import pandas as pd
import numpy as np

def write_binary_grid_to_vtk(
    grid: np.ndarray,
    filepath: str,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    field_name: str = "material",
):
    """
    Write a binary 3D numpy grid (e.g., 80x80x80) to a legacy VTK file readable by ParaView.

    The output is DATASET STRUCTURED_POINTS with POINT_DATA scalars.
    Assumes grid is indexed as grid[x, y, z].

    Parameters
    ----------
    grid : np.ndarray
        3D array of shape (nx, ny, nz) containing 0/1 or bool.
    filepath : str
        Output filename, usually ending in ".vtk".
    spacing : tuple[float, float, float]
        Voxel spacing (dx, dy, dz).
    origin : tuple[float, float, float]
        Grid origin (ox, oy, oz).
    field_name : str
        Name of the scalar field in ParaView.
    """
    g = np.asarray(grid)
    if g.ndim != 3:
        raise ValueError(f"grid must be 3D, got shape {g.shape}")
    nx, ny, nz = g.shape

    # Convert to 0/1 unsigned byte
    g = (g != 0).astype(np.uint8)

    # VTK legacy expects x-fastest ordering; Fortran order achieves that for (nx, ny, nz)
    flat = g.ravel(order="F")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Binary 3D grid\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {field_name} unsigned_char 1\n")
        f.write("LOOKUP_TABLE default\n")

        # Write values with line breaks for readability
        per_line = 20
        for i, v in enumerate(flat, start=1):
            f.write(f"{int(v)} ")
            if i % per_line == 0:
                f.write("\n")
        f.write("\n")

# Example
S = np.load('structures/rtp/349_axis-x.npy')
write_binary_grid_to_vtk(S, filepath='rtp_349.vtk')