import argparse
from pathlib import Path

import pandas as pd

from src.filtrations.pipeline import process_files


def main():
    parser = argparse.ArgumentParser(
        description="Compute cone+PCA directional filtrations for porous structures."
    )
    parser.add_argument("--database",    type=str, required=True,
                        help="Input CSV containing structure paths.")
    parser.add_argument("--npy_col",     type=str, default="path",
                        help="Column name holding .npy paths (default 'path').")
    parser.add_argument("--outputdir",   type=str, required=True)
    parser.add_argument("--radius",      type=int,   default=4,
                        help="PCA sphere radius in voxels (default: 4)")
    parser.add_argument("--cone-radius", type=float, default=3.0,
                        help="Cone base radius in voxels (default: 3.0)")
    parser.add_argument("--cone-height", type=float, default=6.0,
                        help="Cone half-height along loading axis in voxels (default: 6.0)")
    parser.add_argument("--direction",   type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        metavar=("H", "K", "L"),
                        help="Loading direction vector (default: 0 0 1 = z-axis). "
                             "Axis-aligned directions use exact permutation; "
                             "diagonal directions use nearest-neighbour rotation.")

    args = parser.parse_args()
    direction = tuple(args.direction)

    df = pd.read_csv(args.database)
    if args.npy_col not in df.columns:
        raise ValueError(f"Column '{args.npy_col}' not found in {args.database}. "
                         f"Available: {list(df.columns)}")
    paths = df[args.npy_col].tolist()

    print(f"Direction : {direction}")
    print(f"Filtration: radius={args.radius}  cone_radius={args.cone_radius}  cone_height={args.cone_height}")
    print(f"Structures: {len(paths)}")

    process_files(
        input_paths=paths,
        output_dir=Path(args.outputdir),
        radius=args.radius,
        cone_radius=args.cone_radius,
        cone_height=args.cone_height,
        direction=direction,
    )


if __name__ == "__main__":
    main()
