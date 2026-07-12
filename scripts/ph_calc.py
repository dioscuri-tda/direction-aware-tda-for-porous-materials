import argparse
import numpy as np
import os
import gudhi as gd
from gudhi.representations import DiagramSelector, PersistenceImage
from concurrent.futures import ProcessPoolExecutor

def get_vectorized_pers_from_filtration(
    filename, inputdir, outputdir,
    resolution=(10, 10),
    im_range=(0.0, 1.25, 0.0, 1.25),
):
    """ Computes and vectorise persistence diagrams (in 3 dimensions) from a given filtration using Persistence Images."""
    filepath = os.path.join(inputdir, filename)
    outputfilepath = os.path.join(outputdir, filename)

    filtration = np.load(filepath)
    if filtration.ndim == 4:
        filtration = filtration[:,:,:, 0] # Cone-filtration is first channel in multifiltration for ECP
    comp = gd.PeriodicCubicalComplex(top_dimensional_cells=filtration, periodic_dimensions=[False, False, True])
    comp.compute_persistence()
    persist_imager = PersistenceImage(resolution=resolution, im_range=list(im_range))
    gudhi_diag_selector = DiagramSelector(use=True, limit=np.inf, point_type="finite")
    output = []
    for i in range(3):
        intervals = gudhi_diag_selector(comp.persistence_intervals_in_dimension(i))
        if len(intervals) == 0:
             output.append(np.zeros(resolution[0] * resolution[1]))
             continue
        img_pers = persist_imager(intervals)
        output.append(img_pers)
    np.save(outputfilepath, np.array(output))


def process_directory(inputdir, outputdir, max_workers=8, resolution=(10, 10), im_range=(0.0, 1.25, 0.0, 1.25)):
    npy_files = [f for f in os.listdir(inputdir) if f.endswith('.npy')]
    max_workers = min(os.cpu_count(), max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                get_vectorized_pers_from_filtration,
                filename,
                inputdir,
                outputdir,
                resolution,
                im_range,
            )
            for filename in npy_files
        ]

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", type=str, required=True)
    parser.add_argument("--outputdir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=[10, 10],
                        metavar=("H", "W"),
                        help="Persistence image resolution (default: 10 10)")
    parser.add_argument("--im_range", type=float, nargs=4,
                        default=[0.0, 1.25, 0.0, 1.25],
                        metavar=("BIRTH_MIN", "BIRTH_MAX", "DEATH_MIN", "DEATH_MAX"),
                        help="Persistence image birth/death range (default: 0.0 1.25 0.0 1.25)")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_command_line_args()

    os.makedirs(args["outputdir"], exist_ok=True)
    process_directory(
        inputdir=args["inputdir"],
        outputdir=args["outputdir"],
        max_workers=args["max_workers"],
        resolution=tuple(args["resolution"]),
        im_range=tuple(args["im_range"]),
    )

