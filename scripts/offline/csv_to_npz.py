"""
Convert a tactile sensor CSV recording to a compressed .npz archive.

Usage:
    python csv_to_npz.py path/to/session.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

H, W = 288, 384

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "recording"


def load_csv_session(path):
    df = pd.read_csv(path)
    n = len(df)

    time = df["time_s"].to_numpy()
    force = df[["fx", "fy", "fz", "mx", "my", "mz"]].to_numpy(dtype=np.float32)

    depth = np.vstack(
        [np.fromstring(s, sep=";") for s in df["depth_values"]]
    ).reshape(n, H, W)
    def_x = np.vstack(
        [np.fromstring(s, sep=";") for s in df["deformation_x_values"]]
    ).reshape(n, H, W)
    def_y = np.vstack(
        [np.fromstring(s, sep=";") for s in df["deformation_y_values"]]
    ).reshape(n, H, W)
    deformation = np.stack([def_x, def_y], axis=-1)

    return time, force, depth, deformation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="path to the input CSV file")
    args = parser.parse_args()

    time, force, depth, deformation = load_csv_session(args.csv_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (args.csv_path.stem + ".npz")

    np.savez_compressed(
        out_path,
        time=time,
        force=force,
        depth=depth,
        deformation=deformation,
    )
    print(f"Saved {len(time)} frames to {out_path}")


if __name__ == "__main__":
    main()
