import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplcursors
from matplotlib.animation import FuncAnimation
import numpy as np

FORCE_LABELS = ["fx", "fy", "fz", "mx", "my", "mz"]
STEP = 10  # прореживание сетки для стрелок деформации


def plot_forces(time, force):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

    for ax, label, values in zip(axes.flat, FORCE_LABELS, force.T):
        line, = ax.plot(time[3:], values[3:])
        ax.set_title(label)
        ax.set_ylabel(label)
        mplcursors.cursor(line, hover=True)

    for ax in axes[-1]:
        ax.set_xlabel("time_s")

    fig.tight_layout()


def save_frame(depth, index, out_path):
    plt.imsave(out_path, depth[index], cmap="viridis")
    print(f"Saved frame {index} to {out_path}")


def animate_deformation(time, depth, deformation, start_idx, n_frames, fps, out_path):
    H, W = deformation.shape[1:3]
    y, x = np.mgrid[0:H:STEP, 0:W:STEP]

    fig, ax = plt.subplots()
    im = ax.imshow(depth[start_idx], cmap="gray", vmin = -0.1, vmax =0.3)
    u0 = deformation[start_idx, ::STEP, ::STEP, 0]
    v0 = deformation[start_idx, ::STEP, ::STEP, 1]
    quiv = ax.quiver(x, y, u0, v0, color="red")

    def update(frame):
        idx = start_idx + frame
        im.set_data(depth[idx])
        u = deformation[idx, ::STEP, ::STEP, 0]
        v = deformation[idx, ::STEP, ::STEP, 1]
        quiv.set_UVC(u, v)
        ax.set_title(f"t={time[idx]:.2f}s")
        return [im, quiv]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    anim.save(out_path, fps=fps)
    print(f"Saved animation ({n_frames} frames) to {out_path}")
    return anim


def main():
    parser = argparse.ArgumentParser(description="Plot force/torque channels and deformation from a recorded .npz session")
    parser.add_argument("npz_path",  type=Path, help="path to the session .npz file")
    parser.add_argument("output_dir", type=Path, help="directory to save frame/animation into")
    parser.add_argument("--frame-name", default="depth_frame.png", help="filename for the saved single frame")
    parser.add_argument("--anim-name", default="depth.gif", help="filename for the saved animation")
    parser.add_argument("--threshold", type=float, default=-4.0, help="fz threshold used to pick the start frame")
    parser.add_argument("--n-frames", type=int, default=40, help="number of frames in the animation")
    parser.add_argument("--fps", type=float, default=1, help="playback/save speed of the animation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz_path)
    time = data["time"]
    force = data["force"]
    depth = data["depth"]
    deformation = data["deformation"]

    idx = np.where(force[:, 2] < args.threshold)[0]
    print(time[idx])
    i = max(0, idx[2] )

    print (np.min(depth), np.max(depth[4:]))

    save_frame(depth, i, args.output_dir / args.frame_name)

    n_frames = min(args.n_frames, len(depth) - i)
    animate_deformation(
        time, depth, deformation,
        start_idx=i,
        n_frames=n_frames,
        fps=args.fps,
        out_path=args.output_dir / args.anim_name,
    )

    plot_forces(time, force)

    plt.show()


if __name__ == "__main__":
    main()
