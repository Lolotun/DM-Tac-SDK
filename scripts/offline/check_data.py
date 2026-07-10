"""
check_data.py

Quick visual check of a recorded .npz (sensor wrench + gripper).
Shows three stacked subplots vs time:

  1) Commanded gripper force  vs  measured Fz  (two y-axes, different units)
  2) Contact forces Fx, Fy, Fz
  3) Gripper travel: target vs actual position (did it reach the target?)
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/home/physicalai/Denmark/DM-Tac-SDK/data/recording/tactile_20260710_134026.npz"

data = np.load(DATA_PATH)
print("keys:", list(data.keys()))

t = data["time"]                       # (N,)
force = data["force"]                  # (N, 6) -> Fx, Fy, Fz, Mx, My, Mz
fx, fy, fz = force[:, 0], force[:, 1], force[:, 2]

grip_target = data["gripper_target"]   # commanded position (0-255)
grip_pos = data["gripper_pos"]         # actual position    (0-255)
grip_force = data["gripper_force"]     # commanded force param (0-255)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# --- 1) Commanded gripper force vs measured Fz (different units -> twin axis) ---
ax0 = axes[0]
l1, = ax0.plot(t, grip_force, color="tab:red", label="Commanded gripper force (0-255)")
ax0.set_ylabel("Gripper force cmd [0-255]", color="tab:red")
ax0.tick_params(axis="y", labelcolor="tab:red")

ax0b = ax0.twinx()
l2, = ax0b.plot(t, fz, color="tab:blue", label="Measured Fz [N]")
ax0b.set_ylabel("Fz [N]", color="tab:blue")
ax0b.tick_params(axis="y", labelcolor="tab:blue")

ax0.set_title("Commanded gripper force vs measured Fz")
ax0.legend(handles=[l1, l2], loc="upper left")
ax0.grid(True, alpha=0.3)

# --- 2) Contact forces Fx / Fy / Fz ---
ax1 = axes[1]
ax1.plot(t, fx, label="Fx [N]")
ax1.plot(t, fy, label="Fy [N]")
ax1.plot(t, fz, label="Fz [N]")
ax1.set_ylabel("Force [N]")
ax1.set_title("Contact forces")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# --- 3) Gripper travel: target vs actual (did it reach?) ---
ax2 = axes[2]
ax2.plot(t, grip_target, label="Target position", linestyle="--", color="tab:orange")
ax2.plot(t, grip_pos, label="Actual position", color="tab:green")
ax2.set_ylabel("Gripper position [0-255]")
ax2.set_xlabel("Time [s]")
ax2.set_title("Gripper travel: target vs actual")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

# How close did the gripper get to its final target?
final_gap = abs(grip_target[-1] - grip_pos[-1])
print(f"Final target={grip_target[-1]:.0f}  actual={grip_pos[-1]:.0f}  gap={final_gap:.0f}")

fig.suptitle("Recorded sensor forces + gripper state", fontsize=13)
fig.tight_layout()
plt.show()
