#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get_force_record_npz.py

Live visualization (Dear PyGui) for a 6D wrench, with manual start/stop
recording straight to .npz (no intermediate CSV):

  - Force:  Fx, Fy, Fz
  - Torque: Mx, My, Mz

Recording behavior:
  - Nothing is recorded on start; press the button (or hotkey Space) to
    start recording. Press it again to stop and save.
  - Each stop writes one file: data/recording/tactile_<YYYYmmdd_HHMMSS>.npz
    (name = the date/time the recording was started), with the same
    layout consumed by parse_data.py:
        time, force, depth, deformation
  - You can start/stop as many times as you like; the app keeps running
    until you close the window or press Ctrl+C. If a recording is still
    in progress when you quit, it is saved automatically.
"""

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg

from dmrobotics import Sensor, SensorOptions, Mode

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "recording"


def _force_to_wrench(force_arr):
    """Normalize getForce() output into (fx, fy, fz, mx, my, mz)."""
    if isinstance(force_arr, (tuple, list)) and len(force_arr) == 2:
        force_arr = force_arr[1]
    f = np.asarray(force_arr, dtype=np.float32).reshape(-1)
    if f.size < 6:
        raise ValueError(f"wrench size < 6, got shape={np.asarray(force_arr).shape}")
    return tuple(map(float, f[:6]))


def _compute_ylim(ybuf, min_abs=1.0, margin_ratio=0.08, min_margin=1e-3):
    """Y-limits: at least [-1, 1], expand with data + margin."""
    if len(ybuf) == 0:
        return -min_abs, +min_abs

    y = np.asarray(ybuf, dtype=np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    span = max(y_max - y_min, min_margin)
    margin = max(span * margin_ratio, min_margin)

    lo = min(-min_abs, y_min - margin)
    hi = max(+min_abs, y_max + margin)

    if hi - lo < 2 * min_abs:
        lo = min(lo, -min_abs)
        hi = max(hi, +min_abs)

    return lo, hi


def main():
    # ====== Tune these parameters as needed ======
    dev_id = 0
    backend = "cuda"          # "cpu" / "cuda"
    max_fps = 10
    max_points = 1200
    update_hz = 10
    reset_cooldown_s = 0.8
    # ===========================================

    opt = SensorOptions(
        dev_id=dev_id,
        backend=backend,
        mode=Mode.STANDARD,
        show_fps=False,
        max_fps=max_fps,
        enable_raw=False,
        enable_deformation=True,
        enable_shear=False,
        enable_depth=True,
        enable_force=True,
        remote_addr="192.168.127.10:50051",
        pc_host="192.168.127.100",
        pc_port=60001,
    )
    sensor = Sensor(opt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Recording state (nothing is recorded until the user starts it) ----
    is_recording = False
    rec_run_id = None
    rec_t0 = 0.0
    rec_time = []
    rec_force = []
    rec_depth = []
    rec_deformation = []

    channels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    tbuf = deque(maxlen=max_points)
    bufs = {k: deque(maxlen=max_points) for k in channels}

    # ---------- Dear PyGui setup ----------
    dpg.create_context()
    dpg.create_viewport(
        title="dmrobotics Live Wrench (Fx/Fy/Fz + Mx/My/Mz)",
        width=1280,
        height=860,
        resizable=True,
    )

    xaxis_ids = {}
    yaxis_ids = {}
    series_ids = {}

    last_reset_ts = 0.0
    pending_reset = False

    status_text = None
    fps_text = None
    last_text = None
    record_button = None

    def save_recording():
        """Flush the current in-memory recording to a .npz file and clear it."""
        nonlocal rec_run_id, rec_time, rec_force, rec_depth, rec_deformation

        if not rec_time:
            print("[REC] Nothing recorded, skipping save.")
            rec_run_id = None
            return

        out_path = OUTPUT_DIR / f"tactile_{rec_run_id}.npz"
        np.savez_compressed(
            out_path,
            time=np.asarray(rec_time, dtype=np.float32),
            force=np.asarray(rec_force, dtype=np.float32),
            depth=np.asarray(rec_depth, dtype=np.float32),
            deformation=np.asarray(rec_deformation, dtype=np.float32),
        )
        print(f"[REC] Saved {len(rec_time)} frames to {out_path}")
        dpg.set_value(status_text, f"Saved {len(rec_time)} frames: {out_path.name}")

        rec_run_id = None
        rec_time = []
        rec_force = []
        rec_depth = []
        rec_deformation = []

    def start_recording():
        nonlocal is_recording, rec_run_id, rec_t0
        rec_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_t0 = time.time()
        is_recording = True
        dpg.set_item_label(record_button, "Stop Recording (Space)")
        dpg.set_value(status_text, f"Recording started: tactile_{rec_run_id}.npz")
        print(f"[REC] Recording started: tactile_{rec_run_id}.npz")

    def stop_recording():
        nonlocal is_recording
        is_recording = False
        dpg.set_item_label(record_button, "Start Recording (Space)")
        save_recording()

    def toggle_recording():
        if is_recording:
            stop_recording()
        else:
            start_recording()

    def on_record_button(sender, app_data, user_data):
        toggle_recording()

    def do_reset(reason: str):
        nonlocal last_reset_ts, pending_reset
        now = time.time()
        if now - last_reset_ts < reset_cooldown_s:
            dpg.set_value(status_text, f"Reset skipped (cooldown {reset_cooldown_s:.1f}s).")
            return

        pending_reset = True
        try:
            sensor.reset()
            last_reset_ts = now
            dpg.set_value(status_text, f"Reset triggered ({reason}). Keep sensor surface free of contact.")
        except Exception as e:
            dpg.set_value(status_text, f"[WARN] Reset failed: {e}")
        finally:
            pending_reset = False

    def on_reset_button(sender, app_data, user_data):
        do_reset("button")

    def on_key_press(sender, app_data):
        # app_data is key code
        if app_data in (ord("r"), ord("R")):
            do_reset("hotkey")
        elif app_data == 32:  # Spacebar
            toggle_recording()

    def build_plot(parent, name: str):
        with dpg.plot(label=name, height=-1, width=-1, parent=parent):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="t (s)")
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label=name)
            series = dpg.add_line_series([], [], label=name, parent=yaxis)
        return xaxis, yaxis, series

    # Main window (will be made primary so it fills viewport)
    with dpg.window(
        label="Wrench Plots",
        tag="MAIN_WINDOW",
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_collapse=True,
    ):
        # Plots area (leave fixed height for bottom status)
        with dpg.child_window(tag="PLOTS_AREA", border=True, height=-170):
            with dpg.group(tag="GRID_ROOT"):
                with dpg.group(horizontal=True, tag="ROW0"):
                    with dpg.child_window(tag="CELL_00", border=True): pass
                    with dpg.child_window(tag="CELL_01", border=True): pass
                    with dpg.child_window(tag="CELL_02", border=True): pass
                with dpg.group(horizontal=True, tag="ROW1"):
                    with dpg.child_window(tag="CELL_10", border=True): pass
                    with dpg.child_window(tag="CELL_11", border=True): pass
                    with dpg.child_window(tag="CELL_12", border=True): pass

            cell_tags = ["CELL_00", "CELL_01", "CELL_02", "CELL_10", "CELL_11", "CELL_12"]
            for cell, ch in zip(cell_tags, channels):
                xaxis, yaxis, series = build_plot(cell, ch)
                xaxis_ids[ch] = xaxis
                yaxis_ids[ch] = yaxis
                series_ids[ch] = series

        # Bottom status area (always visible)
        with dpg.child_window(tag="STATUS_AREA", border=True, height=-1):
            status_text = dpg.add_text("Starting...")
            fps_text = dpg.add_text("UI FPS: -- | Data: --")
            last_text = dpg.add_text("Last: --")
            dpg.add_separator()
            record_button = dpg.add_button(label="Start Recording (Space)", callback=on_record_button)
            dpg.add_button(label="Reset (R)", callback=on_reset_button)
            dpg.add_text(
                "Tips: Press Space (or button) to start/stop recording to .npz. "
                "Press 'R' to reset. Close window or Ctrl+C to exit "
                "(an in-progress recording is saved automatically).",
                color=(200, 200, 200),
            )

    # Key handler
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press)

    # Disable vsync (may still be limited by compositor/driver)
    dpg.set_viewport_vsync(False)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Make MAIN_WINDOW fill the viewport
    dpg.set_primary_window("MAIN_WINDOW", True)

    def layout_grid():
        """Resize 2x3 cell grid to fill PLOTS_AREA."""
        try:
            w, h = dpg.get_item_rect_size("PLOTS_AREA")
        except Exception:
            return
        if w <= 0 or h <= 0:
            return

        pad = 8
        cell_w = max(80, int((w - pad * 2) / 3))
        cell_h = max(80, int((h - pad * 1) / 2))

        for tag in ["CELL_00", "CELL_01", "CELL_02", "CELL_10", "CELL_11", "CELL_12"]:
            dpg.configure_item(tag, width=cell_w, height=cell_h)

    def fit_main_window_to_viewport():
        """Force MAIN_WINDOW to match the viewport client size."""
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        dpg.configure_item("MAIN_WINDOW", pos=(0, 0), width=vw, height=vh)

    def on_viewport_resize(sender, app_data):
        fit_main_window_to_viewport()
        layout_grid()

    dpg.set_viewport_resize_callback(on_viewport_resize)

    # Ensure correct sizing at startup (before first frames)
    fit_main_window_to_viewport()

    # ---------- Main loop ----------
    t0 = time.time()
    last_plot_update = 0.0

    ui_frame_cnt = 0
    ui_fps_t0 = time.time()
    ui_fps = 0.0

    data_cnt = 0
    data_fps_t0 = time.time()
    data_fps = 0.0

    did_first_layout = False
    last_fid = -1
    try:
        while dpg.is_dearpygui_running():
            now = time.time()
            if sensor.getDevStatus() != 0:
                continue
            if not sensor.wait_for_new(last_fid, timeout_ms=500):
                continue
            last_fid, _ = sensor.getDeformation2D()
            w = sensor.getForce()
            fx, fy, fz, mx, my, mz = _force_to_wrench(w)

            fid, depth = sensor.getDepth()
            _, deformation = sensor.getDeformation2D()

            t = now - t0

            # Only append to the recording buffers while recording is active.
            if is_recording and depth is not None and deformation is not None:
                rec_time.append(now - rec_t0)
                rec_force.append((fx, fy, fz, mx, my, mz))
                rec_depth.append(np.asarray(depth, dtype=np.float32))
                rec_deformation.append(np.asarray(deformation, dtype=np.float32))

            tbuf.append(t)
            bufs["Fx"].append(fx)
            bufs["Fy"].append(fy)
            bufs["Fz"].append(fz)
            bufs["Mx"].append(mx)
            bufs["My"].append(my)
            bufs["Mz"].append(mz)

            data_cnt += 1
            if now - data_fps_t0 >= 1.0:
                data_fps = data_cnt / (now - data_fps_t0)
                data_cnt = 0
                data_fps_t0 = now

            # Update plots at update_hz
            if now - last_plot_update >= 1.0 / max(1, update_hz):
                xs = list(tbuf)

                for ch in channels:
                    ys = list(bufs[ch])
                    dpg.set_value(series_ids[ch], [xs, ys])

                    if len(xs) >= 2:
                        dpg.set_axis_limits(xaxis_ids[ch], xs[0], xs[-1])

                    lo, hi = _compute_ylim(bufs[ch], min_abs=1.0)
                    dpg.set_axis_limits(yaxis_ids[ch], lo, hi)

                dpg.set_value(
                    last_text,
                    f"Last  Fx={fx:+.3f}  Fy={fy:+.3f}  Fz={fz:+.3f}   "
                    f"Mx={mx:+.3f}  My={my:+.3f}  Mz={mz:+.3f}"
                )

                if is_recording:
                    dpg.set_value(
                        status_text,
                        f"Recording... {len(rec_time)} frames "
                        f"({now - rec_t0:.1f}s) -> tactile_{rec_run_id}.npz",
                    )
                elif not pending_reset and (now - last_reset_ts) > 2.0:
                    dpg.set_value(status_text, "Idle. Press Space to start recording.")

                last_plot_update = now

            dpg.render_dearpygui_frame()

            if not did_first_layout:
                layout_grid()
                did_first_layout = True

            ui_frame_cnt += 1
            if now - ui_fps_t0 >= 1.0:
                ui_fps = ui_frame_cnt / (now - ui_fps_t0)
                ui_frame_cnt = 0
                ui_fps_t0 = now
                dpg.set_value(fps_text, f"UI FPS: {ui_fps:.1f} | Data rate: {data_fps:.1f} Hz")

            time.sleep(0.0005)

    except KeyboardInterrupt:
        pass
    finally:
        if is_recording:
            print("[REC] Exiting while recording, saving what we have...")
            save_recording()

        try:
            sensor.disconnect()
        except Exception:
            pass
        dpg.destroy_context()


if __name__ == "__main__":
    main()
