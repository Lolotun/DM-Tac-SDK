import cv2
import numpy as np
import time

# Camera index: try 0, 1, 2 depending on your system
CAMERA_ID = 0

# Parameters
BASELINE_FRAMES = 30
THRESHOLD = 6

# Arrow settings: sparse and readable
GRID_STEP = 30      # larger = fewer arrows
CELL_RADIUS = 8      # local averaging area
ARROW_SCALE = 10.0
MIN_FLOW_MAG = 0.24
MAX_ARROW_LEN = 24.0

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera/sensor stream")

print("Collecting baseline. Do not touch the sensor...")
baseline_frames = []

for _ in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read frame during baseline collection")

    baseline_frames.append(frame.astype(np.float32))
    cv2.imshow("Raw tactile stream", frame)
    cv2.waitKey(1)
    time.sleep(0.02)

I0 = np.mean(baseline_frames, axis=0).astype(np.float32)

print("Baseline collected. Start touching the sensor.")
print("Press 'q' to quit, 'b' to recalibrate baseline.")

prev_gray = None


def draw_arrows_on_frame(frame, flow, contact_mask):
    """
    Draw sparse, locally averaged optical-flow arrows directly
    over the ordinary camera frame.
    """
    vis = frame.copy()
    h, w = contact_mask.shape

    for y in range(GRID_STEP // 2, h, GRID_STEP):
        for x in range(GRID_STEP // 2, w, GRID_STEP):
            y0 = max(0, y - CELL_RADIUS)
            y1 = min(h, y + CELL_RADIUS + 1)
            x0 = max(0, x - CELL_RADIUS)
            x1 = min(w, x + CELL_RADIUS + 1)

            local_mask = contact_mask[y0:y1, x0:x1]

            # Draw arrows only where there is enough contact.
            if np.count_nonzero(local_mask) < 0.30 * local_mask.size:
                continue

            local_flow = flow[y0:y1, x0:x1][local_mask]
            if local_flow.size == 0:
                continue

            dx, dy = np.median(local_flow, axis=0)
            magnitude = float(np.hypot(dx, dy))

            if magnitude < MIN_FLOW_MAG:
                continue

            vx = float(dx) * ARROW_SCALE
            vy = float(dy) * ARROW_SCALE
            length = float(np.hypot(vx, vy))

            if length > MAX_ARROW_LEN:
                scale = MAX_ARROW_LEN / length
                vx *= scale
                vy *= scale

            end = (int(round(x + vx)), int(round(y + vy)))

            # Black outline and white center: readable on a light/dark frame.
            cv2.arrowedLine(
                vis, (x, y), end,
                (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.30
            )
            cv2.arrowedLine(
                vis, (x, y), end,
                (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.30
            )

    return vis


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Difference from the untouched sensor baseline.
    It = frame.astype(np.float32)
    D = np.abs(It - I0)
    D_gray = D.mean(axis=2)
    D_gray = cv2.GaussianBlur(D_gray, (5, 5), 0)

    # Contact region.
    M = D_gray > THRESHOLD

    # Keep your original deformation heatmap.
    D_vis = np.clip(D_gray * 4, 0, 255).astype(np.uint8)
    D_vis = cv2.applyColorMap(D_vis, cv2.COLORMAP_JET)

    # Optical flow between adjacent ordinary camera frames.
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        raw_with_arrows = frame.copy()
    else:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=25,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        # This is the main requested view:
        # ordinary sensor frame + arrows.
        raw_with_arrows = draw_arrows_on_frame(frame, flow, M)

    prev_gray = current_gray

    # Only two windows.
    cv2.imshow("Raw tactile frame + arrows", raw_with_arrows)
    cv2.imshow("Deformation heatmap", D_vis)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("b"):
        print("Recalibrating baseline. Do not touch the sensor...")
        baseline_frames = []

        for _ in range(BASELINE_FRAMES):
            ret, calibration_frame = cap.read()
            if not ret:
                continue

            baseline_frames.append(calibration_frame.astype(np.float32))
            cv2.imshow("Raw tactile frame + arrows", calibration_frame)
            cv2.waitKey(1)
            time.sleep(0.02)

        I0 = np.mean(baseline_frames, axis=0).astype(np.float32)
        prev_gray = None
        print("New baseline collected.")

cap.release()
cv2.destroyAllWindows()
