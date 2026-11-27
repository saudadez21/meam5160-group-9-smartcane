"""Project VL53L8CX zones into the calibrated RGB camera."""

import cv2
import numpy as np

TOF_FOV_DEG = 45.0
TOF_RIGHT_OFFSET_M = 0.035
# The center followed by the four corners of one zone in camera image order.
ZONE_OFFSETS = np.array(
    [[0.0, 0.0], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
)


def camera_order(zones):
    """Return the zone grid in camera image order for this mount."""
    return zones[::-1, ::-1]


def project_tof(frame, matrix, distortion):
    """Return raw camera pixels for zone centers and corners, plus readings."""
    depth = camera_order(frame.distance_mm)
    tentative = camera_order(frame.tentative_mask)
    usable = camera_order(frame.valid_mask) | tentative
    rows, columns = np.nonzero(usable)
    distances = depth[usable]
    if distances.size == 0:
        return np.empty((0, 5, 2)), distances, tentative[usable]

    # Intersect the nominal zone rays with the plane at each axial depth.
    ray_step = np.tan(np.radians(TOF_FOV_DEG / 2)) / 4
    centers = np.column_stack((columns, rows)) - 3.5
    rays = (centers[:, None, :] + ZONE_OFFSETS) * ray_step
    z = distances[:, None] * 0.001
    points = np.empty((*rays.shape[:2], 3))
    points[..., :2] = rays * z[..., None]
    points[..., 0] += TOF_RIGHT_OFFSET_M
    points[..., 2] = z
    zero = np.zeros(3)
    pixels = cv2.projectPoints(points.reshape(-1, 3), zero, zero, matrix, distortion)[0]
    return pixels.reshape(-1, 5, 2), distances, tentative[usable]


def draw_tof_points(image, frame, matrix, distortion, raw_size):
    """Draw the centers and return the visible count and optional outlines."""
    pixels, distances, tentative = project_tof(frame, matrix, distortion)
    raw_size = np.asarray(raw_size)
    center_pixels = pixels[:, 0]
    visible = ((center_pixels >= 0) & (center_pixels <= raw_size - 1)).all(axis=1)
    height, width = image.shape[:2]
    scale = np.array((width, height)) / raw_size
    centers = np.floor((center_pixels[visible] + 0.5) * scale)
    dark = (16, 20, 24)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y), depth, uncertain in zip(
        centers.astype(int), distances[visible], tentative[visible], strict=True
    ):
        point = (int(x), int(y))
        color = (40, 190, 255) if uncertain else (255, 240, 78)
        cv2.circle(image, point, 5, dark, -1, cv2.LINE_AA)
        cv2.circle(image, point, 4, color, 1 if uncertain else -1, cv2.LINE_AA)
        text = str(depth)
        text_width = cv2.getTextSize(text, font, 0.36, 1)[0][0]
        label_x = min(int(x) + 7, width - text_width - 2)
        label_y = min(height - 3, max(11, int(y) + 4))
        cv2.putText(image, text, (label_x, label_y), font, 0.36, dark, 3, cv2.LINE_AA)
        cv2.putText(image, text, (label_x, label_y), font, 0.36, color, 1, cv2.LINE_AA)
    corners = pixels[visible, 1:]
    corners = corners[np.isfinite(corners).all(axis=(1, 2))]
    outlines = ((corners + 0.5) * scale - 0.5).round(1).tolist()
    return len(centers), outlines
