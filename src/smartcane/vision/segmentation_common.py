"""Prepare PP-LiteSeg inputs and render its three class labels."""

import cv2
import numpy as np

MODEL_SIZE = (640, 352)
COLORS = np.array([[0, 200, 0], [200, 0, 200], [0, 0, 255]], dtype=np.uint8)


def preprocess(frame):
    resized = cv2.resize(frame, MODEL_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    return resized, np.ascontiguousarray(tensor)


def overlay(resized, labels):
    return cv2.addWeighted(resized, 0.5, COLORS[labels], 0.5, 0)
