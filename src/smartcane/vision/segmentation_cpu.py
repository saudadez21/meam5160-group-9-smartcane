"""Run the three-class PP-LiteSeg model on the CPU."""

import cv2
import numpy as np

MODEL_SIZE = (640, 352)
COLORS = np.array([[0, 200, 0], [200, 0, 200], [0, 0, 255]], dtype=np.uint8)


def preprocess(frame):
    resized = cv2.resize(frame, MODEL_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    return resized, np.ascontiguousarray(tensor)


class CpuSegmenter:
    def __init__(self, model, threads):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Install ONNX Runtime with uv sync --locked --extra cpu"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        (input_node,) = self.session.get_inputs()
        if input_node.shape != [1, 3, *MODEL_SIZE[::-1]]:
            raise ValueError("Expected a PP-LiteSeg model with input shape 1x3x352x640")
        self.input_name = input_node.name

    def predict(self, frame):
        resized, tensor = preprocess(frame)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        if logits.shape != (1, len(COLORS), *MODEL_SIZE[::-1]):
            raise ValueError("Expected three PP-LiteSeg output channels at 640x352")
        labels = logits[0].argmax(axis=0).astype(np.uint8)
        return resized, labels


def overlay(resized, labels):
    return cv2.addWeighted(resized, 0.5, COLORS[labels], 0.5, 0)
