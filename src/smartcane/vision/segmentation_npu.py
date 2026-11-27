"""Run the three-class INT8 PP-LiteSeg model with A733 VIPLite."""

import ctypes as ct
import os
from pathlib import Path

import numpy as np

from smartcane.vision.segmentation_common import COLORS, MODEL_SIZE, preprocess


class NpuSegmenter:
    def __init__(self, model, library):
        self.model = Path(model).resolve(strict=True)
        self.library = ct.CDLL(os.fspath(Path(library).resolve(strict=True)))
        self.library.smartcane_viplite_open.argtypes = [
            ct.c_char_p,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int32),
            ct.c_char_p,
            ct.c_size_t,
        ]
        self.library.smartcane_viplite_open.restype = ct.c_void_p
        self.library.smartcane_viplite_run.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_char_p,
            ct.c_size_t,
        ]
        self.library.smartcane_viplite_run.restype = ct.c_bool
        self.library.smartcane_viplite_close.argtypes = [ct.c_void_p]
        self.library.smartcane_viplite_close.restype = None
        self.handle = None
        self.error = ct.create_string_buffer(256)
        self.output = np.empty((len(COLORS), *MODEL_SIZE[::-1]), dtype=np.int8)

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("VIPLite session is already open")
        scale, zero_point = ct.c_float(), ct.c_int32()
        self.handle = self.library.smartcane_viplite_open(
            os.fsencode(self.model),
            *MODEL_SIZE,
            len(COLORS),
            ct.byref(scale),
            ct.byref(zero_point),
            self.error,
            len(self.error),
        )
        if self.handle is None:
            raise RuntimeError(self.error.value.decode())
        self.input_scale = scale.value
        self.input_zero_point = zero_point.value
        return self

    def __exit__(self, *exc):
        self.library.smartcane_viplite_close(self.handle)
        self.handle = None

    def predict(self, frame):
        if self.handle is None:
            raise RuntimeError("VIPLite session is closed")
        resized, tensor = preprocess(frame)
        quantized = np.rint(tensor / self.input_scale) + self.input_zero_point
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        if not self.library.smartcane_viplite_run(
            self.handle,
            quantized.ctypes.data,
            self.output.ctypes.data,
            self.error,
            len(self.error),
        ):
            raise RuntimeError(self.error.value.decode())
        # A shared affine scale preserves the ordering of the three logits.
        return resized, self.output.argmax(axis=0).astype(np.uint8)
