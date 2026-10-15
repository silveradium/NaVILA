"""Client side of the NaVILA server: observation history + HTTP calls.

Shared by the Isaac Sim runner and the real-robot (go2_sdk) client. Only depends on numpy, opencv and requests.
"""

from __future__ import annotations

import base64
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import requests

from .mid_level import MidLevelAction, parse_navila_output


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg(data: str) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sample_frames(frames: list, num_frames: int) -> list:
    """Same sampling as ``sample_and_pad_images`` in evaluation/vlnce_baselines/navila_trainer.py.

    Pads with black frames at the front when the history is short, then keeps ``num_frames - 1`` uniformly spaced
    history frames plus the latest frame.
    """
    frames = list(frames)
    if len(frames) < num_frames:
        h, w = frames[-1].shape[:2]
        frames = [np.zeros((h, w, 3), dtype=np.uint8)] * (num_frames - len(frames)) + frames
    idx = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    return [frames[i] for i in idx] + [frames[-1]]


class FrameHistory:
    """RGB observation history of one episode (the 'video of historical observations' in NaVILA's prompt)."""

    def __init__(self, max_len: int = 2000):
        self.frames: list[np.ndarray] = []
        self.max_len = max_len

    def add(self, rgb: np.ndarray):
        self.frames.append(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
        if len(self.frames) > self.max_len:
            # keep the first frame (start view) and drop the oldest of the rest
            del self.frames[1]

    def reset(self):
        self.frames.clear()

    def __len__(self):
        return len(self.frames)


class NaVilaClient:
    """Blocking and async calls to ``navila_go2/server/navila_server.py``."""

    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.info = None

    def wait_ready(self, max_wait: float = 600.0, poll: float = 2.0) -> dict:
        t0 = time.time()
        while True:
            try:
                r = requests.get(f"{self.url}/health", timeout=5)
                if r.ok:
                    self.info = r.json()
                    return self.info
            except requests.RequestException:
                pass
            if time.time() - t0 > max_wait:
                raise TimeoutError(f"NaVILA server at {self.url} not reachable after {max_wait:.0f}s")
            time.sleep(poll)

    @property
    def num_frames(self) -> int:
        return (self.info or {}).get("num_frames", 8)

    def act(self, instruction: str, history: FrameHistory) -> tuple[MidLevelAction, dict]:
        frames = sample_frames(history.frames, self.num_frames)
        payload = {"instruction": instruction, "frames": [encode_jpeg(f) for f in frames]}
        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        out["roundtrip_s"] = time.time() - t0
        return parse_navila_output(out["text"]), out

    def act_async(self, instruction: str, history: FrameHistory) -> Future:
        # snapshot the frame list so the sim thread can keep appending
        snap = FrameHistory(history.max_len)
        snap.frames = list(history.frames)
        return self._pool.submit(self.act, instruction, snap)
