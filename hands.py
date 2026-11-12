"""Webcam hand tracking and gesture classification.

CameraSource runs the webcam and MediaPipe in a background thread.
HandSet turns raw detections into up to two smoothed, persistent hands,
each with a gesture: NONE, PINCH, FIST or OPEN.
"""
import os
import threading
import time
import urllib.request

import numpy as np

NONE, PINCH, FIST, OPEN = "none", "pinch", "fist", "open"

# (tip, middle joint) landmark indices for index, middle, ring and pinky.
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))
PALM = [0, 5, 9, 13, 17]
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def classify(pts, was_pinching=False):
    """Name the gesture shown by 21 hand landmarks (any 2D units)."""
    wrist = pts[0]
    palm = _dist(wrist, pts[9]) or 1.0
    extended = [_dist(pts[tip], wrist) > 1.15 * _dist(pts[mid], wrist) for tip, mid in FINGERS]
    pinch_ratio = _dist(pts[4], pts[8]) / palm
    # Hysteresis: once pinching, the fingers must open wider to let go.
    limit = 0.55 if was_pinching else 0.35
    if pinch_ratio < limit and sum(extended[1:]) >= 2:
        return PINCH
    if not any(extended):
        return FIST
    if all(extended):
        return OPEN
    return NONE


class Hand:
    _next_id = 0

    def __init__(self, pts, now):
        self.id = Hand._next_id
        Hand._next_id += 1
        self.pts = pts.copy()
        self.gesture = NONE
        self.last_seen = now
        self._candidate = NONE
        self._candidate_count = 0

    @property
    def palm(self):
        return self.pts[PALM].mean(axis=0)

    @property
    def pinch_point(self):
        return (self.pts[4] + self.pts[8]) / 2

    def observe(self, pts, now, smoothing=0.5):
        self.pts = smoothing * self.pts + (1 - smoothing) * pts
        self.last_seen = now
        seen = classify(self.pts, self.gesture == PINCH)
        # A new gesture must show up twice in a row before it counts.
        if seen == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = seen, 1
        if self._candidate_count >= 2:
            self.gesture = seen


class HandSet:
    """Keeps up to two hands alive across frames."""

    def __init__(self, max_hands=2, timeout=0.3, match_radius=260):
        self.hands = []
        self.max_hands = max_hands
        self.timeout = timeout
        self.match_radius = match_radius

    def update(self, detections, now):
        """Feed new detections (or None if no new camera frame). Returns (added, removed)."""
        added = []
        if detections is not None:
            free = list(self.hands)
            for pts in detections:
                centre = pts[PALM].mean(axis=0)
                best = min(free, key=lambda h: _dist(h.palm, centre), default=None)
                if best is not None and _dist(best.palm, centre) < self.match_radius:
                    best.observe(pts, now)
                    free.remove(best)
                elif len(self.hands) < self.max_hands:
                    hand = Hand(pts, now)
                    self.hands.append(hand)
                    added.append(hand)
        removed = [h for h in self.hands if now - h.last_seen > self.timeout]
        self.hands = [h for h in self.hands if h not in removed]
        return added, removed


def _make_detector():
    """Return (detect, close). detect maps an RGB image to a list of (21, 2) arrays in 0..1."""
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)

        def detect(rgb, _timestamp_ms):
            result = hands.process(rgb)
            return [np.array([(p.x, p.y) for p in h.landmark])
                    for h in (result.multi_hand_landmarks or [])]
        return detect, hands.close

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(MODEL_FILE):
        print("Downloading the hand tracking model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    with open(MODEL_FILE, "rb") as f:
        model = f.read()
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5))

    def detect(rgb, timestamp_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect_for_video(image, timestamp_ms)
        return [np.array([(p.x, p.y) for p in h]) for h in result.hand_landmarks]
    return detect, landmarker.close


class CameraSource:
    """Reads the webcam and detects hands in a background thread."""

    def __init__(self, size, camera=0, dim=0.4):
        self.size = size
        self.camera = camera
        self.dim = dim
        self.error = None
        self._lock = threading.Lock()
        self._latest = (0, None, [])
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def latest(self):
        """Return (sequence number, dimmed mirrored RGB frame or None, detections in pixels)."""
        with self._lock:
            return self._latest

    def _run(self):
        try:
            import cv2
            detect, close = _make_detector()
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera}. Close other apps using it, "
                    "or try another one with --camera 1.")
            start = time.monotonic()
            seq = 0
            scale = np.array(self.size, dtype=float)
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)  # mirror, so moving right moves right
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = [d * scale for d in detect(rgb, int((time.monotonic() - start) * 1000))]
                shown = cv2.convertScaleAbs(cv2.resize(rgb, self.size), alpha=self.dim)
                seq += 1
                with self._lock:
                    self._latest = (seq, shown, detections)
            cap.release()
            close()
        except Exception as exc:  # reported to the main thread
            self.error = exc


def fake_hand(x, y, pose):
    """Synthetic landmarks for tests. (x, y) is the pinch point for PINCH, else the palm."""
    if pose == PINCH:
        cx, cy = x + 56, y + 44
    else:
        cx, cy = x + 2, y - 18
    pts = np.zeros((21, 2))
    pts[0] = (cx, cy + 90)
    pts[1:5] = [(cx - 30, cy + 60), (cx - 50, cy + 35), (cx - 60, cy + 10), (cx - 65, cy - 10)]
    for i, fx in enumerate((-30, -10, 10, 30)):
        base = 5 + 4 * i
        pts[base] = (cx + fx, cy)
        if pose == OPEN or (pose == PINCH and i > 0):
            pts[base + 1:base + 4] = [(cx + fx, cy - 35), (cx + fx, cy - 60), (cx + fx, cy - 85)]
        else:
            pts[base + 1:base + 4] = [(cx + fx, cy - 30), (cx + fx, cy - 10), (cx + fx, cy + 15)]
    if pose == PINCH:
        pts[6:9] = [(cx - 35, cy - 30), (cx - 45, cy - 40), (cx - 55, cy - 45)]
        pts[4] = (cx - 57, cy - 43)
    return pts
