#!/usr/bin/env python3
"""
plate_blur.py — Automatic license-plate anonymization for road-trip / dashcam video.

Pipeline (per frame):
    1. DETECT   plates with YOLO (recommended) or OpenCV's built-in Haar cascade.
                Detection runs on a low-light-enhanced copy (CLAHE) so neon/night
                frames still produce hits; the output frame is never altered by it.
    2. TRACK    every plate with a constant-velocity Kalman filter. When a detection
                is missed (glare, headlight bloom, motion blur) the track "coasts"
                on its prediction for up to --max-missed frames, so the blur never
                flickers off.
    3. BACKFILL The writer runs --lookback frames behind the detector. When a new
                plate is first detected, its box is also applied to those buffered
                frames, covering the few frames where it was visible but not yet
                detected.
    4. RENDER   Gaussian blur, pixelation, or a solid colored box, feathered at the
                edges so it looks soft rather than like a hard rectangle.
    5. AUDIO    OpenCV writes video only; if ffmpeg is installed the original audio
                is re-muxed and the result is encoded to H.264.

Install:  run scripts/setup.sh (creates a venv with opencv<5, numpy, ultralytics),
          plus ffmpeg (macOS: brew install ffmpeg).

YOLO weights: stock COCO models have NO license-plate class. If --weights is omitted,
a plate-trained YOLO11 model (morsetechlab/yolov11-license-plate-detection, AGPL-3.0)
is downloaded once from Hugging Face into $PLATE_BLUR_HOME/models
(default ~/.local/share/plate-blur/models). Pass --weights to use your own.

Usage:
    # Single clip, YOLO, Gaussian blur ➔ ./anonymized/<name>_blurred.mp4
    python plate_blur.py clip.mp4

    # Whole folder ➔ <folder>/anonymized/, aesthetic solid box instead of blur
    python plate_blur.py ~/Videos/trip --mode box --box-color "#2a1e3c"

    # Also hide fixed regions on every frame, e.g. a dashcam's burned-in stamp
    # (date · time · owner's plate number · GPS · speed) along the bottom-left:
    python plate_blur.py ~/Videos/trip --static-mask 0,0.925,0.62,1

    # Tune settings on a 10-second excerpt with debug boxes drawn
    python plate_blur.py clip.mp4 --start 30 --duration 10 --debug

─────────────────────────────────────────────────────────────────────────────────
ADJUSTING BLUR INTENSITY
─────────────────────────────────────────────────────────────────────────────────
  --blur-kernel  Minimum Gaussian kernel size in px (odd; even values are bumped +1).
                 21 = soft, 51 = default, 99+ = heavy smear.
  --blur-scale   Kernel also grows with plate size: kernel = max(blur-kernel,
                 plate_short_side × blur-scale). Keeps close-up plates unreadable.
  --blur-passes  Repeated blurs compound. 2–3 passes look much smoother than one
                 huge kernel and are cheaper.
  --pad          Expands each box by this fraction of its size (0.25 = 25%) so the
                 plate frame/edges are covered too.
  --feather      Soft edge width in px. 0 = hard rectangle.
  --mode         blur | pixelate | box.  --pixel-size for pixelate, --box-color
                 (hex or "B,G,R") and --box-opacity for box.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    # Detection
    detector: str = "yolo"  # "yolo" | "haar"
    weights: str | None = None  # YOLO .pt / .onnx path
    cascade: str | None = None  # custom Haar XML (default: OpenCV's plate cascade)
    conf: float = 0.30  # YOLO confidence threshold
    imgsz: int = 1280  # YOLO inference size; higher finds smaller/distant plates
    device: str | None = None  # YOLO device: "mps" (Apple GPU), "cuda", "cpu"; None = auto
    detect_every: int = 1  # run detector every N frames (Kalman fills the gaps)
    low_light_boost: bool = True  # CLAHE on the detection copy (helps night/neon)

    # Tracking / temporal consistency
    iou_match: float = 0.15  # min IoU to link a detection to an existing track
    max_missed: int = 10  # frames a track coasts without detections (~0.33s @30fps)
    min_hits: int = 1  # detections needed before a track is rendered (raise to cut false positives)
    lookback: int = 6  # frames to retro-blur before a plate's first detection
    coast_grow: float = 0.04  # box grows 4%/missed frame while coasting (safety margin)

    # Rendering
    mode: str = "blur"  # "blur" | "pixelate" | "box"
    blur_kernel: int = 51
    blur_scale: float = 0.9
    blur_passes: int = 2
    pixel_size: int = 10
    box_color: tuple[int, int, int] = (60, 30, 42)  # BGR — deep plum
    box_opacity: float = 1.0
    pad: float = 0.25
    feather: int = 8

    # Fixed regions blurred on every frame (x1, y1, x2, y2). Values <= 1 are
    # fractions of the frame size, larger values are pixels.
    static_masks: list[tuple[float, float, float, float]] = field(default_factory=list)

    # Debug
    debug: bool = False
    preview: bool = False


# ════════════════════════════════════════════════════════════════════════════
# Detectors — each returns a list of (x1, y1, x2, y2) boxes in frame pixels
# ════════════════════════════════════════════════════════════════════════════


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel: lifts plate contrast in dark/neon scenes without blowing out highlights."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class HaarPlateDetector:
    """Zero-dependency fallback. Fast but noticeably less accurate than YOLO,
    especially on angled or distant plates. Raise --min-hits to 2–3 to suppress
    its false positives."""

    DETECT_WIDTH = 1920  # downscale wider frames for speed (distant plates need resolution)

    def __init__(self, cfg: Config):
        path = cfg.cascade or os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
        if not hasattr(cv2, "CascadeClassifier"):
            # OpenCV 5 moved Haar cascades out of the main package.
            sys.exit(f"OpenCV {cv2.__version__} has no CascadeClassifier: pip install 'opencv-python>=4.8,<5'  (or use --detector yolo)")
        self.cc = cv2.CascadeClassifier(path)
        if self.cc.empty():
            sys.exit(f"Could not load Haar cascade: {path}")

    def __call__(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        s = min(1.0, self.DETECT_WIDTH / w)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        rects = self.cc.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 8))
        return [(x / s, y / s, (x + rw) / s, (y + rh) / s) for (x, y, rw, rh) in rects]


DEFAULT_WEIGHTS_URL = (
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt"
)


def ensure_default_weights() -> Path:
    """Download the default plate model once and cache it under $PLATE_BLUR_HOME/models."""
    home = Path(os.environ.get("PLATE_BLUR_HOME", Path.home() / ".local" / "share" / "plate-blur"))
    dst = home / "models" / DEFAULT_WEIGHTS_URL.rsplit("/", 1)[1]
    if dst.is_file():
        return dst
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading default plate model (AGPL-3.0) ➔ {dst}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(DEFAULT_WEIGHTS_URL, tmp)
    tmp.rename(dst)
    return dst


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloPlateDetector:
    """Ultralytics YOLO with plate-trained weights. Uses CUDA/MPS automatically when available."""

    def __init__(self, cfg: Config):
        if not cfg.weights:
            cfg.weights = str(ensure_default_weights())
        if not Path(cfg.weights).is_file():
            sys.exit(f"Model file not found: {Path(cfg.weights).resolve()}\n"
                     "Download a license-plate YOLO model (.pt) and pass its path with --weights.")
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics is not installed: pip install ultralytics  (or use --detector haar)")
        self.model = YOLO(cfg.weights)
        self.conf = cfg.conf
        self.imgsz = cfg.imgsz
        self.device = cfg.device or _auto_device()
        print(f"YOLO device: {self.device}")

    def __call__(self, frame: np.ndarray) -> list[Box]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]


# ════════════════════════════════════════════════════════════════════════════
# Tracking — Kalman filter per plate
# ════════════════════════════════════════════════════════════════════════════


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Box, b: Box) -> float:
    """Center distance normalized by a's longer side — links fast-moving plates whose boxes no longer overlap."""
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(np.hypot(ax - bx, ay - by) / max(a[2] - a[0], a[3] - a[1], 1.0))


class Track:
    """State = [cx, cy, w, h, vx, vy, vw, vh]; measurement = [cx, cy, w, h]."""

    _next_id = 0

    def __init__(self, box: Box, coast_grow: float):
        self.coast_grow = coast_grow
        self.id = Track._next_id
        Track._next_id += 1

        kf = cv2.KalmanFilter(8, 4)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0  # position += velocity each frame
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        # Process noise: how much we expect motion to change frame-to-frame (px²).
        kf.processNoiseCov = np.diag([4, 4, 2, 2, 1, 1, 0.25, 0.25]).astype(np.float32)
        # Measurement noise: detector box jitter (px²). Higher = smoother but laggier.
        kf.measurementNoiseCov = np.diag([9, 9, 16, 16]).astype(np.float32)
        kf.errorCovPost = np.diag([10, 10, 10, 10, 1000, 1000, 1000, 1000]).astype(np.float32)
        kf.statePost = np.array([*self._to_meas(box), 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        self.kf = kf

        self.hits = 1
        self.missed = 0

    @staticmethod
    def _to_meas(box: Box) -> list[float]:
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

    def predict(self) -> None:
        self.kf.predict()  # OpenCV copies the prediction into statePost

    def correct(self, box: Box) -> None:
        self.kf.correct(np.array(self._to_meas(box), dtype=np.float32).reshape(4, 1))
        self.hits += 1
        self.missed = 0

    @property
    def box(self) -> Box:
        cx, cy, w, h = (float(v) for v in self.kf.statePost[:4, 0])
        grow = 1.0 + self.missed * self.coast_grow
        w, h = max(w, 4.0) * grow, max(h, 2.0) * grow
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class PlateTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[Track] = []

    def step(self, detections: list[Box] | None) -> tuple[list[Track], list[Track]]:
        """Advance one frame. `detections=None` means the detector was skipped this frame.
        Returns (tracks to render, tracks that just became confirmed)."""
        for t in self.tracks:
            t.predict()

        newly_confirmed: list[Track] = []
        if detections is not None:
            # Greedy association: best IoU first, center-distance as tie-breaker/fallback.
            candidates = []
            for ti, t in enumerate(self.tracks):
                tb = t.box
                for di, d in enumerate(detections):
                    ov, dist = iou(tb, d), center_distance(tb, d)
                    if ov >= self.cfg.iou_match or dist < 0.75:
                        candidates.append((ov, -dist, ti, di))
            candidates.sort(reverse=True)

            used_t, used_d = set(), set()
            for _, _, ti, di in candidates:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                t = self.tracks[ti]
                t.correct(detections[di])
                if t.hits == self.cfg.min_hits:
                    newly_confirmed.append(t)

            for ti, t in enumerate(self.tracks):
                if ti not in used_t:
                    t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.cfg.max_missed]

            for di, d in enumerate(detections):
                if di not in used_d:
                    t = Track(d, self.cfg.coast_grow)
                    self.tracks.append(t)
                    if self.cfg.min_hits <= 1:
                        newly_confirmed.append(t)

        visible = [t for t in self.tracks if t.hits >= self.cfg.min_hits]
        return visible, newly_confirmed


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════


def odd(n: float) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def anonymize(frame: np.ndarray, box: Box, cfg: Config, pad: float | None = None) -> None:
    """Applies the chosen effect in place, alpha-blended through a feathered mask."""
    pad = cfg.pad if pad is None else pad
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1, x2, y2 = x1 - bw * cfg.pad / 2, y1 - bh * cfg.pad / 2, x2 + bw * cfg.pad / 2, y2 + bh * cfg.pad / 2
    bw, bh = x2 - x1, y2 - y1

    f = max(0, cfg.feather)
    # Work region = padded box + feather margin, clipped to the frame.
    rx1, ry1 = max(0, int(x1) - f), max(0, int(y1) - f)
    rx2, ry2 = min(W, int(np.ceil(x2)) + f), min(H, int(np.ceil(y2)) + f)
    if rx2 - rx1 < 2 or ry2 - ry1 < 2:
        return
    roi = frame[ry1:ry2, rx1:rx2]

    if cfg.mode == "blur":
        k = odd(max(cfg.blur_kernel, min(bw, bh) * cfg.blur_scale))
        effect = roi
        for _ in range(max(1, cfg.blur_passes)):
            effect = cv2.GaussianBlur(effect, (k, k), 0)
    elif cfg.mode == "pixelate":
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (max(1, rw // cfg.pixel_size), max(1, rh // cfg.pixel_size)), interpolation=cv2.INTER_AREA)
        effect = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:  # box
        solid = np.empty_like(roi)
        solid[:] = cfg.box_color
        effect = cv2.addWeighted(solid, cfg.box_opacity, roi, 1 - cfg.box_opacity, 0)

    # Mask is fully opaque over the padded box; the feather only fades *outside* it,
    # so the plate itself is always 100% covered.
    mask = np.zeros(roi.shape[:2], np.float32)
    cv2.rectangle(mask, (int(x1) - rx1 - f // 2, int(y1) - ry1 - f // 2), (int(x2) - rx1 + f // 2, int(y2) - ry1 + f // 2), 1.0, -1)
    if f > 0:
        mask = cv2.GaussianBlur(mask, (odd(f * 2 + 1), odd(f * 2 + 1)), 0)
        inner = np.zeros_like(mask)
        cv2.rectangle(inner, (int(x1) - rx1, int(y1) - ry1), (int(x2) - rx1, int(y2) - ry1), 1.0, -1)
        mask = np.maximum(mask, inner)
    mask = mask[..., None]

    frame[ry1:ry2, rx1:rx2] = (effect.astype(np.float32) * mask + roi.astype(np.float32) * (1 - mask)).astype(np.uint8)


def _inside(box: Box, region: Box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def draw_debug(frame: np.ndarray, tracks: list[tuple[int, int, Box]]) -> None:
    """tracks: (id, missed, box) snapshots taken when the frame was processed."""
    for tid, missed, box in tracks:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if missed == 0 else (0, 165, 255)  # green = detected, orange = coasting
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{tid} m{missed}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════
# Video I/O
# ════════════════════════════════════════════════════════════════════════════


def process_video(src: Path, dst: Path, cfg: Config, detector, start: float, duration: float | None) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  ! cannot open {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - start * fps)
    if duration is not None:
        total = min(total, int(duration * fps))

    dst.parent.mkdir(parents=True, exist_ok=True)
    have_ffmpeg = shutil.which("ffmpeg") is not None
    silent = dst.with_name(f".{dst.stem}.video_only.mp4") if have_ffmpeg else dst
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    static_boxes: list[Box] = [
        tuple(v * (W if i % 2 == 0 else H) if all(c <= 1 for c in m) else v for i, v in enumerate(m))  # type: ignore[misc]
        for m in cfg.static_masks
    ]

    tracker = PlateTracker(cfg)
    buffer: deque[tuple[np.ndarray, list[Box], list[tuple[int, int, Box]]]] = deque()
    t0, idx, n_plates = time.time(), 0, 0
    quit_requested = False

    def emit(item) -> None:
        nonlocal quit_requested
        frame, boxes, dbg_tracks = item
        for b in static_boxes:
            anonymize(frame, b, cfg, pad=0.0)
        for b in boxes:
            anonymize(frame, b, cfg)
        if cfg.debug:
            draw_debug(frame, dbg_tracks)
        writer.write(frame)
        if cfg.preview:
            cv2.imshow("plate_blur (q to stop)", cv2.resize(frame, (W * 720 // H, 720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_requested = True

    while not quit_requested and (duration is None or idx < total):
        ok, frame = cap.read()
        if not ok:
            break

        detections = None
        if idx % max(1, cfg.detect_every) == 0:
            det_input = enhance_low_light(frame) if cfg.low_light_boost else frame
            # Ignore hits inside static masks (overlay text often fools the detector).
            detections = [d for d in detector(det_input) if not any(_inside(d, m) for m in static_boxes)]

        visible, confirmed = tracker.step(detections)
        boxes = [t.box for t in visible]
        n_plates += len(confirmed)

        # Retro-apply newly confirmed plates to the frames still waiting in the buffer.
        if cfg.lookback:
            for t in confirmed:
                for _, past_boxes, _ in list(buffer)[-cfg.lookback :]:
                    past_boxes.append(t.box)

        buffer.append((frame, boxes, [(t.id, t.missed, t.box) for t in visible] if cfg.debug else []))
        if len(buffer) > cfg.lookback:
            emit(buffer.popleft())

        idx += 1
        if idx % 30 == 0 or idx == total:
            rate = idx / max(time.time() - t0, 1e-6)
            eta = (total - idx) / rate if rate > 0 else 0
            print(f"\r  {idx}/{total} frames · {rate:5.1f} fps · plates seen {n_plates} · ETA {eta:4.0f}s", end="", flush=True)

    while buffer and not quit_requested:
        emit(buffer.popleft())

    cap.release()
    writer.release()
    if cfg.preview:
        cv2.destroyAllWindows()
    print()

    if have_ffmpeg:
        mux_audio(silent, src, dst, start, idx / fps)
        silent.unlink(missing_ok=True)
    else:
        print("  ! ffmpeg not found: output has no audio and uses the mp4v codec (brew install ffmpeg).")


def mux_audio(video: Path, original: Path, dst: Path, start: float, length: float) -> None:
    """Encode to H.264 and copy the original audio track (if any) for the same time range."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", "-movflags", "+faststart",
        str(dst),
    ]  # fmt: skip
    if subprocess.run(cmd).returncode != 0:
        print(f"  ! ffmpeg mux failed; video-only file kept at {video}")
        shutil.copy(video, dst.with_suffix(".video_only.mp4"))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def parse_box(s: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in s.split(","))
    return (x1, y1, x2, y2)


def parse_color(s: str) -> tuple[int, int, int]:
    """'#2a1e3c' (RGB hex) or '60,30,42' (B,G,R) ➔ BGR tuple."""
    s = s.strip()
    if s.startswith("#") and len(s) == 7:
        r, g, b = (int(s[i : i + 2], 16) for i in (1, 3, 5))
        return (b, g, r)
    b, g, r = (int(v) for v in s.split(","))
    return (b, g, r)


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(description="Detect, track and blur license plates in video.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", type=Path, help="video file or folder of videos")
    p.add_argument("-o", "--output", type=Path, help="output file (single input) or folder (default: ./anonymized/)")

    g = p.add_argument_group("detection")
    g.add_argument("--detector", choices=["yolo", "haar"], default=d.detector)
    g.add_argument("--weights", help="YOLO license-plate weights (.pt/.onnx); default: auto-downloaded YOLO11 plate model")
    g.add_argument("--cascade", help="custom Haar cascade XML")
    g.add_argument("--conf", type=float, default=d.conf)
    g.add_argument("--imgsz", type=int, default=d.imgsz)
    g.add_argument("--device", help="YOLO device: mps | cuda | cpu (default: auto)")
    g.add_argument("--detect-every", type=int, default=d.detect_every, help="run detector every N frames (speed-up)")
    g.add_argument("--no-low-light-boost", action="store_true", help="disable CLAHE enhancement before detection")

    g = p.add_argument_group("tracking")
    g.add_argument("--iou-match", type=float, default=d.iou_match)
    g.add_argument("--max-missed", type=int, default=d.max_missed, help="frames to keep blurring after a plate is lost")
    g.add_argument("--min-hits", type=int, default=d.min_hits, help="detections before a track is blurred")
    g.add_argument("--lookback", type=int, default=d.lookback, help="frames to retro-blur before first detection")
    g.add_argument("--coast-grow", type=float, default=d.coast_grow)

    g = p.add_argument_group("rendering (blur intensity — see header docs)")
    g.add_argument("--mode", choices=["blur", "pixelate", "box"], default=d.mode)
    g.add_argument("--blur-kernel", type=int, default=d.blur_kernel)
    g.add_argument("--blur-scale", type=float, default=d.blur_scale)
    g.add_argument("--blur-passes", type=int, default=d.blur_passes)
    g.add_argument("--pixel-size", type=int, default=d.pixel_size)
    g.add_argument("--box-color", type=parse_color, default=d.box_color, help="'#RRGGBB' or 'B,G,R'")
    g.add_argument("--box-opacity", type=float, default=d.box_opacity)
    g.add_argument("--pad", type=float, default=d.pad)
    g.add_argument("--feather", type=int, default=d.feather)

    g.add_argument("--static-mask", type=parse_box, action="append", default=[], metavar="X1,Y1,X2,Y2",
                   help="region blurred on every frame (fractions <=1 or pixels); repeatable")

    g = p.add_argument_group("testing")
    g.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    g.add_argument("--duration", type=float, help="process only this many seconds")
    g.add_argument("--preview", action="store_true", help="show a live preview window")
    g.add_argument("--debug", action="store_true", help="draw track boxes (green=detected, orange=coasting)")
    return p


def main() -> None:
    a = build_parser().parse_args()
    cfg = Config(
        detector=a.detector, weights=a.weights, cascade=a.cascade, conf=a.conf, imgsz=a.imgsz, device=a.device,
        detect_every=a.detect_every, low_light_boost=not a.no_low_light_boost,
        iou_match=a.iou_match, max_missed=a.max_missed, min_hits=a.min_hits, lookback=a.lookback,
        coast_grow=a.coast_grow, mode=a.mode, blur_kernel=a.blur_kernel, blur_scale=a.blur_scale,
        blur_passes=a.blur_passes, pixel_size=a.pixel_size, box_color=a.box_color,
        box_opacity=a.box_opacity, pad=a.pad, feather=a.feather, static_masks=a.static_mask, debug=a.debug, preview=a.preview,
    )  # fmt: skip

    if a.input.is_dir():
        sources = sorted(f for f in a.input.iterdir() if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("."))
        out_dir = a.output or (a.input / "anonymized")
        jobs = [(s, out_dir / f"{s.stem}_blurred.mp4") for s in sources]
    else:
        out = a.output or (Path("anonymized") / f"{a.input.stem}_blurred.mp4")
        jobs = [(a.input, out)]
    if not jobs:
        sys.exit(f"No videos found in {a.input}")

    detector = YoloPlateDetector(cfg) if cfg.detector == "yolo" else HaarPlateDetector(cfg)
    for i, (src, dst) in enumerate(jobs, 1):
        if dst.exists() and len(jobs) > 1:
            print(f"[{i}/{len(jobs)}] skip {src.name} (output exists)")
            continue
        print(f"[{i}/{len(jobs)}] {src.name} ➔ {dst}")
        process_video(src, dst, cfg, detector, a.start, a.duration)


if __name__ == "__main__":
    main()
