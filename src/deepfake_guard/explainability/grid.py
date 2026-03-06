"""Frame-grid utilities for VLM explainability."""
from __future__ import annotations

import base64
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_keyframes(video_path: str, num_frames: int = 6) -> List[np.ndarray]:
    """Uniformly sample *num_frames* frames across a video.

    Frames are returned as RGB numpy arrays.  The sampling avoids the very
    first and last frames by placing sample points at the centres of equal
    temporal buckets.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = num_frames  # fallback for streams that don't report length

    indices = [int(total * (i + 0.5) / num_frames) for i in range(num_frames)]

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def build_grid_image(
    frames: List[np.ndarray],
    cell_size: Tuple[int, int] = (384, 384),
    cols: int = 3,
) -> Image.Image:
    """Stitch frames into a labelled grid image.

    Args:
        frames: List of RGB numpy arrays.
        cell_size: (width, height) of each cell in pixels.
        cols: Number of columns.  Rows are derived automatically.

    Returns:
        A PIL ``Image`` with all frames arranged in a grid and each cell
        labelled "Frame N" below it.
    """
    n = len(frames)
    rows = max(1, (n + cols - 1) // cols)

    cell_w, cell_h = cell_size
    label_h = 28  # pixels reserved beneath each frame for the caption

    grid_w = cols * cell_w
    grid_h = rows * (cell_h + label_h)

    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols

        pil_frame = Image.fromarray(frame).resize(cell_size, Image.LANCZOS)

        x = col * cell_w
        y = row * (cell_h + label_h)

        grid.paste(pil_frame, (x, y))

        label = f"Frame {i + 1}"
        draw.text((x + 6, y + cell_h + 5), label, fill=(200, 200, 200), font=font)

    return grid


def grid_to_base64(grid: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 string (for API payloads)."""
    buf = io.BytesIO()
    grid.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
