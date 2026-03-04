import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Callable

import torch

# ── LipFD weight auto-download constants ─────────────────────────────────────
# Update this URL after uploading lipfd_ckpt.pth to a GitHub Release.
# Expected format: direct-download URL (GitHub Release asset, Google Drive, etc.)
LIPFD_WEIGHTS_URL = (
    "https://github.com/aryanbiswas16/DeepFakeGuard/releases/download/"
    "v0.4.0-weights/lipfd_ckpt.pth"
)

# Default path (inside the installed package, next to dinov3_best_v3.pth)
LIPFD_DEFAULT_PATH = Path(__file__).parent.parent / "weights" / "lipfd_ckpt.pth"


def download_lipfd_weights(
    dest: Optional[str | Path] = None,
    url: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download the LipFD checkpoint to *dest* if it doesn't already exist.

    Parameters
    ----------
    dest : Path, optional
        Where to save the file.  Defaults to ``src/deepfake_guard/weights/lipfd_ckpt.pth``.
    url : str, optional
        Override download URL (defaults to ``LIPFD_WEIGHTS_URL``).
    progress_callback : callable, optional
        ``callback(bytes_downloaded, total_bytes)`` called during download.

    Returns
    -------
    Path
        Absolute path to the (existing or freshly-downloaded) weights file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    dest = Path(dest or LIPFD_DEFAULT_PATH)
    url = url or LIPFD_WEIGHTS_URL

    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pth.part")

    print(f"Downloading LipFD weights (~1.7 GB) to {dest} ...")
    print(f"  URL: {url}")

    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "DeepFakeGuard/0.4"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1 << 20  # 1 MB

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
                    elif total:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / (1 << 20)
                        mb_total = total / (1 << 20)
                        sys.stdout.write(
                            f"\r  [{pct:3d}%] {mb_done:.0f} / {mb_total:.0f} MB"
                        )
                        sys.stdout.flush()

        print()  # newline after progress
        tmp.rename(dest)
        print(f"  ✓ Saved to {dest}")
        return dest

    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download LipFD weights from {url}: {exc}\n"
            f"You can download manually and place the file at:\n  {dest}"
        ) from exc


def resolve_lipfd_weights(weights_path: Optional[str] = None) -> Optional[str]:
    """Resolve the LipFD weights path: explicit → default → auto-download.

    Returns the path string if weights are available, or None if download fails.
    """
    # 1. Explicit path provided and exists
    if weights_path and os.path.exists(weights_path):
        return weights_path

    # 2. Default location already has the file
    if LIPFD_DEFAULT_PATH.exists():
        return str(LIPFD_DEFAULT_PATH)

    # 3. Auto-download
    try:
        path = download_lipfd_weights()
        return str(path)
    except RuntimeError as e:
        warnings.warn(str(e))
        return None


def load_weights(target, path):
    """Load weights into target.

    `target` can be:
      - a full `Detector` instance (with `.head` and `.encoder`),
      - a specific module (e.g., `det.head`), or
      - a nn.Module compatible with a state dict.

    The saved file may be either a plain state_dict or a dict with keys
    like {'head': ..., 'encoder': ...} (as produced by the v3 training script).
    """
    if not os.path.exists(path):
        warnings.warn(f"Weights file not found at {path} — running in uninitialised mode.")
        return False

    try:
        state = torch.load(path, map_location=getattr(target, 'device', 'cpu'))
    except Exception as e:
        warnings.warn(f"Could not load weights file: {e}")
        return False

    # If state contains 'head' or 'encoder', try to load accordingly
    try:
        if isinstance(state, dict) and ('head' in state or 'encoder' in state):
            loaded_any = False
            if hasattr(target, 'head') and 'head' in state:
                try:
                    target.head.load_state_dict(state['head'], strict=False)
                    loaded_any = True
                except Exception as e:
                    warnings.warn(f"Failed to load head weights: {e}")
            if hasattr(target, 'encoder') and 'encoder' in state:
                try:
                    # load encoder tunables only (state['encoder'] may contain only norm layers)
                    target.encoder.model.load_state_dict(state['encoder'], strict=False)
                    loaded_any = True
                except Exception as e:
                    warnings.warn(f"Failed to load encoder weights: {e}")

            # If target itself is an nn.Module and top-level state provided
            if not loaded_any and hasattr(target, 'load_state_dict') and isinstance(state, dict):
                try:
                    target.load_state_dict(state, strict=False)
                    loaded_any = True
                except Exception:
                    pass

            return loaded_any

        # Otherwise assume it's a plain state dict for the given module
        if hasattr(target, 'load_state_dict'):
            try:
                target.load_state_dict(state, strict=False)
                return True
            except Exception as e:
                warnings.warn(f"Failed to apply state dict: {e}")
                return False

    except Exception as e:
        warnings.warn(f"Unexpected error loading weights: {e}")
        return False

    return False
