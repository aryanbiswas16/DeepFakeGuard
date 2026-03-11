"""VLM-based semantic explainability for DeepfakeGuard.

This module provides post-hoc natural language explanations for the ensemble
verdict.  It runs after all detection scores have been finalized and has zero
impact on detection scores.

Supported backends
------------------
- ``openai``: GPT-4o mini via the OpenAI API.
  Requires: ``pip install openai`` and the ``OPENAI_API_KEY`` env var.
  Works on CPU — no local GPU needed.

- ``anthropic``: Claude Opus 4.6 via the Anthropic API.
  Requires: ``pip install anthropic`` and the ``ANTHROPIC_API_KEY`` env var.
  Works on CPU — no local GPU needed.

- ``qwen2vl``: Qwen2-VL-7B-Instruct running locally via HuggingFace
  Transformers.  Requires: ``transformers``, ``accelerate``, ``qwen-vl-utils``,
  and a CUDA-capable GPU (PyTorch 2.4+).

All backends degrade gracefully to ``available=False`` if deps are missing.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VLMExplanation:
    """Result from the VLM semantic explainability pass."""

    available: bool
    """False when the VLM backend could not be loaded or run."""

    explanation: str
    """Natural language summary of findings."""

    artifacts_found: bool
    """True when the VLM detected manipulation artifacts."""

    artifact_categories: List[str]
    """Subset of: anatomical_errors, physics_violations,
    temporal_inconsistencies, ai_generation_artifacts."""

    confidence: str
    """high | medium | low — VLM's self-reported confidence."""

    key_frames: List[int]
    """1-indexed frame numbers showing the strongest artifacts."""

    backend: str
    """Which backend produced this explanation (e.g. 'qwen2vl')."""

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Sentinel returned on failure — zero overhead for callers
# ---------------------------------------------------------------------------

def _unavailable(reason: str, backend: str = "none") -> Dict[str, Any]:
    return VLMExplanation(
        available=False,
        explanation=reason,
        artifacts_found=False,
        artifact_categories=[],
        confidence="low",
        key_frames=[],
        backend=backend,
    ).as_dict()


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class VLMExplainer:
    """Post-hoc VLM semantic explainer for deepfake detections.

    Usage::

        # Anthropic Claude API (no GPU required)
        explainer = VLMExplainer(backend="anthropic", api_key="sk-ant-...")
        result = explainer.explain(video_path, ensemble_score=0.82)

        # Local Qwen2-VL (requires CUDA GPU + PyTorch 2.4+)
        explainer = VLMExplainer(backend="qwen2vl")
        result = explainer.explain(video_path, ensemble_score=0.82)

        # result is always a plain dict with keys: available, explanation, …

    For the ``anthropic`` backend, the API key is resolved in this order:
      1. ``api_key`` argument
      2. ``ANTHROPIC_API_KEY`` environment variable
    """

    QWEN_MODEL_ID    = "Qwen/Qwen2-VL-7B-Instruct"
    ANTHROPIC_MODEL  = "claude-opus-4-6"
    OPENAI_MODEL     = "gpt-4o-mini"

    VALID_BACKENDS = ("openai", "anthropic", "qwen2vl")

    # Maps each API backend to the env var that holds its key
    _API_KEY_ENV: Dict[str, str] = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def __init__(self, backend: str = "openai", api_key: Optional[str] = None) -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Unsupported VLM backend '{backend}'. "
                f"Choose from: {self.VALID_BACKENDS}"
            )
        self.backend = backend
        env_var = self._API_KEY_ENV.get(backend)
        self._api_key = api_key or (os.environ.get(env_var) if env_var else None)

        # Qwen2-VL local model state (lazy-loaded)
        self._model = None
        self._processor = None
        self._device = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        video_path: str,
        ensemble_score: float,
        num_frames: int = 6,
        ensemble_label: str = "FAKE",
        detector_context: str = "",
    ) -> Dict[str, Any]:
        """Produce a semantic explanation for the ensemble verdict.

        Args:
            video_path: Path to the video file (used to extract frames).
            ensemble_score: The ensemble fake-probability score (0–1).
            num_frames: How many frames to sample for the grid.
            ensemble_label: The ensemble verdict string.
            detector_context: Pre-formatted per-detector summary for the prompt.

        Returns:
            A plain dict matching the :class:`VLMExplanation` schema.
            Always safe to access — returns ``available=False`` on any error.
        """
        # ── 1. Extract frames ────────────────────────────────────────────
        try:
            from .grid import extract_keyframes, build_grid_image
            frames = extract_keyframes(video_path, num_frames=num_frames)
        except Exception as exc:
            logger.debug("VLM: frame extraction failed: %s", exc)
            return _unavailable(f"Frame extraction failed: {exc}", self.backend)

        if not frames:
            return _unavailable("No frames could be extracted from the video.", self.backend)

        # ── 2. Build grid image ──────────────────────────────────────────
        try:
            grid_image = build_grid_image(frames, cell_size=(384, 384), cols=3)
        except Exception as exc:
            logger.debug("VLM: grid construction failed: %s", exc)
            return _unavailable(f"Grid construction failed: {exc}", self.backend)

        # ── 3. Run inference via selected backend ────────────────────────
        try:
            if self.backend == "openai":
                raw_output = self._run_openai_inference(
                    grid_image, ensemble_score, len(frames), ensemble_label, detector_context
                )
            elif self.backend == "anthropic":
                raw_output = self._run_anthropic_inference(
                    grid_image, ensemble_score, len(frames), ensemble_label, detector_context
                )
            else:  # qwen2vl
                load_err = self._load_qwen_model()
                if load_err:
                    return _unavailable(load_err, self.backend)
                raw_output = self._run_qwen_inference(
                    grid_image, ensemble_score, len(frames), ensemble_label, detector_context
                )
        except Exception as exc:
            logger.debug("VLM: inference failed: %s", exc)
            return _unavailable(f"VLM inference failed: {exc}", self.backend)

        # ── 5. Parse JSON response ───────────────────────────────────────
        return self._parse_response(raw_output)

    # ------------------------------------------------------------------
    # Private helpers — OpenAI GPT-4o mini backend
    # ------------------------------------------------------------------

    def _run_openai_inference(
        self,
        grid_image,  # PIL.Image
        ensemble_score: float,
        num_frames: int,
        ensemble_label: str = "FAKE",
        detector_context: str = "",
    ) -> str:
        """Call the OpenAI Chat Completions API with the grid image and return raw text."""
        try:
            from openai import OpenAI as _OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not set. Pass api_key= or set OPENAI_API_KEY."
            )

        from .grid import grid_to_base64
        from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        user_text = USER_PROMPT_TEMPLATE.format(
            num_frames=num_frames,
            ensemble_score=ensemble_score,
            verdict=ensemble_label,
            detector_context=detector_context or "No per-detector data available.",
        )

        image_b64 = grid_to_base64(grid_image, fmt="JPEG")

        client = _OpenAI(api_key=self._api_key)

        response = client.chat.completions.create(
            model=self.OPENAI_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Private helpers — Anthropic Claude API backend
    # ------------------------------------------------------------------

    def _run_anthropic_inference(
        self,
        grid_image,  # PIL.Image
        ensemble_score: float,
        num_frames: int,
        ensemble_label: str = "FAKE",
        detector_context: str = "",
    ) -> str:
        """Call the Anthropic Messages API with the grid image and return raw text."""
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK not installed. "
                "Run: pip install anthropic"
            ) from exc

        if not self._api_key:
            raise ValueError(
                "Anthropic API key not set. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        from .grid import grid_to_base64
        from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        user_text = USER_PROMPT_TEMPLATE.format(
            num_frames=num_frames,
            ensemble_score=ensemble_score,
            verdict=ensemble_label,
            detector_context=detector_context or "No per-detector data available.",
        )

        image_b64 = grid_to_base64(grid_image, fmt="JPEG")

        client = _anthropic.Anthropic(api_key=self._api_key)

        response = client.messages.create(
            model=self.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        return response.content[0].text.strip()

    # ------------------------------------------------------------------
    # Private helpers — Qwen2-VL local backend
    # ------------------------------------------------------------------

    def _load_qwen_model(self) -> Optional[str]:
        """Load Qwen2-VL model and processor (lazy, once).  Returns error string on failure."""
        if self._model is not None:
            return None  # already loaded

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            return (
                f"transformers not available ({exc}). "
                "Install with: pip install transformers accelerate qwen-vl-utils"
            )

        try:
            logger.info("VLM: loading %s …", self.QWEN_MODEL_ID)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.QWEN_MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
            )
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(self.QWEN_MODEL_ID)
            self._device = next(self._model.parameters()).device
            logger.info("VLM: model loaded on %s", self._device)
            return None
        except Exception as exc:
            self._model = None
            self._processor = None
            return f"Failed to load {self.QWEN_MODEL_ID}: {exc}"

    def _run_qwen_inference(
        self,
        grid_image,  # PIL.Image
        ensemble_score: float,
        num_frames: int,
        ensemble_label: str = "FAKE",
        detector_context: str = "",
    ) -> str:
        """Run a single forward pass through Qwen2-VL and return raw text."""
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                f"qwen-vl-utils not installed: {exc}. "
                "Install with: pip install qwen-vl-utils"
            ) from exc

        import torch
        from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        user_text = USER_PROMPT_TEMPLATE.format(
            num_frames=num_frames,
            ensemble_score=ensemble_score,
            verdict=ensemble_label,
            detector_context=detector_context or "No per-detector data available.",
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": grid_image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse the JSON payload from the VLM output."""
        # Strip markdown fences if present
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    logger.debug("VLM: could not parse JSON from output: %s", raw[:300])
                    return _unavailable(
                        "VLM returned unparseable output.", self.backend
                    )
            else:
                logger.debug("VLM: no JSON found in output: %s", raw[:300])
                return _unavailable("VLM returned no JSON output.", self.backend)

        valid_categories = {
            "anatomical_errors",
            "physics_violations",
            "temporal_inconsistencies",
            "ai_generation_artifacts",
        }
        raw_cats = data.get("artifact_categories", [])
        artifact_categories = [c for c in raw_cats if c in valid_categories]

        raw_kf = data.get("key_frames", [])
        key_frames = [int(f) for f in raw_kf if str(f).isdigit() or isinstance(f, int)]

        confidence = data.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        return VLMExplanation(
            available=True,
            explanation=str(data.get("explanation", "No explanation provided.")),
            artifacts_found=bool(data.get("artifacts_found", False)),
            artifact_categories=artifact_categories,
            confidence=confidence,
            key_frames=key_frames,
            backend=self.backend,
        ).as_dict()
