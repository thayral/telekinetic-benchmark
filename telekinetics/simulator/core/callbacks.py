from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Optional


def _safe_token(value) -> str:
    """Make identifiers safe for flat filenames."""
    text = "unknown" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text or "unknown"


@dataclass
class StepRenderCallback:
    """Render and save an RGB frame after each MuJoCo step.

    Provide scene_seed, question_id, and interaction_id for flat dataset-debug
    names like:
        scene-123_abcd_correct_step-000001_move_1712345678900000000.png
    """

    output_dir: str | Path
    camera_name: str = "cam_oblique"
    every_n: int = 1
    prefix: str = "frame"
    include_phase: bool = True
    include_timestamp: bool = True
    scene_seed: int | str | None = None
    question_id: str | None = None
    interaction_id: str | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.every_n <= 0:
            raise ValueError("every_n must be positive.")

    def __call__(self, env, step_count: int, phase: Optional[str] = None):
        if step_count % self.every_n != 0:
            return

        rgb = env.render_rgb(camera_name=self.camera_name)
        filename = self._filename(step_count=step_count, phase=phase)
        self._write_image(self.output_dir / filename, rgb)

    def _filename(self, *, step_count: int, phase: Optional[str]) -> str:
        timestamp = str(time.time_ns())
        phase_part = _safe_token(phase or "step")

        if self.scene_seed is not None or self.question_id is not None or self.interaction_id is not None:
            scene = _safe_token(self.scene_seed)
            qid = _safe_token(self.question_id)
            interaction = _safe_token(self.interaction_id)
            parts = [
                f"scene-{scene}",
                qid,
                interaction,
                f"step-{step_count:06d}",
            ]
            if self.include_phase:
                parts.append(phase_part)
            if self.include_timestamp:
                parts.append(timestamp)
            return "_".join(parts) + ".png"

        parts = [self.prefix, f"{step_count:06d}"]
        if self.include_phase and phase:
            parts.append(phase_part)
        if self.include_timestamp:
            parts.append(timestamp)
        return "_".join(parts) + ".png"

    def _write_image(self, path: Path, rgb):
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, rgb)
        except ImportError:
            from PIL import Image

            Image.fromarray(rgb).save(path)
