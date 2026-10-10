"""Vision perception agent — reads the visual_z signal (YOLOv8m + CLIP ViT-B/32)."""

from __future__ import annotations

from src.agents.base import Observation, PerceptionAgent
from src.agents.context import frame_value
from src.agents.registry import register


@register
class VisionAgent(PerceptionAgent):
    """Perception agent for the ``visual_z`` signal."""

    name        = "vision"
    label       = "Visual (CLIP+YOLO)"
    tier        = "starter"
    description = "Detects physical violence, fights, falls and dangerous motion in the image."
    col         = "visual_z"

    def observe(self, ctx: dict, frame: int) -> Observation:
        """Report the z-score of this agent's column at *frame*.

        Triggered when the z-score is at or above the context threshold.
        """
        z = frame_value(ctx, frame, self.col)
        return Observation(self.name, self.label, z, z >= ctx["threshold"])
